"""
本文件用于生成舆情报表（全局/关键词），并提供报表缓存、历史记录与图表数据聚合能力。
主要类/对象:
- `ReportService`: 报表生成与缓存服务
- `report_service`: 全局服务单例
"""

import gc
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

import numpy as np
from sqlalchemy import and_, case, cast, delete, desc, func, literal, or_, select, true
from sqlalchemy.orm import defer
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import AsyncSessionLocal, check_db_connection
from app.core.config import get_settings
from app.core.exceptions import AIConfigurationError
from app.core.logger import logger
from app.core.prompts import prompt_manager
from app.models.news import News
from app.models.report import ReportCache

settings = get_settings()
from app.services.ai_service import ai_service


class ReportService:
    """
    输入:
    - 数据库中的新闻数据与分析条件（时间/分类/地区/来源等）

    输出:
    - 报表结构化数据与历史缓存记录

    作用:
    - 生成舆情分析报表（全局/关键词），并写入数据库缓存供前端展示
    """

    def __init__(self) -> None:
        self._chart_cache: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
        self._global_cache: Optional[Tuple[float, str, Dict[str, Any]]] = None

    def clear_local_cache(self) -> None:
        """
        清空本地内存缓存
        """
        self._chart_cache.clear()
        self._global_cache = None
        
    def _cleanup_chart_cache(self) -> None:
        """
        清理图表缓存：优先移除过期项，若仍超限则移除最早的项
        """
        now = monotonic()
        # 1. 移除超过 300s 的老旧缓存
        keys_to_remove = [k for k, v in self._chart_cache.items() if now - v[0] > 300]
        for k in keys_to_remove:
            self._chart_cache.pop(k, None)

        # 2. 若仍超限，保留最新的 128 条
        if len(self._chart_cache) > 256:
            sorted_items = sorted(self._chart_cache.items(), key=lambda x: x[1][0])
            self._chart_cache = dict(sorted_items[-128:])

    def _build_news_filters(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Any]:
        filters: List[Any] = []
        if keyword:
            filters.append(News.title.ilike(f"%{keyword}%"))
        if category and category != "all":
            if "," in category:
                filters.append(News.category.in_(category.split(",")))
            else:
                filters.append(News.category == category)
        if region and region != "all":
            selected_regions = region.split(",")
            region_conditions = [News.region.ilike(f"%{r}%") for r in selected_regions]
            filters.append(or_(*region_conditions))
        if source and source != "all":
            if "," in source:
                filters.append(News.source.in_(source.split(",")))
            else:
                filters.append(News.source == source)
        if start_date:
            try:
                s_dt = datetime.strptime(start_date, "%Y-%m-%d")
                filters.append(News.publish_date >= s_dt)
            except Exception:
                pass
        if end_date:
            try:
                e_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                filters.append(News.publish_date < e_dt)
            except Exception:
                pass
        return filters

    async def _get_word_cloud_chart_data(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not await check_db_connection(verbose=False):
            return []

        t0 = monotonic()
        filters = self._build_news_filters(
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=None,
        )

        try:
            jsonb_keywords = case(
                (func.jsonb_typeof(cast(News.keywords, JSONB)) == "array", cast(News.keywords, JSONB)),
                else_=cast(literal("[]"), JSONB),
            )
            kw = func.jsonb_array_elements_text(jsonb_keywords).table_valued("value").alias("kw")

            async with AsyncSessionLocal() as db:
                stmt = (
                    select(kw.c.value.label("name"), func.count().label("value"))
                    .select_from(News)
                    .join(kw, true())
                )
                if filters:
                    stmt = stmt.where(and_(*filters))
                stmt = stmt.where(func.length(func.trim(kw.c.value)) > 0)
                stmt = stmt.where(func.lower(func.trim(kw.c.value)).notin_(["无内容", "null", "空", "none", ""]))
                stmt = stmt.group_by(kw.c.value).order_by(desc("value")).limit(50)

                result = await db.execute(stmt)
                rows = result.all()

            data = [{"name": str(name), "value": int(value)} for name, value in rows]

            elapsed = monotonic() - t0
            if elapsed > 0.5:
                logger.info(
                    f"图表数据(词云)慢查询: {elapsed:.2f}s | 数量={len(data)} | 分类={category or ''} | 范围={start_date or ''}~{end_date or ''}"
                )
            return data
        except Exception:
            async with AsyncSessionLocal() as db:
                stmt = select(News.keywords)
                if filters:
                    stmt = stmt.where(and_(*filters))
                result = await db.execute(stmt)
                keywords_values = result.scalars().all()

            all_keywords: List[str] = []
            for kws in keywords_values:
                if not kws or not isinstance(kws, list):
                    continue
                valid_kws = [
                    k.strip()
                    for k in kws
                    if k
                    and isinstance(k, str)
                    and k.strip()
                    and k.strip().lower() not in {"无内容", "null", "空", "none", ""}
                ]
                all_keywords.extend(valid_kws)

            word_counts = Counter(all_keywords)
            data = [{"name": k, "value": v} for k, v in word_counts.most_common(50)]

            elapsed = monotonic() - t0
            if elapsed > 0.5:
                logger.info(
                    f"图表数据(词云)慢查询: {elapsed:.2f}s | 行数={len(keywords_values)} | 分类={category or ''} | 范围={start_date or ''}~{end_date or ''}"
                )
            return data

    async def _get_source_chart_data(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not await check_db_connection(verbose=False):
            return []

        t0 = monotonic()
        filters = self._build_news_filters(
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=None,
        )

        async with AsyncSessionLocal() as db:
            stmt = select(News.source.label("name"), func.count().label("value")).select_from(News)
            if filters:
                stmt = stmt.where(and_(*filters))
            stmt = stmt.where(func.length(func.trim(News.source)) > 0)
            stmt = stmt.group_by(News.source).order_by(desc("value")).limit(10)
            result = await db.execute(stmt)
            rows = result.all()

        data = [{"name": str(name), "value": int(value)} for name, value in rows]
        elapsed = monotonic() - t0
        if elapsed > 0.5:
            logger.info(
                f"图表数据(来源)慢查询: {elapsed:.2f}s | 数量={len(data)} | 分类={category or ''} | 范围={start_date or ''}~{end_date or ''}"
            )
        return data

    async def _get_sentiment_chart_data(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not await check_db_connection(verbose=False):
             return {"sentiment_dist": [], "neg_keywords": []}

        t0 = monotonic()
        filters = self._build_news_filters(
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=None,
        )

        label_expr = case(
            (News.sentiment_label.in_(["正面", "中立", "负面"]), News.sentiment_label),
            else_=literal("中立"),
        ).label("name")

        try:
            jsonb_keywords = case(
                (func.jsonb_typeof(cast(News.keywords, JSONB)) == "array", cast(News.keywords, JSONB)),
                else_=cast(literal("[]"), JSONB),
            )
            kw = func.jsonb_array_elements_text(jsonb_keywords).table_valued("value").alias("kw")

            async with AsyncSessionLocal() as db:
                stmt_dist = select(label_expr, func.count().label("value")).select_from(News)
                if filters:
                    stmt_dist = stmt_dist.where(and_(*filters))
                stmt_dist = stmt_dist.group_by(label_expr)
                dist_rows = (await db.execute(stmt_dist)).all()

                stmt_neg = select(kw.c.value.label("name"), func.count().label("value")).select_from(News).join(kw, true())
                if filters:
                    stmt_neg = stmt_neg.where(and_(*filters))
                stmt_neg = stmt_neg.where(News.sentiment_label == "负面")
                stmt_neg = stmt_neg.where(func.length(func.trim(kw.c.value)) > 0)
                stmt_neg = stmt_neg.where(func.lower(func.trim(kw.c.value)).notin_(["无内容", "null", "空", "none", ""]))
                stmt_neg = stmt_neg.group_by(kw.c.value).order_by(desc("value")).limit(10)
                neg_rows = (await db.execute(stmt_neg)).all()

            dist_map = {str(name): int(value) for name, value in dist_rows}
            sentiment_dist = [
                {"name": "正面", "value": int(dist_map.get("正面", 0))},
                {"name": "中立", "value": int(dist_map.get("中立", 0))},
                {"name": "负面", "value": int(dist_map.get("负面", 0))},
            ]
            neg_keywords = [str(name) for name, _ in neg_rows]

            data = {"sentiment_dist": sentiment_dist, "neg_keywords": neg_keywords}
            elapsed = monotonic() - t0
            if elapsed > 0.5:
                logger.info(
                    f"图表数据(情感分布)慢查询: {elapsed:.2f}s | negN={len(neg_keywords)} | 分类={category or ''} | 范围={start_date or ''}~{end_date or ''}"
                )
            return data
        except Exception:
            async with AsyncSessionLocal() as db:
                stmt = select(News.sentiment_label, News.keywords).select_from(News)
                if filters:
                    stmt = stmt.where(and_(*filters))
                result = await db.execute(stmt)
                rows = result.all()

            sentiment_counts = {"正面": 0, "中立": 0, "负面": 0}
            negative_keywords: List[str] = []
            for label, kws in rows:
                label_str = label if label in sentiment_counts else "中立"
                sentiment_counts[label_str] += 1
                if label_str == "负面" and kws and isinstance(kws, list):
                    valid_kws = [
                        k.strip()
                        for k in kws
                        if k
                        and isinstance(k, str)
                        and k.strip()
                        and k.strip().lower() not in {"无内容", "null", "空", "none", ""}
                    ]
                    negative_keywords.extend(valid_kws)

            sentiment_dist = [
                {"name": "正面", "value": sentiment_counts["正面"]},
                {"name": "中立", "value": sentiment_counts["中立"]},
                {"name": "负面", "value": sentiment_counts["负面"]},
            ]
            neg_keywords = [k for k, _ in Counter(negative_keywords).most_common(10)]

            elapsed = monotonic() - t0
            if elapsed > 0.5:
                logger.info(
                    f"图表数据(情感)慢查询: {elapsed:.2f}s | 行数={len(rows)} | 分类={category or ''} | 范围={start_date or ''}~{end_date or ''}"
                )
            return {"sentiment_dist": sentiment_dist, "neg_keywords": neg_keywords}

    async def _get_list_chart_data(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if not await check_db_connection(verbose=False):
            return []

        t0 = monotonic()
        filters = self._build_news_filters(
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=None,
        )

        async with AsyncSessionLocal() as db:
            stmt = (
                select(
                    News.id,
                    News.title,
                    News.url,
                    News.source,
                    News.heat_score,
                    News.publish_date,
                    News.summary,
                    News.sources,
                    News.category,
                    News.region,
                    News.sentiment_label,
                    News.sentiment_score,
                )
                .select_from(News)
                .order_by(desc(News.heat_score), desc(News.publish_date))
                .limit(limit)
            )
            if filters:
                stmt = stmt.where(and_(*filters))
            result = await db.execute(stmt)
            rows = result.all()

        data = []
        for (
            nid,
            title,
            url,
            source,
            heat_score,
            publish_date,
            summary,
            sources,
            cat,
            reg,
            sentiment_label,
            sentiment_score,
        ) in rows:
            data.append(
                {
                    "id": int(nid),
                    "title": title,
                    "url": url,
                    "source": source,
                    "heat": heat_score,
                    "time": publish_date.isoformat() if publish_date else None,
                    "summary": summary,
                    "sources": sources,
                    "category": cat,
                    "region": reg,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                }
            )

        elapsed = monotonic() - t0
        if elapsed > 0.5:
            logger.info(
                f"chart-data list 慢查询: {elapsed:.2f}s | items={len(data)} | category={category or ''} | range={start_date or ''}~{end_date or ''}"
            )
        return data

    async def generate_and_cache_global_report(self, period: str = "weekly") -> None:
        """
        输入:
        - `period`: 周期（daily/weekly/monthly）

        输出:
        - 无

        作用:
        - 生成指定周期的全局报表并写入缓存
        """
        
        if not await check_db_connection(verbose=False):
            logger.warning("⚠️ 数据库连接不可用，跳过生成全局报表缓存")
            return

        logger.info(f"📊 开始生成全局大盘报表缓存 ({period})...")
        try:
            today = datetime.now()
            end_date_str = today.strftime("%Y-%m-%d")

            if period == "daily":
                start_date_str = today.strftime("%Y-%m-%d")
                # Add 6 days lookback for trend chart in daily report
                data = await self._generate_analysis_data(
                    keyword="", 
                    start_date=start_date_str, 
                    end_date=end_date_str, 
                    generate_ai=True,
                    trend_lookback_days=6
                )
            elif period == "monthly":
                start_date_str = today.replace(day=1).strftime("%Y-%m-%d")
                data = await self._generate_analysis_data(
                    keyword="", start_date=start_date_str, end_date=end_date_str, generate_ai=True
                )
            else:
                start_date_str = (today - timedelta(days=6)).strftime("%Y-%m-%d")
                data = await self._generate_analysis_data(
                    keyword="", start_date=start_date_str, end_date=end_date_str, generate_ai=True
                )

            await self.save_report_cache("global", period, data)
            logger.info(f"✅ 全局报表缓存 ({period}) 已更新")
        except Exception as e:
            logger.error(f"❌ 生成报表缓存失败: {e}")

    async def save_report_cache(self, r_type: str, keyword: str, data: Dict[str, Any]) -> Optional[int]:
        """
        输入:
        - `r_type`: 报表类型（global/keyword）
        - `keyword`: 关键词或周期标识（global 时用 daily/weekly/monthly）
        - `data`: 报表结构化数据

        输出:
        - 无

        作用:
        - 写入报表缓存，并对数量与同日重复缓存进行控制
        """
        if not await check_db_connection(verbose=False):
            return None

        async with AsyncSessionLocal() as db:
            if r_type == "global":
                today_start = datetime.combine(datetime.now().date(), datetime.min.time())
                await db.execute(
                    delete(ReportCache).where(
                        ReportCache.report_type == "global",
                        ReportCache.keyword == keyword,
                        ReportCache.created_at >= today_start,
                    )
                )

            if r_type == "keyword" and keyword:
                stmt = (
                    select(ReportCache.id)
                    .where(ReportCache.report_type == "keyword", ReportCache.keyword == keyword)
                    .order_by(desc(ReportCache.created_at))
                )
                result = await db.execute(stmt)
                ids = result.scalars().all()

                if len(ids) >= 10:
                    ids_to_delete = ids[9:]
                    await db.execute(delete(ReportCache).where(ReportCache.id.in_(ids_to_delete)))

            cache = ReportCache(report_type=r_type, keyword=keyword, data=data, created_at=datetime.now())
            db.add(cache)
            await db.commit()
            await db.refresh(cache)
            return int(cache.id)
        return None

    async def get_recent_reports(self, limit: int = 10, keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        输入:
        - `limit`: 返回数量上限

        输出:
        - 最近关键词报表列表（按关键词去重）

        作用:
        - 为前端展示最近生成的关键词报表入口
        """
        if not await check_db_connection(verbose=False):
            return []

        async with AsyncSessionLocal() as db:
            stmt = select(ReportCache).where(ReportCache.report_type == "keyword")
            if keyword:
                stmt = stmt.where(ReportCache.keyword == keyword)
            stmt = stmt.order_by(desc(ReportCache.created_at)).limit(limit * 5)
            result = await db.execute(stmt)
            reports = result.scalars().all()

            if keyword:
                return [
                    {
                        "keyword": r.keyword,
                        "date": r.created_at.strftime("%Y-%m-%d %H:%M"),
                        "created_at": r.created_at.isoformat(),
                        "id": r.id,
                    }
                    for r in reports[:limit]
                    if r.keyword
                ]

            seen = set()
            recent = []
            for r in reports:
                if r.keyword and r.keyword not in seen:
                    recent.append(
                        {
                            "keyword": r.keyword,
                            "date": r.created_at.strftime("%Y-%m-%d %H:%M"),
                            "created_at": r.created_at.isoformat(),
                            "id": r.id,
                        }
                    )
                    seen.add(r.keyword)
                    if len(recent) >= limit:
                        break
            return recent

    async def get_report_history(self, keyword: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        输入:
        - `keyword`: 关键词

        输出:
        - 该关键词下的历史报表记录列表

        作用:
        - 支持关键词报表历史回溯
        """
        if not await check_db_connection(verbose=False):
            return []

        async with AsyncSessionLocal() as db:
            stmt = (
                select(ReportCache)
                .where(ReportCache.report_type == "keyword", ReportCache.keyword == keyword)
                .order_by(desc(ReportCache.created_at))
            )
            if limit:
                stmt = stmt.limit(limit)
            result = await db.execute(stmt)
            reports = result.scalars().all()
            return [
                {
                    "id": r.id, 
                    "date": r.created_at.strftime("%Y-%m-%d %H:%M"), 
                    "created_at": r.created_at.isoformat(),
                    "keyword": r.keyword
                }
                for r in reports
            ]

    async def get_global_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        输入:
        - 无

        输出:
        - 全局报表历史记录列表

        作用:
        - 支持全局报表历史回溯
        """
        if not await check_db_connection(verbose=False):
            return []

        async with AsyncSessionLocal() as db:
            stmt = (
                select(ReportCache)
                .where(ReportCache.report_type == "global")
                .order_by(desc(ReportCache.created_at))
            )
            stmt = stmt.limit(limit or 100)
            result = await db.execute(stmt)
            reports = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "date": r.created_at.strftime("%Y-%m-%d %H:%M"),
                    "created_at": r.created_at.isoformat(),
                    "type": r.keyword or "weekly",
                }
                for r in reports
            ]

    async def delete_report_cache(self, report_id: int) -> bool:
        if not await check_db_connection(verbose=False):
            return False
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ReportCache).where(ReportCache.id == report_id))
            cached = result.scalar_one_or_none()
            if not cached:
                return False
            await db.delete(cached)
            await db.commit()
            return True

    async def find_latest_report_id(self, keyword: str = "") -> Optional[int]:
        if not await check_db_connection(verbose=False):
            return None
        async with AsyncSessionLocal() as db:
            if keyword:
                stmt = select(ReportCache.id).where(ReportCache.report_type == "keyword", ReportCache.keyword == keyword)
            else:
                stmt = select(ReportCache.id).where(ReportCache.report_type == "global").order_by(desc(ReportCache.created_at))
                # Just take the latest global report
            
            stmt = stmt.order_by(desc(ReportCache.created_at)).limit(1)
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

    async def stream_ai_analysis(self, report_id: int) -> AsyncIterator[str]:
        logger.info(f"🔄 stream_ai_analysis 进入: report_id={report_id}")
        if not await check_db_connection(verbose=False):
            logger.error(f"❌ DB 连接失败: report_id={report_id}")
            yield "数据库连接失败"
            return

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ReportCache).where(ReportCache.id == report_id))
            cached = result.scalar_one_or_none()
            if not cached:
                logger.warning(f"⚠️ 报表未找到: report_id={report_id}")
                yield "报表未找到"
                return

            data = dict(cached.data or {})
            
            # If already done or has content (legacy support), yield result directly
            ai_status = data.get("ai_status")
            ai_analysis = data.get("ai_analysis")
            logger.info(f"ℹ️ 报表状态: id={report_id} status={ai_status} len={len(ai_analysis or '')}")
            
            if (ai_status == "done" or (ai_analysis and len(ai_analysis) > 100)) and ai_analysis:
                # Update status if missing
                if ai_status != "done":
                    data["ai_status"] = "done"
                    cached.data = data
                    await db.commit()
                yield ai_analysis
                return

            # Prepare context
            params = dict(data.get("params") or {})
            keyword = (params.get("keyword") or cached.keyword or "") if cached.report_type == "keyword" else ""
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            category = params.get("category")
            region = params.get("region")
            source = params.get("source")

            scope_parts = []
            if keyword:
                scope_parts.append(f"关键词={keyword}")
            if category and category != "all":
                scope_parts.append(f"领域={category}")
            if region and region != "all":
                scope_parts.append(f"地区={region}")
            if source and source != "all":
                scope_parts.append(f"来源={source}")
            scope_str = "；".join(scope_parts) if scope_parts else "全量样本"

            ai_start = start_date or (datetime.now().date() - timedelta(days=0)).strftime("%Y-%m-%d")
            ai_end = end_date or datetime.now().date().strftime("%Y-%m-%d")
            time_range_label = f"{ai_start} 至 {ai_end}" if ai_start != ai_end else ai_end

            ai_filters = self._build_news_filters(
                keyword=keyword,
                start_date=ai_start,
                end_date=ai_end,
                category=category,
                region=region,
                source=source,
            )

            ai_stmt = (
                select(
                    News.title,
                    func.substr(News.content, 1, 1000).label("content"),
                    News.summary,
                    News.heat_score,
                    News.source,
                    News.url,
                    News.publish_date,
                )
                .select_from(News)
                .order_by(desc(News.heat_score), desc(News.publish_date))
                .limit(100)
            )
            if ai_filters:
                ai_stmt = ai_stmt.where(and_(*ai_filters))
            ai_result = await db.execute(ai_stmt)
            ai_rows = ai_result.all()

            if not ai_rows:
                yield "未找到相关数据，无法进行分析。"
                data["ai_status"] = "done"
                data["ai_analysis"] = "未找到相关数据，无法进行分析。"
                cached.data = data
                await db.commit()
                return

            def compact_text(text: Any, max_len: int = 180) -> str:
                t = (text or "").replace("\r", " ").replace("\n", " ").strip()
                if len(t) > max_len:
                    return t[:max_len] + "…"
                return t

            news_lines = []
            for idx, (title, content, summary, heat, src, url, pub_dt) in enumerate(ai_rows, start=1):
                body = content if content else summary
                body_str = compact_text(body, 220)
                title_str = compact_text(title, 80)
                src_str = compact_text(src, 30)
                time_str = pub_dt.isoformat() if pub_dt else ""
                news_lines.append(
                    f"{idx}. [热度:{(heat or 0):.1f}] [{src_str}] {title_str}\n   时间: {time_str}\n   正文: {body_str}\n   链接: {url}"
                )

            # Build prompt
            if keyword:
                prompt = prompt_manager.get_user_prompt(
                    "report_keyword_analysis",
                    keyword=keyword,
                    time_range_label=time_range_label,
                    scope_str=scope_str,
                    news_lines="\n\n".join(news_lines)
                )
            else:
                prompt = prompt_manager.get_user_prompt(
                    "report_global_analysis",
                    time_range_label=time_range_label,
                    scope_str=scope_str,
                    news_lines="\n\n".join(news_lines)
                )

            # Mark as running
            data["ai_status"] = "running"
            cached.data = data
            await db.commit()

            # Stream
            full_text = ""
            last_flush = monotonic()
            
            try:
                async for chunk in ai_service.stream_completion(prompt, route_key="REPORT"):
                    full_text += chunk
                    yield chunk

                    if len(full_text) > 100000:
                        yield "\n[系统提示: 生成内容过长，已截断]"
                        break
                    
                    # Incremental update
                    now = monotonic()
                    if (now - last_flush) >= 2.0:
                        data["ai_analysis"] = full_text
                        cached.data = dict(data)
                        await db.commit()
                        last_flush = now
                
                # Update DB with full result
                data["ai_analysis"] = full_text
                data["ai_status"] = "done"
                cached.data = dict(data)
                await db.commit()

            except Exception as e:
                logger.error(f"AI Stream failed: {e}")
                yield f"\n[AI 生成失败: {e}]"
                data["ai_status"] = "error"
                cached.data = dict(data)
                await db.commit()
            finally:
                try:
                    # Ensure status is not running
                    await db.refresh(cached)
                    current_data = dict(cached.data or {})
                    if current_data.get("ai_status") == "running":
                        current_data["ai_status"] = "done"
                        if full_text:
                            current_data["ai_analysis"] = full_text
                        cached.data = current_data
                        await db.commit()
                except Exception as e:
                    logger.error(f"Error in finally block: {e}")

    async def generate_report_and_stream_ai(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        logger.info(f"📄 后台生成报表开始: keyword={keyword or '-'}")
        data = await self._generate_analysis_data(
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            limit=limit,
            generate_ai=False,
        )

        news_count = data.get("summary", {}).get("total", 0)
        if news_count == 0:
            data["ai_analysis"] = "未找到相关新闻，无法进行 AI 综述。"
            data["ai_status"] = "done"
            logger.info(f"📄 无新闻，跳过 AI 综述: keyword={keyword or '-'}")
        else:
            data["ai_analysis"] = ""
            data["ai_status"] = "pending"

        if keyword:
            report_id = await self.save_report_cache("keyword", keyword, data)
        else:
            report_id = await self.save_report_cache("global", "weekly", data)

        if not report_id:
            logger.warning("⚠️ 报表缓存写入失败，后台任务结束")
            return

        logger.info(f"📄 报表缓存已写入: id={report_id} keyword={keyword or '-'}")
        
        # Only start background analysis if we have news AND it's not a keyword report (which streams on-demand)
        # OR if we want to support background generation for all.
        # User feedback: "Frontend sees generating, but needs refresh".
        # This means frontend streaming is not working or not connecting to the background stream.
        # To fix this, we disable background generation for keyword reports (interactive),
        # so the frontend triggers it via stream_ai_analysis endpoint.
        # But wait, if we disable it here, the status is 'pending'.
        # The frontend sees 'pending' and calls streamAiAnalysis.
        # stream_ai_analysis endpoint needs to handle 'pending' by STARTING generation.
        
        if news_count > 0 and not keyword:
            # For global reports (usually scheduled or slow), run in background?
            # Or just disable background for all interactive generation?
            # Let's disable for all interactive for now to ensure streaming works.
            # But "generate_report_and_stream_ai" implies it MIGHT be background.
            # If we comment it out, the frontend MUST be open to generate.
            # Let's try commenting it out as requested to fix the "stuck" issue.
            # await self._stream_ai_analysis_to_cache(report_id)
            pass
        
        # Actually, if I comment it out, existing logic in stream_ai_analysis (endpoint) needs to pick it up.
        # Let's check stream_ai_analysis implementation again.
        
        logger.info(f"📄 后台生成报表结束: id={report_id} keyword={keyword or '-'}")

    async def _stream_ai_analysis_to_cache(self, report_id: int) -> None:
        if not await check_db_connection(verbose=False):
            return

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ReportCache).where(ReportCache.id == report_id))
            cached = result.scalar_one_or_none()
            if not cached:
                return

            data = dict(cached.data or {})
            params = dict(data.get("params") or {})
            keyword = (params.get("keyword") or cached.keyword or "") if cached.report_type == "keyword" else ""
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            category = params.get("category")
            region = params.get("region")
            source = params.get("source")

            logger.info(f"🤖 AI 综述生成开始: report_id={report_id} keyword={keyword or '-'}")

            scope_parts = []
            if keyword:
                scope_parts.append(f"关键词={keyword}")
            if category and category != "all":
                scope_parts.append(f"领域={category}")
            if region and region != "all":
                scope_parts.append(f"地区={region}")
            if source and source != "all":
                scope_parts.append(f"来源={source}")
            scope_str = "；".join(scope_parts) if scope_parts else "全量样本"

            ai_start = start_date or (datetime.now().date() - timedelta(days=0)).strftime("%Y-%m-%d")
            ai_end = end_date or datetime.now().date().strftime("%Y-%m-%d")
            time_range_label = f"{ai_start} 至 {ai_end}" if ai_start != ai_end else ai_end

            ai_filters = self._build_news_filters(
                keyword=keyword,
                start_date=ai_start,
                end_date=ai_end,
                category=category,
                region=region,
                source=source,
            )

            ai_stmt = (
                select(
                    News.title,
                    func.substr(News.content, 1, 1000).label("content"),
                    News.summary,
                    News.heat_score,
                    News.source,
                    News.url,
                    News.publish_date,
                )
                .select_from(News)
                .order_by(desc(News.heat_score), desc(News.publish_date))
                .limit(100)
            )
            if ai_filters:
                ai_stmt = ai_stmt.where(and_(*ai_filters))
            ai_result = await db.execute(ai_stmt)
            ai_rows = ai_result.all()

            def compact_text(text: Any, max_len: int = 180) -> str:
                t = (text or "").replace("\r", " ").replace("\n", " ").strip()
                if len(t) > max_len:
                    return t[:max_len] + "…"
                return t

            news_lines = []
            for idx, (title, content, summary, heat, src, url, pub_dt) in enumerate(ai_rows, start=1):
                body = content if content else summary
                body_str = compact_text(body, 220)
                title_str = compact_text(title, 80)
                src_str = compact_text(src, 30)
                time_str = pub_dt.isoformat() if pub_dt else ""
                news_lines.append(
                    f"{idx}. [热度:{(heat or 0):.1f}] [{src_str}] {title_str}\n   时间: {time_str}\n   正文: {body_str}\n   链接: {url}"
                )

            if keyword:
                prompt = prompt_manager.get_user_prompt(
                    "report_keyword_analysis",
                    keyword=keyword,
                    time_range_label=time_range_label,
                    scope_str=scope_str,
                    news_lines="\n\n".join(news_lines) if news_lines else "无可用新闻样本",
                )
            else:
                prompt = prompt_manager.get_user_prompt(
                    "report_global_analysis",
                    time_range_label=time_range_label,
                    scope_str=scope_str,
                    news_lines="\n\n".join(news_lines) if news_lines else "无可用新闻样本",
                )

            data["ai_status"] = "running"
            data["ai_analysis"] = ""
            cached.data = data
            await db.commit()

            full_text = ""
            last_flush = monotonic()
            last_len = 0
            try:
                logger.info(f"DEBUG: Entering stream loop for report {report_id}")
                async for chunk in ai_service.stream_completion(prompt, route_key="REPORT"):
                    # logger.info(f"DEBUG: Got chunk len={len(chunk)} content={chunk[:20]!r}")
                    full_text += chunk
                    
                    if len(full_text) > 100000:
                         logger.warning(f"⚠️ Report {report_id} AI output too long, truncated.")
                         full_text += "\n[截断]"
                         break

                    now = monotonic()
                    if (now - last_flush) >= 1.5 and (len(full_text) - last_len) >= 120:
                        data["ai_analysis"] = full_text
                        cached.data = dict(data)
                        await db.commit()
                        last_flush = now
                        last_len = len(full_text)

                if not full_text.strip():
                    logger.warning(f"⚠️ AI 流式返回为空，尝试降级为非流式生成: report_id={report_id}")
                    fallback = await ai_service.chat_completion(prompt, route_key="REPORT")
                    full_text = (fallback or "").strip()

                # Create a new dict to ensure SQLAlchemy detects the change
                final_data = dict(data)
                final_data["ai_analysis"] = full_text.strip()
                final_data["ai_status"] = "done" if final_data["ai_analysis"] else "error"
                cached.data = final_data
                await db.commit()
            except AIConfigurationError as e:
                logger.warning(f"⚠️ AI 配置不可用: {e}")
                data["ai_analysis"] = str(e)
                data["ai_status"] = "error"
                cached.data = data
                await db.commit()
            except Exception as e:
                logger.error(f"AI分析失败: {e}")
                data["ai_analysis"] = (full_text.strip() or "AI 分析失败").strip()
                data["ai_status"] = "error"
                cached.data = data
                await db.commit()
            finally:
                try:
                    await db.refresh(cached)
                    final_data = dict(cached.data or {})
                    final_status = str(final_data.get("ai_status") or "")
                    final_text = str(final_data.get("ai_analysis") or "")
                    logger.info(
                        f"🤖 AI 综述生成结束: report_id={report_id} status={final_status or '-'} chars={len(final_text)}"
                    )
                except Exception:
                    pass

    async def load_report(self, report_id: int) -> Optional[Dict[str, Any]]:
        """
        输入:
        - `report_id`: 报表缓存 ID

        输出:
        - 报表结构化数据；不存在返回 None

        作用:
        - 读取指定历史报表缓存供前端渲染
        """
        if not await check_db_connection(verbose=False):
            return None

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ReportCache).where(ReportCache.id == report_id))
            cached = result.scalar_one_or_none()
            if cached:
                data = dict(cached.data)
                data["id"] = cached.id
                if cached.report_type == "global":
                    data["period"] = cached.keyword or "weekly"
                return data
            return None


    async def get_analysis_data(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        generate_ai: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        输入:
        - `keyword`: 关键词（为空代表全局）
        - `start_date`/`end_date`: 起止日期（YYYY-MM-DD）
        - `category`/`region`/`source`: 过滤条件
        - `limit`: 取样上限
        - `generate_ai`: 是否生成 AI 文本分析
        - `use_cache`: 是否优先读取缓存

        输出:
        - 报表结构化数据（summary/charts/top_news/ai_analysis）

        作用:
        - 对外提供统一报表数据入口，并在条件允许时命中缓存提升性能
        """

        if use_cache and not keyword and not start_date and not end_date and not category and not region and not source:
            now_ts = monotonic()
            if self._global_cache and (now_ts - self._global_cache[0]) <= 15:
                return self._global_cache[2]
            
            if await check_db_connection(verbose=False):
                async with AsyncSessionLocal() as db:
                    t0 = monotonic()
                    stmt = (
                        select(ReportCache.id, ReportCache.keyword, ReportCache.data)
                        .where(ReportCache.report_type == "global", ReportCache.keyword == "daily")
                        .order_by(desc(ReportCache.created_at))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    row = result.first()

                    if not row:
                        stmt = (
                            select(ReportCache.id, ReportCache.keyword, ReportCache.data)
                            .where(ReportCache.report_type == "global")
                            .order_by(desc(ReportCache.created_at))
                            .limit(1)
                        )
                        result = await db.execute(stmt)
                        row = result.first()

                    if row:
                        rid, kw, data = row
                        data = dict(data)
                        data["id"] = rid
                        elapsed_ms = int((monotonic() - t0) * 1000)
                        logger.info(f"📖 读取全局报表数据库缓存 ({kw or '未知'}) {elapsed_ms}ms")
                        self._global_cache = (monotonic(), str(kw or ""), data)
                        return data

        data = await self._generate_analysis_data(keyword, start_date, end_date, category, region, source, limit, generate_ai)

        if keyword:
            report_id = await self.save_report_cache("keyword", keyword, data)
            if report_id:
                data["id"] = report_id

        return data

    async def _generate_analysis_data(
        self,
        keyword: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        generate_ai: bool = False,
        trend_lookback_days: int = 0, # New parameter
    ) -> Dict[str, Any]:
        """
        输入:
        - `keyword`: 关键词过滤
        - `start_date`/`end_date`: 起止日期过滤
        - `category`/`region`/`source`: 维度过滤
        - `limit`: 数据量上限
        - `generate_ai`: 是否生成 AI 分析文字

        输出:
        - 报表结构化数据（摘要、趋势、来源、词云、情绪分布、Top 新闻等）

        作用:
        - 在数据库中按条件查询新闻，并聚合计算各类统计指标
        """
        
        # 统一返回空数据的闭包函数
        def empty_result(msg: str = "数据库连接不可用"):
             return {
                "params": {
                    "keyword": keyword,
                    "start_date": start_date,
                    "end_date": end_date,
                    "category": category,
                    "region": region,
                    "source": source,
                    "limit": limit
                },
                "summary": {"total": 0, "avg_heat": 0, "max_heat": 0, "sentiment_idx": 0, "risk_count": 0, "time_range": "-"},
                    "charts": {
                        "trend": {},
                        "source": {},
                        "word_cloud": [],
                        "sentiment_dist": [],
                        "neg_keywords": [],
                        "correlation": [],
                        "freq_trend": [],
                        "sentiment_trend": [],
                    },
                    "top_news": [],
                    "ai_analysis": f"{msg}，无法进行分析。",
                }

        if not await check_db_connection(verbose=False):
            return empty_result()

        async with AsyncSessionLocal() as db:
            # 优化：延迟加载大字段（content, embedding），显著降低内存占用
            stmt = select(News).options(
                defer(News.content),
                defer(News.embedding)
            )
            filters = []

            if keyword:
                filters.append(News.title.ilike(f"%{keyword}%"))
            if category and category != "all":
                if "," in category:
                    filters.append(News.category.in_(category.split(",")))
                else:
                    filters.append(News.category == category)
            if region and region != "all":
                selected_regions = region.split(",")
                region_conditions = [News.region.ilike(f"%{r}%") for r in selected_regions]
                filters.append(or_(*region_conditions))
            if source and source != "all":
                if "," in source:
                    filters.append(News.source.in_(source.split(",")))
                else:
                    filters.append(News.source == source)
            if start_date:
                try:
                    s_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    filters.append(News.publish_date >= s_dt)
                except Exception:
                    pass
            if end_date:
                try:
                    e_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                    filters.append(News.publish_date < e_dt)
                except Exception:
                    pass

            count_stmt = select(func.count()).select_from(News)
            if filters:
                # stmt = stmt.where(and_(*filters))  <-- REMOVED: stmt is not defined yet for charts
                count_stmt = count_stmt.where(and_(*filters))

            real_total_count = await db.scalar(count_stmt) or 0
            
            if real_total_count == 0:
                return empty_result(msg="未找到相关数据")

            # -------------------------------------------------------------------------
            # 2. 统计指标 (聚合查询，不加载对象)
            # -------------------------------------------------------------------------
            stats_stmt = select(
                func.avg(News.heat_score),
                func.max(News.heat_score),
                func.min(News.publish_date),
                func.max(News.publish_date)
            )
            if filters:
                stats_stmt = stats_stmt.where(and_(*filters))
            
            stats_res = await db.execute(stats_stmt)
            avg_heat, max_heat, min_date, max_date = stats_res.one()
            avg_heat = float(avg_heat or 0)
            max_heat = float(max_heat or 0)
            time_range = f"{(min_date or datetime.now()).strftime('%Y-%m-%d')} ~ {(max_date or datetime.now()).strftime('%Y-%m-%d')}"

            # -------------------------------------------------------------------------
            # 3. 情感分布 (聚合查询)
            # -------------------------------------------------------------------------
            sentiment_stmt = (
                select(News.sentiment_label, func.count(), func.avg(News.sentiment_score))
                .group_by(News.sentiment_label)
            )
            if filters:
                sentiment_stmt = sentiment_stmt.where(and_(*filters))
            
            sent_res = await db.execute(sentiment_stmt)
            sent_rows = sent_res.all()
            
            sentiment_counts = {"正面": 0, "中立": 0, "负面": 0}
            total_score_sum = 0.0
            total_score_count = 0
            
            for label, count, avg_score in sent_rows:
                l_str = label if label in sentiment_counts else "中立"
                sentiment_counts[l_str] += count
                if avg_score is not None:
                    total_score_sum += float(avg_score) * count
                    total_score_count += count

            sentiment_idx = (total_score_sum / total_score_count) if total_score_count else 50.0
            risk_count = sentiment_counts["负面"]

            sentiment_dist = [
                {"name": "正面", "value": sentiment_counts["正面"]},
                {"name": "中立", "value": sentiment_counts["中立"]},
                {"name": "负面", "value": sentiment_counts["负面"]},
            ]

            # -------------------------------------------------------------------------
            # 4. 来源分布 (Top 10 聚合)
            # -------------------------------------------------------------------------
            source_stmt = (
                select(News.source, func.count().label("cnt"))
                .group_by(News.source)
                .order_by(desc("cnt"))
                .limit(10)
            )
            if filters:
                source_stmt = source_stmt.where(and_(*filters))
            
            source_res = await db.execute(source_stmt)
            chart_source = [{"name": str(src), "value": cnt} for src, cnt in source_res.all() if src]

            # -------------------------------------------------------------------------
            # 5. 趋势图 (聚合查询)
            # -------------------------------------------------------------------------
            # 如果有 trend_lookback_days，我们需要扩展时间范围查询趋势
            # 但为了简化，我们先用 filters 的范围，如果需要扩展，需重新构建 filters
            
            trend_map = defaultdict(lambda: {"count": 0, "heat_sum": 0.0, "categories": Counter()})
            
            # 使用 cast(News.publish_date, Date) 在某些 DB 可能不兼容，但在 PG/SQLite 通常可行
            # 或者直接取 publish_date 并在 Python 端截取日期（聚合后的数据量较小）
            # 这里为了通用性，我们按天聚合
            # 注意：SQLite 不支持 cast(..., Date) 同样语法，Postgres 支持
            # 我们假设环境是 Postgres (TrendSonar 看起来像) 或者兼容
            
            # 尝试使用 func.date_trunc('day', News.publish_date) for PG, or just fetch date
            # 安全起见，我们拉取 (date, category, heat, count) 聚合
            # 但 func.date(...) 依赖方言。
            # 既然已经优化了，我们可以稍微拉取多一点数据：(publish_date, category, heat_score) 
            # 但还是不拉取全量。
            
            # 更好方案：按天分组
            # 针对不同数据库，日期截断写法不同。为了兼容性，我们可以在 Python 做日期聚合，
            # 但只拉取 (publish_date, heat_score, category)，不拉取其他字段。
            
            # 确定趋势查询的时间范围
            trend_filters = list(filters)
            if trend_lookback_days > 0 and start_date:
                # 移除原有的日期 filter，添加新的
                trend_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=trend_lookback_days)).strftime("%Y-%m-%d")
                # 重新构建时间 filter (比较麻烦，因为 filters 是 list)
                # 简单起见，我们单独构建 trend_stmt
                trend_range_filters = self._build_news_filters(
                     keyword=keyword,
                     start_date=trend_start,
                     end_date=end_date,
                     category=category,
                     region=region,
                     source=source
                )
                trend_filters = trend_range_filters

            trend_stmt = select(
                News.publish_date, 
                News.heat_score, 
                News.category
            ).order_by(News.publish_date)
            
            if trend_filters:
                trend_stmt = trend_stmt.where(and_(*trend_filters))
                
            # 这里如果不做 SQL group by，数据量可能还是很大 (比如 10万条记录的日期和热度)
            # 10万条 (date, float, str) 大概占用 100,000 * (10+8+10) bytes ≈ 3MB，完全可接受。
            # 比加载 10万个对象 (几百 MB) 小得多。
            
            trend_res = await db.execute(trend_stmt)
            trend_rows = trend_res.all()
            
            for pub_date, heat, cat in trend_rows:
                if not pub_date: continue
                d_str = pub_date.strftime("%Y-%m-%d")
                trend_map[d_str]["count"] += 1
                trend_map[d_str]["heat_sum"] += (heat or 0.0)
                trend_map[d_str]["categories"][cat or "其他"] += 1

            sorted_dates = sorted(trend_map.keys())
            category_names = set()
            for d in sorted_dates:
                category_names.update(trend_map[d]["categories"].keys())
            category_series = [
                {"name": cat, "type": "bar", "stack": "文章数", "data": [trend_map[d]["categories"][cat] for d in sorted_dates]}
                for cat in sorted(category_names)
            ]

            chart_trend = {
                "dates": sorted_dates,
                "counts": [trend_map[d]["count"] for d in sorted_dates],
                "avg_heats": [
                    (trend_map[d]["heat_sum"] / trend_map[d]["count"]) if trend_map[d]["count"] else 0 for d in sorted_dates
                ],
                "category_series": category_series,
            }

            # -------------------------------------------------------------------------
            # 6. 关键词/共现/每日情感 (采样 Top 2000)
            # -------------------------------------------------------------------------
            # 为了计算词云、共现和每日情感趋势（需要关键词和情感明细），我们只取热度最高的 2000 条
            sample_limit = 2000
            sample_stmt = (
                select(News.keywords, News.sentiment_label, News.sentiment_score, News.publish_date)
                .order_by(desc(News.heat_score))
                .limit(sample_limit)
            )
            if filters:
                sample_stmt = sample_stmt.where(and_(*filters))
                
            sample_res = await db.execute(sample_stmt)
            sample_rows = sample_res.all()

            all_keywords = []
            negative_keywords = []
            co_occurrence = defaultdict(int)
            keyword_freq = defaultdict(int)
            daily_kw_freq = defaultdict(Counter)
            # 重新计算 daily_sentiment 用于趋势图 (基于采样，或者我们可以再做一个 SQL 聚合，
            # 但为了趋势图的平滑性，采样 Top 2000 可能也够了，不过为了准确，最好用 SQL)
            
            # 让我们用 SQL 聚合每日情感，以保证准确性 (step 3 已经是聚合了，但那是总的，我们需要每日的)
            # daily_sentiment_stmt = select(date, label, count, sum(score))...
            # 为了避免复杂 SQL，这里先用采样数据近似每日情感趋势，或者接受采样误差。
            # 考虑到性能，采样 2000 条热点新闻的情感趋势通常能代表整体趋势。
            
            daily_sentiment = defaultdict(lambda: {"正面": 0, "中立": 0, "负面": 0, "score_sum": 0.0, "score_count": 0})

            for kws, label, score, pub_date in sample_rows:
                if not pub_date: continue
                day = pub_date.strftime("%Y-%m-%d")
                
                # 统计每日情感 (Sampled)
                l_str = label if label in ["正面", "中立", "负面"] else "中立"
                daily_sentiment[day][l_str] += 1
                if score is not None:
                    daily_sentiment[day]["score_sum"] += float(score)
                    daily_sentiment[day]["score_count"] += 1
                
                # 统计关键词
                if kws and isinstance(kws, list):
                    valid_kws = [
                        k.strip() for k in kws
                        if k and k.strip() and k.strip().lower() not in {"无内容", "null", "空", "none", ""}
                    ]
                    all_keywords.extend(valid_kws)
                    if l_str == "负面":
                        negative_keywords.extend(valid_kws)
                    
                    unique_kws = list(set(valid_kws))
                    daily_kw_freq[day].update(unique_kws)
                    for kw in unique_kws:
                        keyword_freq[kw] += 1
                    for i in range(len(unique_kws)):
                        for j in range(i + 1, len(unique_kws)):
                            pair = tuple(sorted((unique_kws[i], unique_kws[j])))
                            co_occurrence[pair] += 1

            word_counts = Counter(all_keywords)
            word_cloud = [{"name": k, "value": v} for k, v in word_counts.most_common(50)]
            neg_keywords = [k for k, _ in Counter(negative_keywords).most_common(10)]

            top_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:80]
            node_names = set()
            links = []
            for (a, b), cnt in top_pairs:
                node_names.add(a)
                node_names.add(b)
                links.append({"source": a, "target": b, "value": cnt})
            nodes = [
                {"name": kw, "value": int(keyword_freq.get(kw, 0)), "symbolSize": min(60, 10 + int(keyword_freq.get(kw, 0)) * 2)}
                for kw in node_names
            ]
            correlation = {"nodes": nodes, "links": links}

            top_keywords = [k for k, _ in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:8]]
            freq_trend = {
                "dates": sorted_dates,
                "series": [{"name": kw, "type": "line", "smooth": True, "showSymbol": False, "data": [daily_kw_freq[d][kw] for d in sorted_dates]} for kw in top_keywords],
            }

            # 修正每日情感趋势数据 (对齐 sorted_dates)
            sentiment_trend = {
                "dates": sorted_dates,
                "series": [
                    {"name": "正面", "type": "bar", "stack": "情感", "data": [daily_sentiment[d]["正面"] for d in sorted_dates]},
                    {"name": "中立", "type": "bar", "stack": "情感", "data": [daily_sentiment[d]["中立"] for d in sorted_dates]},
                    {"name": "负面", "type": "bar", "stack": "情感", "data": [daily_sentiment[d]["负面"] for d in sorted_dates]},
                    {
                        "name": "平均情感分",
                        "type": "line",
                        "yAxisIndex": 1,
                        "smooth": True,
                        "showSymbol": False,
                        "data": [
                            (daily_sentiment[d]["score_sum"] / daily_sentiment[d]["score_count"]) if daily_sentiment[d]["score_count"] else 50
                            for d in sorted_dates
                        ],
                    },
                ],
            }

            # -------------------------------------------------------------------------
            # 7. Top News (Limit 10)
            # -------------------------------------------------------------------------
            top_stmt = (
                select(News)
                .options(defer(News.content), defer(News.embedding))
                .order_by(desc(News.heat_score))
                .limit(10)
            )
            if filters:
                top_stmt = top_stmt.where(and_(*filters))
            
            top_res = await db.execute(top_stmt)
            top_news_list = top_res.scalars().all()
            
            top_news = []
            for n in top_news_list:
                top_news.append(
                    {
                        "id": n.id,
                        "title": n.title,
                        "url": n.url,
                        "source": n.source,
                        "heat": n.heat_score,
                        "time": n.publish_date.isoformat() if n.publish_date else "",
                        "summary": n.summary,
                        "sources": n.sources,
                        "category": n.category,
                        "region": n.region,
                        "sentiment_label": n.sentiment_label,
                        "sentiment_score": n.sentiment_score,
                    }
                )



            ai_analysis = ""
            if generate_ai:
                try:
                    scope_parts = []
                    if keyword:
                        scope_parts.append(f"关键词={keyword}")
                    if category and category != "all":
                        scope_parts.append(f"领域={category}")
                    if region and region != "all":
                        scope_parts.append(f"地区={region}")
                    if source and source != "all":
                        scope_parts.append(f"来源={source}")
                    scope_str = "；".join(scope_parts) if scope_parts else "全量样本"

                    # AI 分析的时间范围 logic optimization
                    ai_start = start_date if start_date else (max_date.date() - timedelta(days=0)).strftime("%Y-%m-%d")
                    ai_end = end_date if end_date else max_date.date().strftime("%Y-%m-%d")
                    
                    time_range_label = f"{ai_start} 至 {ai_end}" if ai_start != ai_end else ai_end

                    ai_filters = self._build_news_filters(
                        keyword=keyword,
                        start_date=ai_start,
                        end_date=ai_end,
                        category=category,
                        region=region,
                        source=source,
                    )

                    ai_stmt = (
                        select(
                            News.title,
                            func.substr(News.content, 1, 1000).label("content"),
                            News.summary,
                            News.heat_score,
                            News.source,
                            News.url,
                            News.publish_date,
                        )
                        .select_from(News)
                        .order_by(desc(News.heat_score), desc(News.publish_date))
                        .limit(100)
                    )
                    if ai_filters:
                        ai_stmt = ai_stmt.where(and_(*ai_filters))

                    ai_result = await db.execute(ai_stmt)
                    ai_rows = ai_result.all()

                    def compact_text(text: Any, max_len: int = 180) -> str:
                        t = (text or "").replace("\r", " ").replace("\n", " ").strip()
                        if len(t) > max_len:
                            return t[:max_len] + "…"
                        return t

                    news_lines = []
                    for idx, (title, content, summary, heat, src, url, pub_dt) in enumerate(ai_rows, start=1):
                        body = content if content else summary
                        body_str = compact_text(body, 220)
                        title_str = compact_text(title, 80)
                        src_str = compact_text(src, 30)
                        time_str = pub_dt.isoformat() if pub_dt else ""
                        news_lines.append(
                            f"{idx}. [热度:{(heat or 0):.1f}] [{src_str}] {title_str}\n   时间: {time_str}\n   正文: {body_str}\n   链接: {url}"
                        )

                    # --- AI 提示词分流逻辑 ---
                    # 场景1：关键词深度分析 (Keyword Depth Analysis)
                    if keyword:
                        prompt = prompt_manager.get_user_prompt(
                            "report_keyword_analysis",
                            keyword=keyword,
                            time_range_label=time_range_label,
                            scope_str=scope_str,
                            news_lines="\n\n".join(news_lines) if news_lines else "无可用新闻样本"
                        )
                    # 场景2：全局/大盘综述 (Global Overview)
                    else:
                        prompt = prompt_manager.get_user_prompt(
                            "report_global_analysis",
                            time_range_label=time_range_label,
                            scope_str=scope_str,
                            news_lines="\n\n".join(news_lines) if news_lines else "无可用新闻样本"
                        )

                    ai_analysis = await ai_service.chat_completion(prompt, route_key="REPORT")
                except Exception as e:
                    logger.error(f"AI分析失败: {e}")
                    ai_analysis = "AI 分析失败"
            else:
                ai_analysis = "未开启 AI 分析"

            result_data = {
                "params": {
                    "keyword": keyword,
                    "start_date": start_date,
                    "end_date": end_date,
                    "category": category,
                    "region": region,
                    "source": source,
                    "limit": limit
                },
                "summary": {
                    "total": int(real_total_count or 0),
                    "avg_heat": float(avg_heat),
                    "max_heat": float(max_heat),
                    "sentiment_idx": float(sentiment_idx),
                    "risk_count": int(risk_count),
                    "time_range": time_range,
                },
                "charts": {
                    "trend": chart_trend,
                    "source": chart_source,
                    "word_cloud": word_cloud,
                    "sentiment_dist": sentiment_dist,
                    "neg_keywords": neg_keywords,
                    "correlation": correlation,
                    "freq_trend": freq_trend,
                    "sentiment_trend": sentiment_trend,
                },
                "top_news": top_news,
                "ai_analysis": ai_analysis,
            }
            
            # 显式释放内存，防止大量 News 对象滞留
            if 'news_list' in locals():
                del news_list
            gc.collect()
            
            return result_data

    async def get_chart_data(
        self,
        type: str,
        category: Optional[str] = None,
        region: Optional[str] = None,
        q: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Any:
        """
        输入:
        - `type`: 图表类型
        - `category`/`region`: 过滤条件
        - `q`: 关键词
        - `start_date`/`end_date`: 起止日期

        输出:
        - 对应图表的数据结构

        作用:
        - 为前端按需刷新单一图表提供数据裁剪
        """

        if type == "source":
            cache_key = ("source", q or "", start_date or "", end_date or "", category or "", region or "")
            cached = self._chart_cache.get(cache_key)
            if cached and (monotonic() - cached[0]) < 30:
                return cached[1]

            data = await self._get_source_chart_data(
                keyword=q,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
            )
            self._chart_cache[cache_key] = (monotonic(), data)
            self._cleanup_chart_cache()
            return data

        if type == "sentiment":
            cache_key = ("sentiment", q or "", start_date or "", end_date or "", category or "", region or "")
            cached = self._chart_cache.get(cache_key)
            if cached and (monotonic() - cached[0]) < 30:
                return cached[1]

            data = await self._get_sentiment_chart_data(
                keyword=q,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
            )
            self._chart_cache[cache_key] = (monotonic(), data)
            self._cleanup_chart_cache()
            return data

        if type == "list":
            cache_key = ("list", q or "", start_date or "", end_date or "", category or "", region or "")
            cached = self._chart_cache.get(cache_key)
            if cached and (monotonic() - cached[0]) < 15:
                return cached[1]

            data = await self._get_list_chart_data(
                keyword=q,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
                limit=50,
            )
            self._chart_cache[cache_key] = (monotonic(), data)
            self._cleanup_chart_cache()
            return data

        if type == "word_cloud":
            cache_key = ("word_cloud", q or "", start_date or "", end_date or "", category or "", region or "")
            cached = self._chart_cache.get(cache_key)
            if cached and (monotonic() - cached[0]) < 30:
                return cached[1]

            data = await self._get_word_cloud_chart_data(
                keyword=q,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
            )
            self._chart_cache[cache_key] = (monotonic(), data)
            self._cleanup_chart_cache()
            return data

        data = await self.get_analysis_data(
            keyword=q,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=None,
            limit=None,
            generate_ai=False,
            use_cache=False,
        )

        charts = data.get("charts", {})
        if type == "source":
            return charts.get("source", [])
        if type == "sentiment":
            return {"sentiment_dist": charts.get("sentiment_dist", []), "neg_keywords": charts.get("neg_keywords", [])}
        if type == "list":
            return data.get("top_news", [])
        return charts.get(type, {})


report_service = ReportService()