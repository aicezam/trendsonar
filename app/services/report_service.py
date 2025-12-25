"""
本文件用于生成舆情报表（全局/关键词），并提供报表缓存、历史记录与图表数据聚合能力。
主要类/对象:
- `ReportService`: 报表生成与缓存服务
- `report_service`: 全局服务单例
"""

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import and_, case, cast, delete, desc, func, literal, or_, select, true
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import AsyncSessionLocal, check_db_connection
from app.core.logger import logger
from app.models.news import News
from app.models.report import ReportCache
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
                    f"chart-data word_cloud 慢查询: {elapsed:.2f}s | topN={len(data)} | category={category or ''} | range={start_date or ''}~{end_date or ''}"
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
                    f"chart-data word_cloud 慢查询: {elapsed:.2f}s | rows={len(keywords_values)} | category={category or ''} | range={start_date or ''}~{end_date or ''}"
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
                f"chart-data source 慢查询: {elapsed:.2f}s | topN={len(data)} | category={category or ''} | range={start_date or ''}~{end_date or ''}"
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
                    f"chart-data sentiment 慢查询: {elapsed:.2f}s | negN={len(neg_keywords)} | category={category or ''} | range={start_date or ''}~{end_date or ''}"
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
                    f"chart-data sentiment 慢查询: {elapsed:.2f}s | rows={len(rows)} | category={category or ''} | range={start_date or ''}~{end_date or ''}"
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
            elif period == "monthly":
                start_date_str = today.replace(day=1).strftime("%Y-%m-%d")
            else:
                start_date_str = (today - timedelta(days=6)).strftime("%Y-%m-%d")

            data = await self._generate_analysis_data(
                keyword="", start_date=start_date_str, end_date=end_date_str, generate_ai=True
            )

            await self.save_report_cache("global", period, data)
            logger.info(f"✅ 全局报表缓存 ({period}) 已更新")
        except Exception as e:
            logger.error(f"❌ 生成报表缓存失败: {e}")

    async def save_report_cache(self, r_type: str, keyword: str, data: Dict[str, Any]) -> None:
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
            return

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

    async def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
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
            stmt = (
                select(ReportCache)
                .where(ReportCache.report_type == "keyword")
                .order_by(desc(ReportCache.created_at))
                .limit(limit * 5)
            )
            result = await db.execute(stmt)
            reports = result.scalars().all()

            seen = set()
            recent = []
            for r in reports:
                if r.keyword and r.keyword not in seen:
                    recent.append({"keyword": r.keyword, "date": r.created_at.strftime("%Y-%m-%d %H:%M"), "id": r.id})
                    seen.add(r.keyword)
                    if len(recent) >= limit:
                        break
            return recent

    async def get_report_history(self, keyword: str) -> List[Dict[str, Any]]:
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
            result = await db.execute(stmt)
            reports = result.scalars().all()
            return [{"id": r.id, "date": r.created_at.strftime("%Y-%m-%d %H:%M")} for r in reports]

    async def get_global_history(self) -> List[Dict[str, Any]]:
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
                .limit(100)
            )
            result = await db.execute(stmt)
            reports = result.scalars().all()
            return [
                {"id": r.id, "date": r.created_at.strftime("%Y-%m-%d %H:%M"), "type": r.keyword or "weekly"}
                for r in reports
            ]

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
                data = cached.data
                if cached.report_type == "global":
                    return {**data, "period": cached.keyword or "weekly"}
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
                        select(ReportCache.keyword, ReportCache.data)
                        .where(ReportCache.report_type == "global", ReportCache.keyword == "daily")
                        .order_by(desc(ReportCache.created_at))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    row = result.first()

                    if not row:
                        stmt = (
                            select(ReportCache.keyword, ReportCache.data)
                            .where(ReportCache.report_type == "global")
                            .order_by(desc(ReportCache.created_at))
                            .limit(1)
                        )
                        result = await db.execute(stmt)
                        row = result.first()

                    if row:
                        kw, data = row
                        elapsed_ms = int((monotonic() - t0) * 1000)
                        logger.info(f"📖 读取全局报表数据库缓存 ({kw or 'unknown'}) {elapsed_ms}ms")
                        self._global_cache = (monotonic(), str(kw or ""), data)
                        return data

        data = await self._generate_analysis_data(keyword, start_date, end_date, category, region, source, limit, generate_ai)

        if keyword:
            await self.save_report_cache("keyword", keyword, data)

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
            stmt = select(News)
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
                stmt = stmt.where(and_(*filters))
                count_stmt = count_stmt.where(and_(*filters))

            real_total_count = await db.scalar(count_stmt)

            if limit and limit > 0:
                stmt = stmt.order_by(desc(News.heat_score)).limit(limit)
            else:
                stmt = stmt.order_by(desc(News.publish_date))

            result = await db.execute(stmt)
            news_list = result.scalars().all()

            if not news_list:
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
                    "ai_analysis": "未找到相关数据，无法进行分析。",
                }

            heats = [n.heat_score for n in news_list]
            avg_heat = sum(heats) / len(news_list) if news_list else 0
            max_heat = max(heats) if heats else 0

            sentiment_scores = [n.sentiment_score for n in news_list if n.sentiment_score is not None]
            sentiment_idx = (sum(sentiment_scores) / len(sentiment_scores)) if sentiment_scores else 50
            risk_count = sum(1 for n in news_list if n.sentiment_label == "负面")

            min_date = min(n.publish_date for n in news_list)
            max_date = max(n.publish_date for n in news_list)
            time_range = f"{min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"

            trend_map = defaultdict(lambda: {"count": 0, "heat_sum": 0.0, "categories": Counter()})
            source_map = Counter()

            all_keywords = []
            negative_keywords = []
            sentiment_counts = {"正面": 0, "中立": 0, "负面": 0}

            co_occurrence = defaultdict(int)
            keyword_freq = defaultdict(int)
            daily_kw_freq = defaultdict(Counter)
            daily_sentiment = defaultdict(lambda: {"正面": 0, "中立": 0, "负面": 0, "score_sum": 0.0, "score_count": 0})

            for n in news_list:
                day = n.publish_date.strftime("%Y-%m-%d")
                trend_map[day]["count"] += 1
                trend_map[day]["heat_sum"] += n.heat_score or 0.0
                trend_map[day]["categories"][n.category or "其他"] += 1

                if n.source:
                    source_map[n.source] += 1

                label = n.sentiment_label if n.sentiment_label in sentiment_counts else "中立"
                sentiment_counts[label] += 1
                daily_sentiment[day][label] += 1
                if n.sentiment_score is not None:
                    daily_sentiment[day]["score_sum"] += float(n.sentiment_score)
                    daily_sentiment[day]["score_count"] += 1

                if n.keywords and isinstance(n.keywords, list):
                    valid_kws = [
                        k.strip()
                        for k in n.keywords
                        if k and k.strip() and k.strip().lower() not in {"无内容", "null", "空", "none", ""}
                    ]
                    all_keywords.extend(valid_kws)
                    if label == "负面":
                        negative_keywords.extend(valid_kws)

                    unique_kws = list(set(valid_kws))
                    daily_kw_freq[day].update(unique_kws)
                    for kw in unique_kws:
                        keyword_freq[kw] += 1
                    for i in range(len(unique_kws)):
                        for j in range(i + 1, len(unique_kws)):
                            pair = tuple(sorted((unique_kws[i], unique_kws[j])))
                            co_occurrence[pair] += 1

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

            sorted_sources = sorted(source_map.items(), key=lambda x: x[1], reverse=True)
            chart_source = [{"name": k, "value": v} for k, v in sorted_sources[:10]]

            word_counts = Counter(all_keywords)
            word_cloud = [{"name": k, "value": v} for k, v in word_counts.most_common(50)]

            neg_keywords = [k for k, _ in Counter(negative_keywords).most_common(10)]

            sentiment_dist = [
                {"name": "正面", "value": sentiment_counts["正面"]},
                {"name": "中立", "value": sentiment_counts["中立"]},
                {"name": "负面", "value": sentiment_counts["负面"]},
            ]

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

            top_news = []
            for n in sorted(news_list, key=lambda x: x.heat_score or 0.0, reverse=True)[:10]:
                top_news.append(
                    {
                        "id": n.id,
                        "title": n.title,
                        "url": n.url,
                        "source": n.source,
                        "heat": n.heat_score,
                        "time": n.publish_date.isoformat(),
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
                            News.content,
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
                        prompt = (
                            f"你是一个专业舆情分析师。请基于下面关于“{keyword}”的“Top100 新闻（标题+正文）”输出一份markdown格式的深度舆情分析报告。\n"
                            "必须严格按以下模板格式输出（只输出内容，不要输出任何解释、不要输出原始新闻清单、不要输出“好的”等任何客套话）：\n"
                            "\n"
                            f"# “{keyword}”深度舆情分析报告\n"
                            f"**分析日期**：{time_range_label}  **数据来源**：{scope_str} Top100新闻\n"
                            "## 一、舆情综述\n"
                            f"（简要概括“{keyword}”在当前时间段内的整体舆情态势、热度变化及情感倾向，150字以内）\n"
                            "## 二、核心关注点\n"
                            "（分析围绕该关键词，公众和媒体最关注的具体议题或事件细节。使用有序列表 1. 2. 3. 输出 3-5 点）\n"
                            "1. **[关注点]**：详细描述（引用新闻事实）。\n"
                            "## 三、主要观点与争议\n"
                            "（梳理各方对该关键词相关事件的观点，包括支持、反对或中立的看法）\n"
                            "1. **[观点方向]**：具体内容。\n"
                            "## 四、关联主体与影响\n"
                            "（分析该舆情涉及的关键人物、企业或机构，以及对它们产生的影响）\n"
                            "## 五、研判与建议\n"
                            "（针对该关键词的舆情现状，提出简要的应对或关注建议）\n"
                            "**写作约束（必须遵守）：**\n"
                            f"1. **聚焦**：内容必须紧扣关键词“{keyword}”，无关信息不要写。\n"
                            "2. **去标签化**：禁止在正文中输出“观点方向：”等元数据标签。\n"
                            "3. **关键信息加粗**：人名、机构名、核心数据必须加粗。\n"
                            "4. **真实性**：基于提供的新闻数据分析，不要编造。\n"
                            "5. **格式**：严格按markdown格式输出900-1500字，不要输出任何解释、不要输出原始新闻清单、不要输出“好的”等任何客套话。\n"
                        )
                    # 场景2：全局/大盘综述 (Global Overview)
                    else:
                        prompt = (
                            "你是一个专业舆情分析师。请基于下面“当天 Top100 新闻（标题+正文）”输出一份markdown格式的舆情综述。\n"
                            "必须严格按以下模板格式输出（只输出内容，不要输出任何解释、不要输出原始新闻清单、不要输出“好的”等任何客套话）：\n"
                            "\n"
                            "# 综合舆情分析报告\n"
                            f"**分析日期**：{time_range_label}  **数据来源**：{scope_str} Top100新闻\n"
                            "## 一、热点主题归纳\n"
                            "（首先输出一段不带任何前缀的普通段落作为总述，概括今日整体舆情风向，重点加粗关键领域；总述后空1行）\n"
                            "（接下来用有序列表 1. 2. 3. 输出 4-7 条分主题，每条格式如下：）\n"
                            "1. **[精炼的主题词]**：直接叙述事件详情与核心看点（2-4句）。**禁止**出现“主题名：”“主题概述：”“热度层级”等文字标签。\n"
                            "## 二、关键主体\n"
                            "**国家/地区行为体**：\n"
                            "（3-6 条，格式：1. **[国家/地区名]**：直接描述其立场或动作（2-3句）。不要写“主体名称：”这类前缀）\n"
                            "**企业/机构**：\n"
                            "（3-8 条，格式：1. **[企业/机构名]**：直接描述其动作或面临的影响。重点加粗涉及的金额、产品名或合作方）\n"
                            "**公众人物**：\n"
                            "（2-6 条，格式：1. **[人名]**：直接描述触发点或争议点。不要写“触发点：”这类前缀）\n"
                            "## 三、主要矛盾/驱动因素\n"
                            "（3-6 条，每条格式：1. **[核心驱动力/矛盾点]**：引用新闻中的**事实、具体表述**作为证据线索。不要写“建议/对策”）\n"
                            "## 四、风险点\n"
                            "（3-8 条，每条格式：1. **[风险核心]**：描述触发条件及可能影响。**禁止**出现“风险点：”“触发条件：”等标签，将这些逻辑融入自然语句中）\n"
                            "**重要写作约束（必须遵守）：**\n"
                            "1. **去标签化**：绝对禁止在正文中输出“主题名：”、“证据线索：”、“主体名称：”等元数据标签。请直接用 **加粗名词** + 冒号 + 正文 的形式。\n"
                            "2. **关键信息加粗**：正文中出现的所有 **人名、机构名、核心地名、具体金额、关键数据、重要专有名词** 必须使用 markdown 加粗格式。\n"
                            "3. **结构完整**：必须覆盖上述五个小节，不得新增或缺失。\n"
                            "4. **真实性**：如信息不足请写“数据不足”，不要编造。\n"
                            "5. **篇幅**：总字数控制在 900-1500 字，每条新闻事件需详细描述。\n"
                            "6. **示例格式**：\n"
                            "   ## 一、热点主题归纳\n"
                            "   1. **[主题词1]**：XXXXXX\n"
                            "   2. **[主题词2]**：XXXXXX\n"
                            "   3. ……\n"
                            "   ## 二、关键主体\n"
                            "   **国家/地区**：\n"
                            "   1. **[国家/地区名1]**：XXXXXX\n"
                            "   2. **[国家/地区名2]**：XXXXXX\n"
                            "   3. ……\n"
                            "   **企业/机构**：\n"
                            "   1. **[企业/机构名1]**：XXXXXX\n"
                            "   2. **[企业/机构名2]**：XXXXXX\n"
                            "   3. ……\n"
                            "   **公众人物**：\n"
                            "   1. **[人名1]**：XXXXXX\n"
                            "   2. **[人名2]**：XXXXXX\n"
                            "   3. ……\n"
                            "   ## 三、主要矛盾/驱动因素\n"
                            "   1. **[核心驱动力/矛盾点1]**：XXXXXX\n"
                            "   2. **[核心驱动力/矛盾点2]**：XXXXXX\n"
                            "   3. ……\n"
                            "   ## 四、风险点\n"
                            "   1. **[风险核心1]**：XXXXX\n"
                            "   2. **[风险核心2]**：XXXXXX\n"
                        )
                    
                    # Append news content
                    prompt += (
                        "\n"
                        f"分析日期: {time_range_label}\n"
                        f"筛选条件: {scope_str}\n"
                        "新闻样本（Top100，已按热度排序，仅供你分析）：\n"
                        + ("\n\n".join(news_lines) if news_lines else "无可用新闻样本")
                    )

                    ai_analysis = await ai_service.chat_completion(prompt, route_key="REPORT")
                except Exception as e:
                    logger.error(f"AI分析失败: {e}")
                    ai_analysis = "AI 分析失败"
            else:
                ai_analysis = "未开启 AI 分析"

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
