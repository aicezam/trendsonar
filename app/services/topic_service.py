# app/services/topic_service.py

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from collections import defaultdict
import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal, check_db_connection
from app.core.logger import setup_logger
from app.core.exceptions import AIConfigurationError, AIServiceUnavailableError
from app.core.prompts import prompt_manager
from app.models.news import News
from app.models.topic import Topic, TopicTimelineItem
from app.services.ai_service import AIService
from app.services.crawler_service import crawler_service
from app.services.news_title_service import refine_news_title_if_needed
from app.services.topic_discovery_service import TopicDiscoveryService
from app.utils.retry import retry_async_result
from app.utils.summary_material import build_summary_generation_input, get_existing_summary_material
from app.utils.tools import clean_html_tags

settings = get_settings()
logger = setup_logger("TopicService")


class TopicService:
    def __init__(self, ai: AIService, discovery: Optional[TopicDiscoveryService] = None) -> None:
        self.ai = ai
        self.discovery = discovery or TopicDiscoveryService()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    async def _load_used_news_ids(self, db: AsyncSession, active_only: bool = False) -> Set[int]:
        """
        输入:
        - `db`: 数据库会话
        - `active_only`: 是否只统计活跃专题下的时间轴新闻

        输出:
        - 已归类新闻 ID 集合

        作用:
        - 统一解析 timeline 的主新闻和 sources，避免重复创建或被孤儿记录误伤。
        """

        used_stmt = select(TopicTimelineItem.news_id, TopicTimelineItem.sources)
        if active_only:
            used_stmt = used_stmt.join(Topic, Topic.id == TopicTimelineItem.topic_id).where(Topic.status == "active")
        used_res = await db.execute(used_stmt)

        used_ids: Set[int] = set()
        for nid, srcs in used_res:
            if nid:
                used_ids.add(int(nid))
            if srcs and isinstance(srcs, list):
                for src in srcs:
                    if isinstance(src, dict) and "id" in src:
                        try:
                            used_ids.add(int(src["id"]))
                        except (ValueError, TypeError):
                            pass
        return used_ids

    async def _build_follow_keyword_vecs(self) -> List[List[float]]:
        """
        输入:
        - 无，读取运行时关注关键词配置

        输出:
        - 关注关键词向量列表

        作用:
        - 在新专题发现中继续支持关注范围过滤，减少无关专题。
        """

        follow_keywords = settings.FOLLOW_KEYWORDS
        if not follow_keywords:
            return []
        kw_list = [k.strip() for k in follow_keywords.split(",") if k.strip()]
        if not kw_list:
            return []
        logger.info(f"🔍 [Topic Filter] 启用关键词过滤: {kw_list}")
        kw_embs = await self.ai.get_embeddings(kw_list)
        return [vec for vec in kw_embs if vec]

    async def _build_topic_overview_materials(
        self,
        db: AsyncSession,
        topic_id: int,
        fresh_news: List[News],
        limit: int,
    ) -> List[Dict[str, str]]:
        """
        输入:
        - `db`: 数据库会话
        - `topic_id`: 专题 ID
        - `fresh_news`: 本次新增或确认的新闻
        - `limit`: 最大素材数量

        输出:
        - 供专题综述使用的新闻素材列表

        作用:
        - 优先使用真实新闻标题和摘要，而不是只用时间轴节点二次概括。
        """

        news_map: Dict[int, News] = {int(n.id): n for n in fresh_news if n and n.id}
        all_items_stmt = (
            select(TopicTimelineItem)
            .where(TopicTimelineItem.topic_id == topic_id)
            .order_by(desc(TopicTimelineItem.event_time))
            .limit(max(limit * 3, limit))
        )
        all_items = (await db.execute(all_items_stmt)).scalars().all()

        news_ids: Set[int] = set(news_map.keys())
        for it in all_items:
            if it.news_id:
                news_ids.add(int(it.news_id))
            if it.sources:
                for source in it.sources:
                    if isinstance(source, dict) and source.get("id"):
                        try:
                            news_ids.add(int(source["id"]))
                        except (TypeError, ValueError):
                            pass

        missing_ids = [nid for nid in news_ids if nid not in news_map]
        if missing_ids:
            news_stmt = select(News).where(News.id.in_(missing_ids))
            for news in (await db.execute(news_stmt)).scalars().all():
                news_map[int(news.id)] = news

        sorted_news = sorted(
            news_map.values(),
            key=lambda n: (float(n.heat_score or 0), n.publish_date or datetime.min),
            reverse=True,
        )
        materials: List[Dict[str, str]] = []
        for news in sorted_news[:limit]:
            content = (news.summary or news.content or "").replace("\n", " ").strip()
            materials.append(
                {
                    "title": str(news.title or "")[:160],
                    "content": content[:500],
                }
            )
        return materials

    def _fallback_topic_name(self, evidence: Dict[str, Any]) -> str:
        """
        输入:
        - 候选事件证据包

        输出:
        - 兜底专题名称

        作用:
        - 当 AI 审核未给出名称时，基于证据生成保守名称。
        """

        entities = evidence.get("entities") or []
        keywords = evidence.get("keywords") or []
        title_list = evidence.get("representative_titles") or []
        if entities and keywords:
            return f"{entities[0]}{keywords[0]}进展"[:30]
        if title_list:
            return str(title_list[0])[:30]
        return "热点事件进展"

    def _sort_topic_candidates_for_review(self, candidates: List[Any]) -> None:
        """
        输入:
        - `candidates`: 程序聚类得到的候选专题事件簇列表

        输出:
        - 无，原地排序

        作用:
        - 送 AI 审核前再次按质量分 60%、热度 30%、新闻数 10% 排序，过滤后仍保持同一策略。
        """

        sorter = getattr(self.discovery, "sort_candidates", None)
        if callable(sorter):
            sorter(candidates)
            return

        candidates.sort(
            key=lambda candidate: (
                float((getattr(candidate, "evidence", {}) or {}).get("score", getattr(candidate, "score", 0.0)) or 0.0),
                float((getattr(candidate, "evidence", {}) or {}).get("total_heat", getattr(candidate, "total_heat", 0.0)) or 0.0),
                int((getattr(candidate, "evidence", {}) or {}).get("news_count", len(getattr(candidate, "features", []) or [])) or 0),
            ),
            reverse=True,
        )

    async def refresh_topics(self) -> None:
        """
        专题追踪逻辑：
        1. 找出未归类的新闻（N天内）。
        2. 由程序预聚类形成候选事件簇。
        3. 只把少量证据包交给 AI 审核 create/merge/reject。
        4. 通过审核后再创建或更新专题。
        """
        if not (settings.DATABASE_URL or "").strip():
            return

        with self.ai.task_retry_scope("专题生成任务"):
            await self._refresh_topics_by_discovery()

    async def _refresh_topics_by_discovery(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 新版自动专题发现流程：程序聚类先筛选候选事件，AI 只审核压缩证据包。
        """

        async with AsyncSessionLocal() as db:
            active_topics = (await db.execute(select(Topic).where(Topic.status == "active"))).scalars().all()
            used_ids = await self._load_used_news_ids(db, active_only=True)

            start_date = datetime.now() - timedelta(days=settings.TOPIC_LOOKBACK_DAYS)
            pool_stmt = (
                select(News)
                .where(News.publish_date >= start_date)
                .where(News.id.notin_(used_ids) if used_ids else True)
                .order_by(desc(News.heat_score))
                .limit(settings.TOPIC_RECALL_POOL_SIZE)
            )
            news_pool = (await db.execute(pool_stmt)).scalars().all()
            if not news_pool:
                logger.info("📭 没有待处理的新闻，跳过专题生成")
                return

            min_heat = settings.TOPIC_NEWS_MIN_HEAT
            news_pool = [n for n in news_pool if float(n.heat_score or 0) >= min_heat]
            if not news_pool:
                logger.info(f"📭 经热度过滤(>{min_heat})后，无符合条件的新闻，跳过专题生成")
                return

            logger.info(f"📊 待处理新闻池大小: {len(news_pool)}")
            pool_vecs = await self._ensure_news_embeddings_batch(db, news_pool)
            active_topic_vecs = await self._ensure_topic_embeddings(db, active_topics)

            candidates = self.discovery.build_candidates(news_pool, pool_vecs, active_topic_vecs)
            keyword_vecs = await self._build_follow_keyword_vecs()
            if keyword_vecs:
                before_count = len(candidates)
                candidates = [
                    candidate for candidate in candidates
                    if candidate.centroid and max(self._cosine_similarity(candidate.centroid, kw_vec) for kw_vec in keyword_vecs) >= settings.FOLLOW_KEYWORDS_THRESHOLD
                ]
                logger.info(f"🔍 [Topic Filter] 关注关键词过滤: {before_count} -> {len(candidates)}")

            if not candidates:
                logger.info("📭 程序聚类未发现满足门槛的候选专题")
                return

            self._sort_topic_candidates_for_review(candidates)
            logger.info(f"🧩 程序聚类生成候选专题事件簇: {len(candidates)} 个")
            evidence_list = [candidate.evidence for candidate in candidates]
            candidate_by_id = {candidate.cluster_id: candidate for candidate in candidates}
            candidate_order = {candidate.cluster_id: index for index, candidate in enumerate(candidates)}
            active_by_id = {int(t.id): t for t in active_topics}
            existing_topics_data = [
                {"id": t.id, "name": t.name, "summary": (t.summary or "")[:300]}
                for t in active_topics
            ]
            decisions = await self.ai.evaluate_topic_candidates(evidence_list, existing_topics_data)
            if not decisions:
                logger.info("📭 AI 未放行任何候选专题")
                return

            new_topics_created = 0
            updated_topics_count = 0
            processed_news_ids: Set[int] = set()
            decisions.sort(
                key=lambda item: candidate_order.get(
                    str(item.get("cluster_id") or ""),
                    len(candidate_order),
                )
            )

            for decision in decisions:
                cluster_id = str(decision.get("cluster_id") or "")
                candidate = candidate_by_id.get(cluster_id)
                if not candidate:
                    continue

                action = str(decision.get("decision") or "reject").lower()
                if action == "reject":
                    logger.info(f"   ❌ [候选拒绝] {cluster_id}: {decision.get('reason') or '无理由'}")
                    continue

                confirmed_news = [
                    n for n in candidate.news_items
                    if int(n.id) not in used_ids and int(n.id) not in processed_news_ids
                ]
                if not confirmed_news:
                    logger.info(f"   ⏩ [候选跳过] {cluster_id}: 新闻已在本轮或历史专题中使用")
                    continue

                existing_topic_obj = None
                if action == "merge":
                    existing_id = decision.get("existing_topic_id") or candidate.existing_topic_id
                    existing_topic_obj = active_by_id.get(int(existing_id)) if existing_id else None
                    if not existing_topic_obj:
                        logger.info(f"   ⏩ [合并跳过] {cluster_id}: 未找到可合并的现有专题")
                        continue
                    t_name = existing_topic_obj.name
                    t_desc = str(decision.get("summary") or existing_topic_obj.summary or t_name)
                elif action == "create":
                    if new_topics_created >= settings.TOPIC_CREATE_MAX_PER_RUN:
                        logger.info(f"   ⏩ [创建上限] 本轮已创建 {new_topics_created} 个专题，跳过剩余候选")
                        continue
                    t_name = str(decision.get("name") or "").strip() or self._fallback_topic_name(candidate.evidence)
                    t_desc = str(decision.get("summary") or "").strip() or candidate.evidence.get("fact_brief") or t_name
                else:
                    continue

                t_vec = candidate.centroid
                if not t_vec:
                    emb_text = f"{t_name} {t_desc}"[:1000]
                    t_embs = await self.ai.get_embeddings([emb_text])
                    t_vec = t_embs[0] if t_embs and t_embs[0] else []

                logger.info(f"🔍 [候选通过] {cluster_id} -> {action}: {t_name}")
                used_ids_before_update = set(used_ids)
                result_topic = await self._match_and_update_topic(
                    db,
                    t_name,
                    t_desc,
                    t_vec,
                    existing_topic_obj,
                    confirmed_news,
                    pool_vecs,
                    used_ids,
                    initial_summary=None if existing_topic_obj else t_desc,
                    confirmed_news_override=confirmed_news,
                )

                if result_topic:
                    actual_processed_ids = {int(nid) for nid in used_ids - used_ids_before_update}
                    processed_news_ids.update(actual_processed_ids)
                    if existing_topic_obj:
                        updated_topics_count += 1
                    else:
                        new_topics_created += 1
                        active_by_id[int(result_topic.id)] = result_topic
                        active_topic_vecs.append((result_topic, t_vec))

            logger.info(f"✅ 专题刷新完成，新建 {new_topics_created} 个，更新 {updated_topics_count} 个")

            import gc
            del news_pool
            del pool_vecs
            del active_topics
            del active_topic_vecs
            gc.collect()
            
    async def run_topic_scan_in_background(self, topic_id: int, include_used: bool = False):
        """
        后台任务：为指定专题执行扫描
        """
        try:
            with self.ai.task_retry_scope("专题扫描任务"):
                async with AsyncSessionLocal() as db:
                    topic = await db.get(Topic, topic_id)
                    if topic:
                        await self._trigger_topic_scan(db, topic, include_used=include_used)
        except AIServiceUnavailableError:
            logger.error("后台扫描任务因 AI 服务持续不可用而停止", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"后台扫描任务失败: {e}", exc_info=True)

    async def create_manual_topic(self, db: AsyncSession, name: str, trigger_scan: bool = True) -> Topic:
        """
        手动创建专题
        :param trigger_scan: 是否立即触发扫描（默认 True）。如果在 API 调用中，建议设为 False 并使用后台任务。
        """
        # 1. 检查是否存在同名专题
        existing = (await db.execute(select(Topic).where(Topic.name == name))).scalar_one_or_none()
        if existing:
            raise ValueError(f"专题 '{name}' 已存在，无法重复创建。")

        # 2. 创建新专题
        # 生成向量
        t_embs = await self.ai.get_embeddings([name])
        t_vec = t_embs[0] if t_embs and t_embs[0] else []
        
        new_topic = Topic(
            name=name,
            summary="手动创建专题，正在扫描相关新闻...",
            start_time=datetime.now(),
            updated_time=datetime.now(),
            heat_score=0,
            embedding=t_vec,
            status="active"
        )
        db.add(new_topic)
        await db.flush()
        logger.info(f"手动创建专题 '{name}' 成功 (ID: {new_topic.id})")
        target_topic = new_topic

        # 3. 触发扫描更新 (复用 Phase 2 逻辑)
        if trigger_scan:
            await self._trigger_topic_scan(db, target_topic, include_used=True) # 手动创建默认允许抢占/包含已归类新闻
            
        return target_topic

    async def update_topic_name(self, db: AsyncSession, topic_id: int, new_name: str, trigger_scan: bool = True) -> Topic:
        """
        更新专题名称
        :param trigger_scan: 是否立即触发扫描。
        """
        topic = (await db.execute(select(Topic).where(Topic.id == topic_id))).scalar_one_or_none()
        if not topic:
            raise ValueError("专题不存在")
            
        # 查重
        existing = (await db.execute(select(Topic).where(Topic.name == new_name))).scalar_one_or_none()
        if existing and existing.id != topic_id:
            raise ValueError(f"专题名 '{new_name}' 已被其他专题占用")
            
        old_name = topic.name
        topic.name = new_name
        
        # 重新生成向量
        t_embs = await self.ai.get_embeddings([new_name])
        t_vec = t_embs[0] if t_embs and t_embs[0] else []
        topic.embedding = t_vec
        
        db.add(topic)
        await db.flush()
        
        logger.info(f"专题 '{old_name}' 重命名为 '{new_name}'，已更新向量")
        
        # 触发扫描
        if trigger_scan:
            logger.info("正在触发重新扫描...")
            await self._trigger_topic_scan(db, topic, include_used=True) # 重命名也允许扩大范围
        
        return topic

    async def _trigger_topic_scan(self, db: AsyncSession, target_topic: Topic, include_used: bool = False):
        """
        内部复用逻辑：对指定专题执行 Phase 2 扫描
        :param include_used: 是否包含已经被其他专题归类的新闻。
                             手动触发（创建/改名）时建议为 True，以便将相关新闻“吸纳”进来。
        """
        # 准备排除列表
        exclude_ids = set()
        
        if include_used:
            # 如果允许包含已归类新闻，则只排除 *当前专题* 已经有的新闻（避免重复处理）
            current_stmt = select(TopicTimelineItem.news_id, TopicTimelineItem.sources).where(TopicTimelineItem.topic_id == target_topic.id)
            current_res = await db.execute(current_stmt)
            
            exclude_ids = set()
            for nid, srcs in current_res:
                if nid:
                    exclude_ids.add(nid)
                if srcs and isinstance(srcs, list):
                    for src in srcs:
                        if isinstance(src, dict) and "id" in src:
                            try:
                                exclude_ids.add(int(src["id"]))
                            except (ValueError, TypeError):
                                pass
            logger.info(f"🔍 [Scan] 模式: 包含已归类新闻 (只排除当前专题已有的 {len(exclude_ids)} 条)")
        else:
            # 默认模式：排除所有已归类新闻
            used_stmt = select(TopicTimelineItem.news_id, TopicTimelineItem.sources)
            used_res = await db.execute(used_stmt)
            
            exclude_ids = set()
            for nid, srcs in used_res:
                if nid:
                    exclude_ids.add(nid)
                if srcs and isinstance(srcs, list):
                    for src in srcs:
                        if isinstance(src, dict) and "id" in src:
                            try:
                                exclude_ids.add(int(src["id"]))
                            except (ValueError, TypeError):
                                pass
            logger.info(f"🔍 [Scan] 模式: 排除所有已归类新闻 (共 {len(exclude_ids)} 条)")

        days = settings.TOPIC_LOOKBACK_DAYS
        start_date = datetime.now() - timedelta(days=days)
        
        pool_stmt = (
            select(News)
            .where(News.publish_date >= start_date)
            .where(News.id.notin_(exclude_ids) if exclude_ids else True)
            .order_by(desc(News.heat_score))
            .limit(settings.TOPIC_RECALL_POOL_SIZE)
        )
        news_pool = (await db.execute(pool_stmt)).scalars().all()
        
        if not news_pool:
            logger.info("📭 没有待处理的新闻，跳过专题扫描")
            return

        # 确保池中新闻有向量
        pool_vecs = await self._ensure_news_embeddings_batch(db, news_pool)
        
        # 确保专题有向量
        if not target_topic.embedding:
             logger.warning("专题无向量，跳过扫描")
             return

        # 执行匹配
        logger.info(f"🔍 [Scan] 正在为专题 '{target_topic.name}' 扫描相关新闻 (Pool: {len(news_pool)})...")
        
        # 手动/定向扫描时，适当降低向量匹配阈值（0.6 -> 0.38），依靠 AI 二次核验来把关
        # 这样能召回更多语义相关但向量距离稍远的新闻
        manual_threshold = 0.20
        
        result_topic = await self._match_and_update_topic(
            db, 
            target_topic.name, 
            target_topic.summary or target_topic.name, 
            target_topic.embedding, 
            target_topic, 
            news_pool, 
            pool_vecs, 
            exclude_ids, # 传递作为 used_ids，用于内部过滤
            match_threshold=manual_threshold
        )
        
        if result_topic:
             logger.info(f"✅ 专题 '{target_topic.name}' 扫描更新完成")
        else:
             logger.info(f"专题 '{target_topic.name}' 未匹配到新的相关新闻")

    async def regenerate_topic_overview_action(self, db: AsyncSession, topic_id: int) -> Optional[str]:
        """
        手动触发：重新生成专题综述
        """
        topic = (await db.execute(select(Topic).where(Topic.id == topic_id))).scalar_one_or_none()
        if not topic:
            return None
            
        # 获取该专题下所有关联的新闻
        all_items_stmt = (
            select(TopicTimelineItem)
            .where(TopicTimelineItem.topic_id == topic_id)
            .order_by(desc(TopicTimelineItem.event_time))
            .limit(50)
        )
        all_items = (await db.execute(all_items_stmt)).scalars().all()
        
        overview_materials = []
        for it in all_items:
            overview_materials.append({
                "title": it.news_title,
                "content": it.content or "" 
            })
            
        if not overview_materials:
            return "暂无相关新闻，无法生成综述。"
            
        overview_text = await self.ai.generate_topic_overview(
            topic.name, 
            overview_materials
        )
        
        if overview_text:
            topic.record = overview_text
            # 顺便更新 summary
            summary_prompt = prompt_manager.get_user_prompt("topic_overview_summary", overview_text=overview_text[:2000])
            new_summary = await self.ai.chat_completion(summary_prompt)
            if new_summary:
                topic.summary = new_summary.replace("```", "").strip()
            
            db.add(topic)
            await db.commit()
            
        return overview_text

    async def regenerate_timeline_item_summary_action(
        self,
        db: AsyncSession,
        topic_id: int,
        item_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        输入:
        - `db`: 数据库会话
        - `topic_id`: 专题 ID
        - `item_id`: 时间轴节点 ID

        输出:
        - 重新生成后的节点数据；节点不存在或生成失败返回 None

        作用:
        - 基于时间轴节点关联的新闻，局部刷新该节点摘要并写回数据库。
        """

        topic = (await db.execute(select(Topic).where(Topic.id == topic_id))).scalar_one_or_none()
        if not topic:
            return None

        item = (
            await db.execute(
                select(TopicTimelineItem).where(
                    TopicTimelineItem.id == item_id,
                    TopicTimelineItem.topic_id == topic_id,
                )
            )
        ).scalar_one_or_none()
        if not item:
            return None

        news_ids: Set[int] = set()
        if item.news_id:
            news_ids.add(int(item.news_id))
        if isinstance(item.sources, list):
            for source in item.sources:
                if not isinstance(source, dict) or not source.get("id"):
                    continue
                try:
                    news_ids.add(int(source["id"]))
                except (TypeError, ValueError):
                    continue

        if not news_ids:
            return None

        news_list = (
            await db.execute(
                select(News)
                .where(News.id.in_(list(news_ids)))
                .order_by(desc(News.heat_score), desc(News.publish_date))
            )
        ).scalars().all()
        if not news_list:
            return None

        news_payload = [
            {
                "id": news.id,
                "title": news.title,
                "source": news.source,
                "summary": news.summary or (news.content or "")[:500],
                "content": news.content or "",
            }
            for news in news_list
        ]
        event_time = item.event_time.isoformat() if item.event_time else ""
        new_content = await self.ai.regenerate_timeline_item_summary(
            topic_name=topic.name,
            event_time=event_time,
            current_content=item.content or "",
            news_items=news_payload,
        )
        if not new_content:
            return None

        item.content = new_content
        db.add(item)
        await db.commit()
        await db.refresh(item)

        return {
            "id": item.id,
            "time": item.event_time.isoformat() if item.event_time else None,
            "content": item.content,
            "news_id": item.news_id,
            "news_title": item.news_title,
            "source_name": item.source_name,
            "source_url": item.source_url,
            "sources": item.sources,
        }

    async def _match_and_update_topic(
        self,
        db: AsyncSession,
        t_name: str,
        t_desc: str,
        t_vec: List[float],
        existing_topic_obj: Optional[Topic],
        news_pool: List[News],
        pool_vecs: Dict[int, List[float]],
        used_ids: Set[int],
        match_threshold: float = settings.TOPIC_MATCH_THRESHOLD,
        initial_summary: Optional[str] = None,
        confirmed_news_override: Optional[List[News]] = None
    ) -> Optional[Topic]:
        """
        核心逻辑：根据专题信息（名称、描述、向量），在 news_pool 中寻找匹配新闻，
        经 AI 核验后，创建新专题或更新旧专题。
        """
        is_duplicate = (existing_topic_obj is not None)

        if confirmed_news_override is not None:
            # 自动发现流程已经通过程序聚类和候选簇审核，避免再次逐条消耗 AI token。
            confirmed_news = [n for n in confirmed_news_override if int(n.id) not in used_ids]
            max_sim_found = 1.0
        else:
            # 1. 向量初筛候选新闻
            candidates = []
            max_sim_found = 0.0
            
            for n in news_pool:
                # 跳过已经在当前轮次处理过的新闻
                if n.id in used_ids:
                    continue
                    
                n_vec = pool_vecs.get(n.id)
                if not n_vec:
                    continue
                
                # 计算相似度
                sim = self._cosine_similarity(t_vec, n_vec)
                if sim > max_sim_found:
                    max_sim_found = sim
                
                if sim > match_threshold: # 初筛阈值
                    candidates.append((n, sim))
            
            # 按相似度排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            # 取前 20 个给 AI 核验
            candidates = candidates[:settings.TOPIC_MATCH_MAX_CANDIDATES]
            
            # 优化：如果是旧专题更新，过滤掉发布时间超过 24 小时的新闻
            # 避免将几天前的旧新闻作为“新动态”更新进去
            if is_duplicate:
                cutoff_time = datetime.now() - timedelta(hours=24)
                original_count = len(candidates)
                candidates = [c for c in candidates if c[0].publish_date and c[0].publish_date >= cutoff_time]
                
                if len(candidates) < original_count:
                    logger.info(f"   🧹 [旧专题] 过滤了 {original_count - len(candidates)} 条过旧新闻 (< {cutoff_time.strftime('%m-%d %H:%M')})")
                
                if not candidates:
                    logger.info(f"   ⏩ [旧专题] 无近期候选新闻，跳过")
                    return None
            
            # 如果是新专题，且候选不足，则跳过；如果是合并旧专题，候选不足也无妨（只是本次没更新）
            if not is_duplicate and len(candidates) <= settings.TOPIC_MIN_NEWS_COUNT:
                logger.info(f"   ⚠️ [新专题] 初筛候选新闻不足 ({len(candidates)} <= {settings.TOPIC_MIN_NEWS_COUNT} / MaxSim: {max_sim_found:.3f})，跳过")
                return None
            
            if is_duplicate and not candidates:
                logger.info(f"   ⚠️ [旧专题合并] 无候选新闻 (MaxSim: {max_sim_found:.3f} / Threshold: {match_threshold})，跳过")
                return None

            # 2. AI 批量核验
            verify_tasks = []
            for n, sim in candidates:
                verify_tasks.append({
                    "topic_name": t_name,
                    "topic_summary": t_desc, # 这里用 summary 字段传递 description
                    "news_title": n.title,
                    "news_summary": n.summary or (n.content or "")[:200] or ""
                })
            
            verified_results = await self.ai.verify_topic_match_batch(verify_tasks)
            
            confirmed_news = []
            for idx, (is_match, reason) in enumerate(verified_results):
                if is_match:
                    logger.info(f"   ✅ [Match] {candidates[idx][0].title[:30]}... (Reason: {reason})")
                    confirmed_news.append(candidates[idx][0])
                else:
                    # 可选: 详细模式下记录不匹配信息
                    logger.info(f"   ❌ [Mismatch] {candidates[idx][0].title[:30]}... (Reason: {reason})")

        # === 规则调整：对于已有专题，使用 Top N 热度新闻池进行更新 ===
        if is_duplicate:
             if not confirmed_news:
                 logger.info(f"   ⏩ [旧专题] 在 Top {settings.TOPIC_AGGREGATION_TOP_N} 新闻中未找到匹配项")
                 return None

        # 再次检查数量限制
        # 新专题：必须满足最小数量限制
        # 用户要求：媒体报道 >= 3 (即 count >= 3) => count < 3 则跳过
        if not is_duplicate and len(confirmed_news) < settings.TOPIC_MIN_NEWS_COUNT:
            logger.info(f"   ⚠️ [新专题] AI 核验通过数量不足 ({len(confirmed_news)} < {settings.TOPIC_MIN_NEWS_COUNT})，跳过")
            return None
            
        # 检查热度指标：新专题的总新闻热度必须大于 20
        max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
        total_heat = sum(float(n.heat_score or 0) for n in confirmed_news)
        if confirmed_news_override is None and not is_duplicate and total_heat <= 20:
             logger.info(f"   ⚠️ [新专题] 总新闻热度不足 ({total_heat:.1f} <= 20)，跳过")
             return None
        
        # 旧专题：不限制最小数量，只要有新的就合并
        if is_duplicate and not confirmed_news:
            return None

        confirmed_news = await self._ensure_confirmed_news_content(db, confirmed_news)
        if is_duplicate and not confirmed_news:
            logger.info("   ⏩ [旧专题] 相关新闻正文均无法获取，跳过本次更新")
            return None
        if not is_duplicate and len(confirmed_news) < settings.TOPIC_MIN_NEWS_COUNT:
            logger.info(f"   ⚠️ [新专题] 正文补抓后数量不足 ({len(confirmed_news)} < {settings.TOPIC_MIN_NEWS_COUNT})，跳过")
            return None
        
        # 3. 创建或更新专题
        current_topic_id = None
        topic_obj_to_return = None
        refresh_time = datetime.now()

        if is_duplicate:
            logger.info(f"   🔄 更新旧专题: {existing_topic_obj.name} (新增 {len(confirmed_news)} 条新闻)")
            # 更新旧专题的 update_time 和 heat_score
            max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
            current_max = existing_topic_obj.heat_score or 0
            if max_heat > current_max:
                existing_topic_obj.heat_score = max_heat
            
            # 专题的更新时间表示系统最后一次写入/刷新时间，不再混用新闻发布时间。
            existing_topic_obj.updated_time = refresh_time
            
            current_topic_id = existing_topic_obj.id
            topic_obj_to_return = existing_topic_obj
        else:
            logger.info(f"   ✨ 创建新专题: {t_name} (包含 {len(confirmed_news)} 条新闻)")
            
            # 计算展示热度（取新闻最大热度），准入门槛使用上面的总热度
            max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
            # 最早时间
            start_time = min([n.publish_date for n in confirmed_news if n.publish_date]) if confirmed_news else datetime.now()

            new_topic = Topic(
                name=t_name,
                summary=t_desc,
                start_time=start_time,
                updated_time=refresh_time,
                heat_score=max_heat,
                embedding=t_vec,
                status="active"
            )
            db.add(new_topic)
            await db.flush()
            current_topic_id = new_topic.id
            topic_obj_to_return = new_topic
        
        # 4. 生成标准化的时间轴内容 (按天聚合 + AI 合成)
        # 将 confirmed_news 按日期分组
        news_by_date = defaultdict(list)
        for n in confirmed_news:
            d_str = (n.publish_date or datetime.now()).strftime("%Y-%m-%d")
            news_by_date[d_str].append({
                "id": n.id,
                "title": n.title,
                "summary": n.summary or (n.content or "")[:200],
                "source": n.source,
                "url": n.url,
                "heat_score": float(n.heat_score or 0),
                "publish_date": n.publish_date  # 为了精确时间添加
            })
        
        # 遍历每一天，调用 AI 合成事件
        current_topic_name = topic_obj_to_return.name if topic_obj_to_return else None
        
        for d_str, day_news in news_by_date.items():
            # 1. 获取该天已有的时间轴节点（为了合并更新）
            # 注意：sqlite/pg 兼容性，这里简化处理，假设 event_time 存的是 datetime
            target_date = datetime.strptime(d_str, "%Y-%m-%d").date()
            
            # 构造查询范围：当天 00:00:00 到 23:59:59
            day_start = datetime.combine(target_date, datetime.min.time())
            day_end = datetime.combine(target_date, datetime.max.time())
            
            existing_items_stmt = (
                select(TopicTimelineItem)
                .where(TopicTimelineItem.topic_id == current_topic_id)
                .where(TopicTimelineItem.event_time >= day_start)
                .where(TopicTimelineItem.event_time <= day_end)
            )
            existing_items = (await db.execute(existing_items_stmt)).scalars().all()
            
            # 2. 收集该天所有相关的新闻 ID (旧 + 新)
            all_news_ids = set()
            for n in day_news:
                all_news_ids.add(n["id"])
            
            for it in existing_items:
                if it.news_id:
                    all_news_ids.add(it.news_id)
                if it.sources:
                    for s in it.sources:
                        if isinstance(s, dict) and s.get("id"):
                            all_news_ids.add(s["id"])
                            
            # 3. 如果有旧节点，需要重新拉取所有相关新闻的详情，进行全量重生成
            # 如果没有旧节点，直接用 day_news 即可
            final_news_list = []
            
            if existing_items:
                # 拉取所有涉及的新闻对象
                news_stmt = select(News).where(News.id.in_(list(all_news_ids)))
                all_news_objs = (await db.execute(news_stmt)).scalars().all()
                
                for n in all_news_objs:
                    final_news_list.append({
                        "id": n.id,
                        "title": n.title,
                        "summary": n.summary or (n.content or "")[:200],
                        "source": n.source,
                        "url": n.url,
                        "heat_score": float(n.heat_score or 0),
                        "publish_date": n.publish_date
                    })
            else:
                final_news_list = day_news

            # 4. 调用 AI 合成（全量）
            logger.info(f"   🔄 正在重生成 {d_str} 的时间轴 (基于 {len(final_news_list)} 条新闻)...")
            day_events = await self.ai.generate_daily_timeline_events(d_str, final_news_list, topic_name=current_topic_name)
            
            # 硬性规则：每天最多保留配置数量的节点
            max_day_events = max(1, settings.TOPIC_TIMELINE_MAX_EVENTS_PER_DAY)
            if day_events and len(day_events) > max_day_events:
                logger.info(f"   ⚠️ [Rule] AI 生成了 {len(day_events)} 个节点，强制截取前 {max_day_events} 个")
                day_events = day_events[:max_day_events]

            # 如果 AI 没有生成任何事件（失败或为空），则降级处理：选最重要的 1-2 条作为代表
            if not day_events:
                logger.warning(f"   ⚠️ {d_str} AI 合成事件失败，降级为使用 Top 新闻")
                # 按热度和发布时间排序，取最有代表性的新闻
                final_news_list.sort(
                    key=lambda x: (float(x.get("heat_score") or 0), x.get("publish_date") or datetime.min),
                    reverse=True,
                )
                for n_item in final_news_list[:max_day_events]:
                    day_events.append({
                        "content": n_item["summary"] or n_item["title"],
                        "source_ids": [n_item["id"]]
                    })

            # 5. 删除旧节点（如果存在），写入新节点
            if existing_items:
                for old_it in existing_items:
                    await db.delete(old_it)
                await db.flush() # 立即执行删除

            # 入库 Timeline Items
            for event in day_events:
                content = event.get("content")
                if not content:
                    continue
                
                source_ids = event.get("source_ids", [])
                
                # 构建 sources 列表
                sources_data = []
                # 找出对应的 news item info
                primary_news = None
                
                for nid in source_ids:
                    # 在 final_news_list 中查找
                    found = next((x for x in final_news_list if x["id"] == nid), None)
                    if found:
                        sources_data.append({
                            "id": found["id"],
                            "name": found["source"] or "未知来源",
                            "url": found["url"],
                            "title": found["title"]
                        })
                        if not primary_news:
                            primary_news = found
                
                # 如果 source_ids 为空或没找到，尝试兜底（虽然不应该发生）
                if not primary_news and final_news_list:
                     primary_news = final_news_list[0]

                # 如果可用，从主要新闻确定事件时间
                event_time = datetime.strptime(d_str, "%Y-%m-%d")
                if primary_news and primary_news.get("publish_date"):
                    event_time = primary_news["publish_date"]

                # 创建 item
                item = TopicTimelineItem(
                    topic_id=current_topic_id,
                    event_time=event_time,
                    content=content,
                    # 兼容旧字段，存储主要来源
                    news_id=primary_news["id"] if primary_news else None,
                    news_title=primary_news["title"] if primary_news else None,
                    source_name=primary_news["source"] if primary_news else None,
                    source_url=primary_news["url"] if primary_news else None,
                    # 新字段：多来源
                    sources=sources_data
                )
                db.add(item)
                
                # 标记 used_ids
                for nid in source_ids:
                    used_ids.add(nid)
                if primary_news and primary_news.get("id"):
                    used_ids.add(int(primary_news["id"]))

        await db.flush() # 确保 item 入库

        # 6. 生成/更新专题综述 (Overview) & 简要描述 (Summary)
        # 优化策略：
        # 1. 如果是旧专题更新，且新增新闻很少/热度低，则跳过 Overview 更新（节省大量 Token）
        # 2. 如果是新专题，必须生成 Overview。
        # 3. 生成 Summary 时，如果已有 initial_summary 且是新专题，则直接复用。

        should_update_overview = True
        
        if is_duplicate:
            # 检查新增新闻的重要性
            new_news_count = len(confirmed_news)
            max_new_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
            
            # 阈值：如果新增少于 3 条，且最大热度不超过 6.0，则认为变更不显著，不重写综述
            # (Timeline 已经更新了，用户依然可以看到新动态，只是 Overview 文本不变)
            if new_news_count < 3 and max_new_heat < 6.0:
                logger.info(f"   ⏩ [Overview] 新增内容较少 (Count={new_news_count}, MaxHeat={max_new_heat:.1f})，跳过综述重写")
                should_update_overview = False
        
        if should_update_overview:
            overview_materials = await self._build_topic_overview_materials(
                db,
                current_topic_id,
                confirmed_news,
                limit=settings.TOPIC_OVERVIEW_MAX_NEWS,
            )
            
            if overview_materials:
                # 1. 生成多维度综述
                # 注意：如果是 Existing Topic，名字可能和 t_name 不完全一样（如果是 Phase 2），但通常 Phase 2 传入的 t_name 就是 existing.name
                target_name = existing_topic_obj.name if existing_topic_obj else t_name
                
                overview_text = await self.ai.generate_topic_overview(
                    target_name, 
                    overview_materials
                )
                
                # 2. 更新 summary (简要描述)
                if overview_text:
                    new_summary = None
                    
                    # 优化：如果是新专题，且外部传入了 initial_summary，直接使用，不再调用 AI
                    if not is_duplicate and initial_summary:
                        logger.info("   ✅ [Summary] 复用初始摘要，跳过二次生成")
                        new_summary = initial_summary
                    else:
                        # 为了节省 token，直接让 AI 基于 overview_text 生成 summary
                        summary_prompt = prompt_manager.get_user_prompt("topic_overview_summary", overview_text=overview_text[:2000])
                        new_summary = await self.ai.chat_completion(summary_prompt, route_key="TOPIC_OVERVIEW")
                    
                    # 更新 Topic
                    topic_to_update = existing_topic_obj if is_duplicate else topic_obj_to_return
                    topic_to_update.record = overview_text
                    if new_summary:
                        topic_to_update.summary = new_summary.replace("```", "").strip()
                    
                    db.add(topic_to_update)
                else:
                    logger.warning(f"   ⚠️ 专题综述生成失败 (None)，跳过 Summary 更新")

        await db.commit()
        return topic_obj_to_return
            
    async def scheduled_topic_task(self) -> None:
        """
        后台定时任务：周期性自动执行专题追踪（生成新专题/更新旧专题）。
        作为独立守护进程运行，确保即使没有外部触发，系统也能按配置的时间间隔自动刷新专题数据。
        """
        logger.info("⏰ 专题追踪定时任务启动...")
        while True:
            try:
                if not await check_db_connection():
                    logger.warning("⚠️ 数据库连接异常，专题追踪任务暂停运行，等待恢复...")
                    await asyncio.sleep(60)
                    continue

                if not (settings.DATABASE_URL or "").strip():
                    logger.warning("⚠️ 未配置 DATABASE_URL，专题追踪任务暂停运行")
                    await asyncio.sleep(60)
                    continue

                # 读取配置的间隔时间（小时），默认为 4 小时
                interval = getattr(settings, "TOPIC_SCHEDULE_INTERVAL_HOURS", 4)
                await asyncio.sleep(interval * 3600) 
                
                await self.refresh_topics()
            except AIConfigurationError as e:
                logger.error(f"🛑 配置错误: {e} 请检查 config.yaml 是否配置正确")
                logger.warning("⚠️ 专题追踪任务进入维护模式，每 5 分钟尝试重启服务检查一次...")
                await asyncio.sleep(300)
            except AIServiceUnavailableError as e:
                logger.error(f"🛑 AI 服务持续不可用，本轮专题追踪任务已停止: {e}")
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"专题追踪定时任务错误: {e}")
                await asyncio.sleep(300)

    # 辅助方法
    async def _ensure_news_embeddings_batch(self, db: AsyncSession, news_list: List[News]) -> Dict[int, List[float]]:
        out = {}
        to_embed_indices = []
        texts = []
        
        for idx, n in enumerate(news_list):
            if n.embedding and len(n.embedding) > 0:
                out[n.id] = list(n.embedding)
            else:
                txt = " ".join([n.title or "", n.summary or "", (n.content or "")[:500]]).strip()
                texts.append(txt[:1000] if txt else (n.title or "")[:1000])
                to_embed_indices.append(idx)
        
        if texts:
            # 批量向量化调用（如果需要分块）
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_indices = to_embed_indices[i : i + batch_size]
                try:
                    embs = await self.ai.get_embeddings(batch_texts)
                    for local_idx, emb in enumerate(embs):
                        original_idx = batch_indices[local_idx]
                        n = news_list[original_idx]
                        if emb:
                            n.embedding = emb
                            db.add(n)
                            out[n.id] = emb
                except Exception as e:
                    logger.error(f"   ⚠️ 批量向量化失败: {e}")
            
            await db.flush()
        return out

    async def _ensure_topic_embeddings(self, db: AsyncSession, topics: List[Topic]) -> List[Tuple[Topic, List[float]]]:
        out = []
        to_embed = []
        for idx, t in enumerate(topics):
            if t.embedding and len(t.embedding) > 0:
                out.append((t, list(t.embedding)))
            else:
                txt = f"{t.name} {t.summary}"
                to_embed.append((idx, txt[:1000]))
                out.append((t, []))
        
        if to_embed:
            texts = [x[1] for x in to_embed]
            try:
                embs = await self.ai.get_embeddings(texts)
                for (idx, _), vec in zip(to_embed, embs):
                    if vec:
                        t = topics[idx]
                        t.embedding = vec
                        db.add(t)
                        out[idx] = (t, vec) # 更新输出列表中的元组
            except Exception as e:
                 logger.error(f"   ⚠️ 专题向量化失败: {e}")
            await db.flush()
        return out

    async def _ensure_confirmed_news_content(self, db: AsyncSession, news_list: List[News]) -> List[News]:
        """
        输入:
        - `db`: 数据库会话
        - `news_list`: 已确认归入专题的新闻列表

        输出:
        - 正文可用的新闻列表

        作用:
        - 在创建或更新专题前补全正文；无法获取正文时优先使用已有摘要兜底，减少专题素材流失。
        """

        ready_news: List[News] = []
        use_standalone_crawler = False
        min_content_length = max(10, int(getattr(settings, "CRAWLER_CONTENT_MIN_LENGTH", 30) or 30))
        retry_attempts = max(1, int(getattr(settings, "CRAWLER_RETRY_ATTEMPTS", 2) or 2))
        retry_delay = max(1.0, float(getattr(settings, "CRAWLER_RETRY_DELAY_SECONDS", 8.0) or 8.0))
        fetch_timeout = max(5.0, float(getattr(settings, "CRAWLER_FETCH_TIMEOUT_SECONDS", 45.0) or 45.0))
        async with crawler_service.make_crawler() as crawler:
            for news in news_list:
                cleaned_content = clean_html_tags(news.content or "").strip()
                existing_summary = get_existing_summary_material(news.summary)

                if cleaned_content and len(cleaned_content) >= 100:
                    news.content = cleaned_content
                    ready_news.append(news)
                    continue
                if existing_summary:
                    logger.info(f"   🧾 使用来源自带摘要作为专题素材: {news.title}")
                    ready_news.append(news)
                    continue

                logger.info(f"   📥 正在补全新闻详情: {(news.title or '')[:20]}...")
                retry_with_new_crawler = False

                async def crawl_once() -> Optional[str]:
                    """
                    输入:
                    - 无，闭包读取新闻 URL 与复用爬虫实例

                    输出:
                    - 抓取到的正文；失败返回 None

                    作用:
                    - 为重试工具提供单次正文抓取动作。
                    """

                    if use_standalone_crawler or retry_with_new_crawler:
                        return await crawler_service.crawl_content(news.url)
                    return await crawler_service.crawl_content_with_instance(news.url, crawler)

                async def switch_crawler_before_retry(next_attempt: int) -> None:
                    """
                    输入:
                    - `next_attempt`: 下一次尝试序号

                    输出:
                    - 无

                    作用:
                    - 失败后切换为新爬虫实例重试，避免全局清理浏览器打断其他并发抓取。
                    """

                    nonlocal retry_with_new_crawler, use_standalone_crawler
                    retry_with_new_crawler = True
                    use_standalone_crawler = True

                crawled = await retry_async_result(
                    crawl_once,
                    attempts=retry_attempts,
                    delay_seconds=retry_delay,
                    per_attempt_timeout_seconds=fetch_timeout,
                    min_valid_length=min_content_length,
                    label=f"正文补抓({news.id})",
                    before_retry=switch_crawler_before_retry,
                )
                if not crawled:
                    fallback_summary = get_existing_summary_material(news.summary)
                    if fallback_summary:
                        logger.warning(f"   ⚠️ 无法获取正文，改用已有摘要作为专题素材: {news.title}")
                        ready_news.append(news)
                    else:
                        logger.warning(f"   ⚠️ 无法获取正文，跳过新闻: {news.title}")
                    continue

                news.content = crawled
                if not (news.summary or "").strip():
                    fresh_summary = await self.ai.generate_summary(news.title, news.content, max_words=200)
                    if fresh_summary:
                        news.summary = fresh_summary
                        await refine_news_title_if_needed(news, summary=fresh_summary, content=news.content or "", ai=self.ai)
                db.add(news)
                ready_news.append(news)

        await db.flush()
        return ready_news

    async def _ensure_news_summary(self, db: AsyncSession, news: News) -> None:
        existing_summary = get_existing_summary_material(news.summary)

        cleaned_content = clean_html_tags(news.content or "").strip()

        # 如果缺失，尝试抓取内容
        if (not cleaned_content or len(cleaned_content) < 50) and not existing_summary:
            async def crawl_once() -> Optional[str]:
                """
                输入:
                - 无，闭包读取新闻 URL

                输出:
                - 抓取到的正文；失败返回 None

                作用:
                - 为摘要补全正文提供一次抓取动作。
                """

                return await crawler_service.crawl_content(news.url)

            content = await retry_async_result(
                crawl_once,
                attempts=max(1, int(getattr(settings, "CRAWLER_RETRY_ATTEMPTS", 2) or 2)),
                delay_seconds=max(1.0, float(getattr(settings, "CRAWLER_RETRY_DELAY_SECONDS", 8.0) or 8.0)),
                per_attempt_timeout_seconds=max(5.0, float(getattr(settings, "CRAWLER_FETCH_TIMEOUT_SECONDS", 45.0) or 45.0)),
                min_valid_length=max(10, int(getattr(settings, "CRAWLER_CONTENT_MIN_LENGTH", 30) or 30)),
                label=f"摘要正文补抓({news.id})",
            )
            if content:
                cleaned_content = clean_html_tags(content)
                if len(cleaned_content) > 50:
                    news.content = cleaned_content

        input_content = build_summary_generation_input(
            content=cleaned_content,
            original_summary=news.summary,
        )
        if not input_content:
            return

        try:
            summary = await self.ai.generate_summary(news.title, input_content, max_words=200)
            if summary:
                news.summary = summary
                await refine_news_title_if_needed(news, summary=summary, content=input_content, ai=self.ai)
                db.add(news)
        except Exception:
            pass

# 全局实例
from app.services.ai_service import ai_service
topic_service = TopicService(ai=ai_service)
