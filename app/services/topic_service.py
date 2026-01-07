# app/services/topic_service.py

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from collections import defaultdict
import numpy as np
from sqlalchemy import desc, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal, check_db_connection
from app.core.logger import setup_logger
from app.core.exceptions import AIConfigurationError
from app.core.prompts import prompt_manager
from app.models.news import News
from app.models.topic import Topic, TopicTimelineItem
from app.services.ai_service import AIService
from app.services.crawler_service import crawler_service
from app.utils.tools import clean_html_tags

settings = get_settings()
logger = setup_logger("TopicService")


class TopicService:
    def __init__(self, ai: AIService) -> None:
        self.ai = ai

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

    def _find_candidate_news_by_vector(
        self,
        t_vec: List[float],
        news_pool: List[News],
        pool_vecs: Dict[int, List[float]],
        used_ids: Set[int],
        top_k: int = 10,
        threshold: float = 0.35
    ) -> List[Tuple[News, float]]:
        """
        根据向量相似度查找候选新闻 (不执行 DB 操作，不进行 AI 二次核验)
        """
        candidates = []
        for n in news_pool:
            if n.id in used_ids:
                continue
            n_vec = pool_vecs.get(n.id)
            if not n_vec:
                continue
            
            sim = self._cosine_similarity(t_vec, n_vec)
            if sim > threshold:
                candidates.append((n, sim))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    async def refresh_topics(self) -> None:
        """
        专题追踪逻辑：
        1. 找出未归类的新闻（N天内）。
        2. 聚合标题让 AI 提炼专题。
        3. 对提炼的专题进行向量匹配+AI核验。
        4. 只有新闻数 > 3 的专题才创建。
        5. 补全详情。
        """
        if not (settings.DATABASE_URL or "").strip():
            return

        async with AsyncSessionLocal() as db:
            # 1. 获取已归类的新闻ID集合
            # 同时获取 news_id 和 sources 中的 ID
            used_stmt = select(TopicTimelineItem.news_id, TopicTimelineItem.sources)
            used_res = await db.execute(used_stmt)
            
            used_ids = set()
            for nid, srcs in used_res:
                if nid:
                    used_ids.add(nid)
                if srcs and isinstance(srcs, list):
                    for src in srcs:
                        if isinstance(src, dict) and "id" in src:
                            try:
                                used_ids.add(int(src["id"]))
                            except (ValueError, TypeError):
                                pass
            
            # 2. 获取候选新闻池（N天内，未归类）
            days = settings.TOPIC_LOOKBACK_DAYS
            start_date = datetime.now() - timedelta(days=days)
            
            # 先查所有符合条件的新闻，用于后续向量匹配
            # 限制数量防止内存爆炸，比如取最近 2000 条
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
                
            logger.info(f"📊 待处理新闻池大小: {len(news_pool)}")
            
            # 确保池中新闻有向量（批量处理）
            pool_vecs = await self._ensure_news_embeddings_batch(db, news_pool)

            # 3. 准备 AI 提炼的种子标题（Top 300）
            # news_pool 已经是按 heat_score 排序的
            # 过滤掉低热度新闻
            min_heat = settings.TOPIC_NEWS_MIN_HEAT
            seed_news = [n for n in news_pool if (n.heat_score or 0) >= min_heat][:settings.TOPIC_AGGREGATION_TOP_N]
            
            if not seed_news:
                logger.info(f"📭 经热度过滤(>{min_heat})后，无符合条件的新闻，跳过专题生成")
                return

            # 格式化标题，带上热度信息
            seed_titles = [f"[热度:{float(n.heat_score or 0):.1f}] {(n.title or '').strip()}" for n in seed_news if (n.title or "").strip()]
            
            # 4. AI 提炼专题
            proposed_topics = await self.ai.propose_topics_from_titles(seed_titles)
            if not proposed_topics:
                logger.info("⚠️ AI 未提炼出任何专题")
                proposed_topics = []

            # 获取现有的 Active 专题，用于查重和延伸判断
            active_topics_stmt = select(Topic).where(Topic.status == "active")
            active_topics = (await db.execute(active_topics_stmt)).scalars().all()

            # 4.1 优化：跳过基于初始描述的初步过滤
            # 理由：初始描述是 AI 基于标题生成的，可能存在幻觉。与其浪费 Token 评估幻觉，
            # 不如直接通过向量搜索看是否有真实新闻支撑。如果向量搜不到，自然会被后续逻辑淘汰。
            # existing_topics_data = [...] # 移至需要时再构建
            
            # 确保现有专题有向量
            active_topic_vecs = await self._ensure_topic_embeddings(db, active_topics)

            # 5. 处理每个提炼出的专题
            new_topics_created = 0
            updated_topics_count = 0
            
            # 记录本轮已处理（创建或更新）的专题ID
            processed_topic_ids = set()

            # 准备关注关键词向量 (如果有配置)
            follow_keywords = settings.FOLLOW_KEYWORDS
            keyword_vecs = []
            if follow_keywords:
                kw_list = [k.strip() for k in follow_keywords.split(",") if k.strip()]
                if kw_list:
                    logger.info(f"🔍 [Topic Filter] 启用关键词过滤: {kw_list}")
                    kw_embs = await self.ai.get_embeddings(kw_list)
                    keyword_vecs = [v for v in kw_embs if v]
            
            # === Phase 1: 处理 AI 提炼的潜在专题 ===
            for p_topic in proposed_topics:
                t_name = p_topic.get("name", "")
                t_desc = p_topic.get("description", "") # 初始描述
                
                if not t_name:
                    continue
                    
                logger.info(f"🔍 [Phase 1] 正在评估提炼专题: {t_name}")
                
                # 计算该潜在专题的向量
                t_txt = f"{t_name} {t_desc}"
                t_embs = await self.ai.get_embeddings([t_txt])
                t_vec = t_embs[0] if t_embs and t_embs[0] else []
                
                if not t_vec:
                    logger.warning(f"   ⚠️ 无法生成向量: {t_name}")
                    continue

                # 5.0 关键词过滤
                if keyword_vecs:
                    max_sim = max([self._cosine_similarity(t_vec, kv) for kv in keyword_vecs]) if keyword_vecs else 0
                    if max_sim < settings.FOLLOW_KEYWORDS_THRESHOLD:
                        logger.info(f"   ⏩ 专题 '{t_name}' 与关注关键词相关度不足 ({max_sim:.2f} < {settings.FOLLOW_KEYWORDS_THRESHOLD})，跳过")
                        continue
                
                # --- 新增步骤：基于向量预先查找关联新闻，并生成真实摘要 ---
                # 1. 预查找候选新闻 (Vector Search Only)
                pre_candidates = self._find_candidate_news_by_vector(
                    t_vec, news_pool, pool_vecs, used_ids, top_k=10, threshold=0.35
                )
                
                # 如果候选新闻太少，说明可能是幻觉或无实证的专题，直接跳过
                if len(pre_candidates) < 2:
                    logger.info(f"   ⏩ 专题 '{t_name}' 预查找候选新闻不足 ({len(pre_candidates)} < 2)，跳过")
                    continue
                
                # 2. 生成真实摘要 (Initial Summary Generation)
                logger.info(f"   📝 正在为专题 '{t_name}' 生成基于事实的初始摘要 (Sample: {len(pre_candidates)})...")
                overview_materials = [{"title": n.title, "content": n.content or ""} for n, _ in pre_candidates]
                
                # 使用新的轻量级摘要生成 (默认 main 模型)
                generated_summary = await self.ai.generate_topic_initial_summary(t_name, overview_materials)
                if generated_summary:
                    generated_summary = generated_summary.replace("```", "").strip()
                
                # 使用生成的摘要替换初始描述 (如果生成失败则沿用初始描述)
                final_desc = generated_summary if generated_summary else t_desc
                logger.info(f"   📝 生成摘要: {final_desc[:50]}...")
                
                # 3. 二次质量审核 (Quality Audit with Real Summary)
                # 复用 batch_evaluate 但只传一个
                # 构建 existing_topics_data 供参考 (Lazy build)
                existing_topics_data = [
                    {"name": t.name, "description": (t.summary or "")[:100]} 
                    for t in active_topics
                ]
                
                audit_list = [{"name": t_name, "description": final_desc}]
                valid_topics = await self.ai.batch_evaluate_topic_quality(audit_list, existing_topics=existing_topics_data)
                
                if not valid_topics:
                    logger.info(f"   ❌ 专题 '{t_name}' 在基于真实摘要的二次审核中未通过，跳过")
                    continue
                
                # ----------------------------------------------------

                # 5.1 检查是否与现有专题重复 (使用新的 final_desc)
                existing_topic_obj = None

                for existing_t, existing_vec in active_topic_vecs:
                    sim = self._cosine_similarity(t_vec, existing_vec)
                    # 降低阈值至 0.6 以捕捉更多潜在重复，然后交给 AI 细判
                    if sim > 0.6: 
                        logger.info(f"   🔄 与现有专题 '{existing_t.name}' 相似 (sim={sim:.2f})，正在进行 AI 二次核验...")
                        
                        # 优化：限制传入 AI 的文本长度
                        is_duplicate, reason = await self.ai.check_topic_duplicate(
                            t_name, 
                            final_desc[:1000], 
                            existing_t.name, 
                            (existing_t.summary or "")[:1000]
                        )
                        
                        if is_duplicate:
                            logger.info(f"   ✅ AI 确认重复 (理由: {reason})，将合并到现有专题: {existing_t.name}")
                            existing_topic_obj = existing_t
                            processed_topic_ids.add(existing_t.id)
                            break
                        else:
                            logger.info(f"   ❌ AI 判定为不同事件 (理由: {reason})")
                
                # 执行匹配和更新 (传递 final_desc 作为 summary)
                result_topic = await self._match_and_update_topic(
                    db, t_name, final_desc, t_vec, existing_topic_obj, 
                    news_pool, pool_vecs, used_ids,
                    initial_summary=generated_summary
                )
                
                if result_topic:
                    # 如果是新创建的专题，且没有现成的 record，可以将 initial summary 先存入 record，
                    # 或者保持 record 为空等待后续生成 Overview。
                    # 这里为了数据完整性，暂且将 summary 存入 record，避免为空。
                    if not existing_topic_obj and not result_topic.record and generated_summary:
                         result_topic.record = generated_summary
                         db.add(result_topic)
                         await db.flush()
                    
                    if existing_topic_obj:
                        updated_topics_count += 1
                    else:
                        new_topics_created += 1
                        # 新专题加入 active_topic_vecs 以供后续（虽然本轮 Phase 1 不会再回头，但为了逻辑完整）
                        active_topic_vecs.append((result_topic, t_vec))
                        processed_topic_ids.add(result_topic.id)

            # === Phase 2: 扫描其余现有专题 ===
            logger.info("🔍 [Phase 2] 扫描其余现有专题，寻找潜在更新...")
            
            # 规则：对于最近 N 天内更新过的活跃专题，尝试从高热度新闻池中寻找匹配更新
            check_cutoff_date = datetime.now() - timedelta(days=settings.TOPIC_UPDATE_LOOKBACK_DAYS)
            
            for existing_t, existing_vec in active_topic_vecs:
                if existing_t.id in processed_topic_ids:
                    continue

                # 检查专题更新时间，如果太久没更新（且不是本次新创建的），则跳过以节省资源
                if existing_t.updated_time and existing_t.updated_time < check_cutoff_date:
                    continue
                
                # 使用现有专题的信息进行匹配
                # 注意：现有专题没有 t_desc 变量，使用 summary 或 name
                logger.info(f"   Evaluating existing topic: {existing_t.name}")
                
                result_topic = await self._match_and_update_topic(
                    db, 
                    existing_t.name, 
                    existing_t.summary or existing_t.name, 
                    existing_vec, 
                    existing_t, 
                    news_pool, 
                    pool_vecs, 
                    used_ids
                )
                
                if result_topic:
                    updated_topics_count += 1
                    processed_topic_ids.add(result_topic.id)

            logger.info(f"✅ 专题刷新完成，新建 {new_topics_created} 个，更新 {updated_topics_count} 个")

            # 显式清理大对象，帮助 GC 回收
            logger.info("🧹 [TopicService] 正在执行资源释放与内存清理...")
            if 'news_pool' in locals(): del news_pool
            if 'pool_vecs' in locals(): del pool_vecs
            if 'active_topics' in locals(): del active_topics
            if 'active_topic_vecs' in locals(): del active_topic_vecs
            
            import gc
            gc.collect()
            logger.info("✅ [TopicService] 内存清理完成")
            
    async def run_topic_scan_in_background(self, topic_id: int, include_used: bool = False):
        """
        后台任务：为指定专题执行扫描
        """
        try:
            async with AsyncSessionLocal() as db:
                topic = await db.get(Topic, topic_id)
                if topic:
                    await self._trigger_topic_scan(db, topic, include_used=include_used)
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
        initial_summary: Optional[str] = None
    ) -> Optional[Topic]:
        """
        核心逻辑：根据专题信息（名称、描述、向量），在 news_pool 中寻找匹配新闻，
        经 AI 核验后，创建新专题或更新旧专题。
        """
        is_duplicate = (existing_topic_obj is not None)
        
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
            
        # 检查热度指标 ( > 3.5 以允许普通新闻成专题)
        # 计算热度（取新闻最大热度）
        max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
        if not is_duplicate and max_heat < 3.5:
             logger.info(f"   ⚠️ [新专题] 热度不足 ({max_heat} < 3.5)，跳过")
             return None
        
        # 旧专题：不限制最小数量，只要有新的就合并
        if is_duplicate and not confirmed_news:
            return None
        
        # 3. 创建或更新专题
        current_topic_id = None
        topic_obj_to_return = None

        if is_duplicate:
            logger.info(f"   🔄 更新旧专题: {existing_topic_obj.name} (新增 {len(confirmed_news)} 条新闻)")
            # 更新旧专题的 update_time 和 heat_score
            max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
            current_max = existing_topic_obj.heat_score or 0
            if max_heat > current_max:
                existing_topic_obj.heat_score = max_heat
            
            new_end_time = max([n.publish_date for n in confirmed_news if n.publish_date]) if confirmed_news else None
            if new_end_time and (not existing_topic_obj.updated_time or new_end_time > existing_topic_obj.updated_time):
                existing_topic_obj.updated_time = new_end_time
            
            current_topic_id = existing_topic_obj.id
            topic_obj_to_return = existing_topic_obj
        else:
            logger.info(f"   ✨ 创建新专题: {t_name} (包含 {len(confirmed_news)} 条新闻)")
            
            # 计算热度（取新闻最大热度）
            max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
            # 最早时间
            start_time = min([n.publish_date for n in confirmed_news if n.publish_date]) if confirmed_news else datetime.now()
            # 最新时间
            end_time = max([n.publish_date for n in confirmed_news if n.publish_date]) if confirmed_news else datetime.now()

            new_topic = Topic(
                name=t_name,
                summary=t_desc,
                start_time=start_time,
                updated_time=end_time,
                heat_score=max_heat,
                embedding=t_vec,
                status="active"
            )
            db.add(new_topic)
            await db.flush()
            current_topic_id = new_topic.id
            topic_obj_to_return = new_topic
        
        # 4. 补全新闻详情与生成时间轴
        # 4.1 先检查并补全新闻详情
        async with crawler_service.make_crawler() as crawler:
            for n in confirmed_news:
                if not n.content or len(n.content) < 100:
                    logger.info(f"   📥 正在补全新闻详情: {n.title[:20]}...")
                    try:
                        crawled = await crawler_service.crawl_content_with_instance(n.url, crawler)
                        if crawled and len(crawled) > 50:
                            n.content = crawled
                            # 内容更新了，摘要最好也刷新一下，否则旧摘要可能不准
                            fresh_summary = await self.ai.generate_summary(n.title, n.content, max_words=200)
                            if fresh_summary:
                                n.summary = fresh_summary
                            db.add(n)
                    except Exception as e:
                        logger.warning(f"   ⚠️ 补全详情失败: {e}")
        
        await db.flush()

        # 4.2 生成标准化的时间轴内容 (按天聚合 + AI 合成)
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
                        "publish_date": n.publish_date
                    })
            else:
                final_news_list = day_news

            # 4. 调用 AI 合成（全量）
            logger.info(f"   🔄 正在重生成 {d_str} 的时间轴 (基于 {len(final_news_list)} 条新闻)...")
            day_events = await self.ai.generate_daily_timeline_events(d_str, final_news_list, topic_name=current_topic_name)
            
            # 硬性规则：每天最多保留 2 个节点
            if day_events and len(day_events) > 2:
                logger.info(f"   ⚠️ [Rule] AI 生成了 {len(day_events)} 个节点，强制截取前 2 个")
                day_events = day_events[:2]

            # 如果 AI 没有生成任何事件（失败或为空），则降级处理：选最重要的 1-2 条作为代表
            if not day_events:
                logger.warning(f"   ⚠️ {d_str} AI 合成事件失败，降级为使用 Top 新闻")
                # 按 publish_date 排序，取最新的
                final_news_list.sort(key=lambda x: x.get("publish_date") or datetime.min, reverse=True)
                # 简单取前 2 条
                for n_item in final_news_list[:2]:
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
            # 获取该专题下所有关联的新闻（为了生成全面的综述）
            # 限制数量，取热度最高的 50 条
            all_items_stmt = (
                select(TopicTimelineItem)
                .where(TopicTimelineItem.topic_id == current_topic_id)
                .order_by(desc(TopicTimelineItem.event_time))
                .limit(50)
            )
            all_items = (await db.execute(all_items_stmt)).scalars().all()
            
            # 收集用于生成综述的素材
            overview_materials = []
            for it in all_items:
                # 优化：优先使用 summary 或 content 的截断版本
                # TopicTimelineItem.content 通常是时间轴事件的描述，本身比较精简
                # 但如果它包含很长的引用，还是限制一下为好
                content_val = it.content or ""
                overview_materials.append({
                    "title": it.news_title,
                    "content": content_val[:500] # 使用 timeline 的 AI 摘要作为素材，限制长度
                })
            
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

    async def _ensure_news_summary(self, db: AsyncSession, news: News) -> None:
        if (news.summary or "").strip():
            return

        # 如果缺失，尝试抓取内容
        if not news.content or len(news.content) < 50:
             try:
                content = await crawler_service.crawl_content(news.url)
                if content:
                    # 抓取后立即清洗
                    cleaned = clean_html_tags(content)
                    if len(cleaned) > 50:
                        news.content = cleaned
             except Exception:
                 pass
        
        content = news.content or ""
        if len(content) < 50:
            return # 内容太短，无法总结
            
        try:
            summary = await self.ai.generate_summary(news.title, content, max_words=200)
            if summary:
                news.summary = summary
                db.add(news)
        except Exception:
            pass

# 全局实例
from app.services.ai_service import ai_service
topic_service = TopicService(ai=ai_service)
