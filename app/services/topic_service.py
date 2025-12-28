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
            used_stmt = select(TopicTimelineItem.news_id).where(TopicTimelineItem.news_id.isnot(None))
            used_ids_res = await db.execute(used_stmt)
            used_ids = set(used_ids_res.scalars().all())
            
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
                return

            # 获取现有的 Active 专题，用于查重和延伸判断
            active_topics_stmt = select(Topic).where(Topic.status == "active")
            active_topics = (await db.execute(active_topics_stmt)).scalars().all()

            # 4.1 新增：专题质量评估与过滤
            # 将现有专题转为简单字典供 AI 参考
            existing_topics_data = [{"name": t.name, "description": t.summary or ""} for t in active_topics]
            proposed_topics = await self.ai.batch_evaluate_topic_quality(proposed_topics, existing_topics=existing_topics_data)
            
            if not proposed_topics:
                logger.info("⚠️ 经 AI 评估，所有提炼专题均过于宽泛或质量不佳，跳过")
                return

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
                t_desc = p_topic.get("description", "")
                
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

                # 5.1 检查是否与现有专题重复
                existing_topic_obj = None

                for existing_t, existing_vec in active_topic_vecs:
                    sim = self._cosine_similarity(t_vec, existing_vec)
                    # 降低阈值至 0.6 以捕捉更多潜在重复，然后交给 AI 细判
                    if sim > 0.6: 
                        logger.info(f"   🔄 与现有专题 '{existing_t.name}' 相似 (sim={sim:.2f})，正在进行 AI 二次核验...")
                        
                        is_duplicate, reason = await self.ai.check_topic_duplicate(
                            t_name, t_desc, existing_t.name, existing_t.summary or ""
                        )
                        
                        if is_duplicate:
                            logger.info(f"   ✅ AI 确认重复 (理由: {reason})，将合并到现有专题: {existing_t.name}")
                            existing_topic_obj = existing_t
                            processed_topic_ids.add(existing_t.id)
                            break
                        else:
                            logger.info(f"   ❌ AI 判定为不同事件 (理由: {reason})")
                
                # 执行匹配和更新
                result_topic = await self._match_and_update_topic(
                    db, t_name, t_desc, t_vec, existing_topic_obj, 
                    news_pool, pool_vecs, used_ids
                )
                
                if result_topic:
                    if existing_topic_obj:
                        updated_topics_count += 1
                    else:
                        new_topics_created += 1
                        # 新专题加入 active_topic_vecs 以供后续（虽然本轮 Phase 1 不会再回头，但为了逻辑完整）
                        active_topic_vecs.append((result_topic, t_vec))
                        processed_topic_ids.add(result_topic.id)

            # === Phase 2: 扫描其余现有专题 ===
            logger.info("🔍 [Phase 2] 扫描其余现有专题，寻找潜在更新...")
            for existing_t, existing_vec in active_topic_vecs:
                if existing_t.id in processed_topic_ids:
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
            del news_pool
            del pool_vecs
            del active_topics
            del active_topic_vecs
            import gc
            gc.collect()
            
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
            summary_prompt = (
                "请根据以下专题综述，提炼一段 **高浓缩的事件概览**（100-150字）。\n"
                "要求：\n"
                "1. 包含事件的核心冲突（Who did What）。\n"
                "2. 包含关键的背景信息（如涉及金额、物品名称）。\n"
                "3. 包含当前的最新状态。\n"
                "4. 纯文本，无Markdown。\n\n"
                f"{overview_text[:2000]}"
            )
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
        used_ids: Set[int]
    ) -> Optional[Topic]:
        """
        核心逻辑：根据专题信息（名称、描述、向量），在 news_pool 中寻找匹配新闻，
        经 AI 核验后，创建新专题或更新旧专题。
        """
        is_duplicate = (existing_topic_obj is not None)
        
        # 1. 向量初筛候选新闻
        candidates = []
        for n in news_pool:
            # 跳过已经在当前轮次处理过的新闻
            if n.id in used_ids:
                continue
                
            n_vec = pool_vecs.get(n.id)
            if not n_vec:
                continue
            
            # 计算相似度
            sim = self._cosine_similarity(t_vec, n_vec)
            
            if sim > settings.TOPIC_MATCH_THRESHOLD: # 初筛阈值
                candidates.append((n, sim))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        # 取前 20 个给 AI 核验
        candidates = candidates[:settings.TOPIC_MATCH_MAX_CANDIDATES]
        
        # 如果是新专题，且候选不足，则跳过；如果是合并旧专题，候选不足也无妨（只是本次没更新）
        if not is_duplicate and len(candidates) <= settings.TOPIC_MIN_NEWS_COUNT:
            logger.info(f"   ⚠️ [新专题] 初筛候选新闻不足 ({len(candidates)} <= {settings.TOPIC_MIN_NEWS_COUNT})，跳过")
            return None
        
        if is_duplicate and not candidates:
            logger.info(f"   ⚠️ [旧专题合并] 无候选新闻，跳过")
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
                # Optional: Log mismatch if verbose
                logger.info(f"   ❌ [Mismatch] {candidates[idx][0].title[:30]}... (Reason: {reason})")

        # === 规则调整：对于已有专题，仅更新“今日”的新闻 ===
        if is_duplicate:
            today_date = datetime.now().date()
            today_news = []
            for n in confirmed_news:
                # 假设 publish_date 为空则视为非今日（或保留？通常爬虫数据应有时间）
                if n.publish_date and n.publish_date.date() == today_date:
                    today_news.append(n)
            
            if not today_news:
                logger.info(f"   ⏩ [旧专题] 经日期过滤后无今日新闻，跳过更新")
                return None
            
            if len(today_news) < len(confirmed_news):
                logger.info(f"   🗓️ [Date Filter] 过滤非今日新闻，剩余 {len(today_news)}/{len(confirmed_news)} 条")
            
            confirmed_news = today_news

        # 再次检查数量限制
        # 新专题：必须满足最小数量限制
        # 用户要求：媒体报道 >= 3 (即 count >= 3) => count < 3 则跳过
        if not is_duplicate and len(confirmed_news) < settings.TOPIC_MIN_NEWS_COUNT:
            logger.info(f"   ⚠️ [新专题] AI 核验通过数量不足 ({len(confirmed_news)} < {settings.TOPIC_MIN_NEWS_COUNT})，跳过")
            return None
            
        # 检查热度指标 (用户要求: 热度 > 6)
        # 计算热度（取新闻最大热度）
        max_heat = max([float(n.heat_score or 0) for n in confirmed_news]) if confirmed_news else 0
        if not is_duplicate and max_heat <= 6:
             logger.info(f"   ⚠️ [新专题] 热度不足 ({max_heat} <= 4)，跳过")
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
                "publish_date": n.publish_date  # Added for precise time
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

                # Determine event time from primary news if available
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
            overview_materials.append({
                "title": it.news_title,
                "content": it.content or "" # 使用 timeline 的 AI 摘要作为素材更好
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
                # 为了节省 token，直接让 AI 基于 overview_text 生成 summary
                summary_prompt = (
                    "请根据以下专题综述，提炼一段 **高浓缩的事件概览**（100-150字）。\n"
                    "要求：\n"
                    "1. 包含事件的核心冲突（Who did What）。\n"
                    "2. 包含关键的背景信息（如涉及金额、物品名称）。\n"
                    "3. 包含当前的最新状态。\n"
                    "4. 纯文本，无Markdown。\n"
                    "5. **直接输出**：不要包含任何“好的”、“根据您的要求”等客套话，直接输出摘要内容。\n\n"
                    f"{overview_text[:2000]}"
                )
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
        Scheduled entry point.
        This runs independently if configured, but now we prefer pipeline orchestration.
        We can keep it but maybe it should just call refresh_topics.
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

                # Run every 4 hours or similar
                # But user wants it after summary generation.
                # So this might be just a backup or manual trigger handler
                await asyncio.sleep(4 * 3600) 
                await self.refresh_topics()
            except AIConfigurationError as e:
                logger.error(f"🛑 配置错误: {e} 请检查 config.yaml 是否配置正确")
                logger.warning("⚠️ 专题追踪任务进入维护模式，每 5 分钟尝试重启服务检查一次...")
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Scheduled topic task error: {e}")
                await asyncio.sleep(300)

    # Helper methods
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
            # Batch embedding call (chunking if needed)
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
                        out[idx] = (t, vec) # Update the tuple in out list
            except Exception as e:
                 logger.error(f"   ⚠️ 专题向量化失败: {e}")
            await db.flush()
        return out

    async def _ensure_news_summary(self, db: AsyncSession, news: News) -> None:
        if (news.summary or "").strip():
            return

        # Try to crawl content if missing
        if not news.content or len(news.content) < 50:
             try:
                content = await crawler_service.crawl_content(news.url)
                if content:
                    news.content = content
             except Exception:
                 pass
        
        content = news.content or ""
        if len(content) < 50:
            return # Too short to summarize
            
        try:
            summary = await self.ai.generate_summary(news.title, content, max_words=200)
            if summary:
                news.summary = summary
                db.add(news)
        except Exception:
            pass

# Global instance
from app.services.ai_service import ai_service
topic_service = TopicService(ai=ai_service)
