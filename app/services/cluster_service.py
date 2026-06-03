"""
本文件用于实现新闻聚类与合并逻辑，通过向量相似度与大模型核验降低重复新闻。
主要类/对象:
- `ClusteringService`: 聚类服务实现
- `cluster_service`: 全局服务单例
"""

import json
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.config import BASE_DIR, get_settings
from app.core.database import AsyncSessionLocal
from app.core.logger import setup_logger
from app.models.news import News
from app.models.clustering_history import ClusteringHistory
from app.services.ai_service import AIService, ai_service

settings = get_settings()
logger = setup_logger("ClusteringService")


class ClusteringService:
    """
    输入:
    - 数据库中的新闻向量与相似度阈值配置

    输出:
    - 聚类合并后的新闻记录（更新 sources 与 heat_score）

    作用:
    - 通过向量相似度与 AI 核验，将同一事件的多来源新闻合并
    """

    def __init__(self, ai: AIService) -> None:
        """
        输入:
        - `ai`: AI 服务实例

        输出:
        - 无

        作用:
        - 注入 AI 能力，用于向量生成与聚类核验
        """

        self.ai = ai
        self._source_weight_cache: Dict[str, float] = {}
        self._source_weight_cache_loaded = False

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        输入:
        - `vec1`: 向量1
        - `vec2`: 向量2

        输出:
        - 余弦相似度（0~1）

        作用:
        - 计算两条新闻的语义相似度，用于候选聚类判断
        """

        if not vec1 or not vec2:
            return 0.0
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _get_sources_path_candidates(self) -> List[Path]:
        """
        输入:
        - 无

        输出:
        - 可能的新闻源配置文件路径列表（按优先级）

        作用:
        - 读取新闻源权重时，兼容 `data/` 与项目根目录两种放置方式
        """

        return [
            BASE_DIR / "data" / "news_sources.json",
            BASE_DIR / "news_sources.json",
        ]

    def _refresh_source_weight_cache(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 每轮聚类开始前读取一次新闻源权重，避免合并循环中反复读取配置文件
        """

        weights: Dict[str, float] = {}
        try:
            for path in self._get_sources_path_candidates():
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as f:
                    sources: List[Dict[str, Any]] = json.load(f)
                for src in sources:
                    name = str(src.get("name") or "").strip()
                    if not name:
                        continue
                    weights[name] = float(src.get("weight", 1.0))
                break
        except Exception as e:
            logger.error(f"读取新闻源权重失败: {e}")

        self._source_weight_cache = weights
        self._source_weight_cache_loaded = True

    def _get_source_weight_by_name(self, source_name: str) -> float:
        """
        输入:
        - `source_name`: 新闻来源名称

        输出:
        - 来源权重（未找到返回 1.0）

        作用:
        - 用新闻源配置的权重重算合并后的总热度
        """

        if not self._source_weight_cache_loaded:
            self._refresh_source_weight_cache()
        return float(self._source_weight_cache.get(source_name, 1.0))

    def _needs_summary_regeneration(self, summary: str) -> bool:
        """
        判断摘要是否需要重新生成 (e.g. 包含 HTML 标签)
        """
        if not summary:
            return False
        # 简单判断是否包含 HTML 标签
        return "<" in summary and ">" in summary

    def _build_similarity_candidates(
        self,
        pool: List[Dict[str, Any]],
        failed_pairs: Set[Tuple[int, int]],
    ) -> Tuple[Dict[int, List[Tuple[int, float]]], int]:
        """
        输入:
        - `pool`: 带有新闻 ID 与向量的聚类池
        - `failed_pairs`: 历史 AI 拒绝的新闻 ID 对

        输出:
        - 候选合并映射与被历史记录拦截的数量

        作用:
        - 使用 NumPy 批量计算归一化向量相似度，降低 Python 双重循环开销
        """

        candidates: Dict[int, List[Tuple[int, float]]] = {j: [] for j in range(len(pool))}
        if len(pool) < 2:
            return candidates, 0

        id_to_index = {int(item["id"]): idx for idx, item in enumerate(pool)}
        failed_index_pairs: Set[Tuple[int, int]] = set()
        for id_a, id_b in failed_pairs:
            idx_a = id_to_index.get(int(id_a))
            idx_b = id_to_index.get(int(id_b))
            if idx_a is None or idx_b is None:
                continue
            if idx_a > idx_b:
                idx_a, idx_b = idx_b, idx_a
            failed_index_pairs.add((idx_a, idx_b))

        try:
            vectors = np.asarray([item["vec"] for item in pool], dtype=np.float32)
            if vectors.ndim != 2:
                raise ValueError("新闻向量维度不一致")
        except Exception as e:
            logger.warning(f"向量矩阵构建失败，回退到逐对相似度计算: {e}")
            skipped_count = 0
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    if (i, j) in failed_index_pairs:
                        skipped_count += 1
                        continue
                    sim = self.calculate_cosine_similarity(pool[i]["vec"], pool[j]["vec"])
                    if sim >= settings.CLUSTERING_THRESHOLD:
                        candidates[j].append((i, sim))
            return candidates, skipped_count

        norms = np.linalg.norm(vectors, axis=1)
        valid_mask = norms > 0
        normalized = np.zeros_like(vectors, dtype=np.float32)
        normalized[valid_mask] = vectors[valid_mask] / norms[valid_mask, None]

        skipped_count = len(failed_index_pairs)
        threshold = float(settings.CLUSTERING_THRESHOLD)
        for i in range(len(pool) - 1):
            if not valid_mask[i]:
                continue
            sims = normalized[i + 1 :] @ normalized[i]
            matched_offsets = np.flatnonzero(sims >= threshold)
            for offset in matched_offsets:
                j = i + 1 + int(offset)
                if (i, j) in failed_index_pairs:
                    continue
                candidates[j].append((i, float(sims[offset])))

        return candidates, skipped_count

    async def execute_clustering(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 对时间窗口内新闻进行聚类合并，并写回数据库
        """

        logger.info("🧩 开始执行聚类任务...")
        self._refresh_source_weight_cache()
        async with AsyncSessionLocal() as db:
            from datetime import datetime, timedelta

            time_window = datetime.now() - timedelta(hours=settings.CLUSTERING_TIME_WINDOW_HOURS)
            logger.info(f"   ⏱️ 聚类时间窗口: {settings.CLUSTERING_TIME_WINDOW_HOURS}h (Cutoff: {time_window.strftime('%Y-%m-%d %H:%M:%S')})")
            
            result = await db.execute(
                select(News)
                .options(defer(News.content))
                .where(News.publish_date >= time_window)
                .order_by(News.heat_score.desc())
            )
            items = result.scalars().all()
            if not items:
                return

            items_to_embed = [item for item in items if not item.embedding]
            if items_to_embed:
                logger.info(f"   🧠 计算 {len(items_to_embed)} 条缺失向量...")
                titles = [i.title for i in items_to_embed]
                embeddings = await self.ai.get_embeddings(titles)
                for item, emb in zip(items_to_embed, embeddings):
                    if emb:
                        item.embedding = emb
                await db.commit()

            # 加载历史核验记录 (改为加载最近的 5000 条记录)
            history_result = await db.execute(
                select(ClusteringHistory.news_id_a, ClusteringHistory.news_id_b)
                .order_by(ClusteringHistory.id.desc())
                .limit(5000)
            )
            failed_pairs = set((row.news_id_a, row.news_id_b) for row in history_result)
            if failed_pairs:
                logger.info(f"   📜 已加载 {len(failed_pairs)} 条最近的历史核验记录")

            pool = []
            for item in items:
                if not item.embedding:
                    continue
                pool.append({"id": item.id, "obj": item, "vec": item.embedding, "merged": False})

            logger.info(f"   📊 聚类池大小: {len(pool)}")

            total_items = len(pool)
            logger.info(f"   🔍 开始预扫描 {total_items} 条数据...")

            candidates, skipped_count = self._build_similarity_candidates(pool, failed_pairs)

            total_candidates = sum(len(v) for v in candidates.values())
            processed_candidates = 0
            logger.info(f"   ✅ 预扫描完成 (共发现 {total_candidates} 个潜在合并对, 🛡️ 基于历史拦截跳过 {skipped_count} 对)")

            # ------------------------------------------------

            loop_count = 0
            while True:
                loop_count += 1
                if loop_count % 10 == 0:
                    logger.debug(f"      [聚类循环] 第 {loop_count} 轮扫描...")

                batch_requests = []
                merged_indices_in_this_round = []
                processed_in_this_round = False

                for j in range(len(pool)):
                    follower = pool[j]
                    if follower["merged"]:
                        continue
                    if not candidates[j]:
                        continue

                    leader_idx, sim = candidates[j][0]
                    
                    # 尝试找到最终的 Leader
                    real_l_idx = self._get_active_leader_idx(pool, leader_idx)
                    if real_l_idx is not None and real_l_idx != j:
                        leader_idx = real_l_idx
                    else:
                        candidates[j].pop(0)
                        processed_in_this_round = True
                        continue

                    leader = pool[leader_idx]

                    if leader["merged"]:
                        candidates[j].pop(0)
                        processed_in_this_round = True
                        continue

                    if sim >= 0.98:
                        processed_candidates += len(candidates[j])
                        logger.debug(f"      🔗 高相似度自动合并: [{leader['obj'].title}] <== [{follower['obj'].title}] (Sim: {sim:.4f})")
                        await self._merge_news(db, leader, follower)
                        
                        pool[j]["merged_to"] = leader_idx
                        merged_indices_in_this_round.append(j)
                        
                        candidates[j] = []
                        processed_in_this_round = True
                    else:
                        batch_requests.append(
                            {
                                "leader": leader["obj"].title,
                                "candidate": follower["obj"].title,
                                "leader_idx": leader_idx,
                                "candidate_idx": j,
                            }
                        )
                        # 使用配置的 batch size
                        if len(batch_requests) >= settings.ANALYSIS_BATCH_SIZE:
                            break

                if not batch_requests and not processed_in_this_round:
                    break

                if batch_requests:
                    req_count = len(batch_requests)
                    current_progress_start = processed_candidates + 1
                    current_progress_end = min(processed_candidates + req_count, total_candidates)
                    logger.info(f"   🤖 批量核验 {req_count} 对 (进度: {current_progress_start}-{current_progress_end}/{total_candidates})...")
                    
                    # 批量调用 AI 接口
                    verify_results = await self.ai.verify_cluster_batch(batch_requests)
                    
                    if len(verify_results) != len(batch_requests):
                        logger.error(f"      ❌ AI 返回结果数量不匹配! 请求: {len(batch_requests)}, 返回: {len(verify_results)}")
                    else:
                        logger.debug(f"      🤖 AI 返回 {len(verify_results)} 条核验结果")

                    for idx, (req, is_match) in enumerate(zip(batch_requests, verify_results), 1):
                        l_idx = req["leader_idx"]
                        f_idx = req["candidate_idx"]
                        progress_prefix = f"({idx}/{req_count})"

                        if is_match:
                            processed_candidates += len(candidates[f_idx])
                            logger.debug(
                                f"      {progress_prefix} 🔗 AI确认: [{pool[l_idx]['obj'].title}] <== [{pool[f_idx]['obj'].title}]"
                            )
                            await self._merge_news(db, pool[l_idx], pool[f_idx])
                            
                            pool[f_idx]["merged_to"] = l_idx
                            merged_indices_in_this_round.append(f_idx)
                            
                            candidates[f_idx] = []
                        else:
                            processed_candidates += 1
                            logger.debug(
                                f"      {progress_prefix} 🛡️ AI拦截: [{pool[l_idx]['obj'].title}] vs [{pool[f_idx]['obj'].title}]"
                            )
                            
                            # 记录到历史表，防止下次重复核验
                            l_id = pool[l_idx]["id"]
                            f_id = pool[f_idx]["id"]
                            if l_id > f_id:
                                l_id, f_id = f_id, l_id
                            
                            if (l_id, f_id) not in failed_pairs:
                                # 确保使用 session 中的对象添加
                                try:
                                    history_record = ClusteringHistory(news_id_a=l_id, news_id_b=f_id)
                                    db.add(history_record)
                                    # 立即 flush 确保写入 session，并检测潜在约束冲突
                                    await db.flush()
                                    failed_pairs.add((l_id, f_id))
                                except Exception as e:
                                    logger.error(f"      ❌ 历史记录写入失败 ({l_id}, {f_id}): {e}")

                            if candidates[f_idx]:
                                candidates[f_idx].pop(0)

                # 每一批次后立即提交，防止任务中断导致进度丢失
                try:
                    await db.commit()
                    if batch_requests or merged_indices_in_this_round:
                        logger.debug("      ✅ 批次提交成功")
                except Exception as e:
                    logger.error(f"      ❌ 批次提交失败: {e}")
                    await db.rollback()
                    # 回滚内存状态
                    for idx in merged_indices_in_this_round:
                        pool[idx]["merged"] = False
                        if "merged_to" in pool[idx]:
                            del pool[idx]["merged_to"]

                processed_in_this_round = True

                if not processed_in_this_round:
                    break

            await db.commit()
            logger.info("✅ 聚类完成")

            # 主动释放大对象，防止残留
            del pool
            del candidates
            del items
            gc.collect()

    def _get_active_leader_idx(self, pool: List[Dict], idx: int) -> Optional[int]:
        """递归查找当前节点被合并到的最终 Leader"""
        current = idx
        path = set()
        while pool[current].get("merged"):
            if current in path:
                return None # 环路检测
            path.add(current)
            if "merged_to" not in pool[current]:
                return None # 无法追踪
            current = pool[current]["merged_to"]
        return current

    async def _merge_news(self, db: AsyncSession, leader: Dict[str, Any], follower: Dict[str, Any]) -> None:
        """
        输入:
        - `db`: 数据库会话
        - `leader`: 主新闻（合并目标）
        - `follower`: 从新闻（被合并对象）

        输出:
        - 无

        作用:
        - 合并来源列表与热度，并删除被合并的新闻记录
        """

        leader_obj = leader["obj"]
        f_obj = follower["obj"]
        follower["merged"] = True

        master_sources = list(leader_obj.sources) if leader_obj.sources else []
        if not master_sources:
            master_sources = [{"name": leader_obj.source, "url": leader_obj.url}]

        seen_urls = set(s["url"] for s in master_sources if "url" in s)

        f_sources = list(f_obj.sources) if f_obj.sources else [{"name": f_obj.source, "url": f_obj.url}]
        for s in f_sources:
            if s.get("url") and s["url"] not in seen_urls:
                master_sources.append(s)
                seen_urls.add(s["url"])

        new_total_heat = 0.0
        for s in master_sources:
            w = self._get_source_weight_by_name(s.get("name", ""))
            new_total_heat += w

        if not leader_obj.summary and f_obj.summary:
            leader_obj.summary = f_obj.summary

        leader_obj.sources = master_sources
        leader_obj.heat_score = new_total_heat

        await db.execute(delete(News).where(News.id == f_obj.id))


cluster_service = ClusteringService(ai_service)
