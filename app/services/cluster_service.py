"""
本文件用于实现新闻聚类与合并逻辑，通过向量相似度与大模型核验降低重复新闻。
主要类/对象:
- `ClusteringService`: 聚类服务实现
- `cluster_service`: 全局服务单例
"""

import json
import gc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import BASE_DIR, get_settings
from app.core.database import AsyncSessionLocal
from app.core.logger import setup_logger
from app.models.news import News
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

    def _get_source_weight_by_name(self, source_name: str) -> float:
        """
        输入:
        - `source_name`: 新闻来源名称

        输出:
        - 来源权重（未找到返回 1.0）

        作用:
        - 用新闻源配置的权重重算合并后的总热度
        """

        try:
            for path in self._get_sources_path_candidates():
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as f:
                    sources: List[Dict] = json.load(f)
                for src in sources:
                    if src.get("name") == source_name:
                        return float(src.get("weight", 1.0))
        except Exception as e:
            logger.error(f"读取新闻源权重失败: {e}")
        return 1.0

    def _needs_summary_regeneration(self, summary: str) -> bool:
        """
        判断摘要是否需要重新生成 (e.g. 包含 HTML 标签)
        """
        if not summary:
            return False
        # 简单判断是否包含 HTML 标签
        return "<" in summary and ">" in summary

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
        async with AsyncSessionLocal() as db:
            from datetime import datetime, timedelta

            time_window = datetime.now() - timedelta(hours=settings.CLUSTERING_TIME_WINDOW_HOURS)
            result = await db.execute(
                select(News).where(News.publish_date >= time_window).order_by(News.heat_score.desc())
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

            pool = []
            for item in items:
                if not item.embedding:
                    continue
                pool.append({"id": item.id, "obj": item, "vec": item.embedding, "merged": False})

            logger.info(f"   📊 聚类池大小: {len(pool)}")

            candidates: Dict[int, List] = {j: [] for j in range(len(pool))}
            total_items = len(pool)
            logger.info(f"   🔍 开始预扫描 {total_items} 条数据...")

            for i in range(len(pool)):
                if i > 0 and i % 500 == 0:
                    logger.debug(f"      [预扫描] 已处理 {i}/{total_items} ...")
                for j in range(i + 1, len(pool)):
                    sim = self.calculate_cosine_similarity(pool[i]["vec"], pool[j]["vec"])
                    if sim >= settings.CLUSTERING_THRESHOLD:
                        candidates[j].append((i, sim))

            logger.info("   ✅ 预扫描完成")

            # ------------------------------------------------

            loop_count = 0
            while True:
                loop_count += 1
                if loop_count % 10 == 0:
                    logger.debug(f"      [聚类循环] 第 {loop_count} 轮扫描...")

                batch_requests = []
                processed_in_this_round = False

                for j in range(len(pool)):
                    follower = pool[j]
                    if follower["merged"]:
                        continue
                    if not candidates[j]:
                        continue

                    leader_idx, sim = candidates[j][0]
                    leader = pool[leader_idx]

                    if leader["merged"]:
                        candidates[j].pop(0)
                        processed_in_this_round = True
                        continue

                    if sim >= 0.98:
                        await self._merge_news(db, leader, follower)
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
                    logger.info(f"   🤖 批量核验 {req_count} 对...")
                    verify_results = await self.ai.verify_cluster_batch(batch_requests)

                    for idx, (req, is_match) in enumerate(zip(batch_requests, verify_results), 1):
                        l_idx = req["leader_idx"]
                        f_idx = req["candidate_idx"]
                        progress_prefix = f"({idx}/{req_count})"

                        if is_match:
                            logger.debug(
                                f"      {progress_prefix} 🔗 AI确认: [{pool[l_idx]['obj'].title}] <== [{pool[f_idx]['obj'].title}]"
                            )
                            await self._merge_news(db, pool[l_idx], pool[f_idx])
                            candidates[f_idx] = []
                        else:
                            logger.debug(
                                f"      {progress_prefix} 🛡️ AI拦截: [{pool[l_idx]['obj'].title}] vs [{pool[f_idx]['obj'].title}]"
                            )
                            if candidates[f_idx]:
                                candidates[f_idx].pop(0)

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
