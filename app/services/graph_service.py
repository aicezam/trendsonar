"""
本文件用于构建新闻关键词网络图谱数据，提供总览、节点展开、词项详情和相关新闻查询能力。
主要类:
- `GraphService`: 从新闻关键词/实体中聚合节点、边和词项分析数据
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations
import hashlib
import math
from typing import Any, Iterable, Optional

from sqlalchemy import Text, cast, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.config import get_settings
from app.models.news import News
from app.utils.news_query import build_news_query_filters, serialize_news_item
from app.utils.news_ranking import sort_news_by_composite_score
from app.utils.tools import normalize_regions_to_countries
from app.utils.ttl_cache import TtlMemoryCache

settings = get_settings()

GRAPH_CACHE_TTL_SECONDS = int(getattr(settings, "GRAPH_CACHE_TTL_SECONDS", 600) or 600)
GRAPH_CACHE_SIZE = int(getattr(settings, "GRAPH_CACHE_SIZE", 64) or 64)
GRAPH_MAX_AGGREGATE_NEWS = int(getattr(settings, "GRAPH_MAX_AGGREGATE_NEWS", 50000) or 50000)
GRAPH_TERM_LIMIT_PER_NEWS = int(getattr(settings, "GRAPH_TERM_LIMIT_PER_NEWS", 12) or 12)
GRAPH_OVERVIEW_HUB_COUNT = int(getattr(settings, "GRAPH_OVERVIEW_HUB_COUNT", 8) or 8)
GRAPH_OVERVIEW_MAX_LINKS_PER_NODE = int(getattr(settings, "GRAPH_OVERVIEW_MAX_LINKS_PER_NODE", 10) or 10)


GRAPH_RANGE_ALIASES = {
    "24h": "24h",
    "1d": "24h",
    "day": "24h",
    "today": "24h",
    "7d": "7d",
    "week": "7d",
    "30d": "30d",
    "month": "30d",
    "year": "year",
    "365d": "year",
    "1y": "year",
    "all": "all",
}


INVALID_GRAPH_TERMS = {
    "",
    "null",
    "none",
    "nan",
    "undefined",
    "无",
    "空",
    "其他",
    "其它",
    "未知",
    "无内容",
    "暂无",
    "暂无关键词",
    "暂无实体",
    "分析失败",
    "综合资讯",
    "新华社",
    "新华社音视频部",
    "央视新闻",
    "人民网",
    "36氪",
}


NODE_COLORS = {
    "center": "#0f172a",
    "hub": "#7c3aed",
    "entity": "#2563eb",
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral": "#0891b2",
}


class GraphService:
    """
    输入:
    - 数据库中的新闻、关键词、实体、热度、情绪和时间范围

    输出:
    - 面向前端 Sigma.js 的节点、边、详情和相关新闻数据

    作用:
    - 将新闻集合压缩为可逐层加载的网络图谱，避免前端一次性承载全量新闻。
    """

    def __init__(self) -> None:
        """
        输入:
        - 无

        输出:
        - 初始化后的图谱服务实例

        作用:
        - 创建短期内存缓存，降低重复图谱聚合对数据库和 CPU 的压力。
        """

        self._cache = TtlMemoryCache[dict[str, Any]](
            ttl_seconds=GRAPH_CACHE_TTL_SECONDS,
            max_size=GRAPH_CACHE_SIZE,
        )

    def _clean_cache_part(self, value: Any) -> str:
        """
        输入:
        - `value`: 查询参数值

        输出:
        - 规范化后的缓存键片段

        作用:
        - 让等价筛选条件命中同一份短期缓存。
        """

        return str(value or "").strip().lower()

    def _cache_key(self, name: str, **kwargs: Any) -> str:
        """
        输入:
        - `name`: 缓存场景名
        - `kwargs`: 查询参数

        输出:
        - 稳定缓存键

        作用:
        - 用 JSON 兼容的字符串拼接方式为图谱聚合结果生成缓存键。
        """

        parts = [name]
        for key in sorted(kwargs):
            parts.append(f"{key}={self._clean_cache_part(kwargs[key])}")
        return "|".join(parts)

    def _normalize_range(
        self,
        *,
        range_key: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> tuple[Optional[str], Optional[str], str]:
        """
        输入:
        - `range_key`: 前端快捷范围
        - `start_date`/`end_date`: 可选自定义日期

        输出:
        - 归一化后的开始日期、结束日期和范围标识

        作用:
        - 保证首页图谱默认聚焦近期数据，同时允许查看全年或全部历史。
        """

        normalized = GRAPH_RANGE_ALIASES.get((range_key or "24h").strip().lower(), "24h")
        if start_date or end_date:
            return start_date, end_date, normalized
        if normalized == "all":
            return None, None, normalized
        if normalized in {"24h", "7d", "30d"}:
            return None, None, normalized

        now = datetime.now()
        return (now - timedelta(days=365)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"), normalized

    def _valid_term(self, value: Any) -> Optional[str]:
        """
        输入:
        - `value`: 原始关键词或实体值

        输出:
        - 可用于图谱展示的词项，非法时返回 None

        作用:
        - 过滤占位词、空值和过短噪声，避免图谱被无意义节点污染。
        """

        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in INVALID_GRAPH_TERMS or text in INVALID_GRAPH_TERMS:
            return None
        if len(text) == 1 and not text.isascii():
            return None
        if len(text) > 48:
            return None
        return text

    def _iter_terms(self, keywords: Any, entities: Any, *, limit: int = GRAPH_TERM_LIMIT_PER_NEWS) -> list[tuple[str, str]]:
        """
        输入:
        - `keywords`: 新闻关键词列表
        - `entities`: 新闻实体列表
        - `limit`: 单条新闻参与共现计算的词项上限

        输出:
        - 去重后的 `(词项, 类型)` 列表

        作用:
        - 将新闻关键词和实体统一为图谱节点，限制单条新闻产生的边数量。
        """

        result: list[tuple[str, str]] = []
        seen: set[str] = set()
        for source_items, term_type in ((entities, "entity"), (keywords, "keyword")):
            if not isinstance(source_items, list):
                continue
            for item in source_items:
                term = self._valid_term(item)
                if not term:
                    continue
                key = term.lower()
                if key in seen:
                    continue
                seen.add(key)
                result.append((term, term_type))
                if len(result) >= limit:
                    return result
        return result

    def _term_condition(self, term: str) -> Any:
        """
        输入:
        - `term`: 词项文本

        输出:
        - SQLAlchemy 过滤条件

        作用:
        - 在标题、摘要、关键词和实体字段中召回与词项相关的新闻。
        """

        like = f"%{term}%"
        return or_(
            News.title.ilike(like),
            News.summary.ilike(like),
            cast(News.keywords, Text).ilike(like),
            cast(News.entities, Text).ilike(like),
        )

    def _node_score(self, news_count: int, heat_score: float) -> float:
        """
        输入:
        - `news_count`: 词项出现的新闻数
        - `heat_score`: 累计热度

        输出:
        - 节点排序分

        作用:
        - 综合出现频次和热度挑选图谱中的核心节点。
        """

        return float(news_count) * 2.0 + math.log1p(max(0.0, float(heat_score or 0.0)))

    def _node_size(self, news_count: int, heat_score: float) -> float:
        """
        输入:
        - `news_count`: 新闻数
        - `heat_score`: 累计热度

        输出:
        - 前端节点尺寸

        作用:
        - 将节点影响力压缩到稳定可读的视觉尺寸范围。
        """

        return round(min(16.0, 4.5 + math.sqrt(max(news_count, 1)) * 1.15 + math.log1p(max(heat_score, 0.0)) * 0.16), 2)

    def _node_color(self, term_type: str, avg_sentiment: float) -> tuple[str, str]:
        """
        输入:
        - `term_type`: 词项类型
        - `avg_sentiment`: 平均情绪分

        输出:
        - 节点颜色和语义分组

        作用:
        - 让实体、正向词和负面词在图谱中有稳定的视觉语义。
        """

        if term_type == "entity":
            return NODE_COLORS["entity"], "entity"
        if avg_sentiment > 58:
            return NODE_COLORS["positive"], "positive"
        if avg_sentiment > 0 and avg_sentiment < 45:
            return NODE_COLORS["negative"], "negative"
        return NODE_COLORS["neutral"], "neutral"

    def _stats_avg_sentiment(self, stats: Optional[dict[str, Any]]) -> float:
        """
        输入:
        - `stats`: 词项聚合统计

        输出:
        - 平均情绪分

        作用:
        - 从聚合结构中安全计算情绪均值，供节点和关系语义判断复用。
        """

        if not stats:
            return 0.0
        sentiment_count = int(stats.get("sentiment_count") or 0)
        if sentiment_count <= 0:
            return 0.0
        return float(stats.get("sentiment_sum") or 0.0) / sentiment_count

    def _relation_for_pair(
        self,
        source_stats: Optional[dict[str, Any]],
        target_stats: Optional[dict[str, Any]],
    ) -> tuple[str, str]:
        """
        输入:
        - `source_stats`/`target_stats`: 边两端词项的聚合统计

        输出:
        - 关系类型和前端展示标签

        作用:
        - 在没有显式三元组抽取结果时，基于实体类型和情绪倾向给共现关系补充可读语义。
        """

        source_type = str((source_stats or {}).get("type") or "keyword")
        target_type = str((target_stats or {}).get("type") or "keyword")
        target_sentiment = self._stats_avg_sentiment(target_stats)
        if source_type == "entity" and target_type == "entity":
            return "entity_cooccurrence", "实体共现"
        if target_sentiment > 0 and target_sentiment < 45:
            return "risk", "风险关联"
        if target_sentiment >= 58:
            return "positive", "正向关联"
        if target_type == "entity":
            return "entity_topic", "相关实体"
        if source_type == "entity":
            return "entity_topic", "相关议题"
        return "cooccurrence", "共同出现"

    def _edge_size(self, weight: int) -> float:
        """
        输入:
        - `weight`: 共现次数

        输出:
        - 前端边宽

        作用:
        - 将边权重压缩为适合 WebGL 展示的宽度。
        """

        return round(min(2.8, 0.35 + math.sqrt(max(weight, 1)) * 0.2), 2)

    def _seed_position(self, term: str, radius: float) -> tuple[float, float]:
        """
        输入:
        - `term`: 节点词项
        - `radius`: 散点半径上限

        输出:
        - 确定性初始 `(x, y)` 坐标

        作用:
        - 用词项哈希生成稳定的散点初始坐标，作为前端力导向布局的起点。
          只需打散避免重叠，最终位置由前端 ForceAtlas2 收敛决定，因此不再使用放射状死布局。
        """

        digest = hashlib.sha1(term.encode("utf-8")).hexdigest()
        angle = int(digest[:8], 16) / 0xFFFFFFFF * 2 * math.pi
        dist = (int(digest[8:16], 16) / 0xFFFFFFFF) ** 0.5 * radius
        return math.cos(angle) * dist, math.sin(angle) * dist

    def _make_node(
        self,
        *,
        term: str,
        term_type: str,
        news_count: int,
        heat_score: float,
        sentiment_sum: float,
        sentiment_count: int,
        last_seen_at: Optional[datetime],
        index: int,
        total: int,
        role: str = "node",
        hub_index: Optional[int] = None,
        hub_count: int = 1,
        anchor_index: int = 0,
        anchor_total: int = 1,
        center_term: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        输入:
        - 词项聚合指标、布局序号和知识图谱角色

        输出:
        - 前端图谱节点对象

        作用:
        - 将后端统计数据转成 Sigma.js 可直接导入的节点属性，并生成分层布局坐标。
        """

        avg_sentiment = sentiment_sum / sentiment_count if sentiment_count else 0.0
        color, sentiment_group = self._node_color(term_type, avg_sentiment)
        render_type = "circle"
        if role == "center":
            x = 0.0
            y = 0.0
            color = NODE_COLORS["center"]
        elif center_term:
            # 展开邻域：以中心为原点向外散点，前端 FA2 会进一步收敛
            x, y = self._seed_position(term, radius=2.2)
        elif role == "hub":
            x, y = self._seed_position(term, radius=1.6)
            color = NODE_COLORS["hub"]
        else:
            x, y = self._seed_position(term, radius=4.5)

        return {
            "id": term,
            "label": term,
            "term": term,
            "type": render_type,
            "term_type": term_type,
            "role": role,
            "hub": hub_index,
            "sentiment_group": sentiment_group,
            "x": round(x, 4),
            "y": round(y, 4),
            "size": self._node_size(news_count, heat_score),
            "color": color,
            "weight": round(self._node_score(news_count, heat_score), 4),
            "news_count": int(news_count),
            "heat_score": round(float(heat_score or 0.0), 2),
            "sentiment_avg": round(float(avg_sentiment or 0.0), 2),
            "last_seen_at": last_seen_at.isoformat() if last_seen_at else None,
        }

    def _make_edge(
        self,
        source: str,
        target: str,
        weight: int,
        heat_score: float = 0.0,
        *,
        relation_type: str = "cooccurrence",
        relation_label: str = "共同出现",
        directed: bool = True,
    ) -> dict[str, Any]:
        """
        输入:
        - `source`/`target`: 两端词项
        - `weight`: 共现次数
        - `heat_score`: 共现新闻累计热度
        - `relation_type`/`relation_label`: 关系语义
        - `directed`: 是否按方向渲染

        输出:
        - 前端图谱边对象

        作用:
        - 生成稳定边 ID、关系标签和渲染属性。
        """

        if directed:
            left, right = source, target
            edge_id = f"{left}__{relation_type}__{right}"
        else:
            left, right = sorted((source, target), key=str.lower)
            edge_id = f"{left}__{relation_type}__{right}"
        return {
            "id": edge_id,
            "source": left,
            "target": right,
            "weight": int(weight),
            "size": self._edge_size(weight),
            "heat_score": round(float(heat_score or 0.0), 2),
            "relation_type": relation_type,
            "relation_label": relation_label,
            "label": relation_label,
            "directed": directed,
            "color": "rgba(100, 116, 139, 0.22)",
        }

    async def _load_graph_rows(
        self,
        db: AsyncSession,
        *,
        start_date: Optional[str],
        end_date: Optional[str],
        category: Optional[str],
        region: Optional[str],
        source: Optional[str],
        sample_limit: int,
        date: str = "all",
        sort_by: str = "heat",
        term: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        输入:
        - 图谱筛选条件和采样上限
        - `term`: 可选中心词项

        输出:
        - 新闻聚合所需的轻量行数据

        作用:
        - 只读取图谱计算需要的字段，避免加载正文和向量。
        """

        stmt = build_news_query_filters(
            select(
                News.id,
                News.title,
                News.url,
                News.source,
                News.heat_score,
                News.publish_date,
                News.summary,
                News.category,
                News.region,
                News.sentiment_label,
                News.sentiment_score,
                News.keywords,
                News.entities,
            ),
            date=date,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
        )
        if term:
            stmt = stmt.where(self._term_condition(term))
        if sort_by == "date":
            stmt = stmt.order_by(desc(News.publish_date), desc(News.heat_score))
        else:
            stmt = stmt.order_by(desc(News.heat_score), desc(News.publish_date))
        stmt = stmt.limit(sample_limit)
        return [dict(row) for row in (await db.execute(stmt)).mappings().all()]

    def _aggregate_rows(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        node_limit: int,
        edge_limit: int,
        min_edge_weight: int,
        center_term: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        输入:
        - 新闻轻量行数据
        - 节点、边数量限制和可选中心词

        输出:
        - 聚合后的节点、边和基础统计

        作用:
        - 在 Python 侧基于关键词/实体共现构建受控规模的图谱。
        """

        node_stats: dict[str, dict[str, Any]] = {}
        pair_counter: Counter[tuple[str, str]] = Counter()
        pair_heat: defaultdict[tuple[str, str], float] = defaultdict(float)
        news_scanned = 0

        clean_center = self._valid_term(center_term) if center_term else None
        for item in rows:
            terms = self._iter_terms(item.get("keywords"), item.get("entities"))
            if clean_center and clean_center.lower() not in {term.lower() for term, _type in terms}:
                terms.insert(0, (clean_center, "keyword"))
            if not terms:
                continue

            news_scanned += 1
            heat = float(item.get("heat_score") or 0.0)
            sentiment = float(item.get("sentiment_score") or 0.0) if item.get("sentiment_score") is not None else 0.0
            publish_date = item.get("publish_date")
            unique_terms: list[tuple[str, str]] = []
            seen: set[str] = set()
            for term, term_type in terms:
                key = term.lower()
                if key in seen:
                    continue
                seen.add(key)
                unique_terms.append((term, term_type))
                stats = node_stats.setdefault(
                    term,
                    {
                        "type": term_type,
                        "news_count": 0,
                        "heat_score": 0.0,
                        "sentiment_sum": 0.0,
                        "sentiment_count": 0,
                        "last_seen_at": None,
                    },
                )
                if stats["type"] != term_type:
                    stats["type"] = "entity" if "entity" in {stats["type"], term_type} else "keyword"
                stats["news_count"] += 1
                stats["heat_score"] += heat
                if sentiment > 0:
                    stats["sentiment_sum"] += sentiment
                    stats["sentiment_count"] += 1
                if publish_date and (not stats["last_seen_at"] or publish_date > stats["last_seen_at"]):
                    stats["last_seen_at"] = publish_date

            if clean_center:
                center_matches = [term for term, _type in unique_terms if term.lower() == clean_center.lower()]
                center = center_matches[0] if center_matches else clean_center
                for term, _type in unique_terms:
                    if term.lower() == center.lower():
                        continue
                    pair = (center, term)
                    pair_counter[pair] += 1
                    pair_heat[pair] += heat
            else:
                for left, right in combinations([term for term, _type in unique_terms], 2):
                    pair = tuple(sorted((left, right), key=str.lower))
                    pair_counter[pair] += 1
                    pair_heat[pair] += heat

        ranked_terms = sorted(
            node_stats.items(),
            key=lambda item: self._node_score(item[1]["news_count"], item[1]["heat_score"]),
            reverse=True,
        )
        if clean_center and clean_center in node_stats:
            ranked_terms = [(clean_center, node_stats[clean_center])] + [
                item for item in ranked_terms if item[0] != clean_center
            ]

        selected_terms: list[tuple[str, dict[str, Any]]]
        hub_terms: list[str] = []
        hub_index_by_term: dict[str, int] = {}
        anchor_meta: dict[str, tuple[int, int, int]] = {}
        allowed_pairs: list[tuple[tuple[str, str], int]] = []
        max_links_per_node = max(1, GRAPH_OVERVIEW_MAX_LINKS_PER_NODE)
        if clean_center:
            selected_terms = ranked_terms[:node_limit]
            hub_terms = [clean_center] if selected_terms else []
            hub_index_by_term = {clean_center: 0} if selected_terms else {}
            for index, (term, _stats) in enumerate(selected_terms):
                anchor_meta[term] = (0, index, max(1, len(selected_terms)))
            allowed_pairs = [
                (pair, weight)
                for pair, weight in pair_counter.most_common(edge_limit * 3)
                if weight >= min_edge_weight
            ]
        else:
            hub_count = max(1, min(GRAPH_OVERVIEW_HUB_COUNT, node_limit // 8 or 1, len(ranked_terms)))
            hub_terms = [term for term, _stats in ranked_terms[:hub_count]]
            hub_set = set(hub_terms)
            hub_index_by_term = {term: index for index, term in enumerate(hub_terms)}
            selected_names = set(hub_terms)
            hub_children: defaultdict[str, list[str]] = defaultdict(list)
            node_degrees: Counter[str] = Counter()
            for (left, right), weight in pair_counter.most_common(edge_limit * 6):
                if weight < min_edge_weight:
                    continue
                left_is_hub = left in hub_set
                right_is_hub = right in hub_set
                if left_is_hub and right_is_hub:
                    allowed_pairs.append(((left, right), weight))
                    node_degrees[left] += 1
                    node_degrees[right] += 1
                    continue
                if left_is_hub == right_is_hub:
                    continue
                hub = left if left_is_hub else right
                child = right if left_is_hub else left
                if child not in node_stats:
                    continue
                if node_degrees[hub] >= max_links_per_node * 4 and child not in selected_names:
                    continue
                if len(hub_children[hub]) >= max_links_per_node and child not in selected_names:
                    continue
                allowed_pairs.append(((hub, child), weight))
                node_degrees[hub] += 1
                node_degrees[child] += 1
                if len(selected_names) < node_limit:
                    selected_names.add(child)
                    hub_children[hub].append(child)
                if len(selected_names) >= node_limit and all(len(items) >= max_links_per_node for items in hub_children.values()):
                    break

            if len(selected_names) < node_limit:
                for term, _stats in ranked_terms:
                    if term in selected_names:
                        continue
                    selected_names.add(term)
                    if len(selected_names) >= node_limit:
                        break
            linked_names = {term for pair, _weight in allowed_pairs for term in pair}
            if linked_names:
                selected_names = selected_names.intersection(linked_names)
                selected_names.update(hub_set.intersection(linked_names))
            selected_terms = [(term, stats) for term, stats in ranked_terms if term in selected_names][:node_limit]
            for hub, children in hub_children.items():
                total_children = max(1, len(children))
                for child_index, child in enumerate(children):
                    anchor_meta[child] = (hub_index_by_term[hub], child_index, total_children)

        selected = {term for term, _stats in selected_terms}

        nodes = [
            self._make_node(
                term=term,
                term_type=stats["type"],
                news_count=stats["news_count"],
                heat_score=stats["heat_score"],
                sentiment_sum=stats["sentiment_sum"],
                sentiment_count=stats["sentiment_count"],
                last_seen_at=stats["last_seen_at"],
                index=index,
                total=max(1, len(selected_terms)),
                role="center" if clean_center and term.lower() == clean_center.lower() else "hub" if term in hub_index_by_term else "node",
                hub_index=hub_index_by_term.get(term, anchor_meta.get(term, (None, 0, 1))[0]),
                hub_count=max(1, len(hub_terms)),
                anchor_index=anchor_meta.get(term, (0, index, len(selected_terms)))[1],
                anchor_total=anchor_meta.get(term, (0, index, len(selected_terms)))[2],
                center_term=clean_center,
            )
            for index, (term, stats) in enumerate(selected_terms)
        ]

        pair_items = allowed_pairs if allowed_pairs else list(pair_counter.most_common(edge_limit * 3))
        links = [
            self._make_edge(
                left,
                right,
                weight,
                pair_heat.get((left, right), pair_heat.get((right, left), 0.0)),
                relation_type=relation[0],
                relation_label=relation[1],
                directed=True,
            )
            for (left, right), weight in pair_items
            if weight >= min_edge_weight and left in selected and right in selected
            for relation in [self._relation_for_pair(node_stats.get(left), node_stats.get(right))]
        ][:edge_limit]

        return {
            "nodes": nodes,
            "edges": links,
            "summary": {
                "news_scanned": news_scanned,
                "candidate_nodes": len(node_stats),
                "candidate_edges": len(pair_counter),
                "node_count": len(nodes),
                "edge_count": len(links),
                "hub_count": len(hub_terms),
            },
        }

    async def get_overview(
        self,
        db: AsyncSession,
        *,
        range_key: str = "24h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        sort_by: str = "heat",
        limit: int = 120,
        edge_limit: int = 420,
        min_edge_weight: int = 2,
    ) -> dict[str, Any]:
        """
        输入:
        - 图谱筛选条件、节点上限和边上限

        输出:
        - 全局图谱总览节点和边

        作用:
        - 为图谱首页提供首屏核心词项网络。
        """

        limit = max(20, min(int(limit or 120), 500))
        edge_limit = max(50, min(int(edge_limit or limit * 4), 2000))
        min_edge_weight = max(1, min(int(min_edge_weight or 2), 20))
        start_date, end_date, normalized_range = self._normalize_range(
            range_key=range_key,
            start_date=start_date,
            end_date=end_date,
        )
        cache_key = self._cache_key(
            "overview",
            range=normalized_range,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            sort_by=sort_by,
            limit=limit,
            edge_limit=edge_limit,
            min_edge_weight=min_edge_weight,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            payload = deepcopy(cached)
            payload["summary"]["cached"] = True
            return payload

        rows = await self._load_graph_rows(
            db,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            sample_limit=GRAPH_MAX_AGGREGATE_NEWS,
            date=normalized_range,
        )
        payload = self._aggregate_rows(
            rows,
            node_limit=limit,
            edge_limit=edge_limit,
            min_edge_weight=min_edge_weight,
        )
        payload["scope"] = {
            "range": normalized_range,
            "start_date": start_date,
            "end_date": end_date,
            "category": category,
            "region": normalize_regions_to_countries(region),
            "source": source,
        }
        self._cache.set(cache_key, payload)
        return payload

    async def expand_node(
        self,
        db: AsyncSession,
        *,
        term: str,
        range_key: str = "24h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 90,
        edge_limit: int = 360,
        min_edge_weight: int = 1,
    ) -> dict[str, Any]:
        """
        输入:
        - `term`: 需要展开的中心词
        - 图谱筛选条件和规模限制

        输出:
        - 中心词邻域节点和边

        作用:
        - 支持前端点击或缩放时按需加载隐藏节点。
        """

        clean_term = self._valid_term(term)
        if not clean_term:
            return {"nodes": [], "edges": [], "summary": {"news_scanned": 0, "node_count": 0, "edge_count": 0}}

        limit = max(10, min(int(limit or 90), 300))
        edge_limit = max(20, min(int(edge_limit or limit * 4), 1200))
        min_edge_weight = max(1, min(int(min_edge_weight or 1), 20))
        start_date, end_date, normalized_range = self._normalize_range(
            range_key=range_key,
            start_date=start_date,
            end_date=end_date,
        )
        cache_key = self._cache_key(
            "expand",
            term=clean_term,
            range=normalized_range,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            sort_by=sort_by,
            limit=limit,
            edge_limit=edge_limit,
            min_edge_weight=min_edge_weight,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            payload = deepcopy(cached)
            payload["summary"]["cached"] = True
            return payload

        rows = await self._load_graph_rows(
            db,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            sample_limit=min(GRAPH_MAX_AGGREGATE_NEWS, 12000),
            date=normalized_range,
            term=clean_term,
        )
        payload = self._aggregate_rows(
            rows,
            node_limit=limit,
            edge_limit=edge_limit,
            min_edge_weight=min_edge_weight,
            center_term=clean_term,
        )
        payload["center"] = clean_term
        payload["scope"] = {
            "range": normalized_range,
            "start_date": start_date,
            "end_date": end_date,
            "category": category,
            "region": normalize_regions_to_countries(region),
            "source": source,
        }
        self._cache.set(cache_key, payload)
        return payload

    async def get_node_detail(
        self,
        db: AsyncSession,
        *,
        term: str,
        range_key: str = "24h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        输入:
        - `term`: 词项文本
        - 图谱筛选条件

        输出:
        - 词项统计、趋势、来源分布、分类分布和相关报道样本

        作用:
        - 为图谱右侧详情面板提供关键词分析。
        """

        clean_term = self._valid_term(term)
        if not clean_term:
            return {"found": False, "message": "词项无效"}

        start_date, end_date, normalized_range = self._normalize_range(
            range_key=range_key,
            start_date=start_date,
            end_date=end_date,
        )
        cache_key = self._cache_key(
            "detail",
            term=clean_term,
            range=normalized_range,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            payload = deepcopy(cached)
            payload["summary"]["cached"] = True
            return payload

        condition = self._term_condition(clean_term)
        day_expr = func.date(News.publish_date)
        aggregate_stmt = build_news_query_filters(
            select(
                day_expr.label("day"),
                News.sentiment_label,
                func.count().label("count"),
                func.coalesce(func.sum(News.heat_score), 0.0).label("heat"),
                func.coalesce(func.sum(News.sentiment_score), 0.0).label("sentiment_sum"),
                func.count(News.sentiment_score).label("sentiment_count"),
            ),
            date=normalized_range,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
        ).where(condition).group_by(day_expr, News.sentiment_label).order_by(day_expr)
        aggregate_rows = (await db.execute(aggregate_stmt)).all()

        total = 0
        total_heat = 0.0
        sentiment_sum = 0.0
        sentiment_count = 0
        sentiment_labels: Counter[str] = Counter()
        daily: dict[str, dict[str, Any]] = {}
        for day, label, count, heat, day_sentiment_sum, day_sentiment_count in aggregate_rows:
            row_count = int(count or 0)
            total += row_count
            total_heat += float(heat or 0.0)
            sentiment_sum += float(day_sentiment_sum or 0.0)
            sentiment_count += int(day_sentiment_count or 0)
            label_text = str(label or "中立")
            sentiment_labels[label_text] += row_count
            if day:
                key = str(day)
                item = daily.setdefault(key, {"count": 0, "heat": 0.0})
                item["count"] += row_count
                item["heat"] += float(heat or 0.0)

        sample_stmt = build_news_query_filters(
            select(News)
            .options(defer(News.content), defer(News.embedding))
            .where(condition)
            .order_by(desc(News.heat_score), desc(News.publish_date))
            .limit(300),
            date=normalized_range,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
        )
        sample_news = (await db.execute(sample_stmt)).scalars().all()
        ranked_news = sort_news_by_composite_score(sample_news)
        category_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        neighbor_counter: Counter[str] = Counter()
        for news in sample_news:
            if news.category:
                category_counter[str(news.category)] += 1
            if news.source:
                source_counter[str(news.source)] += 1
            for neighbor, _type in self._iter_terms(news.keywords, news.entities):
                if neighbor.lower() != clean_term.lower():
                    neighbor_counter[neighbor] += 1

        labels = sorted(daily.keys())
        payload = {
            "found": True,
            "term": clean_term,
            "summary": {
                "related_count": total,
                "sample_count": len(sample_news),
                "total_heat": round(total_heat, 2),
                "avg_sentiment": round(sentiment_sum / max(1, sentiment_count), 2) if sentiment_count else 0.0,
                "sentiment_counts": dict(sentiment_labels),
            },
            "trend": {
                "dates": labels,
                "count": [int(daily[label]["count"]) for label in labels],
                "heat": [round(float(daily[label]["heat"]), 2) for label in labels],
            },
            "categories": [{"name": name, "value": value} for name, value in category_counter.most_common(8)],
            "sources": [{"name": name, "value": value} for name, value in source_counter.most_common(8)],
            "neighbors": [{"name": name, "value": value} for name, value in neighbor_counter.most_common(12)],
            "related_news": [serialize_news_item(item) for item in ranked_news[:12]],
            "scope": {
                "range": normalized_range,
                "start_date": start_date,
                "end_date": end_date,
                "category": category,
                "region": normalize_regions_to_countries(region),
                "source": source,
            },
        }
        self._cache.set(cache_key, payload)
        return payload

    async def get_term_news(
        self,
        db: AsyncSession,
        *,
        term: str,
        range_key: str = "24h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        """
        输入:
        - `term`: 词项文本
        - 分页与筛选条件

        输出:
        - 相关新闻分页列表

        作用:
        - 支持图谱详情面板继续查看更多新闻。
        """

        clean_term = self._valid_term(term)
        if not clean_term:
            return {"data": [], "page": 1, "page_size": page_size, "has_more": False}

        start_date, end_date, normalized_range = self._normalize_range(
            range_key=range_key,
            start_date=start_date,
            end_date=end_date,
        )
        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 10), 50))
        offset = (page - 1) * page_size
        stmt = build_news_query_filters(
            select(News)
            .options(defer(News.content), defer(News.embedding))
            .where(self._term_condition(clean_term))
            .order_by(*order_by)
            .offset(offset)
            .limit(page_size + 1),
            date=normalized_range,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
        )
        rows = (await db.execute(stmt)).scalars().all()
        return {
            "data": [serialize_news_item(item) for item in rows[:page_size]],
            "page": page,
            "page_size": page_size,
            "has_more": len(rows) > page_size,
            "scope": {
                "range": normalized_range,
                "start_date": start_date,
                "end_date": end_date,
                "category": category,
                "region": normalize_regions_to_countries(region),
                "source": source,
            },
        }


graph_service = GraphService()
