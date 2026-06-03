# 本文件用于在专题生成前用程序规则发现候选事件簇，减少大模型调用量。

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.core.config import Settings, get_settings
from app.utils.topic_preprocess import (
    TopicNewsFeature,
    UnionFind,
    build_news_feature,
    cosine_similarity,
    days_between,
    group_by_root,
)


@dataclass
class TopicCandidate:
    """
    输入:
    - 聚类后的新闻对象、特征和打分信息

    输出:
    - 可交给 AI 审核的专题候选

    作用:
    - 保存候选事件簇的证据、向量和可能合并的现有专题线索。
    """

    cluster_id: str
    news_items: List[Any]
    features: List[TopicNewsFeature]
    score: float
    centroid: List[float] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    existing_topic_id: Optional[int] = None
    existing_topic_name: str = ""
    existing_topic_similarity: float = 0.0

    @property
    def news_ids(self) -> List[int]:
        return [feature.news_id for feature in self.features]

    @property
    def total_heat(self) -> float:
        return sum(feature.heat_score for feature in self.features)

    @property
    def max_heat(self) -> float:
        return max((feature.heat_score for feature in self.features), default=0.0)

    @property
    def source_count(self) -> int:
        return len({feature.source for feature in self.features if feature.source})

    @property
    def news_count(self) -> int:
        return len(self.features)


class TopicDiscoveryService:
    """
    输入:
    - 待处理新闻池、新闻向量、现有专题及其向量

    输出:
    - 按质量分、热度、新闻数综合排序的候选专题事件簇

    作用:
    - 在不调用大模型的情况下完成事件聚类、准入筛选和证据包压缩。
    """

    def __init__(self, settings_obj: Optional[Settings] = None) -> None:
        self.settings = settings_obj or get_settings()

    def build_candidates(
        self,
        news_pool: List[Any],
        pool_vecs: Dict[int, List[float]],
        active_topic_vecs: Optional[List[Tuple[Any, List[float]]]] = None,
    ) -> List[TopicCandidate]:
        """
        输入:
        - `news_pool`: 未归类新闻池
        - `pool_vecs`: 新闻 ID 到向量的映射
        - `active_topic_vecs`: 现有活跃专题与向量

        输出:
        - 已过滤、已排序的候选事件簇

        作用:
        - 用轻量特征和向量相似度先聚类，避免把所有标题直接交给模型猜专题。
        """

        max_news = max(1, self.settings.TOPIC_DISCOVERY_MAX_NEWS_FOR_CLUSTER)
        scoped_news = sorted(news_pool, key=lambda item: float(getattr(item, "heat_score", 0.0) or 0.0), reverse=True)[:max_news]
        features = [build_news_feature(news, pool_vecs.get(int(getattr(news, "id")))) for news in scoped_news]
        if not features:
            return []

        union_find = UnionFind(len(features))
        for left_idx in range(len(features)):
            for right_idx in range(left_idx + 1, len(features)):
                if self._should_merge(features[left_idx], features[right_idx]):
                    union_find.union(left_idx, right_idx)

        news_by_id = {int(getattr(news, "id")): news for news in scoped_news}
        candidates: List[TopicCandidate] = []
        for root, indexes in group_by_root(union_find, len(features)).items():
            cluster_features = [features[idx] for idx in indexes]
            cluster_news = [news_by_id[feature.news_id] for feature in cluster_features if feature.news_id in news_by_id]
            score = self._score_cluster(cluster_features)
            if not self._passes_gate(cluster_features, score):
                continue

            centroid = self._centroid(cluster_features)
            candidate = TopicCandidate(
                cluster_id=f"cluster_{root}",
                news_items=cluster_news,
                features=cluster_features,
                score=score,
                centroid=centroid,
            )
            self._attach_existing_topic_hint(candidate, active_topic_vecs or [])
            candidate.evidence = self.build_evidence(candidate)
            candidates.append(candidate)

        self.sort_candidates(candidates)
        return candidates[: self.settings.TOPIC_DISCOVERY_MAX_CLUSTERS]

    def sort_candidates(self, candidates: List[TopicCandidate]) -> None:
        """
        输入:
        - `candidates`: 待排序候选事件簇

        输出:
        - 无，原地排序

        作用:
        - 按质量分 60%、热度 30%、新闻数 10% 综合排序，避免单一指标抢占送审名额。
        """

        if not candidates:
            return

        max_score = max((candidate.score for candidate in candidates), default=0.0)
        max_heat = max((candidate.total_heat for candidate in candidates), default=0.0)
        max_news_count = max((candidate.news_count for candidate in candidates), default=0)

        def sort_key(candidate: TopicCandidate) -> Tuple[float, float, float, int, float]:
            score_ratio = candidate.score / max_score if max_score > 0 else 0.0
            heat_ratio = candidate.total_heat / max_heat if max_heat > 0 else 0.0
            news_ratio = candidate.news_count / max_news_count if max_news_count > 0 else 0.0
            weighted_score = score_ratio * 0.6 + heat_ratio * 0.3 + news_ratio * 0.1
            return (
                weighted_score,
                candidate.score,
                candidate.total_heat,
                candidate.news_count,
                candidate.max_heat,
            )

        candidates.sort(key=sort_key, reverse=True)

    def build_evidence(self, candidate: TopicCandidate) -> Dict[str, Any]:
        """
        输入:
        - 候选事件簇

        输出:
        - 压缩后的证据包字典

        作用:
        - 控制发给大模型的字段和长度，实现 token 可控。
        """

        features = sorted(
            candidate.features,
            key=lambda item: (item.heat_score, item.publish_date or datetime.min),
            reverse=True,
        )
        first_time, last_time = self._time_range(features)
        source_counter = Counter(feature.source or "未知来源" for feature in features)
        category = self._most_common_text(feature.category for feature in features)
        region = self._most_common_text(feature.region for feature in features)
        top_entities = self._top_terms([feature.entities for feature in features], limit=8)
        top_keywords = self._top_terms([feature.keywords | feature.action_terms for feature in features], limit=10)
        representative_titles = self._representative_titles(features)
        fact_brief = self._fact_brief(features)

        evidence: Dict[str, Any] = {
            "cluster_id": candidate.cluster_id,
            "news_ids": [feature.news_id for feature in features],
            "news_count": len(features),
            "source_count": len(source_counter),
            "sources": [source for source, _ in source_counter.most_common(6)],
            "score": round(candidate.score, 2),
            "total_heat": round(sum(feature.heat_score for feature in features), 2),
            "max_heat": round(max((feature.heat_score for feature in features), default=0.0), 2),
            "time_range": {
                "start": first_time.isoformat() if first_time else "",
                "end": last_time.isoformat() if last_time else "",
            },
            "category": category,
            "region": region,
            "entities": top_entities,
            "keywords": top_keywords,
            "representative_titles": representative_titles,
            "fact_brief": fact_brief,
        }
        if candidate.existing_topic_id:
            evidence["existing_topic_hint"] = {
                "id": candidate.existing_topic_id,
                "name": candidate.existing_topic_name,
                "similarity": round(candidate.existing_topic_similarity, 3),
            }
        return evidence

    def _should_merge(self, left: TopicNewsFeature, right: TopicNewsFeature) -> bool:
        """
        输入:
        - 两条新闻特征

        输出:
        - 是否应合并为同一事件

        作用:
        - 用时间、向量、主体词和动作词共同约束事件边界。
        """

        if days_between(left.publish_date, right.publish_date) > self.settings.TOPIC_CLUSTER_MAX_DAYS_SPAN:
            return False

        entity_overlap = self._jaccard(left.entities, right.entities)
        keyword_overlap = self._jaccard(left.keywords | left.title_terms, right.keywords | right.title_terms)
        action_overlap = bool(left.action_terms and right.action_terms and left.action_terms & right.action_terms)
        same_category = bool(left.category and right.category and left.category == right.category)
        same_region = bool(left.region and right.region and left.region == right.region)
        sim = cosine_similarity(left.embedding, right.embedding)

        if sim >= self.settings.TOPIC_CLUSTER_STRONG_SIM_THRESHOLD:
            return self._has_boundary_support(left, right, entity_overlap, keyword_overlap, action_overlap)

        if sim >= self.settings.TOPIC_CLUSTER_SIM_THRESHOLD:
            support = 0
            if entity_overlap >= 0.2:
                support += 2
            if keyword_overlap >= 0.18:
                support += 1
            if action_overlap:
                support += 1
            if same_category:
                support += 1
            if same_region:
                support += 1
            return support >= 3

        if entity_overlap >= 0.45 and keyword_overlap >= 0.2 and (action_overlap or same_category):
            return True
        return False

    def _has_boundary_support(
        self,
        left: TopicNewsFeature,
        right: TopicNewsFeature,
        entity_overlap: float,
        keyword_overlap: float,
        action_overlap: bool,
    ) -> bool:
        """
        输入:
        - 两条高语义相似新闻及重合度特征

        输出:
        - 是否有足够实体或动作支撑合并

        作用:
        - 防止同类泛话题因向量相近被误合并。
        """

        if entity_overlap >= 0.12 or keyword_overlap >= 0.22 or action_overlap:
            return True
        if left.category and right.category and left.category == right.category and left.region and left.region == right.region:
            return True
        return False

    def _score_cluster(self, features: List[TopicNewsFeature]) -> float:
        """
        输入:
        - 事件簇特征列表

        输出:
        - 0 到 100 左右的候选质量分

        作用:
        - 综合报道量、来源数、热度、实体清晰度和时间集中度判断是否值得送审。
        """

        news_count = len(features)
        source_count = len({feature.source for feature in features if feature.source})
        total_heat = sum(feature.heat_score for feature in features)
        max_heat = max((feature.heat_score for feature in features), default=0.0)
        entity_count = len(set().union(*(feature.entities for feature in features))) if features else 0
        action_count = len(set().union(*(feature.action_terms for feature in features))) if features else 0
        first_time, last_time = self._time_range(features)
        span_days = max(days_between(first_time, last_time), 0)

        score = 0.0
        score += min(news_count, 8) * 8.0
        score += min(source_count, 5) * 7.0
        score += min(total_heat, 60.0) * 0.45
        score += min(max_heat, 15.0) * 0.8
        score += min(entity_count, 5) * 2.5
        score += min(action_count, 3) * 4.0
        if 1 <= span_days <= self.settings.TOPIC_CLUSTER_MAX_DAYS_SPAN:
            score += 5.0
        if news_count == 1:
            score -= 30.0
        if source_count < self.settings.TOPIC_MIN_SOURCE_COUNT:
            score -= 18.0
        return max(0.0, min(score, 120.0))

    def _passes_gate(self, features: List[TopicNewsFeature], score: float) -> bool:
        """
        输入:
        - 事件簇特征与质量分

        输出:
        - 是否进入 AI 审核

        作用:
        - 明确回答“不会所有新闻都生成专题”，单条弱事件在这里被拦截。
        """

        news_count = len(features)
        source_count = len({feature.source for feature in features if feature.source})
        if news_count < self.settings.TOPIC_MIN_NEWS_COUNT:
            return False
        if source_count < self.settings.TOPIC_MIN_SOURCE_COUNT:
            return False
        if score < self.settings.TOPIC_CLUSTER_MIN_SCORE:
            return False
        return True

    def _centroid(self, features: List[TopicNewsFeature]) -> List[float]:
        """
        输入:
        - 事件簇特征列表

        输出:
        - 平均向量

        作用:
        - 为后续新专题 embedding 或现有专题相似度判断提供低成本向量。
        """

        vectors = [feature.embedding for feature in features if feature.embedding]
        if not vectors:
            return []
        length = len(vectors[0])
        valid_vectors = [vec for vec in vectors if len(vec) == length]
        if not valid_vectors:
            return []
        centroid = np.asarray(valid_vectors, dtype=np.float32).mean(axis=0)
        return centroid.astype(float).tolist()

    def _attach_existing_topic_hint(
        self,
        candidate: TopicCandidate,
        active_topic_vecs: List[Tuple[Any, List[float]]],
    ) -> None:
        """
        输入:
        - 候选簇和现有专题向量

        输出:
        - 无，原地写入合并提示

        作用:
        - 给 AI 审核提供可能合并的现有专题，减少重复创建。
        """

        if not candidate.centroid or not active_topic_vecs:
            return
        best_topic: Optional[Any] = None
        best_sim = 0.0
        for topic, topic_vec in active_topic_vecs:
            sim = cosine_similarity(candidate.centroid, topic_vec)
            if sim > best_sim:
                best_topic = topic
                best_sim = sim
        if best_topic and best_sim >= self.settings.TOPIC_EXISTING_MERGE_HINT_THRESHOLD:
            candidate.existing_topic_id = int(getattr(best_topic, "id"))
            candidate.existing_topic_name = str(getattr(best_topic, "name", "") or "")
            candidate.existing_topic_similarity = best_sim

    def _time_range(self, features: List[TopicNewsFeature]) -> Tuple[Optional[datetime], Optional[datetime]]:
        times = [feature.publish_date for feature in features if feature.publish_date]
        if not times:
            return None, None
        return min(times), max(times)

    def _representative_titles(self, features: List[TopicNewsFeature]) -> List[str]:
        limit = self.settings.TOPIC_EVIDENCE_MAX_TITLES
        titles: List[str] = []
        seen: Set[str] = set()
        for feature in sorted(features, key=lambda item: item.heat_score, reverse=True):
            title = feature.clean_title or feature.title
            if not title or title in seen:
                continue
            seen.add(title)
            titles.append(title[:120])
            if len(titles) >= limit:
                break
        return titles

    def _fact_brief(self, features: List[TopicNewsFeature]) -> str:
        parts: List[str] = []
        char_limit = self.settings.TOPIC_EVIDENCE_MAX_CHARS
        for feature in sorted(features, key=lambda item: item.heat_score, reverse=True):
            summary = (feature.summary or "").replace("\n", " ").strip()
            text = feature.clean_title
            if summary:
                text = f"{feature.clean_title}：{summary[:180]}"
            if not text:
                continue
            if sum(len(part) for part in parts) + len(text) > char_limit:
                break
            parts.append(text)
        return "\n".join(parts)

    def _top_terms(self, term_groups: List[Set[str]], limit: int) -> List[str]:
        counter: Counter[str] = Counter()
        for terms in term_groups:
            counter.update(term for term in terms if term)
        return [term for term, _ in counter.most_common(limit)]

    def _most_common_text(self, values: Any) -> str:
        counter = Counter(value for value in values if value)
        if not counter:
            return ""
        return counter.most_common(1)[0][0]

    def _jaccard(self, left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        inter = len(left & right)
        union = len(left | right)
        if union <= 0:
            return 0.0
        return inter / union
