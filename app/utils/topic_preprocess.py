# 本文件用于专题发现前的新闻清洗、轻量特征提取和并查集合并工具。

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np


_NOISE_PATTERNS = [
    r"\s+-\s+May\s+\d{1,2},\s+\d{4}$",
    r"\s+-\s+\d{4}年\d{1,2}月\d{1,2}日$",
    r"\s*[\[\(（【].{0,12}(快讯|滚动|组图|视频|图集).{0,12}[\]\)）】]\s*",
    r"\s*[-_—|｜]\s*(新华网|央视新闻|中新网|环球网|观察者网).*$",
]

_PUNCT_PATTERN = re.compile(r"[\s,，。.!！?？:：;；、/\\|｜\-_=+*#《》<>“”\"'‘’`~]+")
_STOP_WORDS = {
    "the",
    "and",
    "with",
    "for",
    "from",
    "about",
    "新闻",
    "快讯",
    "消息",
    "最新",
    "报道",
    "回应",
    "表示",
    "称",
    "指出",
    "有关",
    "相关",
}
_ACTION_HINTS = {
    "冲突",
    "空袭",
    "袭击",
    "谈判",
    "停火",
    "爆炸",
    "调查",
    "通报",
    "制裁",
    "反制",
    "发射",
    "发布",
    "召回",
    "裁员",
    "起诉",
    "审判",
    "坠毁",
    "火灾",
    "地震",
    "洪水",
    "选举",
    "抗议",
    "会谈",
    "演习",
    "封锁",
    "撤离",
}


@dataclass
class TopicNewsFeature:
    """
    输入:
    - 单条新闻对象和可选向量

    输出:
    - 可用于聚类和打分的轻量特征对象

    作用:
    - 避免后续逻辑反复解析 ORM 对象字段。
    """

    news_id: int
    title: str
    clean_title: str
    summary: str
    source: str
    category: str
    region: str
    publish_date: Optional[datetime]
    heat_score: float
    keywords: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    title_terms: Set[str] = field(default_factory=set)
    action_terms: Set[str] = field(default_factory=set)
    embedding: List[float] = field(default_factory=list)

    @property
    def merge_terms(self) -> Set[str]:
        return self.keywords | self.entities | self.title_terms | self.action_terms


class UnionFind:
    """
    输入:
    - 元素数量

    输出:
    - 支持 find/union 的并查集结构

    作用:
    - 将满足同一事件条件的新闻下标合并为事件簇。
    """

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def clean_title(title: str) -> str:
    """
    输入:
    - 原始新闻标题

    输出:
    - 去除来源、日期和常见噪声后的标题

    作用:
    - 提升标题词重合、去重和证据包质量。
    """

    text = str(title or "").strip()
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -_—|｜")


def normalize_term(value: str) -> str:
    """
    输入:
    - 原始关键词、实体或标题切分词

    输出:
    - 规范化后的词

    作用:
    - 过滤过短、无信息量的词，减少误合并。
    """

    text = str(value or "").strip().lower()
    text = _PUNCT_PATTERN.sub("", text)
    if len(text) < 2 or text in _STOP_WORDS:
        return ""
    return text


def split_terms(value: Any) -> Set[str]:
    """
    输入:
    - 字符串、列表或其他可能承载关键词的值

    输出:
    - 规范化词集合

    作用:
    - 兼容新闻模型中的 keywords/entities/region 多种存储形态。
    """

    if value is None:
        return set()
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,，、\s]+", str(value))
    terms = {normalize_term(str(item)) for item in raw_items}
    return {term for term in terms if term}


def title_terms(title: str) -> Set[str]:
    """
    输入:
    - 清洗后的标题

    输出:
    - 标题中的候选词集合

    作用:
    - 在缺少模型关键词时提供最低成本的召回线索。
    """

    chunks = _PUNCT_PATTERN.split(title)
    terms = split_terms(chunks)
    compact = normalize_term(title)
    if 2 <= len(compact) <= 18:
        terms.add(compact)
    return terms


def action_terms(text: str) -> Set[str]:
    """
    输入:
    - 标题或摘要文本

    输出:
    - 命中的事件动作词

    作用:
    - 用主体加动作约束事件边界，减少泛话题聚合。
    """

    return {term for term in _ACTION_HINTS if term in text}


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """
    输入:
    - 两个向量

    输出:
    - 余弦相似度

    作用:
    - 供专题聚类在纯程序阶段复用。
    """

    if not left or not right or len(left) != len(right):
        return 0.0
    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def days_between(left: Optional[datetime], right: Optional[datetime]) -> int:
    """
    输入:
    - 两个发布时间

    输出:
    - 日期跨度天数

    作用:
    - 约束事件合并不要跨越过长时间。
    """

    if not left or not right:
        return 0
    return abs((left.date() - right.date()).days)


def build_news_feature(news: Any, embedding: Optional[List[float]] = None) -> TopicNewsFeature:
    """
    输入:
    - 新闻 ORM 对象或同字段对象
    - 可选新闻向量

    输出:
    - `TopicNewsFeature`

    作用:
    - 将新闻对象转换为聚类可消费的统一特征。
    """

    raw_title = str(getattr(news, "title", "") or "")
    clean = clean_title(raw_title)
    summary = str(getattr(news, "summary", "") or getattr(news, "content", "") or "")
    keywords = split_terms(getattr(news, "keywords", None))
    entities = split_terms(getattr(news, "entities", None))
    terms = title_terms(clean)
    actions = action_terms(f"{clean} {summary[:200]}")
    return TopicNewsFeature(
        news_id=int(getattr(news, "id")),
        title=raw_title,
        clean_title=clean,
        summary=summary,
        source=str(getattr(news, "source", "") or ""),
        category=str(getattr(news, "category", "") or ""),
        region=str(getattr(news, "region", "") or ""),
        publish_date=getattr(news, "publish_date", None),
        heat_score=float(getattr(news, "heat_score", 0.0) or 0.0),
        keywords=keywords,
        entities=entities,
        title_terms=terms,
        action_terms=actions,
        embedding=list(embedding or getattr(news, "embedding", None) or []),
    )


def group_by_root(union_find: UnionFind, size: int) -> Dict[int, List[int]]:
    """
    输入:
    - 并查集和元素数量

    输出:
    - 根节点到成员下标的映射

    作用:
    - 将合并结果转换为事件簇。
    """

    groups: Dict[int, List[int]] = {}
    for idx in range(size):
        root = union_find.find(idx)
        groups.setdefault(root, []).append(idx)
    return groups
