"""
本文件用于提供新闻语义召回与多关键词模糊检索能力，供首页接口和智能体工具复用。
主要函数:
- `semantic_news_search`: 基于文本命中、向量相似度、时间新鲜度和热度综合召回新闻
- `build_soft_search_query`: 将关键词、分类和地区筛选转换为适合语义召回的查询文本
- `normalize_similarity_terms`: 清洗新闻关键词和实体，供相似新闻检索使用
"""

from __future__ import annotations

import re
import time as perf_time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from sqlalchemy import Text, cast, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.exceptions import AIConfigurationError
from app.core.logger import logger
from app.models.news import News
from app.services.ai_service import ai_service
from app.utils.tools import normalize_regions_to_countries

MILITARY_INTENT_TERMS = [
    "军事",
    "军方",
    "军队",
    "防务",
    "国防",
    "军演",
    "武器",
    "导弹",
    "美军",
    "军用",
    "无人机",
    "战机",
    "军舰",
    "航母",
    "部队",
    "五角大楼",
    "国防部",
]
US_INTENT_TERMS = ["美国", "美方", "美军", "五角大楼", "美国防部", "美国国防部", "华盛顿"]
CHINA_INTENT_TERMS = ["中国", "中方", "解放军"]


@dataclass
class NewsSearchResult:
    """
    输入:
    - 搜索召回后的新闻、总数、耗时和调试信息

    输出:
    - 结构化的搜索结果对象

    作用:
    - 让接口和智能体工具共享相同的返回结构，减少字段口径分叉。
    """

    items: list[tuple[float, News]]
    total: int
    elapsed_ms: float
    query: str
    terms: list[str]
    used_embedding: bool
    relaxed: bool = False


def normalize_similarity_terms(items: Any, limit: int = 16) -> list[str]:
    """
    输入:
    - `items`: 关键词或实体列表
    - `limit`: 最大保留数量

    输出:
    - 清洗去重后的词项列表

    作用:
    - 为相似新闻和智能体搜索构造稳定的关键词集合。
    """

    if not isinstance(items, list):
        return []

    values: list[str] = []
    seen: set[str] = set()
    ignored = {"无内容", "分析失败", "暂无关键词", "其他", "其它", "unknown", "none", "null"}
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value.lower() in ignored:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
        if len(values) >= limit:
            break
    return values


def split_search_terms(query_text: str, text_terms: Optional[list[str]] = None) -> list[str]:
    """
    输入:
    - `query_text`: 用户查询文本
    - `text_terms`: 可选的外部关键词列表

    输出:
    - 去重后的搜索词列表

    作用:
    - 支持空格、逗号、顿号、斜杠等分隔的多关键词检索，并保留完整查询句。
    """

    raw_terms = text_terms or re.split(r"[\s,，、/|;；]+", query_text or "")
    terms: list[str] = []
    full_query = (query_text or "").strip().lower()
    if full_query and text_terms is None:
        terms.append(full_query)
    for term in raw_terms:
        value = (term or "").strip().lower()
        if value and value not in terms:
            terms.append(value)
    return terms


def expand_search_terms(
    terms: list[str],
    *,
    category: Optional[str] = None,
    region: Optional[str] = None,
) -> list[str]:
    """
    输入:
    - `terms`: 原始搜索词
    - `category`/`region`: 可选分类和地区

    输出:
    - 增强后的搜索词列表

    作用:
    - 将“军事”等用户短词扩展到“时政军事、防务、军方”等同义线索，提高工具召回率。
    """

    expanded: list[str] = []
    seen: set[str] = set()

    def add(value: Optional[str]) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        expanded.append(text)

    for term in terms:
        add(term)

    has_military_intent = any(
        any(alias in term for alias in ["军事", "军方", "军队", "防务", "国防", "军演", "武器", "导弹", "美军"])
        for term in expanded
    )
    if has_military_intent:
        for alias in ["时政军事", "政治军事", "军事", "军方", "军队", "防务", "国防", "军演", "武器", "导弹"]:
            add(alias)

    has_us_intent = any(
        any(alias in term for alias in ["美国", "美方", "美军", "五角大楼", "华盛顿", "美国防部", "美国国防部"])
        for term in expanded
    )
    if has_us_intent:
        for alias in ["美国", "美方", "美军", "五角大楼", "美国防部", "美国国防部", "华盛顿"]:
            add(alias)

    has_china_intent = any(any(alias in term for alias in CHINA_INTENT_TERMS) for term in expanded)
    if has_china_intent:
        for alias in ["中国", "中方", "解放军", "国防部", "外交部"]:
            add(alias)

    category_text = str(category or "").strip()
    if category_text and category_text != "all":
        for item in re.split(r"[,，、/|;；\s]+", category_text):
            add(item)
            if "军事" in item:
                for alias in ["时政军事", "政治军事", "军事", "军方", "军队", "防务", "国防", "军演", "武器", "导弹"]:
                    add(alias)
            if item == "科技":
                add("科技科学")
            if item == "财经":
                add("财经商业")
            if item == "社会":
                add("社会民生")

    normalized_region = normalize_regions_to_countries(region) if region and region != "all" else ""
    if normalized_region and normalized_region not in {"其他", "全球"}:
        for item in normalized_region.split(","):
            add(item)
            if item == "美国":
                for alias in ["美国", "美方", "美军", "五角大楼", "美国防部", "美国国防部", "华盛顿"]:
                    add(alias)
            if item == "中国":
                for alias in ["中国", "中方", "解放军", "国防部", "外交部"]:
                    add(alias)

    return expanded


def detect_query_intents(terms: list[str]) -> dict[str, bool]:
    """
    输入:
    - `terms`: 查询词列表

    输出:
    - 查询中识别出的主题意图

    作用:
    - 判断是否需要对军事、国家等概念做结果覆盖度约束，避免宽泛分类命中过多无关新闻。
    """

    return {
        "military": any(any(alias in term for alias in MILITARY_INTENT_TERMS) for term in terms),
        "us": any(any(alias in term for alias in US_INTENT_TERMS) for term in terms),
        "china": any(any(alias in term for alias in CHINA_INTENT_TERMS) for term in terms),
    }


def _joined_news_text(news: News, *, include_category: bool = False) -> str:
    """
    输入:
    - `news`: 新闻对象
    - `include_category`: 是否纳入分类字段

    输出:
    - 用于意图命中的合并文本

    作用:
    - 将标题、摘要、关键词和实体聚合，避免只靠宽泛分类判断新闻主题。
    """

    parts = [news.title or "", news.summary or "", news.source or "", news.region or ""]
    if include_category:
        parts.append(news.category or "")
    for item in (news.keywords or []) + (news.entities or []):
        if isinstance(item, str):
            parts.append(item)
    return " ".join(parts)


def _has_any(text: str, aliases: list[str]) -> bool:
    """
    输入:
    - `text`: 待匹配文本
    - `aliases`: 候选词列表

    输出:
    - 是否命中任一候选词

    作用:
    - 统一处理意图覆盖判断。
    """

    return any(alias and alias in text for alias in aliases)


def matches_query_intents(news: News, intents: dict[str, bool]) -> bool:
    """
    输入:
    - `news`: 新闻对象
    - `intents`: 查询意图

    输出:
    - 新闻是否覆盖所有强查询意图

    作用:
    - 对“美国军事新闻”这类多条件查询，要求结果同时具备国家和主题线索。
    """

    text = _joined_news_text(news, include_category=False)
    if intents.get("military") and not _has_any(text, MILITARY_INTENT_TERMS):
        return False
    if intents.get("us") and not _has_any(text, US_INTENT_TERMS):
        return False
    if intents.get("china") and not _has_any(text, CHINA_INTENT_TERMS + ["国防部"]):
        return False
    return True


def intent_coverage_score(news: News, intents: dict[str, bool]) -> float:
    """
    输入:
    - `news`: 新闻对象
    - `intents`: 查询意图

    输出:
    - 0 到 1 之间的意图覆盖分

    作用:
    - 当强覆盖结果不足时，用覆盖分参与排序，而不是直接返回完全无关内容。
    """

    active = [key for key, enabled in intents.items() if enabled]
    if not active:
        return 1.0

    text = _joined_news_text(news, include_category=False)
    category_text = news.category or ""
    score = 0.0
    for key in active:
        if key == "military":
            if _has_any(text, MILITARY_INTENT_TERMS):
                score += 1.0
            elif "军事" in category_text:
                score += 0.15
        elif key == "us":
            score += 1.0 if _has_any(text, US_INTENT_TERMS) else 0.0
        elif key == "china":
            score += 1.0 if _has_any(text, CHINA_INTENT_TERMS + ["国防部"]) else 0.0
    return score / len(active)


def build_soft_search_query(
    q: str,
    *,
    category: Optional[str] = None,
    region: Optional[str] = None,
) -> tuple[str, list[str]]:
    """
    输入:
    - `q`: 用户搜索词
    - `category`/`region`: 可选筛选条件

    输出:
    - 组合后的查询文本和增强词列表

    作用:
    - 将硬筛选条件转化为语义检索线索，避免分类或地区字段不精确时直接返回 0。
    """

    base_terms = split_search_terms(q)
    terms = expand_search_terms(base_terms, category=category, region=region)
    query_parts: list[str] = []
    base_text = " ".join(str(value or "").strip() for value in [q, region, category] if str(value or "").strip())
    for value in [q, region, category]:
        text = str(value or "").strip()
        if text and text not in query_parts and not any(text in existing for existing in query_parts):
            query_parts.append(text)
    for term in terms:
        text = str(term or "").strip()
        if text and text not in query_parts and text not in base_text:
            query_parts.append(text)
    return " ".join(query_parts).strip(), terms


def text_match_score(news: News, query_text: str, terms: list[str]) -> float:
    """
    输入:
    - `news`: 新闻对象
    - `query_text`: 完整查询文本
    - `terms`: 搜索词列表

    输出:
    - 0 到 1 之间的文本相关性分数

    作用:
    - 综合标题、摘要、来源、分类、地区、关键词和实体命中情况，作为语义召回的基础分。
    """

    lowered_q = (query_text or "").strip().lower()
    title = (news.title or "").lower()
    summary = (news.summary or "").lower()
    source = (news.source or "").lower()
    category = (news.category or "").lower()
    region = (news.region or "").lower()
    item_terms = [
        t.strip().lower()
        for t in (news.keywords or []) + (news.entities or [])
        if isinstance(t, str) and t.strip()
    ]

    score = 0.0
    if lowered_q:
        if lowered_q in title:
            score += 0.72
        if lowered_q in summary:
            score += 0.18
        if lowered_q in source or lowered_q in category or lowered_q in region:
            score += 0.2
        if lowered_q in item_terms:
            score += 0.35

    for term in terms:
        value = str(term or "").strip().lower()
        if not value:
            continue
        if value in title:
            score += 0.14
        if value in summary:
            score += 0.06
        if value in source or value in category or value in region:
            score += 0.09
        if value in item_terms:
            score += 0.12

    return min(score, 1.0)


def _recency_score(publish_date: Optional[datetime], now: datetime) -> float:
    """
    输入:
    - `publish_date`: 新闻发布时间
    - `now`: 当前时间

    输出:
    - 时间新鲜度分数

    作用:
    - 让近期新闻在语义结果中有适度优先级。
    """

    if not publish_date:
        return 0.0
    try:
        age_days = max(0.0, (now - publish_date).total_seconds() / 86400)
    except TypeError:
        return 0.0
    if age_days <= 1:
        return 1.0
    if age_days <= 3:
        return 0.86
    if age_days <= 7:
        return 0.72
    if age_days <= 30:
        return 0.52
    if age_days <= 180:
        return 0.28
    if age_days <= 365:
        return 0.16
    return 0.08


def _heat_score(heat: Optional[float], max_heat: float) -> float:
    """
    输入:
    - `heat`: 新闻热度
    - `max_heat`: 候选池最大热度

    输出:
    - 归一化热度分数

    作用:
    - 在相关性相近时，让高热新闻更靠前。
    """

    value = max(0.0, float(heat or 0.0))
    if max_heat <= 0:
        return 0.0
    return min(1.0, float(np.log1p(value) / np.log1p(max_heat)))


def combined_search_score(news: News, relevance: float, max_heat: float, now: datetime) -> float:
    """
    输入:
    - `news`: 新闻对象
    - `relevance`: 文本和向量相关性
    - `max_heat`: 候选池最大热度
    - `now`: 当前时间

    输出:
    - 综合排序分数

    作用:
    - 按相关性、时效性和热度综合排序，避免纯热度覆盖用户意图。
    """

    relevance = max(0.0, min(1.0, float(relevance or 0.0)))
    freshness = _recency_score(news.publish_date, now)
    heat = _heat_score(news.heat_score, max_heat)
    relevance_gate = min(1.0, relevance / 0.45)
    return relevance * 0.55 + freshness * 0.35 * relevance_gate + heat * 0.10 * relevance_gate


def _build_text_conditions(terms: list[str]) -> list[Any]:
    """
    输入:
    - `terms`: 搜索词列表

    输出:
    - SQLAlchemy 文本匹配条件列表

    作用:
    - 在多个字段中构造模糊匹配候选召回条件。
    """

    conditions: list[Any] = []
    for term in terms:
        value = str(term or "").strip()
        if not value:
            continue
        like = f"%{value}%"
        conditions.extend(
            [
                News.title.ilike(like),
                News.summary.ilike(like),
                News.source.ilike(like),
                News.category.ilike(like),
                News.region.ilike(like),
                cast(News.keywords, Text).ilike(like),
                cast(News.entities, Text).ilike(like),
            ]
        )
    return conditions


async def _load_embedding(query_text: str, log_prefix: str) -> Optional[np.ndarray]:
    """
    输入:
    - `query_text`: 查询文本
    - `log_prefix`: 日志前缀

    输出:
    - 查询向量或 None

    作用:
    - 封装向量生成失败时的降级逻辑，保证搜索仍可用。
    """

    try:
        q_emb_list = await ai_service.get_embeddings([query_text])
        return np.array(q_emb_list[0]) if q_emb_list and q_emb_list[0] else None
    except AIConfigurationError as e:
        logger.warning(f"{log_prefix} 语义搜索不可用，退回文本匹配: {e}")
        return None


async def semantic_news_search(
    db: AsyncSession,
    stmt: Any,
    query_text: str,
    *,
    offset: int = 0,
    limit: int = 20,
    candidate_limit: int = 2000,
    min_score: float = 0.2,
    text_terms: Optional[list[str]] = None,
    log_prefix: str = "新闻搜索",
) -> NewsSearchResult:
    """
    输入:
    - `db`: 数据库会话
    - `stmt`: 已追加基础过滤的 SQLAlchemy 查询
    - `query_text`: 查询文本
    - `offset`/`limit`: 分页参数
    - `candidate_limit`: 候选池上限
    - `min_score`: 最低相关性阈值
    - `text_terms`: 可选关键词列表
    - `log_prefix`: 日志前缀

    输出:
    - `NewsSearchResult`

    作用:
    - 同时进行多关键词文本召回、近期高热候选召回和向量相似度重排，支持首页和智能体工具统一搜索。
    """

    search_start = perf_time.perf_counter()
    query_text = (query_text or "").strip()
    if not query_text:
        return NewsSearchResult([], 0, 0.0, query_text, [], False)

    normalized_terms = expand_search_terms(split_search_terms(query_text, text_terms))
    intents = detect_query_intents(normalized_terms)
    text_conditions = _build_text_conditions(normalized_terms)

    q_vec = await _load_embedding(query_text, log_prefix)
    candidate_options = [defer(News.content)]
    if q_vec is None:
        candidate_options.append(defer(News.embedding))

    candidate_by_id: dict[int, News] = {}

    if text_conditions:
        text_stmt = (
            stmt.options(*candidate_options)
            .where(or_(*text_conditions))
            .order_by(desc(News.publish_date), desc(News.heat_score))
            .limit(max(candidate_limit, offset + limit))
        )
        text_result = await db.execute(text_stmt)
        for item in text_result.scalars().all():
            candidate_by_id[item.id] = item

    broad_stmt = (
        stmt.options(*candidate_options)
        .order_by(desc(News.publish_date), desc(News.heat_score))
        .limit(candidate_limit)
    )
    broad_result = await db.execute(broad_stmt)
    for item in broad_result.scalars().all():
        candidate_by_id[item.id] = item

    candidates = list(candidate_by_id.values())
    now = datetime.now()
    max_heat = max((float(n.heat_score or 0.0) for n in candidates), default=0.0)

    norm_q = float(np.linalg.norm(q_vec)) if q_vec is not None else 0.0
    q_dim = len(q_vec) if q_vec is not None else 0
    scored_news: list[tuple[float, News]] = []
    for item in candidates:
        coverage = intent_coverage_score(item, intents)
        if intents and any(intents.values()) and coverage <= 0:
            continue
        score = text_match_score(item, query_text, normalized_terms)

        if q_vec is not None and item.embedding and len(item.embedding) == q_dim:
            n_vec = np.array(item.embedding)
            norm_n = np.linalg.norm(n_vec)
            if norm_q > 0 and norm_n > 0:
                sim = np.dot(q_vec, n_vec) / (norm_q * norm_n)
                score += max(0.0, float(sim))

        if score >= min_score:
            final_score = combined_search_score(item, score, max_heat, now)
            if intents and any(intents.values()):
                final_score *= 0.35 + coverage * 0.65
            scored_news.append((final_score, item))

    scored_news.sort(key=lambda x: x[0], reverse=True)
    if intents and any(intents.values()):
        strict_news = [(score, item) for score, item in scored_news if matches_query_intents(item, intents)]
        if len(strict_news) >= min(limit, 3):
            scored_news = strict_news
    elapsed_ms = (perf_time.perf_counter() - search_start) * 1000
    return NewsSearchResult(
        items=scored_news[offset : offset + limit],
        total=len(scored_news),
        elapsed_ms=elapsed_ms,
        query=query_text,
        terms=normalized_terms,
        used_embedding=q_vec is not None,
    )
