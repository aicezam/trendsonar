"""
本文件用于提供新闻语义召回与多关键词模糊检索能力，供首页接口和智能体工具复用。
主要函数:
- `semantic_news_search`: 基于文本命中、向量相似度、时间新鲜度和热度综合召回新闻
- `build_soft_search_query`: 将关键词、分类和地区筛选转换为适合语义召回的查询文本
- `build_search_query_variants`: 为智能体生成多组可复用搜索词，提升复杂问法召回率
- `normalize_similarity_terms`: 清洗新闻关键词和实体，供相似新闻检索使用
"""

from __future__ import annotations

import re
import time as perf_time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from sqlalchemy import Text, cast, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.exceptions import AIConfigurationError
from app.core.logger import logger
from app.models.news import News
from app.services.ai_service import ai_service
from app.utils.tools import normalize_regions_to_countries

GENERIC_QUERY_TERMS = {
    "新闻",
    "报道",
    "相关",
    "近期",
    "最近",
    "一个月",
    "近一个月",
    "过去一个月",
    "本月",
    "今天",
    "昨天",
    "昨日",
    "本周",
    "今年",
    "热点",
    "事件",
    "情况",
    "信息",
    "内容",
    "国际",
    "国内",
    "时政",
    "政治",
    "财经",
    "商业",
    "科技",
    "科学",
    "社会",
    "民生",
    "文娱",
    "体育",
    "综合",
    "会晤",
    "访问",
    "外国",
    "别国",
    "国家",
    "时政军事",
    "财经商业",
    "科技科学",
    "社会民生",
    "文娱体育",
    "其他",
    "全球",
}

TERM_SPLIT_PATTERN = r"[\s,，、/|;；]+"
MAX_COMPACT_PHRASE_LENGTH = 24
MAX_CORE_TERM_LENGTH = 18


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

    raw_terms = text_terms or re.split(TERM_SPLIT_PATTERN, query_text or "")
    terms: list[str] = []
    full_query = (query_text or "").strip().lower()
    if full_query and text_terms is None:
        terms.append(full_query)
        compact_query = compact_search_phrase(full_query)
        split_count = len([term for term in raw_terms if str(term or "").strip()])
        if (
            compact_query
            and compact_query != full_query
            and compact_query not in terms
            and 1 < split_count <= 2
            and len(compact_query) <= MAX_COMPACT_PHRASE_LENGTH
        ):
            terms.append(compact_query)
    for term in raw_terms:
        value = (term or "").strip().lower()
        if value and value not in terms:
            terms.append(value)
    return terms


def compact_search_phrase(value: str) -> str:
    """
    输入:
    - `value`: 原始搜索文本

    输出:
    - 去除空白和常见分隔符后的短语

    作用:
    - 兼容中文短语中间带空格或分隔符的写法差异，避免精确事件被拆词结果淹没。
    """

    text = str(value or "").strip().lower()
    if not text:
        return ""
    compact = re.sub(TERM_SPLIT_PATTERN, "", text)
    if compact == text:
        return text
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", compact))
    return compact if has_cjk and len(compact) >= 2 else text


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
    - 合并用户关键词、分类和地区线索，不在通用搜索层写死具体领域同义词。
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

    category_text = str(category or "").strip()
    if category_text and category_text != "all":
        for item in re.split(TERM_SPLIT_PATTERN, category_text):
            add(item)

    normalized_region = normalize_regions_to_countries(region) if region and region != "all" else ""
    if normalized_region and normalized_region not in {"其他", "全球"}:
        for item in normalized_region.split(","):
            add(item)

    return expanded


def detect_query_intents(terms: list[str]) -> dict[str, bool]:
    """
    输入:
    - `terms`: 查询词列表

    输出:
    - 查询中识别出的强约束意图

    作用:
    - 保留扩展点，默认不对任何领域做硬编码强约束。
    """

    return {}


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
    - 保留强约束过滤扩展点，默认不过滤任何领域。
    """

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

    return 1.0


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


def _is_generic_query_term(term: str) -> bool:
    """
    输入:
    - `term`: 候选查询词

    输出:
    - 是否属于通用占位词

    作用:
    - 生成查询变体时保留人名、国家、机构等主体词，过滤“新闻、最近、访问”等宽泛词。
    """

    value = str(term or "").strip().lower()
    if not value:
        return True
    return value in GENERIC_QUERY_TERMS or value.isdigit()


def _is_usable_core_term(term: str) -> bool:
    """
    输入:
    - `term`: 候选核心词

    输出:
    - 是否适合作为相关性锚点

    作用:
    - 过滤整句、过长串和通用占位词，保留用户明确给出的人名、机构、地点、产品或事件词。
    """

    value = str(term or "").strip()
    if not value or _is_generic_query_term(value):
        return False
    if len(value) > MAX_CORE_TERM_LENGTH:
        return False
    if any(separator in value for separator in [" ", ",", "，", "、", "/", "|", ";", "；"]):
        return False
    return True


def _important_query_terms(terms: list[str], limit: int = 4) -> list[str]:
    """
    输入:
    - `terms`: 原始查询词列表
    - `limit`: 最多保留的主体词数量

    输出:
    - 去重后的重要主体词

    作用:
    - 从用户问题里提取可作为查询变体锚点的非泛化关键词。
    """

    important: list[str] = []
    seen: set[str] = set()

    def add_important(value: str) -> None:
        text = str(value or "").strip()
        if not text or _is_generic_query_term(text):
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        important.append(text)

    for term in terms:
        value = str(term or "").strip()
        if not _is_usable_core_term(value):
            continue
        add_important(value)
        if len(important) >= limit:
            break
    return important


def _term_bigrams(terms: list[str], limit: int = 3, *, anchor_to_first: bool = False) -> list[str]:
    """
    输入:
    - `terms`: 关键词列表
    - `limit`: 最多生成数量
    - `anchor_to_first`: 是否只保留包含首个核心词的组合

    输出:
    - 相邻关键词组合列表

    作用:
    - 为“主体 + 动作/对象”生成更精确的中文短语变体，并避免后半段宽泛组合覆盖主查询意图。
    """

    values = [term for term in terms if _is_usable_core_term(term)]
    bigrams: list[str] = []
    seen: set[str] = set()
    anchor = values[0] if values and anchor_to_first else ""
    for left, right in zip(values, values[1:]):
        if anchor and anchor not in {left, right}:
            continue
        compact = compact_search_phrase(f"{left} {right}")
        if not compact or compact in seen or compact in {left, right}:
            continue
        seen.add(compact)
        bigrams.append(compact)
        if len(bigrams) >= limit:
            break
    return bigrams


def build_search_query_variants(
    q: str,
    *,
    category: Optional[str] = None,
    region: Optional[str] = None,
    max_variants: int = 5,
) -> list[tuple[str, list[str]]]:
    """
    输入:
    - `q`: 用户或智能体传入的搜索文本
    - `category`/`region`: 可选分类和地区线索
    - `max_variants`: 最大查询变体数量

    输出:
    - `(查询文本, 查询词列表)` 元组列表

    作用:
    - 让单次智能体工具调用内部执行多组检索，例如同时尝试原始问题、去空格短语和重要词组合。
    """

    query = str(q or "").strip()
    variants: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    def add(raw_query: str, raw_terms: Optional[list[str]] = None) -> None:
        text = str(raw_query or "").strip()
        if not text:
            return
        query_text, terms = build_soft_search_query(text, category=category, region=region)
        if raw_terms:
            terms = expand_search_terms(split_search_terms(query_text, raw_terms), category=category, region=region)
        key = query_text.lower()
        if not key or key in seen:
            return
        seen.add(key)
        variants.append((query_text, terms))

    raw_terms = [term.strip() for term in re.split(TERM_SPLIT_PATTERN, query) if term.strip()]
    important_terms = _important_query_terms(raw_terms, limit=12)
    visible_terms = important_terms[:4 if len(raw_terms) > 4 else 6]
    if len(raw_terms) > 4 and important_terms:
        add(" ".join(visible_terms), important_terms)
    else:
        add(query)

    if len(important_terms) > 2:
        for phrase in _term_bigrams(important_terms[:6], anchor_to_first=True):
            add(phrase, [phrase])

    if len(raw_terms) > 4:
        add(query)
    elif 1 < len(important_terms) <= 6:
        add(" ".join(important_terms), important_terms)

    return variants[: max(1, max_variants)]


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
    has_specific_terms = any(not _is_generic_query_term(term) for term in terms)

    score = 0.0
    if lowered_q:
        if lowered_q in title:
            score += 0.72
        if lowered_q in summary:
            score += 0.18
        if lowered_q in source or lowered_q in category or lowered_q in region:
            score += 0.06
        if lowered_q in item_terms:
            score += 0.35

        compact_q = compact_search_phrase(lowered_q)
        if compact_q and compact_q != lowered_q:
            compact_title = compact_search_phrase(title)
            compact_summary = compact_search_phrase(summary)
            compact_item_terms = [compact_search_phrase(term) for term in item_terms]
            if compact_q in compact_title:
                score += 0.86
            if compact_q in compact_summary:
                score += 0.26
            if compact_q in compact_item_terms:
                score += 0.42

    for term in terms:
        value = str(term or "").strip().lower()
        if not value:
            continue
        if has_specific_terms and _is_generic_query_term(value):
            continue
        if value in title:
            score += 0.18
        if value in summary:
            score += 0.07
        if value in source or value in category or value in region:
            score += 0.025
        if value in item_terms:
            score += 0.16

        compact_value = compact_search_phrase(value)
        if compact_value and compact_value != value:
            if compact_value in compact_search_phrase(title):
                score += 0.22
            if compact_value in compact_search_phrase(summary):
                score += 0.09

    return min(score, 1.0)


def _core_query_terms(terms: list[str], limit: int = 6) -> list[str]:
    """
    输入:
    - `terms`: 搜索词列表
    - `limit`: 最大核心词数量

    输出:
    - 去重后的核心词列表

    作用:
    - 抽取用户显式提供的短主体词，用于限制向量和热度对搜索结果的过度影响。
    """

    core_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = str(term or "").strip().lower()
        if not _is_usable_core_term(value):
            continue
        key = compact_search_phrase(value)
        if not key or key in seen:
            continue
        seen.add(key)
        core_terms.append(value)
        if len(core_terms) >= limit:
            break
    return core_terms


def _anchor_query_term(terms: list[str]) -> str:
    """
    输入:
    - `terms`: 搜索词列表

    输出:
    - 第一个可用核心词，若不存在则返回空字符串

    作用:
    - 将用户问题中靠前的主体词作为召回锚点，避免动作词或地点词单独命中导致结果跑偏。
    """

    core_terms = _core_query_terms(terms, limit=1)
    return core_terms[0] if core_terms else ""


def anchor_term_matched(news: News, terms: list[str]) -> bool:
    """
    输入:
    - `news`: 新闻对象
    - `terms`: 搜索词列表

    输出:
    - 新闻文本是否命中查询锚点词

    作用:
    - 对多核心词查询要求主体锚点出现，降低只命中动作或地点的弱相关结果。
    """

    anchor = _anchor_query_term(terms)
    if not anchor:
        return True
    text = compact_search_phrase(_joined_news_text(news, include_category=False).lower())
    compact_anchor = compact_search_phrase(anchor)
    return bool(compact_anchor and compact_anchor in text)


def core_term_coverage(news: News, terms: list[str]) -> tuple[float, int, int]:
    """
    输入:
    - `news`: 新闻对象
    - `terms`: 搜索词列表

    输出:
    - 覆盖率、命中数量、核心词总数

    作用:
    - 衡量新闻正文是否真正覆盖用户给出的核心词，避免宽泛语义召回压过精确查询。
    """

    core_terms = _core_query_terms(terms)
    if not core_terms:
        return 1.0, 0, 0

    parts = [news.title or "", news.summary or ""]
    for item in (news.keywords or []) + (news.entities or []):
        if isinstance(item, str):
            parts.append(item)
    text = compact_search_phrase(" ".join(parts).lower())
    matched = 0
    for term in core_terms:
        compact_term = compact_search_phrase(term)
        if compact_term and compact_term in text:
            matched += 1
    return matched / len(core_terms), matched, len(core_terms)


def strong_core_term_coverage(news: News, terms: list[str]) -> tuple[float, int, int]:
    """
    输入:
    - `news`: 新闻对象
    - `terms`: 搜索词列表

    输出:
    - 标题、关键词、实体字段中的核心词覆盖率、命中数量、核心词总数

    作用:
    - 区分标题/标签强相关和摘要中的顺带提及，提升特定事件搜索的结果精度。
    """

    core_terms = _core_query_terms(terms)
    if not core_terms:
        return 1.0, 0, 0

    parts = [news.title or ""]
    for item in (news.keywords or []) + (news.entities or []):
        if isinstance(item, str):
            parts.append(item)
    text = compact_search_phrase(" ".join(parts).lower())
    matched = 0
    for term in core_terms:
        compact_term = compact_search_phrase(term)
        if compact_term and compact_term in text:
            matched += 1
    return matched / len(core_terms), matched, len(core_terms)


def _meets_core_term_requirement(matched: int, total: int) -> bool:
    """
    输入:
    - `matched`: 已命中的核心词数量
    - `total`: 查询中的核心词总数

    输出:
    - 是否达到最低核心词覆盖要求

    作用:
    - 单核心词查询允许命中 1 个；多核心词查询至少命中 2 个，避免单个宽泛词把无关新闻带入结果。
    """

    if total <= 0:
        return True
    return matched > 0


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
    return relevance * 0.7 + freshness * 0.2 * relevance_gate + heat * 0.1 * relevance_gate


def _apply_search_score_modifiers(
    news: News,
    relevance: float,
    *,
    max_heat: float,
    now: datetime,
    coverage: float,
    has_intents: bool,
    core_total: int,
    core_coverage: float,
    strong_matched: int,
    strong_coverage: float,
) -> float:
    """
    输入:
    - `news`: 候选新闻
    - `relevance`: 文本和向量融合后的相关性分
    - `max_heat`/`now`: 热度和时间归一化所需上下文
    - `coverage`/`has_intents`: 查询意图覆盖情况
    - `core_total`/`core_coverage`/`strong_matched`/`strong_coverage`: 核心词命中情况

    输出:
    - 应用核心词和意图修正后的最终排序分

    作用:
    - 统一文本预排序与向量重排的打分修正逻辑，避免两个阶段排序口径发散。
    """

    final_score = combined_search_score(news, relevance, max_heat, now)
    if core_total:
        final_score *= 0.6 + core_coverage * 0.5
        if strong_matched:
            final_score *= 0.92 + strong_coverage * 0.25
        else:
            final_score *= 0.65
    if has_intents:
        final_score *= 0.35 + coverage * 0.65
    return final_score


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


def _build_core_text_conditions(terms: list[str]) -> list[Any]:
    """
    输入:
    - `terms`: 搜索词列表

    输出:
    - 针对核心词的 SQLAlchemy 文本匹配条件列表

    作用:
    - 为特定事件搜索提供锚点候选，避免只从宽泛分类、地区或热榜候选中取数。
    """

    conditions: list[Any] = []
    for term in _core_query_terms(terms):
        like = f"%{term}%"
        conditions.extend(
            [
                News.title.ilike(like),
                News.summary.ilike(like),
                cast(News.keywords, Text).ilike(like),
                cast(News.entities, Text).ilike(like),
            ]
        )
    return conditions


def _build_single_term_text_conditions(term: str) -> list[Any]:
    """
    输入:
    - `term`: 单个查询词

    输出:
    - 单词跨字段模糊匹配条件

    作用:
    - 让多关键词查询中的每个词都能独立进入候选池，避免宽 OR 查询按时间截断时漏掉目标新闻。
    """

    value = str(term or "").strip()
    if not value:
        return []
    like = f"%{value}%"
    return [
        News.title.ilike(like),
        News.summary.ilike(like),
        News.source.ilike(like),
        News.category.ilike(like),
        News.region.ilike(like),
        cast(News.keywords, Text).ilike(like),
        cast(News.entities, Text).ilike(like),
    ]


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


def _coerce_query_vector(query_vector: Any) -> Optional[np.ndarray]:
    """
    输入:
    - `query_vector`: 外部传入的查询向量，可能是列表或 NumPy 数组

    输出:
    - 可参与相似度计算的一维 NumPy 数组，非法时返回 None

    作用:
    - 允许调用方复用已有新闻向量，避免相似新闻查询重复请求 embedding 接口。
    """

    if query_vector is None:
        return None
    try:
        vector = np.asarray(query_vector, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector


def cosine_similarity(left: Any, right: Any) -> float:
    """
    输入:
    - `left`/`right`: 两个待比较的向量

    输出:
    - 余弦相似度，无法计算时返回 0

    作用:
    - 为新闻搜索、相似报道和后过滤统一提供轻量向量相似度计算。
    """

    left_vec = _coerce_query_vector(left)
    right_vec = _coerce_query_vector(right)
    if left_vec is None or right_vec is None or left_vec.size != right_vec.size:
        return 0.0
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


async def _load_embeddings_for_candidates(
    db: AsyncSession,
    candidates: list[tuple[float, News, dict[str, Any]]],
) -> dict[int, Any]:
    """
    输入:
    - `db`: 数据库会话
    - `candidates`: 需要参与向量重排的候选新闻

    输出:
    - `{新闻 ID: embedding}` 映射

    作用:
    - 只为文本预排序靠前的一小批候选批量加载向量，减少远程数据库传输和 Python 向量计算量。
    """

    ids = [int(item.id) for _score, item, _meta in candidates if getattr(item, "id", None)]
    if not ids:
        return {}
    rows = await db.execute(select(News.id, News.embedding).where(News.id.in_(ids)))
    return {int(news_id): embedding for news_id, embedding in rows.all() if embedding is not None}


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
    use_embedding: bool = True,
    query_vector: Any = None,
    extra_candidate_limit: int = 0,
    max_core_terms: int = 12,
    per_term_candidate_limit: Optional[int] = None,
    min_text_candidate_limit: int = 120,
    embedding_rerank_limit: int = 80,
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
    - `use_embedding`: 是否启用向量重排
    - `query_vector`: 可选的预置查询向量，传入后优先复用
    - `extra_candidate_limit`: 额外召回高热候选的数量，用于向量兜底重排
    - `max_core_terms`: 最多按多少个核心词分别查询候选
    - `per_term_candidate_limit`: 单个核心词候选上限
    - `min_text_candidate_limit`: 文本候选查询的最低上限，默认保持首页搜索行为
    - `embedding_rerank_limit`: 文本预排序后最多加载多少条候选向量参与重排

    输出:
    - `NewsSearchResult`

    作用:
    - 同时进行多关键词文本召回、近期高热候选召回和向量相似度重排，支持首页和智能体工具统一搜索。
    """

    search_start = perf_time.perf_counter()
    query_text = (query_text or "").strip()
    if not query_text:
        return NewsSearchResult([], 0, 0.0, query_text, [], False)

    base_terms = split_search_terms(query_text, text_terms)
    normalized_terms = expand_search_terms(base_terms)
    intents = detect_query_intents(normalized_terms)
    text_conditions = _build_text_conditions(normalized_terms)
    core_terms = _core_query_terms(normalized_terms, limit=max(0, max_core_terms))

    q_vec = _coerce_query_vector(query_vector)
    if q_vec is None and use_embedding:
        q_vec = await _load_embedding(query_text, log_prefix)
    candidate_options = [defer(News.content), defer(News.embedding)]

    candidate_by_id: dict[int, News] = {}

    if text_conditions:
        text_limit = max(min_text_candidate_limit, min(candidate_limit, offset + limit * 6))
        text_stmt = (
            stmt.options(*candidate_options)
            .where(or_(*text_conditions))
            .order_by(desc(News.heat_score), desc(News.publish_date))
            .limit(text_limit)
        )
        text_result = await db.execute(text_stmt)
        for item in text_result.scalars().all():
            candidate_by_id[item.id] = item

    if per_term_candidate_limit is None:
        per_term_limit = max(100, min(260, offset + limit * 5))
    else:
        per_term_limit = max(1, min(candidate_limit, per_term_candidate_limit))
    for term in core_terms:
        single_conditions = _build_single_term_text_conditions(term)
        if not single_conditions:
            continue
        single_stmt = (
            stmt.options(*candidate_options)
            .where(or_(*single_conditions))
            .order_by(desc(News.heat_score), desc(News.publish_date))
            .limit(per_term_limit)
        )
        single_result = await db.execute(single_stmt)
        for item in single_result.scalars().all():
            candidate_by_id[item.id] = item

    if q_vec is not None and extra_candidate_limit > 0:
        broad_limit = max(1, min(candidate_limit, extra_candidate_limit))
        broad_stmt = (
            stmt.options(*candidate_options)
            .order_by(desc(News.heat_score), desc(News.publish_date))
            .limit(broad_limit)
        )
        broad_result = await db.execute(broad_stmt)
        for item in broad_result.scalars().all():
            candidate_by_id[item.id] = item

    if not text_conditions:
        broad_stmt = (
            stmt.options(*candidate_options)
            .order_by(desc(News.heat_score), desc(News.publish_date))
            .limit(candidate_limit)
        )
        broad_result = await db.execute(broad_stmt)
        for item in broad_result.scalars().all():
            candidate_by_id[item.id] = item

    candidates = list(candidate_by_id.values())
    now = datetime.now()
    max_heat = max((float(n.heat_score or 0.0) for n in candidates), default=0.0)

    has_intents = bool(intents and any(intents.values()))
    scored_candidates: list[tuple[float, News, dict[str, Any]]] = []
    for item in candidates:
        coverage = intent_coverage_score(item, intents)
        if has_intents and coverage <= 0:
            continue
        score = text_match_score(item, query_text, normalized_terms)
        core_coverage, core_matched, core_total = core_term_coverage(item, normalized_terms)
        strong_coverage, strong_matched, _strong_total = strong_core_term_coverage(item, normalized_terms)
        if not _meets_core_term_requirement(core_matched, core_total):
            continue

        if score >= min_score:
            final_score = _apply_search_score_modifiers(
                item,
                score,
                max_heat=max_heat,
                now=now,
                coverage=coverage,
                has_intents=has_intents,
                core_total=core_total,
                core_coverage=core_coverage,
                strong_matched=strong_matched,
                strong_coverage=strong_coverage,
            )
            scored_candidates.append(
                (
                    final_score,
                    item,
                    {
                        "text_score": score,
                        "coverage": coverage,
                        "core_total": core_total,
                        "core_coverage": core_coverage,
                        "strong_matched": strong_matched,
                        "strong_coverage": strong_coverage,
                    },
                )
            )

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    if q_vec is not None and scored_candidates and embedding_rerank_limit > 0:
        rerank_limit = min(len(scored_candidates), max(offset + limit * 3, embedding_rerank_limit))
        embedding_by_id = await _load_embeddings_for_candidates(db, scored_candidates[:rerank_limit])
        norm_q = float(np.linalg.norm(q_vec))
        q_dim = len(q_vec)
        reranked_candidates: list[tuple[float, News, dict[str, Any]]] = []
        for final_score, item, meta in scored_candidates:
            embedding = embedding_by_id.get(int(item.id))
            if embedding is None or len(embedding) != q_dim or norm_q <= 0:
                reranked_candidates.append((final_score, item, meta))
                continue
            n_vec = np.asarray(embedding, dtype=np.float32)
            norm_n = float(np.linalg.norm(n_vec))
            if norm_n <= 0:
                reranked_candidates.append((final_score, item, meta))
                continue
            sim = float(np.dot(q_vec, n_vec) / (norm_q * norm_n))
            semantic_score = float(meta["text_score"]) + max(0.0, sim)
            reranked_score = _apply_search_score_modifiers(
                item,
                semantic_score,
                max_heat=max_heat,
                now=now,
                coverage=float(meta["coverage"]),
                has_intents=has_intents,
                core_total=int(meta["core_total"]),
                core_coverage=float(meta["core_coverage"]),
                strong_matched=int(meta["strong_matched"]),
                strong_coverage=float(meta["strong_coverage"]),
            )
            reranked_candidates.append((reranked_score, item, meta))
        reranked_candidates.sort(key=lambda x: x[0], reverse=True)
        scored_candidates = reranked_candidates

    scored_news = [(score, item) for score, item, _meta in scored_candidates]
    if has_intents:
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
