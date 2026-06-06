# 本文件用于封装首页新闻详情中的相似报道召回、过滤与重排业务逻辑。

from __future__ import annotations

import re
import time as perf_time
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from sqlalchemy.orm.attributes import set_committed_value

from app.core.logger import logger
from app.models.news import News
from app.utils.news_search import (
    NewsSearchResult,
    combined_search_score,
    core_term_coverage,
    cosine_similarity,
    normalize_similarity_terms,
    strong_core_term_coverage,
    text_match_score,
)

SIMILAR_NEWS_LIMIT = 10
SIMILAR_NEWS_KEYWORD_LIMIT = 8
SIMILAR_NEWS_ENTITY_LIMIT = 6
SIMILAR_NEWS_TEXT_RECALL_LIMIT = 60
SIMILAR_NEWS_EMBEDDING_RERANK_LIMIT = 24
SIMILAR_NEWS_VECTOR_STRONG_THRESHOLD = 0.62
SIMILAR_NEWS_VECTOR_TEXT_THRESHOLD = 0.5
SIMILAR_NEWS_TITLE_OVERLAP_THRESHOLD = 0.46
SIMILAR_NEWS_TITLE_TERM_LIMIT = 7
SIMILAR_NEWS_LOOKBACK_DAYS = 30
SIMILAR_NEWS_LOOKAHEAD_DAYS = 7


def _add_unique(values: list[str], seen: set[str], value: Any) -> None:
    """
    输入:
    - `values`: 待写入的词项列表
    - `seen`: 已出现词项集合
    - `value`: 候选词项

    输出:
    - 无

    作用:
    - 按原顺序清洗去重，保证查询词项稳定且不会重复扩大候选池。
    """

    text = str(value or "").strip()
    if not text:
        return
    key = text.lower()
    if key in seen:
        return
    seen.add(key)
    values.append(text)


def build_similar_news_query(news: News) -> tuple[str, list[str]]:
    """
    输入:
    - `news`: 当前详情页主新闻

    输出:
    - 查询文本与文本召回词项

    作用:
    - 使用标题、摘要、关键词和实体构造与首页搜索一致的语义召回输入，同时控制词项数量以兼顾速度。
    """

    terms: list[str] = []
    seen: set[str] = set()

    for keyword in normalize_similarity_terms(news.keywords, limit=SIMILAR_NEWS_KEYWORD_LIMIT):
        _add_unique(terms, seen, keyword)
    for entity in normalize_similarity_terms(news.entities, limit=SIMILAR_NEWS_ENTITY_LIMIT):
        _add_unique(terms, seen, entity)

    title = str(news.title or "").strip()
    summary = str(news.summary or "").strip()
    if title and (not terms or len(title) <= 18):
        _add_unique(terms, seen, title)
    if not terms and title:
        _add_unique(terms, seen, title)

    query_parts: list[str] = []
    for value in [title, summary[:180], " ".join(terms)]:
        text = str(value or "").strip()
        if text:
            query_parts.append(text)

    return " ".join(query_parts).strip(), terms


def _is_embedding_unloaded(news: News) -> bool:
    """
    输入:
    - `news`: 新闻对象

    输出:
    - embedding 字段是否处于延迟加载状态

    作用:
    - 避免在异步上下文中误触发 ORM 懒加载，导致相似报道接口变慢或报错。
    """

    try:
        state = sqlalchemy_inspect(news, raiseerr=False)
    except Exception:
        return False
    return bool(state is not None and "embedding" in state.unloaded)


def _loaded_embedding(news: News) -> Any:
    """
    输入:
    - `news`: 新闻对象

    输出:
    - 已加载的 embedding，未加载或不存在时返回 None

    作用:
    - 只读取内存中已有向量，不触发数据库懒加载。
    """

    if _is_embedding_unloaded(news):
        return None
    return getattr(news, "embedding", None)


async def _ensure_source_embedding(db: AsyncSession, source_news: News) -> Any:
    """
    输入:
    - `db`: 数据库会话
    - `source_news`: 当前详情页主新闻

    输出:
    - 主新闻向量或 None

    作用:
    - 当调用方延迟加载了主新闻向量时，只补一次单条查询。
    """

    embedding = _loaded_embedding(source_news)
    if embedding is not None or not _is_embedding_unloaded(source_news):
        return embedding

    row = await db.execute(select(News.embedding).where(News.id == source_news.id))
    embedding = row.scalar_one_or_none()
    set_committed_value(source_news, "embedding", embedding)
    return embedding


async def _load_candidate_embeddings(
    db: AsyncSession,
    candidates: list[tuple[float, News]],
) -> None:
    """
    输入:
    - `db`: 数据库会话
    - `candidates`: 需要参与向量重排的候选新闻

    输出:
    - 无

    作用:
    - 只为少量文本候选批量补齐向量，避免首轮 SQL 加载大量 embedding。
    """

    ids = [int(item.id) for _score, item in candidates if getattr(item, "id", None)]
    if not ids:
        return

    rows = await db.execute(select(News.id, News.embedding).where(News.id.in_(ids)))
    embedding_by_id = {int(news_id): embedding for news_id, embedding in rows.all()}
    for _score, item in candidates:
        set_committed_value(item, "embedding", embedding_by_id.get(int(item.id)))


def _title_recall_terms(terms: list[str]) -> list[str]:
    """
    输入:
    - `terms`: 相似报道查询词项

    输出:
    - 适合标题模糊召回的短词项

    作用:
    - 只保留少量明确主体/动作词，避免长标题或过多 OR 条件拖慢 SQLite 查询。
    """

    values: list[str] = []
    seen: set[str] = set()
    for term in terms:
        text = str(term or "").strip()
        if len(text) < 2 or len(text) > 24:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
        if len(values) >= SIMILAR_NEWS_TITLE_TERM_LIMIT:
            break
    return values


async def _recall_similar_text_candidates(
    db: AsyncSession,
    source_news: News,
    query_text: str,
    query_terms: list[str],
    *,
    limit: int,
) -> NewsSearchResult:
    """
    输入:
    - `db`: 数据库会话
    - `source_news`: 当前详情页主新闻
    - `query_text`: 相似报道查询上下文
    - `query_terms`: 关键词和实体词项
    - `limit`: 文本候选数量上限

    输出:
    - 轻量文本召回结果

    作用:
    - 用标题模糊匹配和发布时间窗口快速召回候选，避免通用搜索在 SQLite 中扫描摘要和 JSON 字段。
    """

    start_time = perf_time.perf_counter()
    recall_terms = _title_recall_terms(query_terms)
    stmt = (
        select(News)
        .options(defer(News.content), defer(News.embedding))
        .where(News.id != source_news.id)
    )

    if source_news.publish_date:
        window_start = source_news.publish_date - timedelta(days=SIMILAR_NEWS_LOOKBACK_DAYS)
        window_end = source_news.publish_date + timedelta(days=SIMILAR_NEWS_LOOKAHEAD_DAYS)
        stmt = stmt.where(News.publish_date >= window_start, News.publish_date <= window_end)

    if recall_terms:
        title_conditions = [News.title.ilike(f"%{term}%") for term in recall_terms]
        stmt = stmt.where(or_(*title_conditions))

    result = await db.execute(
        stmt.order_by(desc(News.heat_score), desc(News.publish_date)).limit(max(1, limit))
    )
    candidates = result.scalars().all()
    if len(candidates) < min(8, limit) and recall_terms:
        fallback_stmt = (
            select(News)
            .options(defer(News.content), defer(News.embedding))
            .where(News.id != source_news.id)
        )
        if source_news.publish_date:
            fallback_stmt = fallback_stmt.where(
                News.publish_date >= source_news.publish_date - timedelta(days=SIMILAR_NEWS_LOOKBACK_DAYS),
                News.publish_date <= source_news.publish_date + timedelta(days=SIMILAR_NEWS_LOOKAHEAD_DAYS),
            )
        fallback_result = await db.execute(
            fallback_stmt.order_by(desc(News.heat_score), desc(News.publish_date)).limit(80)
        )
        candidate_by_id = {int(item.id): item for item in candidates}
        for item in fallback_result.scalars().all():
            candidate_by_id.setdefault(int(item.id), item)
        candidates = list(candidate_by_id.values())

    scored: list[tuple[float, News]] = []
    now = datetime.now()
    max_heat = max((float(item.heat_score or 0.0) for item in candidates), default=0.0)
    for item in candidates:
        score = text_match_score(item, query_text, query_terms)
        core_coverage, core_matched, core_total = core_term_coverage(item, query_terms)
        strong_coverage, strong_matched, _ = strong_core_term_coverage(item, query_terms)
        title_overlap = _title_overlap_score(source_news, item)
        if core_total and core_matched <= 0 and title_overlap < SIMILAR_NEWS_TITLE_OVERLAP_THRESHOLD:
            continue
        if score <= 0 and title_overlap <= 0:
            continue

        final_score = combined_search_score(item, score, max_heat, now)
        if core_total:
            final_score *= 0.7 + core_coverage * 0.45
            if strong_matched:
                final_score *= 0.95 + strong_coverage * 0.18
            else:
                final_score *= 0.72
        if title_overlap > 0:
            final_score += min(0.25, title_overlap * 0.2)
        scored.append((final_score, item))

    scored.sort(key=lambda value: value[0], reverse=True)
    elapsed_ms = (perf_time.perf_counter() - start_time) * 1000
    return NewsSearchResult(
        items=scored[: max(1, limit)],
        total=len(scored),
        elapsed_ms=elapsed_ms,
        query=query_text,
        terms=query_terms,
        used_embedding=False,
    )


def _title_overlap_score(source_news: News, candidate: News) -> float:
    """
    输入:
    - `source_news`: 当前详情页主新闻
    - `candidate`: 候选相似新闻

    输出:
    - 标题字符片段重合分，范围 0 到 1

    作用:
    - 为没有向量的历史新闻提供严格文本兜底，只在标题高度相近时保留候选。
    """

    def normalize(text: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text or "").lower()

    def bigrams(text: str) -> set[str]:
        value = normalize(text)
        if len(value) < 4:
            return {value} if value else set()
        return {value[index : index + 2] for index in range(len(value) - 1)}

    left = bigrams(source_news.title or "")
    right = bigrams(candidate.title or "")
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap < 6:
        return 0.0
    return overlap / max(1, min(len(left), len(right)))


def _passes_similarity_filter(
    source_news: News,
    candidate: News,
    terms: list[str],
) -> tuple[bool, float]:
    """
    输入:
    - `source_news`: 当前详情页主新闻
    - `candidate`: 候选相似新闻
    - `terms`: 相似报道查询词项

    输出:
    - 是否通过后过滤，以及候选与主新闻的向量相似度

    作用:
    - 在统一搜索召回后增加相似报道专用过滤，降低只命中单个宽泛词或热度较高导致的误召回。
    """

    source_embedding = _loaded_embedding(source_news)
    candidate_embedding = _loaded_embedding(candidate)
    vector_similarity = cosine_similarity(source_embedding, candidate_embedding)
    core_coverage, core_matched, core_total = core_term_coverage(candidate, terms)
    strong_coverage, strong_matched, _ = strong_core_term_coverage(candidate, terms)
    has_vector = source_embedding is not None and candidate_embedding is not None and vector_similarity > 0
    title_overlap = _title_overlap_score(source_news, candidate)

    if has_vector and vector_similarity >= SIMILAR_NEWS_VECTOR_STRONG_THRESHOLD:
        return True, vector_similarity

    if has_vector and vector_similarity >= SIMILAR_NEWS_VECTOR_TEXT_THRESHOLD:
        if core_total == 0 or core_matched > 0 or strong_matched > 0:
            return True, vector_similarity

    if title_overlap >= SIMILAR_NEWS_TITLE_OVERLAP_THRESHOLD:
        return True, vector_similarity

    if core_total == 0:
        return False, vector_similarity

    if strong_matched > 0 and core_coverage >= 0.34:
        return True, vector_similarity

    if core_matched >= 2 and strong_coverage > 0:
        return True, vector_similarity

    return False, vector_similarity


def rerank_similar_news(
    source_news: News,
    search_result: NewsSearchResult,
    *,
    limit: int = SIMILAR_NEWS_LIMIT,
) -> list[tuple[float, News]]:
    """
    输入:
    - `source_news`: 当前详情页主新闻
    - `search_result`: 统一搜索函数返回的候选结果
    - `limit`: 返回数量上限

    输出:
    - 已过滤并重排的相似报道列表

    作用:
    - 融合首页搜索综合分和主新闻向量相似度，让同事件报道优先，同时保留无向量时的文本兜底。
    """

    reranked: list[tuple[float, News]] = []
    for score, item in search_result.items:
        passed, vector_similarity = _passes_similarity_filter(source_news, item, search_result.terms)
        if not passed:
            continue
        final_score = float(score)
        if vector_similarity > 0:
            final_score = final_score * 0.62 + vector_similarity * 0.38
        reranked.append((final_score, item))

    reranked.sort(key=lambda value: value[0], reverse=True)
    return reranked[: max(1, limit)]


async def find_similar_news(
    db: AsyncSession,
    source_news: News,
    *,
    limit: int = SIMILAR_NEWS_LIMIT,
) -> NewsSearchResult:
    """
    输入:
    - `db`: 数据库会话
    - `source_news`: 当前详情页主新闻
    - `limit`: 返回数量上限

    输出:
    - 使用统一搜索结果结构承载的相似报道结果

    作用:
    - 先复用首页搜索做轻量文本召回，再只为少量候选补向量重排，兼顾弹窗加载速度和同事件精度。
    """

    query_text, query_terms = build_similar_news_query(source_news)
    if not query_text:
        return NewsSearchResult([], 0, 0.0, "", [], False)

    source_vector_start = perf_time.perf_counter()
    source_embedding = await _ensure_source_embedding(db, source_news)
    source_vector_ms = (perf_time.perf_counter() - source_vector_start) * 1000

    search_result = await _recall_similar_text_candidates(
        db,
        source_news,
        query_text,
        query_terms=query_terms,
        limit=max(limit * 4, SIMILAR_NEWS_TEXT_RECALL_LIMIT),
    )
    candidate_vector_ms = 0.0
    if source_embedding is not None:
        candidate_vector_start = perf_time.perf_counter()
        await _load_candidate_embeddings(
            db,
            search_result.items[:SIMILAR_NEWS_EMBEDDING_RERANK_LIMIT],
        )
        candidate_vector_ms = (perf_time.perf_counter() - candidate_vector_start) * 1000
    items = rerank_similar_news(source_news, search_result, limit=limit)
    logger.info(
        f"/api/news/{source_news.id}/similar 相似报道重排完成: terms={len(search_result.terms)}, "
        f"matched={search_result.total}, returned={len(items)}, embedding={search_result.used_embedding}, "
        f"text_elapsed={search_result.elapsed_ms:.1f}ms, source_vector={source_vector_ms:.1f}ms, "
        f"candidate_vectors={candidate_vector_ms:.1f}ms"
    )
    return NewsSearchResult(
        items=items,
        total=len(items),
        elapsed_ms=search_result.elapsed_ms,
        query=search_result.query,
        terms=search_result.terms,
        used_embedding=search_result.used_embedding,
        relaxed=search_result.relaxed,
    )
