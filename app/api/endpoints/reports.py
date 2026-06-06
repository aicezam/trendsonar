"""
本文件用于提供报告相关 API：生成全局报告、生成关键词报告、读取历史与图表数据等。
主要函数:
- `get_recent_reports`: 获取最近生成的报告
- `generate_global_report`: 生成并缓存全局报告
- `get_report_history`: 获取报告历史
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Optional, List
import hashlib
import json
import re
import time as perf_time

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import Text, cast, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import verify_admin_access
from app.core.database import get_db
from app.models.news import News
from app.services.report_service import report_service
from app.services.task_manager import task_manager
from app.utils.news_query import build_news_query_filters
from app.utils.news_ranking import sort_news_by_composite_score
from app.utils.ttl_cache import TtlMemoryCache
from app.utils.tools import normalize_regions_to_countries

router = APIRouter(prefix="/api/report", tags=["report"])
TERM_ANALYSIS_CACHE_TTL_SECONDS = 300
TERM_ANALYSIS_CACHE_SIZE = 96
TERM_ANALYSIS_SAMPLE_LIMIT = 600
TERM_ANALYSIS_COOCCURRENCE_SAMPLE_LIMIT = 800
_TERM_ANALYSIS_CACHE = TtlMemoryCache[dict[str, Any]](
    ttl_seconds=TERM_ANALYSIS_CACHE_TTL_SECONDS,
    max_size=TERM_ANALYSIS_CACHE_SIZE,
)


def _clean_cache_part(value: Optional[str]) -> str:
    """
    输入:
    - `value`: 可选查询参数

    输出:
    - 适合作为缓存键的规范化字符串

    作用:
    - 统一清洗查询参数，避免大小写或空白差异造成重复缓存。
    """

    return str(value or "").strip().lower()


def _term_analysis_cache_key(
    *,
    query: str,
    start_date: Optional[str],
    end_date: Optional[str],
    category: Optional[str],
    region: Optional[str],
    source: Optional[str],
    range_key: str,
) -> tuple[str, str, str, str, str, str, str]:
    """
    输入:
    - 词项分析接口的筛选参数

    输出:
    - 稳定的缓存键元组

    作用:
    - 将同一词项和同一筛选条件的重复请求映射到同一个短期缓存结果。
    """

    return (
        _clean_cache_part(query),
        _clean_cache_part(start_date),
        _clean_cache_part(end_date),
        _clean_cache_part(category),
        _clean_cache_part(region),
        _clean_cache_part(source),
        _clean_cache_part(range_key),
    )


def _valid_report_term(value: str) -> bool:
    text = (value or "").strip()
    return bool(text and text.lower() not in {"无内容", "分析失败", "暂无关键词", "其他", "其它", "null", "none"})


def _news_terms(news: Any, limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    if isinstance(news, dict):
        raw_terms = (news.get("keywords") or []) + (news.get("entities") or [])
    else:
        raw_terms = (news.keywords or []) + (news.entities or [])
    for item in raw_terms:
        if not isinstance(item, str) or not _valid_report_term(item):
            continue
        value = item.strip()
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(value)
        if len(terms) >= limit:
            break
    return terms


def _label_sentiment(label: str) -> str:
    text = (label or "").strip()
    if text == "正面":
        return "positive"
    if text == "负面":
        return "negative"
    return "neutral"


def _db_dialect_name(db: AsyncSession) -> str:
    """
    输入:
    - `db`: 数据库会话

    输出:
    - 当前数据库方言名称

    作用:
    - 让词项分析在 PostgreSQL 上使用更精确的短英文词匹配策略，同时兼容 SQLite。
    """

    try:
        bind = db.get_bind()
    except Exception:
        return ""
    return str(getattr(getattr(bind, "dialect", None), "name", "") or "").lower()


def _is_short_latin_term(query: str) -> bool:
    """
    输入:
    - `query`: 词项文本

    输出:
    - 是否为容易误命中的短英文/数字词

    作用:
    - 对 AI、GDP 等短词避免直接使用 `%term%` 扫描摘要造成误召回和慢查询。
    """

    return bool(re.fullmatch(r"[A-Za-z0-9]{1,3}", (query or "").strip()))


def _latin_word_match(column: Any, query: str) -> Any:
    """
    输入:
    - `column`: 待匹配的 SQLAlchemy 字段
    - `query`: 短英文词项

    输出:
    - PostgreSQL 正则匹配条件

    作用:
    - 用非字母数字边界匹配短英文词，减少 `said`、`main` 等普通词片段误命中 `AI`。
    """

    escaped = re.escape((query or "").strip())
    pattern = rf"(^|[^[:alnum:]_]){escaped}([^[:alnum:]_]|$)"
    return column.op("~*")(pattern)


def _term_match_conditions(query: str, *, dialect_name: str = "") -> Any:
    like = f"%{query}%"
    if "postgresql" in dialect_name and _is_short_latin_term(query):
        return or_(
            _latin_word_match(News.title, query),
            cast(News.keywords, Text).ilike(like),
            cast(News.entities, Text).ilike(like),
        )
    return or_(
        News.title.ilike(like),
        News.summary.ilike(like),
        cast(News.keywords, Text).ilike(like),
        cast(News.entities, Text).ilike(like),
    )


def _score_term_news(item: Any, query: str) -> float:
    def get_value(name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    lowered = query.lower()
    score = 0.0
    title = (get_value("title", "") or "").lower()
    summary = (get_value("summary", "") or "").lower()
    terms = [
        t.strip().lower()
        for t in (get_value("keywords", []) or []) + (get_value("entities", []) or [])
        if isinstance(t, str) and t.strip()
    ]
    if lowered in terms:
        score += 5.0
    if any(lowered in term for term in terms):
        score += 2.0
    if lowered in title:
        score += 1.5
    if lowered in summary:
        score += 0.5
    score += min(float(get_value("heat_score", 0.0) or 0.0) / 1000, 1.0)
    return score


def _serialize_term_news_row(item: dict[str, Any]) -> dict[str, Any]:
    publish_date = item.get("publish_date")
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "url": item.get("url"),
        "source": item.get("source"),
        "heat": item.get("heat_score"),
        "time": publish_date.isoformat() if publish_date else None,
        "summary": item.get("summary"),
        "sources": item.get("sources"),
        "category": item.get("category"),
        "region": normalize_regions_to_countries(item.get("region")),
        "sentiment_label": item.get("sentiment_label"),
        "sentiment_score": item.get("sentiment_score"),
    }


@router.get("/recent")
async def get_recent_reports(
    limit: int = Query(10, ge=1, le=50),
    keyword: Optional[str] = None,
):
    """
    输入:
    - `limit`: 返回数量上限
    - `keyword`: 关键词（可选；若指定则返回该关键词下最近记录）

    输出:
    - 最近关键词报告列表

    作用:
    - 为前端展示最近生成的关键词报告入口
    """
    return await report_service.get_recent_reports(limit, keyword)


@router.get("/history")
async def get_report_history(
    limit: int = Query(20, ge=1, le=100),
    report_type: str = Query("keyword", regex="^(global|keyword)$"),
    keyword: Optional[str] = None,
):
    """
    输入:
    - `limit`: 返回数量
    - `report_type`: 报告类型 (global / keyword)
    - `keyword`: 关键词 (当 report_type=keyword 时需提供)

    输出:
    - 历史报告列表

    作用:
    - 管理后台查看历史生成记录
    """
    if keyword:
        return await report_service.get_report_history(keyword, limit)

    return await report_service.get_global_history(limit)


@router.get("/load/{report_id}")
async def load_report(report_id: int):
    """
    输入:
    - `report_id`: 报告 ID

    输出:
    - 报告详情数据

    作用:
    - 获取指定历史报告的详细数据供前端渲染
    """
    report = await report_service.load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/analysis")
async def get_report_analysis(
    q: Optional[str] = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    limit: Optional[int] = None,
    generate_ai: Optional[bool] = False,
    ids: Optional[str] = None,
    nocache: Optional[bool] = False,
):
    """
    输入:
    - `q`: 关键词（可选）
    - `start_date`/`end_date`: 起止日期（可选）
    - `category`/`region`/`source`: 过滤条件（可选）
    - `limit`: 取样上限（可选）
    - `generate_ai`: 是否生成 AI 分析文字结论

    输出:
    - 报告分析数据（摘要、图表数据、Top 新闻、AI 分析）

    作用:
    - 按条件生成报告数据，供前端图表渲染与下载
    """
    news_ids: Optional[List[int]] = None
    if ids:
        try:
            news_ids = [int(x) for x in ids.split(",") if x.strip()]
        except Exception:
            news_ids = None

    return await report_service.get_analysis_data(
        keyword=q,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
        limit=limit,
        generate_ai=generate_ai,
        use_cache=False if news_ids else True,
        news_ids=news_ids,
        save_cache=False if nocache else True,
    )


@router.get("/chart-data")
async def get_report_chart_data(
    type: str = Query(..., min_length=1, max_length=40),
    q: Optional[str] = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    输入:
    - `type`: 图表类型
    - `q`/`start_date`/`end_date`/`category`/`region`: 报告筛选条件
    - `limit`/`offset`: 列表类图表分页参数

    输出:
    - 指定图表或列表数据

    作用:
    - 支持报告页单个图表刷新，以及热门舆情列表继续加载。
    """

    return await report_service.get_chart_data(
        type=type,
        q=q or "",
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
        limit=limit,
        offset=offset,
    )


@router.get("/term-analysis")
async def get_report_term_analysis(
    term: str = Query(..., min_length=1, max_length=80),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    range: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - `term`: 需要分析的词项
    - `start_date`/`end_date`: 可选日期范围
    - `category`/`region`/`source`: 可选筛选条件
    - `range`: 前端快捷范围
    - `db`: 数据库会话

    输出:
    - 词项概览、趋势、情绪趋势、共现关系和相关报道样本

    作用:
    - 为首页和报告页词项弹窗提供聚合分析，并使用短缓存和受控样本量降低远程数据库与 Python 计算压力。
    """

    query = (term or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="term is required")
    elapsed_start = perf_time.perf_counter()
    range_key = (range or "year").strip().lower()
    if not start_date and not end_date and range_key != "all":
        now = datetime.now()
        if range_key == "year":
            start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

    cache_key = _term_analysis_cache_key(
        query=query,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
        range_key=range_key,
    )
    cached = _TERM_ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        payload = deepcopy(cached)
        summary = payload.setdefault("summary", {})
        summary["cached"] = True
        summary["elapsed_ms"] = round((perf_time.perf_counter() - elapsed_start) * 1000, 1)
        return payload

    term_condition = _term_match_conditions(query, dialect_name=_db_dialect_name(db))

    day_expr = func.date(News.publish_date)
    aggregate_stmt = build_news_query_filters(
        select(
            day_expr.label("day"),
            News.sentiment_label.label("sentiment_label"),
            func.count().label("count"),
            func.coalesce(func.sum(News.heat_score), 0.0).label("heat"),
            func.coalesce(func.sum(News.sentiment_score), 0.0).label("sentiment_sum"),
            func.count(News.sentiment_score).label("sentiment_count"),
        ),
        date="all",
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
    ).where(term_condition).group_by(day_expr, News.sentiment_label).order_by(day_expr)
    aggregate_rows = (await db.execute(aggregate_stmt)).all()

    total = 0
    total_heat = 0.0
    total_sentiment_sum = 0.0
    total_sentiment_count = 0
    sentiment_counts: Counter[str] = Counter()
    daily: dict[str, dict[str, Any]] = {}
    for day, label, count, heat, sentiment_sum, sentiment_count in aggregate_rows:
        row_count = int(count or 0)
        row_heat = float(heat or 0.0)
        row_sentiment_sum = float(sentiment_sum or 0.0)
        row_sentiment_count = int(sentiment_count or 0)

        total += row_count
        total_heat += row_heat
        total_sentiment_sum += row_sentiment_sum
        total_sentiment_count += row_sentiment_count
        sentiment_counts[_label_sentiment(label)] += row_count

        if day:
            day_key = str(day)
            item = daily.setdefault(
                day_key,
                {"count": 0, "heat": 0.0, "sentiment_sum": 0.0, "sentiment_count": 0},
            )
            item["count"] += row_count
            item["heat"] += row_heat
            item["sentiment_sum"] += row_sentiment_sum
            item["sentiment_count"] += row_sentiment_count

    sample_base_stmt = build_news_query_filters(
        select(
            News.id,
            News.title,
            News.url,
            News.source,
            News.heat_score,
            News.publish_date,
            News.summary,
            News.sources,
            News.category,
            News.region,
            News.sentiment_label,
            News.sentiment_score,
            News.keywords,
            News.entities,
        ),
        date="all",
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
    ).where(term_condition)
    heat_rows = [
        dict(row)
        for row in (
            await db.execute(
                sample_base_stmt.order_by(desc(News.heat_score), desc(News.publish_date)).limit(
                    TERM_ANALYSIS_SAMPLE_LIMIT
                )
            )
        ).mappings().all()
    ]
    recent_rows = [
        dict(row)
        for row in (
            await db.execute(
                sample_base_stmt.order_by(desc(News.publish_date), desc(News.heat_score)).limit(
                    TERM_ANALYSIS_SAMPLE_LIMIT
                )
            )
        ).mappings().all()
    ]
    sample_map: dict[int, dict[str, Any]] = {}
    for item in heat_rows + recent_rows:
        news_id = int(item.get("id") or 0)
        if news_id:
            sample_map[news_id] = item
    sample_rows = sort_news_by_composite_score(sample_map.values())

    related_news = [_serialize_term_news_row(item) for item in sample_rows[:10]]
    keyword_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()

    for item in sample_rows[:TERM_ANALYSIS_COOCCURRENCE_SAMPLE_LIMIT]:
        terms = _news_terms(item)
        for kw in terms:
            keyword_counter[kw] += 1
        for left, right in combinations(sorted(set(terms), key=str.lower)[:8], 2):
            pair_counter[(left, right)] += 1

    labels = sorted(daily.keys())
    trend = {
        "dates": labels,
        "heat": [round(float(daily[label]["heat"]), 2) for label in labels],
        "count": [int(daily[label]["count"]) for label in labels],
    }
    sentiment_trend = {
        "dates": labels,
        "series": [
            {
                "name": "情绪均值",
                "type": "line",
                "smooth": True,
                "data": [
                    round(float(daily[label]["sentiment_sum"]) / max(1, int(daily[label]["sentiment_count"])), 2)
                    if int(daily[label]["sentiment_count"]) > 0 else None
                    for label in labels
                ],
            }
        ],
    }

    top_terms = keyword_counter.most_common(18)
    nodes = [{"name": name, "value": int(value), "symbolSize": min(58, 16 + int(value) * 4)} for name, value in top_terms]
    known = {node["name"] for node in nodes}
    links = [
        {"source": left, "target": right, "value": int(value)}
        for (left, right), value in pair_counter.most_common(30)
        if left in known and right in known
    ]

    payload = {
        "term": query,
        "summary": {
            "related_count": total,
            "returned_count": len(sample_rows),
            "total_heat": round(float(total_heat or 0.0), 2),
            "avg_sentiment": round(total_sentiment_sum / max(1, total_sentiment_count), 2)
            if total_sentiment_count > 0 else 0.0,
            "sentiment_counts": dict(sentiment_counts),
            "elapsed_ms": round((perf_time.perf_counter() - elapsed_start) * 1000, 1),
        },
        "related_news": related_news,
        "trend": trend,
        "sentiment_trend": sentiment_trend,
        "cooccurrence": {"nodes": nodes, "links": links},
    }
    _TERM_ANALYSIS_CACHE.set(cache_key, payload)
    return payload


@router.post("/generate")
async def generate_report_background(
    q: Optional[str] = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    输入:
    - 报告生成参数

    输出:
    - 任务提交状态

    作用:
    - 异步触发报告生成任务
    """
    params = {
        "q": q or "",
        "start_date": start_date or "",
        "end_date": end_date or "",
        "category": category or "",
        "region": region or "",
        "source": source or "",
        "limit": limit or 0,
    }
    task_hash = hashlib.sha1(json.dumps(params, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    task_name = f"report:{task_hash}"

    async def run_report_task() -> None:
        await report_service.generate_report_and_stream_ai(
            keyword=q,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            limit=limit,
        )

    if q:
        report_id = await report_service.generate_report_and_stream_ai(
            keyword=q,
            start_date=start_date,
            end_date=end_date,
            category=category,
            region=region,
            source=source,
            limit=limit,
        )
        if not report_id:
            raise HTTPException(status_code=500, detail="Report generation failed")
        return {"status": "pending", "report_id": report_id}

    status = await task_manager.start_background(
        task_name,
        run_report_task,
        progress="报告生成中",
    )
    return {"status": "running" if status.get("running") else status.get("status"), "task": status}


@router.delete("/cache/{report_id}", dependencies=[Depends(verify_admin_access)])
async def delete_report_cache(report_id: int):
    ok = await report_service.delete_report_cache(report_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"status": "ok"}


@router.get("/stream_ai")
async def stream_ai_report(
    q: Optional[str] = "",
    report_id: Optional[int] = None,
):
    """
    输入:
    - `q`: 关键词 (可选)
    - `report_id`: 报告ID (可选, 优先使用)

    输出:
    - AI 分析流式响应 (text/plain)

    作用:
    - 实时流式输出 AI 综述内容
    """
    from app.core.logger import logger
    logger.info(f"📡 收到流式 AI 请求: report_id={report_id} q={q}")

    final_report_id = report_id
    if not final_report_id and q:
        final_report_id = await report_service.find_latest_report_id(q)
    
    if not final_report_id:
        logger.warning(f"⚠️ 流式请求未找到报告ID: q={q}")
        async def empty_generator():
            yield "报告未生成，请先点击生成报告"
        return StreamingResponse(empty_generator(), media_type="text/plain")
    
    logger.info(f"🚀 开始流式传输: report_id={final_report_id}")
    return StreamingResponse(
        report_service.stream_ai_analysis(final_report_id),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
