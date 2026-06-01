"""
本文件用于提供报表相关 API：生成全局报表、生成关键词报表、读取历史与图表数据等。
主要函数:
- `get_recent_reports`: 获取最近生成的报表
- `generate_global_report`: 生成并缓存全局报表
- `get_report_history`: 获取报表历史
"""

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Optional, List
import hashlib
import json

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import verify_admin_access
from app.api.endpoints.news import _semantic_news_search
from app.core.database import get_db
from app.models.news import News
from app.services.report_service import report_service
from app.services.task_manager import task_manager
from app.utils.news_query import build_news_query_filters, serialize_news_item

router = APIRouter(prefix="/api/report", tags=["report"])


def _valid_report_term(value: str) -> bool:
    text = (value or "").strip()
    return bool(text and text.lower() not in {"无内容", "分析失败", "暂无关键词", "其他", "其它", "null", "none"})


def _news_terms(news: News, limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in (news.keywords or []) + (news.entities or []):
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
    - 最近关键词报表列表

    作用:
    - 为前端展示最近生成的关键词报表入口
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
    - `report_type`: 报表类型 (global / keyword)
    - `keyword`: 关键词 (当 report_type=keyword 时需提供)

    输出:
    - 历史报表列表

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
    - `report_id`: 报表 ID

    输出:
    - 报表详情数据

    作用:
    - 获取指定历史报表的详细数据供前端渲染
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
    - 报表分析数据（摘要、图表数据、Top 新闻、AI 分析）

    作用:
    - 按条件生成报表数据，供前端图表渲染与下载
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


@router.get("/term-analysis")
async def get_report_term_analysis(
    term: str = Query(..., min_length=1, max_length=80),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = (term or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="term is required")

    stmt = build_news_query_filters(
        select(News),
        date="all",
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
    )
    scored, total, elapsed_ms = await _semantic_news_search(
        db,
        stmt,
        query,
        offset=0,
        limit=30,
        candidate_limit=2000,
        min_score=0.2,
        text_terms=[query],
    )
    news_list = [item for _, item in scored]

    related_news = [serialize_news_item(item) for item in news_list[:10]]
    dates: list[str] = []
    daily: dict[str, dict[str, float | int]] = defaultdict(lambda: {"heat": 0.0, "count": 0, "sentiment_sum": 0.0, "sentiment_count": 0})
    sentiment_counts: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()

    for item in news_list:
        day = (item.publish_date or datetime.now()).date().isoformat()
        dates.append(day)
        bucket = daily[day]
        bucket["count"] = int(bucket["count"]) + 1
        bucket["heat"] = float(bucket["heat"]) + float(item.heat_score or 0.0)
        if item.sentiment_score is not None:
            bucket["sentiment_sum"] = float(bucket["sentiment_sum"]) + float(item.sentiment_score)
            bucket["sentiment_count"] = int(bucket["sentiment_count"]) + 1
        sentiment_counts[_label_sentiment(item.sentiment_label)] += 1

        terms = _news_terms(item)
        for kw in terms:
            keyword_counter[kw] += 1
        for left, right in combinations(sorted(set(terms), key=str.lower)[:8], 2):
            pair_counter[(left, right)] += 1

    labels = sorted(set(dates))
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
                    if int(daily[label]["sentiment_count"]) else None
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

    return {
        "term": query,
        "summary": {
            "related_count": total,
            "returned_count": len(news_list),
            "total_heat": round(sum(float(item.heat_score or 0.0) for item in news_list), 2),
            "avg_sentiment": round(
                sum(float(item.sentiment_score or 0.0) for item in news_list if item.sentiment_score is not None)
                / max(1, len([item for item in news_list if item.sentiment_score is not None])),
                2,
            ),
            "sentiment_counts": dict(sentiment_counts),
            "elapsed_ms": round(elapsed_ms, 1),
        },
        "related_news": related_news,
        "trend": trend,
        "sentiment_trend": sentiment_trend,
        "cooccurrence": {"nodes": nodes, "links": links},
    }


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
    - 报表生成参数

    输出:
    - 任务提交状态

    作用:
    - 异步触发报表生成任务
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

    status = await task_manager.start_background(
        task_name,
        run_report_task,
        progress="报表生成中",
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
    - `report_id`: 报表ID (可选, 优先使用)

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
        logger.warning(f"⚠️ 流式请求未找到报表ID: q={q}")
        async def empty_generator():
            yield "报表未生成，请先点击生成报表"
        return StreamingResponse(empty_generator(), media_type="text/plain")
    
    logger.info(f"🚀 开始流式传输: report_id={final_report_id}")
    return StreamingResponse(
        report_service.stream_ai_analysis(final_report_id),
        media_type="text/plain"
    )
