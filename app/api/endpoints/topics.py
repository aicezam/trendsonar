"""
本文件用于提供“专题追踪”相关 API：专题列表、专题详情（时间轴/完整记录/来源新闻）。
主要函数:
- `get_topics`: 获取专题列表
- `get_topic_detail`: 获取专题详情
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy import desc, select, asc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.deps import verify_admin_access
from app.models.news import News
from app.models.topic import Topic, TopicTimelineItem
from app.services.topic_service import topic_service

router = APIRouter(prefix="/api", tags=["topics"])


from pydantic import BaseModel

class TopicCreate(BaseModel):
    name: str

class TopicUpdate(BaseModel):
    name: str


def _collect_timeline_news_ids(items: List[TopicTimelineItem]) -> set[int]:
    """
    输入:
    - `items`: 专题时间轴条目

    输出:
    - 时间轴关联到的新闻 ID 集合

    作用:
    - 兼容旧字段 news_id 和新字段 sources，供详情和趋势接口复用。
    """

    news_ids: set[int] = set()
    for item in items:
        if item.news_id:
            news_ids.add(item.news_id)
        if item.sources:
            for source in item.sources:
                if isinstance(source, dict) and source.get("id"):
                    try:
                        news_ids.add(int(source["id"]))
                    except (TypeError, ValueError):
                        continue
    return news_ids


def _date_range(start: datetime, end: datetime) -> list[str]:
    days = max(1, min((end.date() - start.date()).days + 1, 120))
    return [(start.date() + timedelta(days=i)).isoformat() for i in range(days)]

@router.post("/topics/manual_create")
async def manual_create_topic(
    payload: TopicCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_admin_access)
) -> Dict:
    """
    手动创建专题并触发扫描（仅管理员）
    """
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="专题名称不能为空")
        
    try:
        # 创建专题但不立即扫描，避免阻塞接口
        topic = await topic_service.create_manual_topic(db, payload.name.strip(), trigger_scan=False)
        
        # 显式提交事务，确保后台任务能读到最新数据
        await db.commit()
        await db.refresh(topic)
        
        # 添加后台任务进行扫描，并允许包含已归类的新闻
        background_tasks.add_task(topic_service.run_topic_scan_in_background, topic.id, include_used=True)
        
    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "message": "专题创建成功，系统正在后台扫描匹配相关新闻...",
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "status": topic.status
        }
    }

@router.patch("/topics/{topic_id}")
async def update_topic(
    topic_id: int,
    payload: TopicUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_admin_access)
) -> Dict:
    """
    更新专题名称（仅管理员）
    """
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="专题名称不能为空")
        
    try:
        # 更新名称但不立即扫描
        topic = await topic_service.update_topic_name(db, topic_id, payload.name.strip(), trigger_scan=False)
        
        # 显式提交事务，确保后台任务能读到最新数据
        await db.commit()
        await db.refresh(topic)
        
        # 添加后台任务进行扫描，并允许包含已归类的新闻
        background_tasks.add_task(topic_service.run_topic_scan_in_background, topic.id, include_used=True)
        
    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
        
    return {
        "message": "专题名称已更新，系统正在后台重新扫描匹配相关新闻...",
        "topic": {
            "id": topic.id,
            "name": topic.name
        }
    }


@router.get("/topics/list")
async def get_topics_list(
    page: int = 1,
    size: int = 20,
    db: AsyncSession = Depends(get_db)
) -> Dict:
    stmt = select(Topic).where(Topic.status == "active").order_by(desc(Topic.updated_time))
    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    
    items = (await db.execute(stmt.offset((page - 1) * size).limit(size))).scalars().all()
    
    return {
        "total": total,
        "items": [
            {
                "id": t.id,
                "name": t.name,
                "summary": t.summary,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "updated_time": t.updated_time.isoformat() if t.updated_time else None,
                "heat_score": float(t.heat_score or 0.0),
            }
            for t in items
        ],
        "page": page,
        "size": size
    }

@router.delete("/topics/{topic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_topic(
    topic_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_admin_access)
):
    """
    删除专题（仅管理员）
    """
    topic = (await db.execute(select(Topic).where(Topic.id == topic_id))).scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="专题不存在")
        
    await db.delete(topic)
    await db.commit()
    return None


@router.get("/topics/{topic_id}")
async def get_topic_detail(
    topic_id: int, 
    sort: str = Query("asc", pattern="^(asc|desc)$"),
    db: AsyncSession = Depends(get_db)
) -> Dict:
    topic = (await db.execute(select(Topic).where(Topic.id == topic_id))).scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="专题不存在")

    # 默认按时间正序排列
    sort_func = asc if sort == "asc" else desc
    items_stmt = (
        select(TopicTimelineItem)
        .where(TopicTimelineItem.topic_id == topic_id)
        .order_by(sort_func(TopicTimelineItem.event_time))
        .limit(200)
    )
    items = (await db.execute(items_stmt)).scalars().all()

    # 收集所有相关的新闻 ID，包括旧的 news_id 和新的 sources 字段
    news_ids = _collect_timeline_news_ids(items)

    news_cards: List[Dict] = []
    if news_ids:
        news_stmt = select(News).where(News.id.in_(list(news_ids))).order_by(desc(News.publish_date))
        news_list = (await db.execute(news_stmt)).scalars().all()
        
        for n in news_list:
            news_cards.append(
                {
                    "id": n.id,
                    "title": n.title,
                    "url": n.url,
                    "source": n.source,
                    "time": n.publish_date.isoformat() if n.publish_date else None,
                    "heat": float(n.heat_score or 0.0),
                    "summary": n.summary,
                    "sources": n.sources,
                    "category": n.category,
                    "region": n.region,
                    "sentiment_label": n.sentiment_label,
                    "sentiment_score": float(n.sentiment_score or 0.0),
                }
            )

    timeline: List[Dict] = []
    for it in items:
        timeline.append(
            {
                "id": it.id,
                "time": it.event_time.isoformat() if it.event_time else None,
                "content": it.content,
                "news_id": it.news_id,
                "news_title": it.news_title,
                "source_name": it.source_name,
                "source_url": it.source_url,
                "sources": it.sources,
            }
        )

    return {
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "summary": topic.summary,
            "record": topic.record,
            "start_time": topic.start_time.isoformat() if topic.start_time else None,
            "updated_time": topic.updated_time.isoformat() if topic.updated_time else None,
            "heat_score": float(topic.heat_score or 0.0),
            "summary": topic.summary,
            "record": topic.record,
        },
        "timeline": timeline,
        "news": news_cards,
    }


@router.get("/topics/{topic_id}/trends")
async def get_topic_trends(topic_id: int, db: AsyncSession = Depends(get_db)) -> Dict:
    """
    输入:
    - `topic_id`: 专题 ID

    输出:
    - 专题趋势仪表盘数据

    作用:
    - 聚合专题相关新闻的热度、数量、情感、来源和关键词变化。
    """

    topic = (await db.execute(select(Topic).where(Topic.id == topic_id))).scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="专题不存在")

    timeline_items = (
        await db.execute(
            select(TopicTimelineItem)
            .where(TopicTimelineItem.topic_id == topic_id)
            .order_by(asc(TopicTimelineItem.event_time))
            .limit(500)
        )
    ).scalars().all()

    news_ids = _collect_timeline_news_ids(timeline_items)
    news_list: List[News] = []
    if news_ids:
        news_list = (
            await db.execute(select(News).where(News.id.in_(list(news_ids))).order_by(asc(News.publish_date)))
        ).scalars().all()

    if news_list:
        min_time = min((n.publish_date for n in news_list if n.publish_date), default=topic.start_time or datetime.now())
        max_time = max((n.publish_date for n in news_list if n.publish_date), default=topic.updated_time or datetime.now())
    else:
        min_time = topic.start_time or datetime.now()
        max_time = topic.updated_time or min_time

    labels = _date_range(min_time, max_time)
    daily = {
        label: {
            "news_count": 0,
            "heat": 0.0,
            "sentiment_total": 0.0,
            "sentiment_count": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        }
        for label in labels
    }

    source_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    source_daily: dict[str, Counter[str]] = defaultdict(Counter)
    keyword_daily: dict[str, Counter[str]] = defaultdict(Counter)

    for item in news_list:
        if not item.publish_date:
            continue
        day = item.publish_date.date().isoformat()
        if day not in daily:
            daily[day] = {
                "news_count": 0,
                "heat": 0.0,
                "sentiment_total": 0.0,
                "sentiment_count": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
            }
            labels.append(day)
        bucket = daily[day]
        bucket["news_count"] += 1
        bucket["heat"] += float(item.heat_score or 0.0)
        if item.sentiment_score is not None:
            bucket["sentiment_total"] += float(item.sentiment_score or 0.0)
            bucket["sentiment_count"] += 1

        label = (item.sentiment_label or "中立").strip()
        if label == "正面":
            bucket["positive"] += 1
        elif label == "负面":
            bucket["negative"] += 1
        else:
            bucket["neutral"] += 1

        source = item.source or "未知来源"
        source_counter[source] += 1
        source_daily[source][day] += 1

        for kw in (item.keywords or []):
            if isinstance(kw, str) and kw.strip():
                key = kw.strip()
                keyword_counter[key] += 1
                keyword_daily[key][day] += 1

    labels = sorted(set(labels))
    avg_sentiment = []
    for label in labels:
        bucket = daily[label]
        if bucket["sentiment_count"]:
            avg_sentiment.append(round(bucket["sentiment_total"] / bucket["sentiment_count"], 2))
        else:
            avg_sentiment.append(None)

    top_sources = source_counter.most_common(8)
    top_keywords = keyword_counter.most_common(8)

    source_series = [
        {"name": name, "type": "line", "smooth": True, "data": [source_daily[name].get(label, 0) for label in labels]}
        for name, _ in top_sources[:5]
    ]
    keyword_series = [
        {"name": name, "type": "line", "smooth": True, "data": [keyword_daily[name].get(label, 0) for label in labels]}
        for name, _ in top_keywords[:5]
    ]

    peak_day = None
    if labels:
        peak_day = max(labels, key=lambda label: (daily[label]["heat"], daily[label]["news_count"]))

    return {
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "heat_score": float(topic.heat_score or 0.0),
            "start_time": topic.start_time.isoformat() if topic.start_time else None,
            "updated_time": topic.updated_time.isoformat() if topic.updated_time else None,
        },
        "metrics": {
            "news_count": len(news_list),
            "timeline_count": len(timeline_items),
            "source_count": len(source_counter),
            "keyword_count": len(keyword_counter),
            "total_heat": round(sum(float(n.heat_score or 0.0) for n in news_list), 2),
            "avg_sentiment": round(
                sum(float(n.sentiment_score or 0.0) for n in news_list if n.sentiment_score is not None)
                / max(1, len([n for n in news_list if n.sentiment_score is not None])),
                2,
            ),
            "peak_day": peak_day,
        },
        "dates": labels,
        "series": {
            "news_count": [daily[label]["news_count"] for label in labels],
            "heat": [round(daily[label]["heat"], 2) for label in labels],
            "avg_sentiment": avg_sentiment,
            "positive": [daily[label]["positive"] for label in labels],
            "neutral": [daily[label]["neutral"] for label in labels],
            "negative": [daily[label]["negative"] for label in labels],
            "source": source_series,
            "keyword": keyword_series,
        },
        "rankings": {
            "sources": [{"name": name, "value": count} for name, count in top_sources],
            "keywords": [{"name": name, "value": count} for name, count in top_keywords],
        },
    }


@router.post("/topics/{topic_id}/regenerate_overview")
async def regenerate_topic_overview(topic_id: int, db: AsyncSession = Depends(get_db)) -> Dict:
    """
    手动触发重新生成专题综述
    """
    new_record = await topic_service.regenerate_topic_overview_action(db, topic_id)
    if new_record is None:
        raise HTTPException(status_code=404, detail="专题不存在或生成失败")
    
    return {"message": "专题综述生成成功", "record": new_record}
