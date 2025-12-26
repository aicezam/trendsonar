"""
本文件用于提供“专题追踪”相关 API：专题列表、专题详情（时间轴/完整记录/来源新闻）。
主要函数:
- `get_topics`: 获取专题列表
- `get_topic_detail`: 获取专题详情
"""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.news import News
from app.models.topic import Topic, TopicTimelineItem
from app.services.topic_service import topic_service

router = APIRouter(prefix="/api", tags=["topics"])


@router.get("/topics")
async def get_topics(db: AsyncSession = Depends(get_db)) -> Dict[str, List[Dict]]:
    stmt = select(Topic).where(Topic.status == "active").order_by(desc(Topic.updated_time)).limit(100)
    topics = (await db.execute(stmt)).scalars().all()
    data: List[Dict] = []
    for t in topics:
        data.append(
            {
                "id": t.id,
                "name": t.name,
                "summary": t.summary,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "updated_time": t.updated_time.isoformat() if t.updated_time else None,
                "heat_score": float(t.heat_score or 0.0),
            }
        )
    return {"data": data}


@router.get("/topics/{topic_id}")
async def get_topic_detail(
    topic_id: int, 
    sort: str = Query("asc", regex="^(asc|desc)$"),
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

    # Collect all related news IDs from both legacy news_id and new sources field
    news_ids = set()
    for i in items:
        if i.news_id:
            news_ids.add(i.news_id)
        if i.sources:
            for s in i.sources:
                if isinstance(s, dict) and s.get("id"):
                     news_ids.add(s["id"])

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


@router.post("/topics/{topic_id}/regenerate_overview")
async def regenerate_topic_overview(topic_id: int, db: AsyncSession = Depends(get_db)) -> Dict:
    """
    手动触发重新生成专题综述
    """
    new_record = await topic_service.regenerate_topic_overview_action(db, topic_id)
    if new_record is None:
        raise HTTPException(status_code=404, detail="专题不存在或生成失败")
    
    return {"message": "专题综述生成成功", "record": new_record}

