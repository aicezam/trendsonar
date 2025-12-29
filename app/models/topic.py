"""
本文件用于定义“专题追踪”相关 ORM 模型，用于对持续性新闻事件进行归档与时间轴展示。
主要类:
- `Topic`: 专题主体
- `TopicTimelineItem`: 专题时间轴条目
"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()
# SQLite 不支持 ARRAY 类型，自动回退到 JSON
if "sqlite" in (settings.DATABASE_URL or "").lower():
    VectorType = JSON
else:
    VectorType = ARRAY(Float)


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)

    start_time = Column(DateTime, default=datetime.now, index=True)
    updated_time = Column(DateTime, default=datetime.now, index=True)

    status = Column(String, default="active", index=True)
    heat_score = Column(Float, default=0.0, index=True)

    summary = Column(Text, nullable=True)
    record = Column(Text, nullable=True)
    keywords = Column(JSON, default=list)

    embedding = Column(VectorType, nullable=True)

    timeline_items = relationship("TopicTimelineItem", back_populates="topic", cascade="all, delete-orphan")


class TopicTimelineItem(Base):
    __tablename__ = "topic_timeline_items"

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id", ondelete="CASCADE"), index=True, nullable=False)

    event_time = Column(DateTime, default=datetime.now, index=True)
    content = Column(Text, nullable=False)

    news_id = Column(Integer, nullable=True, index=True)
    news_title = Column(String, nullable=True)
    source_name = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    
    # 新增字段以支持每个事件多个来源
    sources = Column(JSON, default=list)  # {name, url, title, id} 的列表

    created_at = Column(DateTime, default=datetime.now, index=True)

    topic = relationship("Topic", back_populates="timeline_items")

