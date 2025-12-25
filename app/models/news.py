"""
本文件用于定义 `news` 表的 ORM 模型，承载新闻抓取、聚类与分析后的结构化字段。
主要类:
- `News`: 新闻数据模型
"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.dialects.postgresql import ARRAY

from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()
# SQLite 不支持 ARRAY 类型，自动回退到 JSON
if "sqlite" in (settings.DATABASE_URL or "").lower():
    VectorType = JSON
else:
    VectorType = ARRAY(Float)


class News(Base):
    """
    输入:
    - 新闻抓取与分析后的结构化字段（标题、来源、热度、时间、向量等）

    输出:
    - 数据库 `news` 表的 ORM 映射对象

    作用:
    - 统一存储新闻原始信息、聚类合并信息以及 AI 分析结果
    """

    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, index=True, nullable=False)
    source = Column(String, nullable=False)

    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    is_ai_summary = Column(Boolean, default=False)

    heat_score = Column(Float, default=0.0, index=True)
    publish_date = Column(DateTime, default=datetime.now, index=True)
    crawled_at = Column(DateTime, default=datetime.now)

    sources = Column(JSON, default=list)

    embedding = Column(VectorType, nullable=True)

    sentiment_score = Column(Float, default=50.0)
    sentiment_label = Column(String, default="中立")
    category = Column(String, default="其他")
    region = Column(String, default="其他")
    keywords = Column(JSON, default=list)
    entities = Column(JSON, default=list)
