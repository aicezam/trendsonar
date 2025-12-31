"""
本文件用于定义 `clustering_history` 表的 ORM 模型，记录聚类核验历史。
主要类:
- `ClusteringHistory`: 聚类核验历史记录
"""

from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, UniqueConstraint

from app.core.database import Base


class ClusteringHistory(Base):
    """
    输入:
    - `news_id_a`: 新闻ID A (较小ID)
    - `news_id_b`: 新闻ID B (较大ID)

    输出:
    - 数据库 `clustering_history` 表的 ORM 映射对象

    作用:
    - 记录 AI 判定为“不匹配”的新闻对，避免重复核验
    """

    __tablename__ = "clustering_history"

    id = Column(Integer, primary_key=True, index=True)
    news_id_a = Column(Integer, nullable=False, index=True)
    news_id_b = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint('news_id_a', 'news_id_b', name='uix_clustering_history_a_b'),
    )
