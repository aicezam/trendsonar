"""
本文件用于定义 `report_cache` 表的 ORM 模型，用于缓存全局/关键词报表数据。
主要类:
- `ReportCache`: 报表缓存模型
"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String

from app.core.database import Base


class ReportCache(Base):
    """
    输入:
    - `report_type`: 报表类型（global/keyword）
    - `keyword`: 关键词（当 report_type=keyword 时使用）
    - `data`: 报表结构化数据

    输出:
    - 数据库 `report_cache` 表的 ORM 映射对象

    作用:
    - 缓存全局报表与关键词报表，便于前端快速加载与历史回溯
    """

    __tablename__ = "report_cache"

    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String, index=True, nullable=False)
    keyword = Column(String, index=True, nullable=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now, index=True)
