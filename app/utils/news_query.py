"""
本文件用于提供新闻查询构造和响应序列化工具，减少多个新闻接口中的重复筛选逻辑。
主要函数:
- `build_news_query_filters`: 为新闻查询统一追加时间、分类、地区和来源过滤条件
- `serialize_news_item`: 将 News ORM 对象转换为接口响应字典
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Optional

from sqlalchemy import or_
from sqlalchemy.sql import Select

from app.models.news import News
from app.utils.tools import normalize_regions_to_countries


def build_news_query_filters(
    stmt: Select[Any],
    *,
    date: str = "24h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
) -> Select[Any]:
    """
    输入:
    - `stmt`: SQLAlchemy 查询对象
    - `date`: 快捷时间范围，支持 24h/3d/7d/week/month/year/all 或 YYYY-MM-DD
    - `start_date`/`end_date`: 自定义起止日期，格式为 YYYY-MM-DD
    - `category`/`region`/`source`: 新闻分类、地区和来源筛选条件

    输出:
    - 追加过滤条件后的查询对象

    作用:
    - 统一新闻列表和 Top 新闻接口的过滤口径，避免筛选逻辑在多个接口中重复发散
    """

    now = datetime.now()
    if start_date or end_date:
        try:
            if start_date:
                s_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                start = datetime.combine(s_date, time.min)
                stmt = stmt.where(News.publish_date >= start)
            if end_date:
                e_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                end = datetime.combine(e_date, time.max)
                stmt = stmt.where(News.publish_date <= end)
        except ValueError:
            pass
    else:
        try:
            if date == "24h":
                start = now - timedelta(hours=24)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "today":
                start = datetime.combine(now.date(), time.min)
                end = datetime.combine(now.date(), time.max)
                stmt = stmt.where(News.publish_date >= start, News.publish_date <= end)
            elif date == "yesterday":
                target = now.date() - timedelta(days=1)
                start = datetime.combine(target, time.min)
                end = datetime.combine(target, time.max)
                stmt = stmt.where(News.publish_date >= start, News.publish_date <= end)
            elif date == "3d":
                start = now - timedelta(days=3)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "7d":
                start = now - timedelta(days=7)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "week":
                start = now - timedelta(days=now.weekday())
                start = datetime.combine(start.date(), time.min)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "month":
                start = now.replace(day=1)
                start = datetime.combine(start.date(), time.min)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "year":
                start = now.replace(month=1, day=1)
                start = datetime.combine(start.date(), time.min)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "all":
                pass
            elif date:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                start = datetime.combine(target_date, time.min)
                end = datetime.combine(target_date, time.max)
                stmt = stmt.where(News.publish_date >= start, News.publish_date <= end)
        except ValueError:
            pass

    if category and category != "all":
        if "," in category:
            stmt = stmt.where(News.category.in_([c.strip() for c in category.split(",") if c.strip()]))
        else:
            stmt = stmt.where(News.category == category)

    normalized_region = normalize_regions_to_countries(region) if region and region != "all" else ""
    if normalized_region and normalized_region not in {"其他", "全球"}:
        selected_regions = [r for r in normalized_region.split(",") if r]
        if selected_regions:
            conditions = [News.region.ilike(f"%{r}%") for r in selected_regions]
            stmt = stmt.where(or_(*conditions))

    if source and source != "all":
        if "," in source:
            stmt = stmt.where(News.source.in_([s.strip() for s in source.split(",") if s.strip()]))
        else:
            stmt = stmt.where(News.source == source)

    return stmt


def serialize_news_item(news: News) -> dict[str, Any]:
    """
    输入:
    - `news`: 新闻 ORM 对象

    输出:
    - 面向前端接口的新闻字典

    作用:
    - 统一新闻列表、Top 新闻和搜索结果的响应字段，避免接口之间字段口径不一致
    """

    return {
        "id": news.id,
        "title": news.title,
        "url": news.url,
        "source": news.source,
        "heat": news.heat_score,
        "time": news.publish_date.isoformat(),
        "summary": news.summary,
        "sources": news.sources,
        "category": news.category,
        "region": normalize_regions_to_countries(news.region),
        "sentiment_label": news.sentiment_label,
        "sentiment_score": news.sentiment_score,
        "keywords": news.keywords or [],
        "entities": news.entities or [],
    }
