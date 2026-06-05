"""
本文件用于提供新闻排序工具，统一报告页与词项分析中的热点新闻排序口径。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional, TypeVar

T = TypeVar("T")


def get_news_field(item: Any, name: str, default: Any = None) -> Any:
    """
    输入:
    - `item`: 新闻 ORM 对象、SQLAlchemy Row 映射或普通字典
    - `name`: 字段名
    - `default`: 字段不存在时的默认值

    输出:
    - 指定字段值

    作用:
    - 兼容不同数据结构的新闻行，便于排序与序列化共用同一套取值逻辑。
    """

    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def normalize_datetime(value: Any) -> Optional[datetime]:
    """
    输入:
    - `value`: datetime、ISO 字符串或其他可选时间值

    输出:
    - 解析后的 datetime；无法解析时返回 None

    作用:
    - 将新闻发布时间统一转换成可比较的时间对象。
    """

    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def calculate_news_composite_scores(
    items: Iterable[T],
    *,
    heat_weight: float = 0.4,
    time_weight: float = 0.6,
) -> dict[int, float]:
    """
    输入:
    - `items`: 待排序新闻集合
    - `heat_weight`: 热度权重
    - `time_weight`: 时间权重

    输出:
    - `{新闻id: 综合分}` 映射

    作用:
    - 在候选集合内部将热度和发布时间归一化，计算“越新且越热越靠前”的综合分。
    """

    rows = list(items)
    heat_values = [float(get_news_field(item, "heat_score", 0.0) or 0.0) for item in rows]
    time_values = [
        normalize_datetime(get_news_field(item, "publish_date")) or datetime.min
        for item in rows
    ]

    min_heat = min(heat_values, default=0.0)
    max_heat = max(heat_values, default=0.0)
    min_ts = min((dt.timestamp() for dt in time_values if dt != datetime.min), default=0.0)
    max_ts = max((dt.timestamp() for dt in time_values if dt != datetime.min), default=0.0)

    scores: dict[int, float] = {}
    for item, heat, publish_time in zip(rows, heat_values, time_values):
        news_id = int(get_news_field(item, "id", 0) or 0)
        heat_score = (heat - min_heat) / (max_heat - min_heat) if max_heat > min_heat else 1.0
        if publish_time == datetime.min:
            time_score = 0.0
        elif max_ts > min_ts:
            time_score = (publish_time.timestamp() - min_ts) / (max_ts - min_ts)
        else:
            time_score = 1.0
        scores[news_id] = heat_weight * heat_score + time_weight * time_score
    return scores


def sort_news_by_composite_score(
    items: Iterable[T],
    *,
    heat_weight: float = 0.4,
    time_weight: float = 0.6,
) -> list[T]:
    """
    输入:
    - `items`: 待排序新闻集合
    - `heat_weight`/`time_weight`: 热度与时间权重

    输出:
    - 按综合分倒序排列的新闻列表

    作用:
    - 统一按“热度 40% + 时间 60%”排序，避免最新新闻因热度偏低被排到末尾。
    """

    rows = list(items)
    scores = calculate_news_composite_scores(
        rows,
        heat_weight=heat_weight,
        time_weight=time_weight,
    )
    return sorted(
        rows,
        key=lambda item: (
            scores.get(int(get_news_field(item, "id", 0) or 0), 0.0),
            normalize_datetime(get_news_field(item, "publish_date")) or datetime.min,
            float(get_news_field(item, "heat_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
