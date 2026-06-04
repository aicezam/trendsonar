# 本文件用于规范化 JSON 新闻源载荷，兼容多种 API 返回结构和摘要字段命名。

from __future__ import annotations

import email.utils
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, TypedDict

NEWS_CONTAINER_KEYS = ("data", "items", "stories", "news", "list", "results", "articles", "records")
TITLE_FIELDS = ("title", "headline", "name")
LINK_FIELDS = ("url", "link", "share_url", "mobileUrl", "mobile_url", "uri")
SUMMARY_FIELDS = ("content", "description", "summary", "digest", "desc", "abstract", "brief", "text")
DATE_FIELDS = ("publish_time", "published_at", "publish_date", "pub_date", "created_at", "updated_at", "time", "date")


class NormalizedJsonNewsItem(TypedDict, total=False):
    """
    输入:
    - JSON 新闻条目中的标准字段集合

    输出:
    - 供抓取服务统一消费的类型声明

    作用:
    - 明确 JSON 新闻源规范化后的字段名称，避免服务层依赖不同平台的原始字段。
    """

    title: str
    link: str
    original_link: str
    summary: str
    content: str
    publish_date: datetime


def _stringify_value(value: Any) -> str:
    """
    输入:
    - `value`: JSON 字段原始值

    输出:
    - 可用于标题、链接或正文素材的字符串

    作用:
    - 统一把字符串、数字、列表和对象转换为文本，避免平台字段类型差异导致解析失败。
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _first_text(item: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    """
    输入:
    - `item`: 单条 JSON 新闻对象
    - `fields`: 按优先级排列的候选字段名

    输出:
    - 第一个非空字段文本；不存在时返回空字符串

    作用:
    - 在多个平台字段命名不一致时统一抽取标题、链接和摘要素材。
    """

    for field in fields:
        text = _stringify_value(item.get(field))
        if text:
            return text
    return ""


def _looks_like_news_item(value: Any) -> bool:
    """
    输入:
    - `value`: JSON 结构中的任意节点

    输出:
    - 是否像一条新闻对象

    作用:
    - 判断递归展开时哪些字典应作为新闻条目保留。
    """

    if not isinstance(value, dict):
        return False
    has_title = any(_stringify_value(value.get(field)) for field in TITLE_FIELDS)
    has_link = any(_stringify_value(value.get(field)) for field in LINK_FIELDS)
    has_material = any(_stringify_value(value.get(field)) for field in SUMMARY_FIELDS)
    return has_title and (has_link or has_material)


def extract_json_news_items(payload: Any, *, max_depth: int = 5) -> List[Dict[str, Any]]:
    """
    输入:
    - `payload`: JSON 解析后的原始载荷
    - `max_depth`: 递归展开的最大层级

    输出:
    - 扁平化后的新闻条目字典列表

    作用:
    - 兼容顶层列表、`data/items/stories` 列表，以及 `data -> 平台名 -> 列表` 这类嵌套结构。
    """

    def collect(value: Any, depth: int) -> List[Dict[str, Any]]:
        """
        输入:
        - `value`: 当前递归节点
        - `depth`: 当前递归深度

        输出:
        - 当前节点下收集到的新闻条目

        作用:
        - 优先识别容器字段，再兜底扫描子节点，保持原始列表顺序。
        """

        if depth > max_depth:
            return []
        if _looks_like_news_item(value):
            return [dict(value)]
        if isinstance(value, list):
            items: List[Dict[str, Any]] = []
            for child in value:
                items.extend(collect(child, depth + 1))
            return items
        if isinstance(value, dict):
            items = []
            for key in NEWS_CONTAINER_KEYS:
                if key in value:
                    items.extend(collect(value[key], depth + 1))
            if items:
                return items
            for child in value.values():
                if isinstance(child, (dict, list)):
                    items.extend(collect(child, depth + 1))
            return items
        return []

    return collect(payload, 0)


def parse_json_datetime(value: Any) -> Optional[datetime]:
    """
    输入:
    - `value`: JSON 中的发布时间字段，可能是 ISO 字符串、常见时间字符串或时间戳

    输出:
    - 解析后的 `datetime`；无法解析时返回 `None`

    作用:
    - 统一兼容不同 JSON 新闻 API 的发布时间格式。
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        try:
            return datetime.fromtimestamp(timestamp)
        except (OSError, ValueError):
            return None

    text = _stringify_value(value)
    if not text:
        return None
    if text.isdigit():
        return parse_json_datetime(float(text))

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    parsed_tuple = email.utils.parsedate_tz(text)
    if parsed_tuple:
        try:
            return datetime.fromtimestamp(email.utils.mktime_tz(parsed_tuple))
        except (OSError, ValueError):
            return None
    return None


def _iter_date_candidates(item: Mapping[str, Any]) -> List[Any]:
    """
    输入:
    - `item`: 单条 JSON 新闻对象

    输出:
    - 按优先级排列的发布时间候选值

    作用:
    - 兼容 NewsNow 等接口把发布时间放在 `extra.date` 的结构。
    """

    candidates = [item.get(field) for field in DATE_FIELDS]
    extra = item.get("extra")
    if isinstance(extra, Mapping):
        candidates.extend(extra.get(field) for field in DATE_FIELDS)
    return candidates


def normalize_json_news_item(item: Mapping[str, Any]) -> Optional[NormalizedJsonNewsItem]:
    """
    输入:
    - `item`: 原始 JSON 新闻条目

    输出:
    - 标准化后的新闻条目；标题或链接缺失时返回 `None`

    作用:
    - 将 `content/description/summary/digest` 统一识别为来源自带正文摘要素材，供后续整理流程直接使用。
    """

    title = _first_text(item, TITLE_FIELDS)
    link = _first_text(item, LINK_FIELDS)
    if not title or not link:
        return None

    material = _first_text(item, SUMMARY_FIELDS)
    normalized: NormalizedJsonNewsItem = {
        "title": title,
        "link": link,
        "original_link": link,
    }
    if material:
        normalized["summary"] = material
        normalized["content"] = material

    for date_value in _iter_date_candidates(item):
        parsed_date = parse_json_datetime(date_value)
        if parsed_date:
            normalized["publish_date"] = parsed_date
            break

    return normalized


def ensure_unique_json_item_links(items: List[NormalizedJsonNewsItem]) -> List[NormalizedJsonNewsItem]:
    """
    输入:
    - `items`: 标准化后的 JSON 新闻条目列表

    输出:
    - URL 在列表内唯一的新闻条目列表

    作用:
    - 处理部分热榜 API 多条新闻共用首页 URL 的情况，避免数据库 URL 唯一约束导致同源热榜只入库一条。
    """

    seen: set[str] = set()
    unique_items: List[NormalizedJsonNewsItem] = []
    for item in items:
        link = item.get("link", "").strip()
        if not link:
            continue
        original_link = item.get("original_link") or link
        if link in seen:
            fingerprint_raw = "|".join(
                [
                    item.get("title", ""),
                    item.get("summary", ""),
                    str(item.get("publish_date", "")),
                ]
            )
            fingerprint = hashlib.sha1(fingerprint_raw.encode("utf-8")).hexdigest()[:12]
            separator = "&" if "?" in link else "#"
            link = f"{link}{separator}ts-news={fingerprint}"
            item["link"] = link
            item["original_link"] = original_link
        seen.add(link)
        unique_items.append(item)
    return unique_items
