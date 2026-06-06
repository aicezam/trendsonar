"""
本文件用于封装 TrendSonar 智能体可调用工具，统一复用新闻、专题与报告服务能力。
主要类/对象:
- `AgentToolService`: 智能体工具服务
- `agent_tool_service`: 全局工具服务单例
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from sqlalchemy import Text, cast, desc, func, or_, select
from sqlalchemy.orm import defer

from app.core.database import AsyncSessionLocal
from app.core.logger import setup_logger
from app.models.news import News
from app.models.topic import Topic, TopicTimelineItem
from app.services.report_service import report_service
from app.services.task_manager import task_manager
from app.services.topic_service import topic_service
from app.utils.agent_tool_config import delete_custom_agent_tool, load_custom_agent_tools, save_custom_agent_tool
from app.utils.agent_web import compact_web_text, ensure_public_web_url, simple_web_search
from app.utils.news_query import build_news_query_filters, serialize_news_item
from app.utils.news_search import (
    build_search_query_variants,
    build_soft_search_query,
    semantic_news_search,
    strong_core_term_coverage,
)
from app.utils.news_image import generate_news_text_image
from app.utils.tools import normalize_regions_to_countries
from app.services.crawler_service import crawler_service

logger = setup_logger("AgentToolService")
REPORT_CREATE_COOLDOWN_SECONDS = 60
CUSTOM_TOOL_TIMEOUT_SECONDS = 20
CUSTOM_TOOL_MAX_RESPONSE_CHARS = 60000
CUSTOM_TOOL_MAX_FIELD_CHARS = 3000
CUSTOM_TOOL_MAX_OBJECT_KEYS = 80
CUSTOM_TOOL_MAX_ITEMS = 20
BLOCKED_CUSTOM_TOOL_HOSTS = {"metadata.google.internal"}
BLOCKED_CUSTOM_TOOL_IPS = {"0.0.0.0", "169.254.169.254"}
WEB_CRAWL_MAX_CHARS = 12000


BUILTIN_AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_top_news",
        "title": "热点新闻查询",
        "description": "查询指定时间范围内的热点新闻 TopN，只返回新闻 ID 和标题。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "limit": {"type": "integer", "default": 20, "description": "返回数量，默认 20。"},
            "date": {
                "type": "string",
                "default": "today",
                "description": "today/24h/3d/7d/30d/week/month/year/all；30d 是滚动最近30天，month 是本月。",
            },
            "sort_by": {"type": "string", "default": "heat", "description": "heat 或 date。"},
            "category": {"type": "string", "default": "", "description": "可选分类。"},
            "region": {"type": "string", "default": "", "description": "可选地区。"},
            "source": {"type": "string", "default": "", "description": "可选来源。"},
        },
    },
    {
        "name": "search_news",
        "title": "关键词新闻搜索",
        "description": "按多个关键词语义召回新闻，只返回新闻 ID 和标题。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "q": {"type": "string", "default": "", "description": "搜索关键词，可包含多个词或自然语言查询。"},
            "limit": {"type": "integer", "default": 100, "description": "返回数量，默认 100；宽泛枚举或查漏可提高到 300。"},
            "date": {"type": "string", "default": "all", "description": "时间范围；最近一个月请用 30d，本月请用 month。"},
            "sort_by": {"type": "string", "default": "heat", "description": "heat 或 date。"},
        },
    },
    {
        "name": "get_news_detail",
        "title": "新闻详情读取",
        "description": "读取一个或多个新闻 ID 的详情、摘要、来源、关键词和实体。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "news_id": {"type": "integer", "default": 0, "description": "单个新闻 ID。"},
            "news_ids": {"type": "array", "default": [], "description": "多个新闻 ID，一次最多 30 个。"},
        },
    },
    {
        "name": "list_topics",
        "title": "专题列表查询",
        "description": "查询已有活跃专题。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "q": {"type": "string", "default": "", "description": "专题关键词。"},
            "limit": {"type": "integer", "default": 20, "description": "返回数量。"},
            "date": {"type": "string", "default": "all", "description": "时间范围。"},
            "sort_by": {"type": "string", "default": "updated", "description": "updated 或 heat。"},
        },
    },
    {
        "name": "get_topic_detail",
        "title": "专题详情读取",
        "description": "读取指定专题 ID 的详情、时间轴和相关新闻。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "topic_id": {"type": "integer", "default": 0, "description": "专题 ID。"},
            "timeline_limit": {"type": "integer", "default": 80, "description": "时间轴节点数量。"},
        },
    },
    {
        "name": "get_report_analysis",
        "title": "报告分析数据",
        "description": "获取报告摘要、图表、词云和 Top 新闻。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "q": {"type": "string", "default": "", "description": "关键词，留空为全局。"},
            "limit": {"type": "integer", "default": 50, "description": "分析样本数量。"},
            "generate_ai": {"type": "boolean", "default": False, "description": "测试时建议保持 false。"},
        },
    },
    {
        "name": "create_keyword_report",
        "title": "创建关键词报告",
        "description": "生成指定关键词的报告缓存，属于写操作。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": False,
        "parameters": {"keyword": {"type": "string", "default": "", "description": "报告关键词。"}},
    },
    {
        "name": "create_event_topic",
        "title": "创建事件专题",
        "description": "创建某个新闻事件专题，属于写操作。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": False,
        "parameters": {"name": {"type": "string", "default": "", "description": "专题名称。"}},
    },
    {
        "name": "get_term_analysis",
        "title": "词项分析",
        "description": "查询关键词或实体的趋势、相关新闻、情感和共现词。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "term": {"type": "string", "default": "", "description": "词项名称。"},
            "range": {"type": "string", "default": "year", "description": "30d/year/all。"},
        },
    },
    {
        "name": "web_search",
        "title": "网页搜索",
        "description": "查询公开网页搜索结果，只返回标题、URL 和简短摘要；适合补充站外资料。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "q": {"type": "string", "default": "TrendSonar 新闻", "description": "搜索关键词或自然语言问题。"},
            "limit": {"type": "integer", "default": 5, "description": "返回数量，最多 10。"},
        },
    },
    {
        "name": "web_crawl_page",
        "title": "网页正文抓取",
        "description": "使用轻量抓取、Crawl4AI 和 Playwright 兜底读取公开网页正文。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "url": {"type": "string", "default": "https://example.com", "description": "公开 http/https 页面地址。"},
            "max_chars": {"type": "integer", "default": 8000, "description": "返回正文最大字符数，最多 12000。"},
        },
    },
    {
        "name": "generate_news_image",
        "title": "新闻图片生成",
        "description": "将新闻 ID 或文字摘要渲染为 PNG 图片卡片，并返回可访问 URL。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "news_id": {"type": "integer", "default": 0, "description": "可选新闻 ID，传入后自动读取标题、摘要、来源和时间。"},
            "title": {"type": "string", "default": "TrendSonar 新闻图片", "description": "未传 news_id 时使用的图片标题。"},
            "body": {"type": "string", "default": "这里是一段新闻摘要，可由智能体根据上下文生成。", "description": "未传 news_id 时使用的正文。"},
            "source": {"type": "string", "default": "TrendSonar", "description": "来源。"},
            "time_label": {"type": "string", "default": "", "description": "时间标签。"},
            "theme": {"type": "string", "default": "default", "description": "default/dark/warm。"},
        },
    },
]


def _safe_limit(value: int, *, default: int, minimum: int = 1, maximum: int = 100) -> int:
    """
    输入:
    - `value`: 用户或模型传入的数量
    - `default`: 默认数量
    - `minimum`/`maximum`: 数量上下限

    输出:
    - 经过边界保护后的整数

    作用:
    - 统一保护智能体工具的查询规模，避免一次调用拉取过多数据。
    """

    try:
        raw = int(value or default)
    except (TypeError, ValueError):
        raw = default
    return max(minimum, min(raw, maximum))


def _safe_news_ids(news_id: Optional[int] = None, news_ids: Optional[list[int]] = None, *, maximum: int = 30) -> list[int]:
    """
    输入:
    - `news_id`: 单个新闻 ID
    - `news_ids`: 多个新闻 ID
    - `maximum`: 最多允许查询数量

    输出:
    - 清洗去重后的新闻 ID 列表

    作用:
    - 兼容单条详情读取和批量详情读取，避免模型一次请求过多记录。
    """

    values: list[int] = []
    seen: set[int] = set()
    raw_values: list[Any] = []
    if news_ids:
        raw_values.extend(news_ids)
    if news_id:
        raw_values.append(news_id)

    for value in raw_values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        values.append(item)
        if len(values) >= maximum:
            break
    return values


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    """
    输入:
    - `value`: 可选文本

    输出:
    - 去除首尾空白后的文本；空值返回 None

    作用:
    - 统一处理工具入参中的空字符串，减少下游重复判断。
    """

    text = str(value or "").strip()
    return text or None


def _serialize_news_brief(news: News) -> dict[str, Any]:
    """
    输入:
    - `news`: 新闻对象

    输出:
    - 仅包含新闻 ID 与标题的轻量字典

    作用:
    - 降低智能体候选召回工具的输出体积；摘要与来源等详情统一由详情工具按 ID 读取。
    """

    return {"id": news.id, "title": news.title or ""}


def _render_template_value(value: Any, args: Dict[str, Any]) -> Any:
    """
    输入:
    - `value`: 自定义工具执行器中的模板值
    - `args`: 工具调用参数

    输出:
    - 用参数替换后的值

    作用:
    - 支持在 URL、查询参数、请求体和请求头中使用 `{name}` 形式引用工具入参。
    """

    if isinstance(value, str):
        try:
            template_args = defaultdict(str, {key: "" if val is None else val for key, val in args.items()})
            return value.format_map(template_args)
        except Exception:
            return value
    if isinstance(value, list):
        return [_render_template_value(item, args) for item in value]
    if isinstance(value, dict):
        return {str(key): _render_template_value(item, args) for key, item in value.items()}
    return value


def _drop_empty_values(data: Any) -> Any:
    """
    输入:
    - `data`: 字典、列表或普通值

    输出:
    - 删除空字符串和空值后的结构

    作用:
    - 避免把可选参数的空值发送给外部 HTTP 接口。
    """

    if isinstance(data, dict):
        return {
            key: _drop_empty_values(value)
            for key, value in data.items()
            if value not in (None, "")
        }
    if isinstance(data, list):
        return [_drop_empty_values(item) for item in data if item not in (None, "")]
    return data


def _ensure_safe_custom_tool_url(url: str) -> str:
    """
    输入:
    - `url`: 自定义 HTTP 工具目标地址

    输出:
    - 校验后的 URL

    作用:
    - 限制自定义工具只能访问 http/https，并拦截高风险元数据地址。
    """

    clean_url = str(url or "").strip()
    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("自定义工具 URL 必须是合法的 http/https 地址")
    host = (parsed.hostname or "").lower()
    if host in BLOCKED_CUSTOM_TOOL_HOSTS or host in BLOCKED_CUSTOM_TOOL_IPS:
        raise ValueError("自定义工具 URL 指向被禁止的地址")
    return clean_url


def _extract_result_path(data: Any, path: str) -> Any:
    """
    输入:
    - `data`: HTTP 响应 JSON
    - `path`: 点号分隔路径，例如 `results`

    输出:
    - 路径对应的数据，路径不存在时返回原数据

    作用:
    - 让自定义工具可以从外部接口响应中截取主要结果数组。
    """

    current = data
    for part in [item for item in str(path or "").split(".") if item]:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return data
        else:
            return data
        if current is None:
            return data
    return current


def _compact_custom_tool_result(data: Any, *, result_path: str = "", item_fields: Optional[list[str]] = None, limit: int = 10) -> Any:
    """
    输入:
    - `data`: 外部 HTTP 接口响应
    - `result_path`: 可选结果路径
    - `item_fields`: 可选字段白名单
    - `limit`: 数组返回上限

    输出:
    - 控制体积后的工具结果

    作用:
    - 减少自定义工具返回给智能体的 token 体积。
    """

    def truncate_value(value: Any) -> Any:
        if isinstance(value, str):
            return value[:CUSTOM_TOOL_MAX_FIELD_CHARS] if len(value) > CUSTOM_TOOL_MAX_FIELD_CHARS else value
        if isinstance(value, list):
            return [truncate_value(item) for item in value[:safe_limit]]
        if isinstance(value, dict):
            return {
                str(key): truncate_value(item)
                for key, item in list(value.items())[:CUSTOM_TOOL_MAX_OBJECT_KEYS]
            }
        return value

    selected = _extract_result_path(data, result_path) if result_path else data
    safe_limit = _safe_limit(limit, default=10, maximum=CUSTOM_TOOL_MAX_ITEMS)
    fields = [str(field) for field in (item_fields or []) if str(field).strip()]
    if isinstance(selected, list):
        rows = selected[:safe_limit]
        if fields:
            return [
                {field: truncate_value(item.get(field)) for field in fields if isinstance(item, dict) and field in item}
                if isinstance(item, dict)
                else truncate_value(item)
                for item in rows
            ]
        return truncate_value(rows)
    if isinstance(selected, dict) and fields:
        return {field: truncate_value(selected.get(field)) for field in fields if field in selected}
    return truncate_value(selected)


def _build_custom_tool_args(tool: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入:
    - `tool`: 自定义工具配置
    - `args`: 调用方显式传入的参数

    输出:
    - 已合并参数默认值的调用参数

    作用:
    - 让管理端配置的参数默认值在智能体调用时同样生效，避免 URL 模板缺少 `base_url` 等必填默认参数。
    """

    parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    defaults = {
        key: meta.get("default")
        for key, meta in parameters.items()
        if isinstance(meta, dict) and "default" in meta
    }
    merged = dict(defaults)
    if isinstance(args, dict):
        merged.update(args)
    return merged


def _serialize_topic(topic: Topic, effective_updated_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    输入:
    - `topic`: 专题 ORM 对象
    - `effective_updated_time`: 可选的有效更新时间

    输出:
    - 智能体工具返回的专题摘要字典

    作用:
    - 让专题列表工具输出稳定、可读的结构化数据。
    """

    updated_time = effective_updated_time or topic.updated_time
    return {
        "id": topic.id,
        "name": topic.name,
        "summary": topic.summary,
        "start_time": topic.start_time.isoformat() if topic.start_time else None,
        "updated_time": updated_time.isoformat() if updated_time else None,
        "heat_score": float(topic.heat_score or 0.0),
        "status": topic.status,
    }


def _collect_timeline_news_ids(items: List[TopicTimelineItem]) -> set[int]:
    """
    输入:
    - `items`: 专题时间轴节点列表

    输出:
    - 节点关联的新闻 ID 集合

    作用:
    - 兼容旧字段 news_id 和新字段 sources，供专题详情工具读取相关新闻。
    """

    news_ids: set[int] = set()
    for item in items:
        if item.news_id:
            news_ids.add(int(item.news_id))
        if isinstance(item.sources, list):
            for source in item.sources:
                if not isinstance(source, dict) or not source.get("id"):
                    continue
                try:
                    news_ids.add(int(source["id"]))
                except (TypeError, ValueError):
                    continue
    return news_ids


class AgentToolService:
    """
    输入:
    - 无，运行时按工具调用打开数据库会话

    输出:
    - 面向智能体的结构化工具结果

    作用:
    - 将项目已有接口和服务包装成低耦合、可审计、可复用的智能体工具。
    """

    def __init__(self) -> None:
        """
        输入:
        - 无

        输出:
        - 工具服务实例

        作用:
        - 初始化写操作限流状态，避免智能体并发创建报告压垮服务。
        """

        self._report_create_lock = asyncio.Lock()
        self._last_report_create_at = 0.0

    def list_tool_definitions(self) -> Dict[str, Any]:
        """
        输入:
        - 无

        输出:
        - 内置工具与自定义工具元数据

        作用:
        - 为管理端提供工具查看、提示词编辑入口和自定义工具列表。
        """

        custom_tools = load_custom_agent_tools()
        return {
            "builtin_tools": BUILTIN_AGENT_TOOLS,
            "custom_tools": custom_tools,
            "total": len(BUILTIN_AGENT_TOOLS) + len(custom_tools),
        }

    async def test_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入:
        - `name`: 工具名称
        - `args`: 测试入参

        输出:
        - 测试执行结果或阻止原因

        作用:
        - 为管理端提供只读工具测试，避免误触发报告/专题创建等写操作。
        """

        tool_name = str(name or "").strip()
        meta = next((item for item in BUILTIN_AGENT_TOOLS if item.get("name") == tool_name), None)
        if not meta:
            custom_tool = next((item for item in load_custom_agent_tools() if item.get("name") == tool_name), None)
            if custom_tool:
                started = datetime.now()
                result = await self.run_custom_tool(tool_name=tool_name, args=args or {})
                elapsed_ms = round((datetime.now() - started).total_seconds() * 1000, 1)
                return {"ok": bool(result.get("ok")), "tool": tool_name, "elapsed_ms": elapsed_ms, "result": result}
            return {"ok": False, "message": "工具不存在"}

        if not meta.get("safe_to_test"):
            return {"ok": False, "message": "该工具会产生写入或后台任务，管理端测试已阻止。"}

        started = datetime.now()
        try:
            result = await self._execute_builtin_test_tool(tool_name, args or {})
            elapsed_ms = round((datetime.now() - started).total_seconds() * 1000, 1)
            return {"ok": True, "tool": tool_name, "elapsed_ms": elapsed_ms, "result": result}
        except Exception as exc:
            logger.error(f"智能体工具测试失败: tool={tool_name}, error={exc}", exc_info=True)
            return {"ok": False, "tool": tool_name, "message": str(exc)}

    async def _execute_builtin_test_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入:
        - `name`: 内置工具名称
        - `args`: 测试参数

        输出:
        - 工具返回结果

        作用:
        - 集中路由管理端工具测试调用，限制只读工具的参数规模。
        """

        if name == "get_top_news":
            return await self.get_top_news(
                limit=_safe_limit(args.get("limit", 20), default=20, maximum=30),
                date=str(args.get("date") or "today"),
                start_date=_clean_optional_text(args.get("start_date")),
                end_date=_clean_optional_text(args.get("end_date")),
                sort_by=str(args.get("sort_by") or "heat"),
                category=_clean_optional_text(args.get("category")),
                region=_clean_optional_text(args.get("region")),
                source=_clean_optional_text(args.get("source")),
            )
        if name == "search_news":
            return await self.search_news(
                q=str(args.get("q") or "").strip(),
                limit=_safe_limit(args.get("limit", 100), default=100, maximum=300),
                date=str(args.get("date") or "all"),
                start_date=_clean_optional_text(args.get("start_date")),
                end_date=_clean_optional_text(args.get("end_date")),
                sort_by=str(args.get("sort_by") or "heat"),
                category=_clean_optional_text(args.get("category")),
                region=_clean_optional_text(args.get("region")),
                source=_clean_optional_text(args.get("source")),
            )
        if name == "get_news_detail":
            return await self.get_news_detail(
                news_id=int(args.get("news_id") or 0),
                news_ids=args.get("news_ids") if isinstance(args.get("news_ids"), list) else None,
            )
        if name == "list_topics":
            return await self.list_topics(
                q=str(args.get("q") or ""),
                limit=_safe_limit(args.get("limit", 20), default=20, maximum=30),
                date=str(args.get("date") or "all"),
                min_heat=float(args.get("min_heat") or 0.0),
                sort_by=str(args.get("sort_by") or "updated"),
            )
        if name == "get_topic_detail":
            return await self.get_topic_detail(
                topic_id=int(args.get("topic_id") or 0),
                timeline_limit=_safe_limit(args.get("timeline_limit", 80), default=80, maximum=120),
            )
        if name == "get_report_analysis":
            return await self.get_report_analysis(
                q=str(args.get("q") or ""),
                start_date=_clean_optional_text(args.get("start_date")),
                end_date=_clean_optional_text(args.get("end_date")),
                category=_clean_optional_text(args.get("category")),
                region=_clean_optional_text(args.get("region")),
                source=_clean_optional_text(args.get("source")),
                limit=_safe_limit(args.get("limit", 50), default=50, maximum=80),
                generate_ai=False,
            )
        if name == "get_term_analysis":
            return await self.get_term_analysis(
                term=str(args.get("term") or "").strip(),
                start_date=_clean_optional_text(args.get("start_date")),
                end_date=_clean_optional_text(args.get("end_date")),
                category=_clean_optional_text(args.get("category")),
                region=_clean_optional_text(args.get("region")),
                source=_clean_optional_text(args.get("source")),
                range=str(args.get("range") or "year"),
            )
        if name == "web_search":
            return await self.web_search(
                q=str(args.get("q") or "").strip(),
                limit=_safe_limit(args.get("limit", 5), default=5, maximum=10),
            )
        if name == "web_crawl_page":
            return await self.web_crawl_page(
                url=str(args.get("url") or "").strip(),
                max_chars=_safe_limit(args.get("max_chars", 8000), default=8000, maximum=WEB_CRAWL_MAX_CHARS),
            )
        if name == "generate_news_image":
            return await self.generate_news_image(
                news_id=int(args.get("news_id") or 0),
                title=str(args.get("title") or ""),
                body=str(args.get("body") or ""),
                source=str(args.get("source") or ""),
                time_label=str(args.get("time_label") or ""),
                theme=str(args.get("theme") or "default"),
            )
        return {"message": "工具未实现测试路由"}

    def save_custom_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入:
        - `tool`: 自定义工具配置

        输出:
        - 保存后的工具配置

        作用:
        - 为管理端新增工具提供持久化入口。
        """

        saved = save_custom_agent_tool(tool)
        logger.info(f"保存自定义智能体工具: name={saved.get('name')}, enabled={saved.get('enabled')}")
        return saved

    def delete_custom_tool(self, name: str) -> bool:
        """
        输入:
        - `name`: 自定义工具名称

        输出:
        - 是否删除成功

        作用:
        - 为管理端删除自定义工具配置。
        """

        ok = delete_custom_agent_tool(name)
        if ok:
            logger.info(f"删除自定义智能体工具: name={name}")
        return ok

    async def run_custom_tool(self, *, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入:
        - `tool_name`: 自定义工具名称
        - `args`: 智能体或管理端传入的工具参数

        输出:
        - 外部 HTTP 工具执行结果

        作用:
        - 执行管理端配置的 HTTP 自定义工具，使用户新增工具可被测试并被智能体调用。
        """

        clean_name = str(tool_name or "").strip()
        tool = next((item for item in load_custom_agent_tools() if item.get("name") == clean_name), None)
        if not tool:
            return {"ok": False, "message": "自定义工具不存在"}
        if tool.get("enabled") is False:
            return {"ok": False, "message": "自定义工具已停用"}

        executor = tool.get("executor") if isinstance(tool.get("executor"), dict) else {}
        if not executor:
            return {"ok": False, "message": "自定义工具缺少 executor 执行器配置"}
        executor_type = str(executor.get("type") or "http").strip().lower()
        if executor_type != "http":
            return {"ok": False, "message": f"暂不支持的自定义工具执行器类型: {executor_type}"}

        method = str(executor.get("method") or "GET").strip().upper()
        if method not in {"GET", "POST"}:
            return {"ok": False, "message": "自定义 HTTP 工具只支持 GET/POST"}

        raw_args = _build_custom_tool_args(tool, args if isinstance(args, dict) else {})
        rendered_url = _render_template_value(executor.get("url") or "", raw_args)
        try:
            url = _ensure_safe_custom_tool_url(str(rendered_url))
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}

        query_params = _drop_empty_values(_render_template_value(executor.get("query") or {}, raw_args))
        headers = _drop_empty_values(_render_template_value(executor.get("headers") or {}, raw_args))
        body = _drop_empty_values(_render_template_value(executor.get("body") or {}, raw_args))
        result_path = str(executor.get("result_path") or "")
        item_fields = executor.get("item_fields") if isinstance(executor.get("item_fields"), list) else []
        limit_value = raw_args.get("limit") or executor.get("limit") or 10
        timeout_seconds = _safe_limit(executor.get("timeout", CUSTOM_TOOL_TIMEOUT_SECONDS), default=CUSTOM_TOOL_TIMEOUT_SECONDS, maximum=60)

        started = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if method == "POST":
                    response_ctx = session.post(url, params=query_params, json=body or None, headers=headers or None)
                else:
                    response_ctx = session.get(url, params=query_params, headers=headers or None)
                async with response_ctx as resp:
                    text = await resp.text()
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                    if len(text) > CUSTOM_TOOL_MAX_RESPONSE_CHARS:
                        text = text[:CUSTOM_TOOL_MAX_RESPONSE_CHARS]
                    try:
                        payload: Any = await resp.json(content_type=None)
                    except Exception:
                        payload = text
                    compact = _compact_custom_tool_result(
                        payload,
                        result_path=result_path,
                        item_fields=item_fields,
                        limit=_safe_limit(limit_value, default=10, maximum=CUSTOM_TOOL_MAX_ITEMS),
                    )
                    if isinstance(compact, str) and len(compact) > CUSTOM_TOOL_MAX_RESPONSE_CHARS:
                        compact = compact[:CUSTOM_TOOL_MAX_RESPONSE_CHARS]
                    return {
                        "ok": 200 <= resp.status < 300,
                        "tool": clean_name,
                        "status": resp.status,
                        "elapsed_ms": elapsed_ms,
                        "items": compact if isinstance(compact, list) else None,
                        "data": None if isinstance(compact, list) else compact,
                    }
        except Exception as exc:
            logger.warning(f"自定义工具执行失败: tool={clean_name}, url={url}, error={exc}")
            return {"ok": False, "tool": clean_name, "message": str(exc)}

    async def web_search(self, *, q: str, limit: int = 5) -> Dict[str, Any]:
        """
        输入:
        - `q`: 搜索关键词或问题
        - `limit`: 返回数量

        输出:
        - 网页搜索结果列表

        作用:
        - 为智能体提供轻量外部网页查询能力，适合查找公开页面入口。
        """

        safe_limit = _safe_limit(limit, default=5, maximum=10)
        result = await simple_web_search(q, limit=safe_limit)
        logger.info(f"智能体工具 web_search 完成: q={q}, returned={len(result.get('items') or [])}")
        return result

    async def web_crawl_page(self, *, url: str, max_chars: int = 8000) -> Dict[str, Any]:
        """
        输入:
        - `url`: 公开网页地址
        - `max_chars`: 返回正文最大字符数

        输出:
        - 抓取到的网页正文片段

        作用:
        - 复用项目 Crawl4AI/Playwright 正文抓取链路，供智能体读取公开网页内容。
        """

        safe_chars = _safe_limit(max_chars, default=8000, maximum=WEB_CRAWL_MAX_CHARS)
        try:
            clean_url = await ensure_public_web_url(url)
        except ValueError as exc:
            return {"ok": False, "message": str(exc)}

        started = time.perf_counter()
        content = await crawler_service.crawl_content(clean_url)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        text = compact_web_text(content or "", safe_chars)
        return {
            "ok": bool(text),
            "url": clean_url,
            "elapsed_ms": elapsed_ms,
            "content": text,
            "truncated": bool(content and len(" ".join(str(content).split())) > len(text)),
            "message": "" if text else "未抓取到可用正文",
        }

    async def generate_news_image(
        self,
        *,
        news_id: int = 0,
        title: str = "",
        body: str = "",
        source: str = "",
        time_label: str = "",
        theme: str = "default",
    ) -> Dict[str, Any]:
        """
        输入:
        - `news_id`: 可选新闻 ID
        - `title`/`body`/`source`/`time_label`: 图片文字内容
        - `theme`: 图片主题

        输出:
        - 生成的图片 URL 与文件信息

        作用:
        - 将新闻文字或指定新闻记录渲染成 PNG 图片，方便用户分享或下载。
        """

        clean_title = str(title or "").strip()
        clean_body = str(body or "").strip()
        clean_source = str(source or "").strip()
        clean_time = str(time_label or "").strip()
        source_news_id = int(news_id or 0)

        if source_news_id > 0:
            async with AsyncSessionLocal() as db:
                row = (
                    await db.execute(
                        select(News)
                        .options(defer(News.embedding))
                        .where(News.id == source_news_id)
                    )
                ).scalar_one_or_none()
            if not row:
                return {"ok": False, "message": "新闻不存在"}
            clean_title = clean_title or row.title or f"新闻 {source_news_id}"
            clean_body = clean_body or row.summary or row.content or ""
            clean_source = clean_source or row.source or "TrendSonar"
            if not clean_time and row.publish_date:
                clean_time = row.publish_date.strftime("%Y-%m-%d %H:%M")

        if not clean_title and not clean_body:
            return {"ok": False, "message": "title/body 不能为空"}

        try:
            result = generate_news_text_image(
                title=clean_title,
                body=clean_body,
                source=clean_source,
                time_label=clean_time,
                theme=theme,
            )
            result["news_id"] = source_news_id or None
            logger.info(f"智能体工具 generate_news_image 完成: news_id={source_news_id}, url={result.get('url')}")
            return result
        except Exception as exc:
            logger.warning(f"新闻图片生成失败: news_id={source_news_id}, error={exc}")
            return {"ok": False, "message": str(exc)}

    async def get_top_news(
        self,
        *,
        limit: int = 20,
        date: str = "24h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "heat",
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        输入:
        - 热榜查询参数，包括数量、时间、分类、地区和来源

        输出:
        - 热点新闻列表

        作用:
        - 查询指定时间范围内热度最高或最新的新闻，适合回答“今天热点”“近 24 小时新闻”等问题。
        """

        safe_limit = _safe_limit(limit, default=20, maximum=100)
        async with AsyncSessionLocal() as db:
            stmt = build_news_query_filters(
                select(News).options(defer(News.content), defer(News.embedding)),
                date=date or "24h",
                start_date=_clean_optional_text(start_date),
                end_date=_clean_optional_text(end_date),
                category=_clean_optional_text(category),
                region=_clean_optional_text(region),
                source=_clean_optional_text(source),
            )
            if sort_by == "date":
                stmt = stmt.order_by(desc(News.publish_date))
            else:
                stmt = stmt.order_by(desc(News.heat_score), desc(News.publish_date))

            rows = (await db.execute(stmt.limit(safe_limit))).scalars().all()
            relaxed_search: Optional[dict[str, Any]] = None
            if not rows and (category or region):
                query_text, terms = build_soft_search_query("", category=category, region=region)
                if query_text:
                    relaxed_stmt = build_news_query_filters(
                        select(News),
                        date=date or "24h",
                        start_date=_clean_optional_text(start_date),
                        end_date=_clean_optional_text(end_date),
                        category=None,
                        region=None,
                        source=_clean_optional_text(source),
                    )
                    search_result = await semantic_news_search(
                        db,
                        relaxed_stmt,
                        query_text,
                        offset=0,
                        limit=safe_limit,
                        candidate_limit=2000,
                        min_score=0.16,
                        text_terms=terms,
                        log_prefix="智能体工具 get_top_news 软召回",
                    )
                    rows = [row for _, row in search_result.items]
                    if rows:
                        relaxed_search = {
                            "reason": "严格分类/地区筛选无结果，已改用多关键词语义召回。",
                            "query": search_result.query,
                            "terms": search_result.terms[:20],
                            "matched": search_result.total,
                            "used_embedding": search_result.used_embedding,
                        }
            logger.info(
                f"智能体工具 get_top_news 完成: date={date}, start={start_date}, end={end_date}, "
                f"limit={safe_limit}, returned={len(rows)}, sort_by={sort_by}, relaxed={bool(relaxed_search)}"
            )
            return [_serialize_news_brief(row) for row in rows]

    async def search_news(
        self,
        *,
        q: str,
        limit: int = 100,
        date: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "heat",
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        输入:
        - `q`: 搜索关键词
        - 其他筛选参数：时间、分类、地区、来源和排序

        输出:
        - 仅包含 ID 和标题的新闻搜索结果

        作用:
        - 基于标题、摘要、来源、分类、地区、关键词、实体和向量相似度做多关键词语义召回，供智能体先定位候选新闻。
        """

        query = (q or "").strip()
        safe_limit = _safe_limit(limit, default=100, maximum=300)
        if not query:
            return []

        async with AsyncSessionLocal() as db:
            stmt = build_news_query_filters(
                select(News),
                date=date or "all",
                start_date=_clean_optional_text(start_date),
                end_date=_clean_optional_text(end_date),
                category=None,
                region=None,
                source=_clean_optional_text(source),
            )
            query_variants = build_search_query_variants(
                query,
                category=_clean_optional_text(category),
                region=_clean_optional_text(region),
                max_variants=5,
            )
            if not query_variants:
                query_variants = [build_soft_search_query(query, category=category, region=region)]

            merged: dict[int, tuple[float, News, list[str]]] = {}
            total_elapsed_ms = 0.0
            used_embedding = False

            for index, (query_text, terms) in enumerate(query_variants):
                effective_query = query_text or query
                search_result = await semantic_news_search(
                    db,
                    stmt,
                    effective_query,
                    offset=0,
                    limit=max(safe_limit * 12, 300),
                    candidate_limit=3000,
                    min_score=0.16 if index == 0 else 0.14,
                    text_terms=terms,
                    log_prefix=f"智能体工具 search_news#{index + 1}",
                    use_embedding=False,
                )
                total_elapsed_ms += search_result.elapsed_ms
                used_embedding = used_embedding or search_result.used_embedding
                variant_bonus = max(0.0, 0.08 - index * 0.015)
                for score, row in search_result.items:
                    existing = merged.get(int(row.id))
                    merged_score = float(score) + variant_bonus
                    if existing is None or merged_score > existing[0]:
                        merged[int(row.id)] = (merged_score, row, search_result.terms[:24])

            merged_items = sorted(merged.values(), key=lambda item: item[0], reverse=True)
            strong_items = [
                (score, row, terms)
                for score, row, terms in merged_items
                if strong_core_term_coverage(row, terms)[1] > 0
            ]
            effective_items = strong_items if strong_items else merged_items
            selected = effective_items[:safe_limit]
            rows = [row for _, row, _terms in selected]
            if sort_by == "date":
                rows = sorted(rows, key=lambda item: item.publish_date or datetime.min, reverse=True)
            weak_only = bool(merged_items) and not strong_items
            logger.info(
                f"智能体工具 search_news 完成: q={query}, date={date}, limit={safe_limit}, "
                f"returned={len(rows)}, matched={len(merged)}, attempts={len(query_variants)}, "
                f"embedding={used_embedding}, sort_by={sort_by}, weak_only={weak_only}"
            )
            return [_serialize_news_brief(row) for row in rows]

    async def get_news_detail(self, *, news_id: int = 0, news_ids: Optional[list[int]] = None) -> Dict[str, Any]:
        """
        输入:
        - `news_id`: 单个新闻 ID
        - `news_ids`: 可选的多个新闻 ID

        输出:
        - 单条或多条新闻详情、关联来源和基础状态

        作用:
        - 读取新闻的可复核信息，供智能体在轻量搜索后按 ID 批量获取摘要和详情。
        """

        ids = _safe_news_ids(news_id, news_ids)
        if not ids:
            return {"found": False, "items": [], "message": "news_id/news_ids 不能为空"}

        def build_detail(news: News) -> dict[str, Any]:
            related_sources = []
            if isinstance(news.sources, list):
                for source in news.sources:
                    if not isinstance(source, dict):
                        continue
                    related_sources.append(
                        {
                            "id": source.get("id"),
                            "name": source.get("name") or source.get("source") or "未知来源",
                            "title": source.get("title") or "",
                            "url": source.get("url") or "",
                        }
                    )
            if not any(item.get("url") == news.url for item in related_sources):
                related_sources.insert(
                    0,
                    {"id": news.id, "name": news.source or "主报道", "title": news.title or "", "url": news.url or ""},
                )

            return {
                "news": {
                    **serialize_news_item(news),
                    "is_ai_summary": bool(news.is_ai_summary),
                    "keywords": news.keywords or [],
                    "entities": news.entities or [],
                },
                "related_sources": related_sources[:20],
                "content_status": {
                    "has_summary": bool(news.summary),
                    "related_source_count": len(related_sources),
                },
            }

        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(
                    select(News)
                    .options(defer(News.content), defer(News.embedding))
                    .where(News.id.in_(ids))
                )
            ).scalars().all()
            row_by_id = {int(row.id): row for row in rows}
            items = [build_detail(row_by_id[item_id]) for item_id in ids if item_id in row_by_id]
            missing_ids = [item_id for item_id in ids if item_id not in row_by_id]

            if len(ids) == 1:
                if not items:
                    return {"found": False, "message": "新闻不存在", "missing_ids": missing_ids}
                return {"found": True, **items[0], "missing_ids": missing_ids}

            return {
                "found": bool(items),
                "total": len(items),
                "requested": len(ids),
                "missing_ids": missing_ids,
                "items": items,
            }

    async def list_topics(
        self,
        *,
        q: str = "",
        limit: int = 20,
        date: str = "all",
        min_heat: float = 0.0,
        sort_by: str = "updated",
    ) -> Dict[str, Any]:
        """
        输入:
        - 专题查询参数，包括关键词、时间范围、热度下限和排序方式

        输出:
        - 活跃专题列表

        作用:
        - 查询已有专题，供智能体回答专题相关问题或创建专题前查重。
        """

        safe_limit = _safe_limit(limit, default=20, maximum=80)
        query = (q or "").strip()
        async with AsyncSessionLocal() as db:
            stmt = select(Topic).where(Topic.status == "active")
            if query:
                like = f"%{query}%"
                stmt = stmt.where(or_(Topic.name.ilike(like), Topic.summary.ilike(like), Topic.record.ilike(like)))
            if min_heat and min_heat > 0:
                stmt = stmt.where(Topic.heat_score >= float(min_heat))

            now = datetime.now()
            if date == "today":
                stmt = stmt.where(Topic.updated_time >= datetime.combine(now.date(), datetime.min.time()))
            elif date == "24h":
                stmt = stmt.where(Topic.updated_time >= now - timedelta(hours=24))
            elif date == "3d":
                stmt = stmt.where(Topic.updated_time >= now - timedelta(days=3))
            elif date == "7d":
                stmt = stmt.where(Topic.updated_time >= now - timedelta(days=7))

            if sort_by == "heat":
                stmt = stmt.order_by(desc(Topic.heat_score), desc(Topic.updated_time), desc(Topic.id))
            else:
                stmt = stmt.order_by(desc(Topic.updated_time), desc(Topic.heat_score), desc(Topic.id))

            rows = (await db.execute(stmt.limit(safe_limit))).scalars().all()
            return {"q": query, "total": len(rows), "items": [_serialize_topic(row) for row in rows]}

    async def get_topic_detail(self, *, topic_id: int, timeline_limit: int = 80) -> Dict[str, Any]:
        """
        输入:
        - `topic_id`: 专题 ID
        - `timeline_limit`: 时间轴节点上限

        输出:
        - 专题详情、时间轴和关联新闻

        作用:
        - 读取某个专题的完整上下文，供智能体总结事件进展。
        """

        safe_limit = _safe_limit(timeline_limit, default=80, maximum=200)
        async with AsyncSessionLocal() as db:
            topic = (await db.execute(select(Topic).where(Topic.id == int(topic_id)))).scalar_one_or_none()
            if not topic:
                return {"found": False, "message": "专题不存在"}

            items = (
                await db.execute(
                    select(TopicTimelineItem)
                    .where(TopicTimelineItem.topic_id == int(topic_id))
                    .order_by(desc(TopicTimelineItem.event_time))
                    .limit(safe_limit)
                )
            ).scalars().all()

            news_cards: List[Dict[str, Any]] = []
            news_ids = _collect_timeline_news_ids(items)
            if news_ids:
                news_rows = (
                    await db.execute(
                        select(News)
                        .options(defer(News.content), defer(News.embedding))
                        .where(News.id.in_(list(news_ids)))
                        .order_by(desc(News.publish_date))
                        .limit(80)
                    )
                ).scalars().all()
                news_cards = [serialize_news_item(row) for row in news_rows]

            timeline = [
                {
                    "id": item.id,
                    "time": item.event_time.isoformat() if item.event_time else None,
                    "content": item.content,
                    "news_id": item.news_id,
                    "news_title": item.news_title,
                    "source_name": item.source_name,
                    "source_url": item.source_url,
                    "sources": item.sources or [],
                }
                for item in items
            ]

            return {
                "found": True,
                "topic": _serialize_topic(topic),
                "record": topic.record,
                "timeline": timeline,
                "news": news_cards,
            }

    async def get_report_analysis(
        self,
        *,
        q: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        generate_ai: bool = False,
    ) -> Dict[str, Any]:
        """
        输入:
        - 报告分析筛选参数

        输出:
        - 报告摘要、图表和热点新闻

        作用:
        - 获取舆情报告结构化数据；默认不直接生成 AI 长文，避免工具调用过慢。
        """

        data = await report_service.get_analysis_data(
            keyword=(q or "").strip(),
            start_date=_clean_optional_text(start_date),
            end_date=_clean_optional_text(end_date),
            category=_clean_optional_text(category),
            region=_clean_optional_text(region),
            source=_clean_optional_text(source),
            limit=_safe_limit(limit, default=50, maximum=200),
            generate_ai=bool(generate_ai),
            use_cache=True,
            save_cache=False,
        )
        return self._compact_report_analysis(data)

    async def create_keyword_report(
        self,
        *,
        keyword: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        输入:
        - `keyword`: 关键词
        - 报告筛选参数

        输出:
        - 已存在或新创建的报告 ID 与状态

        作用:
        - 先查询该关键词现有报告；存在则返回最新报告，不重复创建。不存在则创建报告缓存。
        """

        query = (keyword or "").strip()
        if not query:
            return {"created": False, "message": "keyword 不能为空"}

        existing = await report_service.get_report_history(query, limit=1)
        if existing:
            return {
                "created": False,
                "message": "已存在该关键词报告，返回最新报告",
                "report": existing[0],
            }

        async with self._report_create_lock:
            now = time.monotonic()
            remaining = REPORT_CREATE_COOLDOWN_SECONDS - (now - self._last_report_create_at)
            if remaining > 0:
                retry_after = int(remaining) + 1
                logger.info(f"智能体创建报告被限流: keyword={query}, retry_after={retry_after}s")
                return {
                    "created": False,
                    "rate_limited": True,
                    "retry_after_seconds": retry_after,
                    "message": f"报告创建过于频繁，请约 {retry_after} 秒后再试。系统限制为每 1 分钟创建 1 个新报告。",
                }
            self._last_report_create_at = now

        report_id = await report_service.generate_report_and_stream_ai(
            keyword=query,
            start_date=_clean_optional_text(start_date),
            end_date=_clean_optional_text(end_date),
            category=_clean_optional_text(category),
            region=_clean_optional_text(region),
            source=_clean_optional_text(source),
            limit=_safe_limit(limit, default=100, maximum=200),
        )
        if not report_id:
            return {"created": False, "message": "报告创建失败"}
        return {
            "created": True,
            "message": "报告已创建",
            "report": {"id": report_id, "keyword": query},
        }

    async def create_event_topic(self, *, name: str, is_admin: bool = False) -> Dict[str, Any]:
        """
        输入:
        - `name`: 新闻事件专题名称
        - `is_admin`: 当前请求是否已管理员登录

        输出:
        - 已存在或新创建的专题信息

        作用:
        - 创建指定新闻事件专题；未登录时拒绝创建，创建前查询现有专题避免重复创建。
        """

        if not is_admin:
            logger.info(f"智能体创建专题被拦截: name={name}, reason=未登录")
            return {
                "created": False,
                "auth_required": True,
                "message": "创建专题需要先登录管理账号，请登录后再试。",
            }

        topic_name = (name or "").strip()
        if not topic_name:
            return {"created": False, "message": "name 不能为空"}

        existing = await self.list_topics(q=topic_name, limit=5, date="all")
        exact = [item for item in existing.get("items", []) if item.get("name") == topic_name]
        if exact:
            return {
                "created": False,
                "message": "已存在同名专题",
                "topic": exact[0],
                "similar_topics": existing.get("items", []),
            }

        async with AsyncSessionLocal() as db:
            try:
                topic = await topic_service.create_manual_topic(db, topic_name, trigger_scan=False)
                await db.commit()
                await db.refresh(topic)
            except ValueError as exc:
                await db.rollback()
                return {
                    "created": False,
                    "message": str(exc),
                    "similar_topics": existing.get("items", []),
                }

        async def run_topic_scan() -> None:
            """
            输入:
            - 无，闭包读取专题 ID

            输出:
            - 无

            作用:
            - 在后台为智能体创建的专题扫描相关新闻，避免聊天请求长期阻塞。
            """

            await topic_service.run_topic_scan_in_background(int(topic.id), include_used=True)

        task_name = f"agent_topic_scan:{topic.id}"
        status = await task_manager.start_background(task_name, run_topic_scan, progress=f"专题“{topic_name}”扫描中")
        return {
            "created": True,
            "message": "专题已创建，相关新闻扫描已在后台启动",
            "topic": _serialize_topic(topic),
            "task": status,
            "similar_topics": existing.get("items", []),
        }

    async def get_term_analysis(
        self,
        *,
        term: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
        range: str = "year",
    ) -> Dict[str, Any]:
        """
        输入:
        - `term`: 词项或关键词
        - 可选筛选范围

        输出:
        - 词项相关统计、趋势、情感和共现数据

        作用:
        - 查询某个关键词的词项分析，支撑智能体回答“这个词最近趋势如何”等问题。
        """

        query = (term or "").strip()
        if not query:
            return {"found": False, "message": "term 不能为空"}

        range_key = (range or "year").strip().lower()
        final_start = _clean_optional_text(start_date)
        final_end = _clean_optional_text(end_date)
        if not final_start and not final_end and range_key != "all":
            now = datetime.now()
            if range_key == "year":
                final_start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
            else:
                final_start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            final_end = now.strftime("%Y-%m-%d")

        async with AsyncSessionLocal() as db:
            term_condition = self._term_match_conditions(query)
            stats_stmt = build_news_query_filters(
                select(
                    func.count(),
                    func.coalesce(func.sum(News.heat_score), 0.0),
                    func.avg(News.sentiment_score),
                ),
                date="all",
                start_date=final_start,
                end_date=final_end,
                category=_clean_optional_text(category),
                region=_clean_optional_text(region),
                source=_clean_optional_text(source),
            ).where(term_condition)
            total, total_heat, avg_sentiment = (await db.execute(stats_stmt)).one()

            day_expr = func.date(News.publish_date)
            trend_stmt = build_news_query_filters(
                select(
                    day_expr.label("day"),
                    func.count().label("count"),
                    func.coalesce(func.sum(News.heat_score), 0.0).label("heat"),
                    func.avg(News.sentiment_score).label("avg_sentiment"),
                ),
                date="all",
                start_date=final_start,
                end_date=final_end,
                category=_clean_optional_text(category),
                region=_clean_optional_text(region),
                source=_clean_optional_text(source),
            ).where(term_condition).group_by(day_expr).order_by(day_expr)
            trend_rows = (await db.execute(trend_stmt)).all()

            sample_stmt = build_news_query_filters(
                select(News)
                .options(defer(News.content), defer(News.embedding))
                .where(term_condition)
                .order_by(desc(News.heat_score), desc(News.publish_date))
                .limit(200),
                date="all",
                start_date=final_start,
                end_date=final_end,
                category=_clean_optional_text(category),
                region=_clean_optional_text(region),
                source=_clean_optional_text(source),
            )
            sample_news = (await db.execute(sample_stmt)).scalars().all()

        labels = [str(day) for day, _count, _heat, _avg_sentiment in trend_rows if day]
        daily = {
            str(day): {
                "count": int(count or 0),
                "heat": float(heat or 0.0),
                "avg_sentiment": float(avg or 0.0) if avg is not None else None,
            }
            for day, count, heat, avg in trend_rows
            if day
        }

        sentiment_counts: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        pair_counter: Counter[tuple[str, str]] = Counter()
        for item in sample_news:
            label = (item.sentiment_label or "中立").strip()
            if label == "正面":
                sentiment_counts["positive"] += 1
            elif label == "负面":
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
            terms = self._news_terms(item)
            for kw in terms:
                keyword_counter[kw] += 1
            for left, right in combinations(sorted(set(terms), key=str.lower)[:8], 2):
                pair_counter[(left, right)] += 1

        top_terms = keyword_counter.most_common(18)
        known = {name for name, _value in top_terms}
        return {
            "found": True,
            "term": query,
            "summary": {
                "related_count": int(total or 0),
                "sample_count": len(sample_news),
                "total_heat": round(float(total_heat or 0.0), 2),
                "avg_sentiment": round(float(avg_sentiment or 0.0), 2),
                "sentiment_counts": dict(sentiment_counts),
            },
            "trend": {
                "dates": labels,
                "heat": [round(float(daily[label]["heat"]), 2) for label in labels],
                "count": [int(daily[label]["count"]) for label in labels],
                "avg_sentiment": [daily[label]["avg_sentiment"] for label in labels],
            },
            "related_news": [serialize_news_item(item) for item in sample_news[:10]],
            "cooccurrence": {
                "nodes": [{"name": name, "value": int(value)} for name, value in top_terms],
                "links": [
                    {"source": left, "target": right, "value": int(value)}
                    for (left, right), value in pair_counter.most_common(30)
                    if left in known and right in known
                ],
            },
        }

    def _compact_report_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入:
        - `data`: 报告服务返回的完整数据

        输出:
        - 适合放入模型上下文的精简报告数据

        作用:
        - 控制工具输出体积，同时保留摘要、核心图表和 Top 新闻。
        """

        charts = data.get("charts") or {}
        return {
            "id": data.get("id"),
            "params": data.get("params") or {},
            "summary": data.get("summary") or {},
            "top_news": (data.get("top_news") or [])[:20],
            "charts": {
                "source": (charts.get("source") or [])[:10],
                "word_cloud": (charts.get("word_cloud") or [])[:20],
                "sentiment_dist": charts.get("sentiment_dist") or [],
                "neg_keywords": (charts.get("neg_keywords") or [])[:10],
                "pos_keywords": (charts.get("pos_keywords") or [])[:10],
                "trend": charts.get("trend") or {},
            },
            "ai_analysis": data.get("ai_analysis"),
        }

    def _term_match_conditions(self, query: str) -> Any:
        """
        输入:
        - `query`: 词项关键词

        输出:
        - SQLAlchemy 文本匹配条件

        作用:
        - 统一词项分析中标题、摘要、关键词和实体的匹配口径。
        """

        like = f"%{query}%"
        return or_(
            News.title.ilike(like),
            News.summary.ilike(like),
            cast(News.keywords, Text).ilike(like),
            cast(News.entities, Text).ilike(like),
        )

    def _news_terms(self, news: News, limit: int = 12) -> List[str]:
        """
        输入:
        - `news`: 新闻 ORM 对象
        - `limit`: 最大词项数量

        输出:
        - 去重后的关键词和实体列表

        作用:
        - 为词项分析构建共现网络，过滤无意义占位词。
        """

        terms: List[str] = []
        seen: set[str] = set()
        ignored = {"无内容", "分析失败", "暂无关键词", "其他", "其它", "null", "none"}
        for item in (news.keywords or []) + (news.entities or []):
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value or value.lower() in ignored:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(value)
            if len(terms) >= limit:
                break
        return terms


agent_tool_service = AgentToolService()
