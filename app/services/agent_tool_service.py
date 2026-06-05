"""
本文件用于封装 TrendSonar 智能体可调用工具，统一复用新闻、专题与报告服务能力。
主要类/对象:
- `AgentToolService`: 智能体工具服务
- `agent_tool_service`: 全局工具服务单例
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional

from sqlalchemy import Text, cast, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.database import AsyncSessionLocal
from app.core.logger import setup_logger
from app.models.news import News
from app.models.topic import Topic, TopicTimelineItem
from app.services.report_service import report_service
from app.services.task_manager import task_manager
from app.services.topic_service import topic_service
from app.utils.agent_tool_config import delete_custom_agent_tool, load_custom_agent_tools, save_custom_agent_tool
from app.utils.news_query import build_news_query_filters, serialize_news_item
from app.utils.tools import normalize_regions_to_countries

logger = setup_logger("AgentToolService")


BUILTIN_AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_top_news",
        "title": "热点新闻查询",
        "description": "查询指定时间范围内的热点新闻 TopN。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "limit": {"type": "integer", "default": 20, "description": "返回数量，默认 20。"},
            "date": {"type": "string", "default": "today", "description": "today/24h/3d/7d/week/month/year/all。"},
            "sort_by": {"type": "string", "default": "heat", "description": "heat 或 date。"},
            "category": {"type": "string", "default": "", "description": "可选分类。"},
            "region": {"type": "string", "default": "", "description": "可选地区。"},
            "source": {"type": "string", "default": "", "description": "可选来源。"},
        },
    },
    {
        "name": "search_news",
        "title": "关键词新闻搜索",
        "description": "按关键词搜索新闻并支持时间、分类、地区和来源筛选。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {
            "q": {"type": "string", "default": "", "description": "搜索关键词。"},
            "limit": {"type": "integer", "default": 20, "description": "返回数量。"},
            "date": {"type": "string", "default": "all", "description": "时间范围。"},
            "sort_by": {"type": "string", "default": "heat", "description": "heat 或 date。"},
        },
    },
    {
        "name": "get_news_detail",
        "title": "新闻详情读取",
        "description": "读取指定新闻 ID 的详情、摘要、来源、关键词和实体。",
        "kind": "builtin",
        "enabled": True,
        "safe_to_test": True,
        "parameters": {"news_id": {"type": "integer", "default": 0, "description": "新闻 ID。"}},
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


def _date_label(date: str, start_date: Optional[str], end_date: Optional[str]) -> str:
    """
    输入:
    - `date`: 快捷时间范围
    - `start_date`/`end_date`: 自定义日期

    输出:
    - 可读的时间范围说明

    作用:
    - 让智能体工具返回明确说明，避免用户误以为“今天”和“24 小时”相同。
    """

    if start_date or end_date:
        return f"{start_date or '不限'} 至 {end_date or '不限'}"
    labels = {
        "today": "今天",
        "yesterday": "昨天",
        "24h": "最近24小时",
        "3d": "最近3天",
        "7d": "最近7天",
        "week": "本周",
        "month": "本月",
        "year": "今年",
        "all": "所有时间",
    }
    return labels.get(str(date or "").strip(), str(date or "24h"))


def _summarize_rows(rows: list[News]) -> dict[str, Any]:
    """
    输入:
    - `rows`: 新闻 ORM 列表

    输出:
    - 工具结果统计说明

    作用:
    - 为智能体提供摘要覆盖率和结果数量，帮助其解释输出范围。
    """

    return {
        "returned": len(rows),
        "summary_available": sum(1 for row in rows if bool(row.summary)),
        "summary_missing": sum(1 for row in rows if not row.summary),
    }


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

    def list_tool_definitions(self) -> Dict[str, Any]:
        """
        输入:
        - 无

        输出:
        - 内置工具与自定义工具元数据

        作用:
        - 为管理端提供工具查看、提示词编辑入口和新增工具草案列表。
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
                return {
                    "ok": True,
                    "dry_run": True,
                    "message": "自定义工具当前作为草案保存，尚未绑定后端执行器。",
                    "tool": custom_tool,
                    "args": args or {},
                }
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
                limit=_safe_limit(args.get("limit", 20), default=20, maximum=30),
                date=str(args.get("date") or "all"),
                start_date=_clean_optional_text(args.get("start_date")),
                end_date=_clean_optional_text(args.get("end_date")),
                sort_by=str(args.get("sort_by") or "heat"),
                category=_clean_optional_text(args.get("category")),
                region=_clean_optional_text(args.get("region")),
                source=_clean_optional_text(args.get("source")),
            )
        if name == "get_news_detail":
            return await self.get_news_detail(news_id=int(args.get("news_id") or 0))
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
        - 为管理端删除自定义工具草案。
        """

        ok = delete_custom_agent_tool(name)
        if ok:
            logger.info(f"删除自定义智能体工具: name={name}")
        return ok

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
    ) -> Dict[str, Any]:
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
            logger.info(
                f"智能体工具 get_top_news 完成: date={date}, start={start_date}, end={end_date}, "
                f"limit={safe_limit}, returned={len(rows)}, sort_by={sort_by}"
            )
            return {
                "total": len(rows),
                "limit": safe_limit,
                "date": date,
                "time_range_label": _date_label(date or "24h", start_date, end_date),
                "stats": _summarize_rows(rows),
                "items": [serialize_news_item(row) for row in rows],
            }

    async def search_news(
        self,
        *,
        q: str,
        limit: int = 20,
        date: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "heat",
        category: Optional[str] = None,
        region: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        输入:
        - `q`: 搜索关键词
        - 其他筛选参数：时间、分类、地区、来源和排序

        输出:
        - 新闻搜索结果

        作用:
        - 基于标题、摘要、来源、分类、地区、关键词和实体做文本检索，供智能体查找特定事件。
        """

        query = (q or "").strip()
        safe_limit = _safe_limit(limit, default=20, maximum=80)
        if not query:
            return {"total": 0, "items": [], "message": "q 不能为空"}

        terms = [item.strip().lower() for item in query.replace("，", " ").replace(",", " ").split() if item.strip()]
        if query.lower() not in terms:
            terms.insert(0, query.lower())

        async with AsyncSessionLocal() as db:
            stmt = build_news_query_filters(
                select(News).options(defer(News.content), defer(News.embedding)),
                date=date or "all",
                start_date=_clean_optional_text(start_date),
                end_date=_clean_optional_text(end_date),
                category=_clean_optional_text(category),
                region=_clean_optional_text(region),
                source=_clean_optional_text(source),
            )
            conditions = []
            for term in terms:
                like = f"%{term}%"
                conditions.extend(
                    [
                        News.title.ilike(like),
                        News.summary.ilike(like),
                        News.source.ilike(like),
                        News.category.ilike(like),
                        News.region.ilike(like),
                        cast(News.keywords, Text).ilike(like),
                        cast(News.entities, Text).ilike(like),
                    ]
                )
            stmt = stmt.where(or_(*conditions))
            if sort_by == "date":
                stmt = stmt.order_by(desc(News.publish_date), desc(News.heat_score))
            else:
                stmt = stmt.order_by(desc(News.heat_score), desc(News.publish_date))
            rows = (await db.execute(stmt.limit(safe_limit))).scalars().all()
            logger.info(
                f"智能体工具 search_news 完成: q={query}, date={date}, limit={safe_limit}, "
                f"returned={len(rows)}, sort_by={sort_by}"
            )
            return {
                "q": query,
                "total": len(rows),
                "limit": safe_limit,
                "time_range_label": _date_label(date or "all", start_date, end_date),
                "stats": _summarize_rows(rows),
                "items": [serialize_news_item(row) for row in rows],
            }

    async def get_news_detail(self, *, news_id: int) -> Dict[str, Any]:
        """
        输入:
        - `news_id`: 新闻 ID

        输出:
        - 新闻详情、关联来源和基础状态

        作用:
        - 读取单条新闻的可复核信息，供智能体基于具体事件继续分析。
        """

        async with AsyncSessionLocal() as db:
            news = await db.get(News, int(news_id), options=[defer(News.content), defer(News.embedding)])
            if not news:
                return {"found": False, "message": "新闻不存在"}

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
                "found": True,
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

    async def create_event_topic(self, *, name: str) -> Dict[str, Any]:
        """
        输入:
        - `name`: 新闻事件专题名称

        输出:
        - 已存在或新创建的专题信息

        作用:
        - 创建指定新闻事件专题；创建前查询现有专题，避免重复创建。
        """

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
