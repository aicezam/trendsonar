"""
本文件用于构建 TrendSonar 智能体，提供连续对话、工具调用和可观察事件流。
主要类/对象:
- `AgentService`: 智能体对话服务
- `agent_service`: 全局智能体服务单例
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    TextPartDelta,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.run import AgentRunResultEvent

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.services.agent_tool_service import agent_tool_service

logger = setup_logger("AgentService")

AGENT_SYSTEM_PROMPT = """
你是 TrendSonar 新闻智能体。你可以自主分解用户任务，并一次或多次调用工具完成问题。

工作原则:
1. 回答必须使用中文，优先基于工具返回的事实，不编造新闻、专题、报告 ID 或链接。
2. 用户问“今天”“昨日”“本周”等相对日期时，优先调用新闻或报告工具确认数据。
3. 需要创建报告或专题时，必须先调用对应工具完成查重；工具会负责发现已存在记录时避免重复创建。
4. 如果需要同时了解新闻、专题、报告或词项趋势，可以连续调用多个工具，最后给出整合结论。
5. 当工具返回数据为空时，要说明没有查到，并给出下一步可尝试的筛选条件。
6. 最终回复要简洁、可执行；涉及列表时保留标题、来源、时间、热度或 ID 等关键信息。
"""


@dataclass
class AgentConversation:
    """
    输入:
    - `conversation_id`: 会话 ID

    输出:
    - 保存消息历史和最近访问时间的会话状态

    作用:
    - 为首页智能体提供轻量级连续对话能力。
    """

    conversation_id: str
    messages: List[ModelMessage] = field(default_factory=list)
    turn_count: int = 0


@dataclass
class AgentDeps:
    """
    输入:
    - `conversation_id`: 当前会话 ID

    输出:
    - 工具调用上下文依赖

    作用:
    - 向 PydanticAI 工具函数注入项目服务与会话信息，避免工具直接依赖全局状态。
    """

    conversation_id: str


class AgentService:
    """
    输入:
    - OpenAI 兼容模型配置和项目工具服务

    输出:
    - 智能体事件流和对话历史

    作用:
    - 统一管理智能体实例、连续对话历史和工具调用事件。
    """

    def __init__(self) -> None:
        self._conversations: Dict[str, AgentConversation] = {}

    def create_conversation_id(self) -> str:
        """
        输入:
        - 无

        输出:
        - 新会话 ID

        作用:
        - 为前端未传会话 ID 的请求生成稳定会话标识。
        """

        return uuid4().hex

    def _get_conversation(self, conversation_id: str) -> AgentConversation:
        """
        输入:
        - `conversation_id`: 会话 ID

        输出:
        - 会话状态对象

        作用:
        - 获取或创建会话，并限制历史消息长度，避免上下文无限增长。
        """

        conv = self._conversations.get(conversation_id)
        if not conv:
            conv = AgentConversation(conversation_id=conversation_id)
            self._conversations[conversation_id] = conv
        if len(conv.messages) > 24:
            conv.messages = conv.messages[-24:]
        return conv

    def _build_model(self, use_backup: bool = False) -> OpenAIChatModel:
        """
        输入:
        - `use_backup`: 是否强制使用备用模型

        输出:
        - PydanticAI OpenAI 聊天模型

        作用:
        - 复用项目现有主/备用 OpenAI-compatible 模型配置。
        """

        settings = get_settings()
        route_prefers_backup = (settings.AI_ROUTE or {}).get("CHAT") == "backup"
        model_type = "backup" if use_backup or route_prefers_backup else "main"
        if model_type == "backup":
            api_key = settings.BACKUP_AI_API_KEY
            base_url = settings.BACKUP_AI_BASE_URL
            model_name = settings.BACKUP_AI_MODEL
            if not (api_key and base_url and model_name):
                api_key = settings.MAIN_AI_API_KEY
                base_url = settings.MAIN_AI_BASE_URL
                model_name = settings.MAIN_AI_MODEL
        else:
            api_key = settings.MAIN_AI_API_KEY
            base_url = settings.MAIN_AI_BASE_URL
            model_name = settings.MAIN_AI_MODEL

        if not (api_key and base_url and model_name):
            raise RuntimeError("AI 服务暂时不可用，请先在管理页完善 AI 配置")

        provider = OpenAIProvider(api_key=str(api_key), base_url=str(base_url))
        return OpenAIChatModel(str(model_name), provider=provider)

    def _build_agent(self, use_backup: bool = False) -> Agent[AgentDeps, str]:
        """
        输入:
        - `use_backup`: 是否使用备用模型

        输出:
        - 已注册 TrendSonar 工具的 PydanticAI Agent

        作用:
        - 每次请求构建轻量 Agent，确保模型配置变更后即时生效。
        """

        agent: Agent[AgentDeps, str] = Agent(
            self._build_model(use_backup),
            deps_type=AgentDeps,
            instructions=AGENT_SYSTEM_PROMPT,
            tool_timeout=180,
            retries=1,
            max_concurrency=4,
        )
        self._register_tools(agent)
        return agent

    def _register_tools(self, agent: Agent[AgentDeps, str]) -> None:
        """
        输入:
        - `agent`: PydanticAI Agent 实例

        输出:
        - 无

        作用:
        - 注册新闻、专题、报告和词项分析工具。
        """

        @agent.tool(description="查询指定时间范围内的热点新闻 TopN，适合今天热点、24小时热点、本周热点等问题。")
        async def get_top_news(
            ctx: RunContext[AgentDeps],
            limit: int = 20,
            date: str = "24h",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            sort_by: str = "heat",
            category: Optional[str] = None,
            region: Optional[str] = None,
            source: Optional[str] = None,
        ) -> Dict[str, Any]:
            return await agent_tool_service.get_top_news(
                limit=limit,
                date=date,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by,
                category=category,
                region=region,
                source=source,
            )

        @agent.tool(description="按关键词搜索新闻，可按时间、分类、地区、来源筛选。")
        async def search_news(
            ctx: RunContext[AgentDeps],
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
            return await agent_tool_service.search_news(
                q=q,
                limit=limit,
                date=date,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by,
                category=category,
                region=region,
                source=source,
            )

        @agent.tool(description="读取指定新闻 ID 的详情、摘要、来源和关键词实体。")
        async def get_news_detail(ctx: RunContext[AgentDeps], news_id: int) -> Dict[str, Any]:
            return await agent_tool_service.get_news_detail(news_id=news_id)

        @agent.tool(description="查询已有活跃专题。创建专题前必须先用这个能力或 create_event_topic 的查重能力确认是否已存在。")
        async def list_topics(
            ctx: RunContext[AgentDeps],
            q: str = "",
            limit: int = 20,
            date: str = "all",
            min_heat: float = 0.0,
            sort_by: str = "updated",
        ) -> Dict[str, Any]:
            return await agent_tool_service.list_topics(q=q, limit=limit, date=date, min_heat=min_heat, sort_by=sort_by)

        @agent.tool(description="读取指定专题 ID 的详情、时间轴和相关新闻。")
        async def get_topic_detail(
            ctx: RunContext[AgentDeps],
            topic_id: int,
            timeline_limit: int = 80,
        ) -> Dict[str, Any]:
            return await agent_tool_service.get_topic_detail(topic_id=topic_id, timeline_limit=timeline_limit)

        @agent.tool(description="获取报告分析数据，包括摘要、来源分布、词云、情感分布、趋势和 Top 新闻。")
        async def get_report_analysis(
            ctx: RunContext[AgentDeps],
            q: str = "",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            category: Optional[str] = None,
            region: Optional[str] = None,
            source: Optional[str] = None,
            limit: int = 50,
            generate_ai: bool = False,
        ) -> Dict[str, Any]:
            return await agent_tool_service.get_report_analysis(
                q=q,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
                source=source,
                limit=limit,
                generate_ai=generate_ai,
            )

        @agent.tool(description="生成指定关键词的报告缓存。工具内部会先查询现有报告，已存在时不会重复创建。")
        async def create_keyword_report(
            ctx: RunContext[AgentDeps],
            keyword: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            category: Optional[str] = None,
            region: Optional[str] = None,
            source: Optional[str] = None,
            limit: int = 100,
        ) -> Dict[str, Any]:
            return await agent_tool_service.create_keyword_report(
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
                source=source,
                limit=limit,
            )

        @agent.tool(description="创建某个新闻事件专题。工具内部会先查询现有专题，已存在同名专题时不会重复创建。")
        async def create_event_topic(ctx: RunContext[AgentDeps], name: str) -> Dict[str, Any]:
            return await agent_tool_service.create_event_topic(name=name)

        @agent.tool(description="查询某个关键词或词项的词项分析，包括热度趋势、相关新闻、情感和共现词。")
        async def get_term_analysis(
            ctx: RunContext[AgentDeps],
            term: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            category: Optional[str] = None,
            region: Optional[str] = None,
            source: Optional[str] = None,
            range: str = "year",
        ) -> Dict[str, Any]:
            return await agent_tool_service.get_term_analysis(
                term=term,
                start_date=start_date,
                end_date=end_date,
                category=category,
                region=region,
                source=source,
                range=range,
            )

    async def stream_chat(
        self,
        *,
        query: str,
        conversation_id: Optional[str] = None,
        use_backup: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        输入:
        - `query`: 用户问题
        - `conversation_id`: 可选会话 ID
        - `use_backup`: 是否强制备用模型

        输出:
        - 智能体事件流字典

        作用:
        - 运行 PydanticAI 智能体，并把工具调用、工具结果和回答增量转换成前端可显示事件。
        """

        user_query = (query or "").strip()
        if not user_query:
            yield {"type": "error", "message": "问题不能为空"}
            return

        cid = conversation_id or self.create_conversation_id()
        conv = self._get_conversation(cid)
        yield {"type": "conversation", "conversation_id": cid}

        try:
            agent = self._build_agent(use_backup=use_backup)
            deps = AgentDeps(conversation_id=cid)
            answer_parts: List[str] = []
            final_output = ""
            async with agent.run_stream_events(
                user_query,
                deps=deps,
                message_history=conv.messages,
                conversation_id=cid,
            ) as stream:
                async for event in stream:
                    for payload in self._convert_event(event):
                        if payload.get("type") == "answer_delta":
                            answer_parts.append(str(payload.get("content") or ""))
                        if payload.get("type") == "agent_result":
                            final_output = str(payload.get("content") or "")
                            messages = payload.get("messages")
                            if isinstance(messages, list):
                                conv.messages = messages
                                conv.turn_count += 1
                            continue
                        yield payload
            yield {"type": "answer_done", "content": final_output or "".join(answer_parts)}
        except Exception as exc:
            logger.error(f"智能体对话失败: {exc}", exc_info=True)
            yield {"type": "error", "message": str(exc)}

    def _convert_event(self, event: AgentStreamEvent) -> List[Dict[str, Any]]:
        """
        输入:
        - `event`: PydanticAI 原始流式事件

        输出:
        - 前端可渲染的事件列表

        作用:
        - 将框架事件映射成稳定的 JSONL 协议。
        """

        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            return [
                {
                    "type": "tool_call",
                    "tool_name": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "args": part.args,
                    "args_valid": event.args_valid,
                }
            ]

        if isinstance(event, FunctionToolResultEvent):
            part = event.part
            if not part:
                return []
            return [
                {
                    "type": "tool_result",
                    "tool_name": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "content": self._json_safe(part.content),
                    "outcome": getattr(part, "outcome", "success"),
                }
            ]

        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            return [{"type": "answer_delta", "content": event.delta.content_delta}]

        if isinstance(event, AgentRunResultEvent):
            return [
                {
                    "type": "agent_result",
                    "content": str(event.result.output or ""),
                    "messages": list(event.result.all_messages()),
                }
            ]

        return []

    def _json_safe(self, value: Any) -> Any:
        """
        输入:
        - `value`: 任意工具返回值

        输出:
        - 可 JSON 序列化且大小受控的值

        作用:
        - 保护前端工具详情展示，避免返回体过大或不可序列化。
        """

        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
        if len(text) > 12000:
            text = text[:12000] + "...(已截断)"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


agent_service = AgentService()
