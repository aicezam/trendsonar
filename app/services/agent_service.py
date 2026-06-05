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
from app.core.prompts import prompt_manager
from app.services.agent_tool_service import agent_tool_service

logger = setup_logger("AgentService")

AGENT_SYSTEM_PROMPT = """
你是 TrendSonar 新闻智能体。你可以自主分解用户任务，并一次或多次调用工具完成问题。

工作原则:
1. 回答必须使用中文，优先基于工具返回的事实，不编造新闻、专题、报告 ID 或链接。
2. 用户问“今天”“昨日”“本周”等相对日期时，优先调用新闻或报告工具确认数据；“今天有哪些热点新闻”默认使用 get_top_news(date="today", limit=20)，不要自行降到 10 条。
3. 列出新闻时必须尽量写出标题、来源、时间、热度、摘要和引用标记；引用格式使用 [新闻:ID]，其中 ID 必须来自工具返回。
4. 如果新闻工具返回 summary 字段，回答必须输出摘要；summary 为空时才写“暂无摘要”。
5. 用户用自然语言查找某类新闻时，优先把地区、主题、同义词合并到 search_news 的 q 中做语义召回；不要只依赖 category/region 硬过滤。例如“美国近7天军事新闻”可调用 search_news(q="美国 军事 美军 防务", date="7d")。
6. 需要创建报告或专题时，必须先调用对应工具完成查重；工具会负责发现已存在记录时避免重复创建。创建专题必须用户已登录管理账号；创建报告受到每 1 分钟 1 个新报告的限流，遇到 rate_limited 时要提示用户稍后再试。
7. 如果需要同时了解新闻、专题、报告或词项趋势，可以连续调用多个工具，最后给出整合结论。
8. 当工具返回数据为空时，要说明没有查到，并给出下一步可尝试的筛选条件。
9. 新闻列表、热点梳理或趋势分析类回答，必须在正文末尾单独输出“下一步建议：”，给出 2-4 条可继续追问的建议；每条建议使用 <<建议:建议文本>> 标记，不要把建议混在正文里。
10. 不要在正文末尾额外输出关键词、实体或标签清单；可点击词项由前端从正文自动识别。
11. 最终回复要简洁、可执行；涉及列表时保留标题、来源、时间、热度或 ID 等关键信息。
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
    is_admin: bool = False


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
            instructions=self._get_system_prompt(),
            tool_timeout=180,
            retries=1,
            max_concurrency=4,
        )
        self._register_tools(agent)
        return agent

    def _get_system_prompt(self) -> str:
        """
        输入:
        - 无

        输出:
        - 智能体系统提示词

        作用:
        - 优先使用管理端可编辑提示词，缺失时回退到代码内置默认值。
        """

        prompt = prompt_manager.get_system_prompt("agent_system").strip()
        return prompt or AGENT_SYSTEM_PROMPT

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

        @agent.tool(description="生成指定关键词的报告缓存。工具内部会先查询现有报告，已存在时不会重复创建；新建报告限制为每 1 分钟 1 个。")
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

        @agent.tool(description="创建某个新闻事件专题。必须用户已登录管理账号；工具内部会先查询现有专题，已存在同名专题时不会重复创建。")
        async def create_event_topic(ctx: RunContext[AgentDeps], name: str) -> Dict[str, Any]:
            return await agent_tool_service.create_event_topic(name=name, is_admin=ctx.deps.is_admin)

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
        is_admin: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        输入:
        - `query`: 用户问题
        - `conversation_id`: 可选会话 ID
        - `use_backup`: 是否强制备用模型
        - `is_admin`: 当前请求是否已管理员登录

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
        logger.info(
            f"智能体收到对话请求: conversation_id={cid}, query={user_query}, "
            f"use_backup={use_backup}, is_admin={is_admin}"
        )
        yield {"type": "conversation", "conversation_id": cid}

        try:
            agent = self._build_agent(use_backup=use_backup)
            deps = AgentDeps(conversation_id=cid, is_admin=bool(is_admin))
            answer_parts: List[str] = []
            final_output = ""
            emitted_meta: Dict[str, Any] = {"references": [], "terms": []}
            async with agent.run_stream_events(
                user_query,
                deps=deps,
                message_history=conv.messages,
                conversation_id=cid,
            ) as stream:
                async for event in stream:
                    for payload in self._convert_event(event):
                        if payload.get("type") == "tool_result":
                            emitted_meta = self._merge_meta_payload(emitted_meta, payload.get("meta"))
                            emitted_meta = self._merge_tool_meta(emitted_meta, payload.get("content"))
                            meta_event = self._build_meta_event(emitted_meta)
                            if meta_event:
                                yield meta_event
                        if payload.get("type") == "answer_delta":
                            answer_parts.append(str(payload.get("content") or ""))
                        if payload.get("type") == "agent_result":
                            final_output = str(payload.get("content") or "")
                            messages = payload.get("messages")
                            if isinstance(messages, list):
                                conv.messages = messages
                                conv.turn_count += 1
                            logger.info(
                                f"智能体对话完成: conversation_id={cid}, turns={conv.turn_count}, "
                                f"answer_length={len(final_output or ''.join(answer_parts))}"
                            )
                            continue
                        yield payload
            final_answer = final_output or "".join(answer_parts)
            final_meta = self._build_meta_event(emitted_meta)
            if final_meta:
                yield final_meta
            yield {"type": "answer_done", "content": final_answer}
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
            logger.info(f"智能体调用工具: name={part.tool_name}, call_id={part.tool_call_id}, args={self._json_safe(part.args, max_chars=2000)}")
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
            preview = self._json_safe(part.content, max_chars=1200)
            logger.info(f"智能体工具返回: name={part.tool_name}, call_id={part.tool_call_id}, outcome={getattr(part, 'outcome', 'success')}, preview={preview}")
            raw_meta = self._merge_tool_meta({"references": [], "terms": []}, part.content)
            return [
                {
                    "type": "tool_result",
                    "tool_name": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "content": self._json_safe(part.content),
                    "meta": raw_meta,
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

    def _json_safe(self, value: Any, max_chars: int = 12000) -> Any:
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
        if max_chars > 0 and len(text) > max_chars:
            try:
                return {
                    "preview": text[:max_chars],
                    "truncated": True,
                    "original_length": len(text),
                    "message": "工具结果较长，前端仅展示预览；模型侧已收到完整工具结果。",
                }
            except Exception:
                text = text[:max_chars] + "...(已截断)"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _merge_meta_payload(self, current: Dict[str, Any], meta: Any) -> Dict[str, Any]:
        """
        输入:
        - `current`: 已累计元数据
        - `meta`: 单次工具事件携带的元数据

        输出:
        - 去重合并后的元数据

        作用:
        - 合并直接从原始工具返回中抽取的引用与词项，避免前端预览截断影响点击能力。
        """

        if not isinstance(meta, dict):
            return current
        merged = {
            "references": list(current.get("references") or []),
            "terms": list(current.get("terms") or []),
        }
        seen_refs = {str(item.get("id")) for item in merged["references"] if item.get("id") is not None}
        for item in meta.get("references") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id"))
            if not key or key in seen_refs:
                continue
            merged["references"].append(item)
            seen_refs.add(key)

        seen_terms = {str(item.get("term") or "").lower() for item in merged["terms"]}
        for item in meta.get("terms") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("term") or "").strip().lower()
            if not key or key in seen_terms:
                continue
            merged["terms"].append(item)
            seen_terms.add(key)
        return {"references": merged["references"][:80], "terms": merged["terms"][:120]}

    def _merge_tool_meta(self, current: Dict[str, Any], content: Any) -> Dict[str, Any]:
        """
        输入:
        - `current`: 已累计的元数据
        - `content`: 工具返回内容

        输出:
        - 合并后的引用与词项元数据

        作用:
        - 从新闻类工具结果里抽取可点击引用和词项，供前端即时增强回答。
        """

        references = list(current.get("references") or [])
        terms = list(current.get("terms") or [])
        seen_refs = {str(item.get("id")) for item in references if item.get("id") is not None}
        seen_terms = {str(item.get("term") or "").lower() for item in terms}

        def add_news(item: Any) -> None:
            if not isinstance(item, dict):
                return
            news_id = item.get("id")
            if news_id is None or str(news_id) in seen_refs:
                return
            references.append(
                {
                    "id": news_id,
                    "title": item.get("title") or "",
                    "source": item.get("source") or "",
                    "time": item.get("time") or item.get("publish_date"),
                    "url": item.get("url") or "",
                    "summary": item.get("summary") or "",
                }
            )
            seen_refs.add(str(news_id))

        def add_term(value: Any, term_type: str) -> None:
            if not isinstance(value, str):
                return
            text = value.strip()
            if len(text) < 2:
                return
            key = text.lower()
            if key in seen_terms:
                return
            terms.append({"term": text, "type": term_type})
            seen_terms.add(key)

        if isinstance(content, dict):
            for item in content.get("items") or []:
                add_news(item)
                if isinstance(item, dict):
                    for kw in item.get("keywords") or []:
                        add_term(kw, "keyword")
                    for entity in item.get("entities") or []:
                        add_term(entity, "entity")

            news = content.get("news")
            if isinstance(news, dict):
                add_news(news)
                for kw in news.get("keywords") or []:
                    add_term(kw, "keyword")
                for entity in news.get("entities") or []:
                    add_term(entity, "entity")

            for item in content.get("related_news") or []:
                add_news(item)

        return {"references": references[:80], "terms": terms[:120]}

    def _build_meta_event(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        输入:
        - `meta`: 引用与词项元数据

        输出:
        - 前端可消费的元数据事件或 None

        作用:
        - 统一构造智能体回答增强信息，减少前端从工具 JSON 中重复解析。
        """

        references = meta.get("references") or []
        terms = meta.get("terms") or []
        if not references and not terms:
            return None
        return {
            "type": "agent_meta",
            "references": references,
            "terms": terms,
        }


agent_service = AgentService()
