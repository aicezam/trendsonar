"""
本文件用于封装与外部 AI 服务的交互，包括对话、摘要、分析与向量化等能力。
主要类/对象:
- `AIService`: AI 能力封装（主/备模型切换、并发控制、失败降级）
- `ai_service`: 全局服务单例
"""

import asyncio
import json
import re
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
from openai import AsyncOpenAI, APIStatusError, RateLimitError, APIConnectionError

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.core.exceptions import AIConfigurationError
from app.core.prompts import prompt_manager
from app.utils.tools import normalize_regions_to_countries

settings = get_settings()
logger = setup_logger("AIService")


class AIService:
    """
    输入:
    - AI 配置（主/备模型、并发限制、Embedding 配置）

    输出:
    - 大模型对话/摘要/分析结果，以及向量化结果

    作用:
    - 封装与外部 AI 服务的交互，提供统一、可降级的调用入口
    """

    def __init__(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 初始化主/备通道并发控制信号量
        """

        self.main_sem = asyncio.Semaphore(settings.MAIN_AI_CONCURRENCY)
        self.backup_sem = asyncio.Semaphore(settings.BACKUP_AI_CONCURRENCY)

    def reload_config(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 重新加载全局配置（用于配置更新后刷新本地引用）
        """
        global settings
        from app.core.config import get_settings
        settings = get_settings()
        
        # 重新初始化信号量（并发配置可能改变）
        self.main_sem = asyncio.Semaphore(settings.MAIN_AI_CONCURRENCY)
        self.backup_sem = asyncio.Semaphore(settings.BACKUP_AI_CONCURRENCY)
        logger.info("🔄 AIService 配置已刷新")

    def _has_main_llm(self) -> bool:
        return bool((settings.MAIN_AI_API_KEY or "").strip()) and bool((settings.MAIN_AI_BASE_URL or "").strip()) and bool((settings.MAIN_AI_MODEL or "").strip())

    def _has_backup_llm(self) -> bool:
        return bool((settings.BACKUP_AI_API_KEY or "").strip()) and bool((settings.BACKUP_AI_BASE_URL or "").strip()) and bool((settings.BACKUP_AI_MODEL or "").strip())

    def _has_embedding(self) -> bool:
        return bool((settings.SILICONFLOW_API_KEY or "").strip()) and bool((settings.SILICONFLOW_BASE_URL or "").strip()) and bool((settings.EMBEDDING_MODEL or "").strip())

    def _iter_llm_routes(self, prefer_backup: bool) -> List[Dict[str, str]]:
        routes: List[Dict[str, str]] = []
        if prefer_backup:
            if self._has_backup_llm():
                routes.append(
                    {
                        "base_url": str(settings.BACKUP_AI_BASE_URL),
                        "api_key": str(settings.BACKUP_AI_API_KEY),
                        "model": str(settings.BACKUP_AI_MODEL),
                        "type": "backup",
                    }
                )
            if self._has_main_llm():
                routes.append(
                    {
                        "base_url": str(settings.MAIN_AI_BASE_URL),
                        "api_key": str(settings.MAIN_AI_API_KEY),
                        "model": str(settings.MAIN_AI_MODEL),
                        "type": "main",
                    }
                )
            return routes

        if self._has_main_llm():
            routes.append(
                {
                    "base_url": str(settings.MAIN_AI_BASE_URL),
                    "api_key": str(settings.MAIN_AI_API_KEY),
                    "model": str(settings.MAIN_AI_MODEL),
                    "type": "main",
                }
            )
        if self._has_backup_llm():
            routes.append(
                {
                    "base_url": str(settings.BACKUP_AI_BASE_URL),
                    "api_key": str(settings.BACKUP_AI_API_KEY),
                    "model": str(settings.BACKUP_AI_MODEL),
                    "type": "backup",
                }
            )
        return routes

    async def _call_llm(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: str,
        system: str = "",
        semaphore: asyncio.Semaphore | None = None,
    ) -> Optional[str]:
        """
        输入:
        - `client`: OpenAI 兼容客户端
        - `model`: 模型名称
        - `prompt`: 用户提示词
        - `system`: 系统提示词
        - `semaphore`: 并发控制（可选）

        输出:
        - 模型返回文本；失败返回 None

        作用:
        - 统一封装 LLM 调用、并发控制与异常处理
        """

        try:
            extra_body = {}
            if "modelscope" in str(client.base_url):
                extra_body["enable_thinking"] = False

            if semaphore is None:
                if str(settings.MAIN_AI_BASE_URL) in str(client.base_url):
                    semaphore = self.main_sem
                elif str(settings.BACKUP_AI_BASE_URL) in str(client.base_url):
                    semaphore = self.backup_sem

            # 调试日志：记录提示词
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔵 [LLM 请求] 模型: {model}\n系统提示词: {system}\n用户提示词: {prompt[:2000]}...")

            async def do_call():
                return await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.6,
                    timeout=120,
                    extra_body=extra_body if extra_body else None,
                )

            # 重试逻辑
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    if semaphore:
                        async with semaphore:
                            response = await do_call()
                    else:
                        response = await do_call()
                    
                    content = response.choices[0].message.content
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"🟢 [LLM 响应] 模型: {model}\n内容: {content[:2000]}...")
                    
                    if not content:
                         logger.warning(f"⚠️ AI 返回内容为空 ({model})")
                         if attempt < max_retries - 1:
                             logger.info(f"   🔄 空内容重试 ({attempt + 1}/{max_retries})...")
                             await asyncio.sleep(1)
                             continue
                         return None

                    return content
                
                except (RateLimitError, APIConnectionError) as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = 2 * (attempt + 1)
                    logger.warning(f"⚠️ AI 调用受限或网络波动 ({model}): {e}，将在 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                
                except APIStatusError as e:
                    # 401: API Key 无效 - 致命错误
                    if e.status_code == 401:
                        logger.error(f"❌ AI 认证失败 (401) - API Key 无效 ({model}): {e}")
                        raise AIConfigurationError(f"AI API Key 无效 ({model})")

                    # 400 Bad Request 通常意味着内容过滤或参数无效
                    if e.status_code == 400:
                        logger.warning(f"❌ AI 请求被拒绝 (400) - 可能触发敏感词过滤 ({model}): {e}")
                        # 触发外部的切换逻辑，如果是路由模式，会捕获 None 然后切换
                        return None 
                    
                    # 服务端错误，重试可能有效
                    if e.status_code >= 500:
                        if attempt == max_retries - 1:
                            raise e
                        wait_time = 2 * (attempt + 1)
                        logger.warning(f"⚠️ AI 服务端错误 ({model}): {e}，将在 {wait_time} 秒后重试")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e

        except AIConfigurationError:
            raise
        except Exception as e:
            logger.error(f"❌ AI 调用异常 ({model}): {e}")
            return None

    async def chat_completion(self, prompt: str, system_prompt: str = "", route_key: str = None) -> str:
        """
        输入:
        - `prompt`: 用户提示词
        - `system_prompt`: 系统提示词（可选）
        - `route_key`: 配置路由键（可选，如 "REPORT", "SUMMARY" 等）

        输出:
        - 模型回复文本（保证返回字符串）

        作用:
        - 执行一次对话补全；根据 route_key 选择主/备通道策略
        """
        prefer_backup = False
        if route_key:
            prefer_backup = self._get_prefer_backup(route_key)
        
        res = await self._call_llm_with_routes(prompt, system_prompt, prefer_backup=prefer_backup)
        if res:
            return res

        return "AI 服务暂时不可用（请先在管理页完善 AI 配置）"

    async def stream_completion(self, prompt: str, system_prompt: str = "", route_key: Optional[str] = None) -> AsyncIterator[str]:
        prefer_backup = False
        if route_key:
            prefer_backup = self._get_prefer_backup(route_key)

        routes = self._iter_llm_routes(prefer_backup)
        if not routes:
            yield "AI 服务暂时不可用（请先在管理页完善 AI 配置）"
            return
            # raise AIConfigurationError("AI 服务暂时不可用（请先在管理页完善 AI 配置）")
        last_error: Optional[Exception] = None

        for idx, route in enumerate(routes):
            try:
                async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                    extra_body = {}
                    if "modelscope" in str(client.base_url):
                        extra_body["enable_thinking"] = False

                    stream = await client.chat.completions.create(
                        model=route["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        stream=True,
                        temperature=0.6,
                        timeout=120,
                        extra_body=extra_body if extra_body else None,
                    )

                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                    return
            except Exception as e:
                last_error = e
                logger.error(f"流式补全路由失败: {e}")
                if idx < len(routes) - 1:
                    next_route = routes[idx + 1]
                    logger.warning(
                        f"⚠️ 流式路由 {route['model']} ({route['type']}) 失败，切换到 -> {next_route['model']} ({next_route['type']})"
                    )

        if last_error:
            logger.error(f"所有流式路由均失败: {last_error}")

    async def batch_evaluate_topic_quality(self, topics: List[Dict[str, str]], existing_topics: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        输入:
        - topics: 专题列表 [{"name":..., "description":...}]
        - existing_topics: 现有专题列表 [{"name":..., "description":...}] （可选）

        输出:
        - 经过筛选的有效专题列表

        作用:
        - 批量评估专题质量，过滤掉过于宽泛、非具体事件、或纯粹行业趋势的专题
        - 检查是否与现有专题重复或属于现有专题的延伸
        """
        if not topics:
            return []

        logger.info(f"🤖 批量评估专题质量: {len(topics)} 个")
        
        existing_info = ""
        if existing_topics:
            # 限制现有专题数量以防止 Prompt 过长，取最近 50 个即可（假设按时间倒序或相关性排序）
            # 这里调用者传入的通常是 active 专题，数量可能较多，截取一下比较安全
            limit_existing = existing_topics[:50]
            existing_info = "【已存在的专题列表（用于查重和判断延伸关系）】:\n"
            for t in limit_existing:
                existing_info += f"- {t.get('name')}: {t.get('description')}\n"
            existing_info += "\n"

        # 获取质量等级，默认为 3
        quality_level = getattr(settings, "TOPIC_QUALITY_LEVEL", 3)
        logger.info(f"🔍 专题审核质量等级: {quality_level}")

        # 默认使用等级 3
        criteria_text = prompt_manager.get_user_prompt(f"topic_quality_criteria_{quality_level}")
        if not criteria_text:
             criteria_text = prompt_manager.get_user_prompt("topic_quality_criteria_3")

        # 构建禁止项
        forbidden_common = prompt_manager.get_user_prompt("topic_quality_forbidden_common")
        forbidden_strict = prompt_manager.get_user_prompt("topic_quality_forbidden_strict")
        forbidden_loose = prompt_manager.get_user_prompt("topic_quality_forbidden_loose")

        if quality_level >= 3:
            forbidden_text = forbidden_strict + "\n" + forbidden_common
        else:
            forbidden_text = forbidden_loose + "\n" + forbidden_common

        system_prompt = prompt_manager.get_system_prompt(
            "topic_quality_eval_base",
            quality_level=quality_level,
            criteria_text=criteria_text,
            forbidden_text=forbidden_text
        )
        
        topics_text = ""
        for i, t in enumerate(topics):
            topics_text += f"[{i}] 名称：{t.get('name')}\n    描述：{t.get('description')}\n\n"

        user_prompt = prompt_manager.get_user_prompt(
            "topic_quality_eval_base",
            existing_info=existing_info,
            topics_text=topics_text
        )
            
        prefer_backup = self._get_prefer_backup("TOPIC_EVAL")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        
        if not res:
            logger.warning("⚠️ 专题质量评估失败，默认全部保留")
            return topics
            
        try:
            clean = res.replace("```json", "").replace("```", "").strip()
            # 尝试提取 JSON 数组
            start = clean.find("[")
            end = clean.rfind("]")
            if start != -1 and end != -1:
                clean = clean[start : end + 1]
                
            results = json.loads(clean)
            
            valid_topics = []
            if isinstance(results, list):
                for item in results:
                    idx = item.get("index")
                    is_valid = item.get("valid")
                    reason = item.get("reason", "无理由")
                    
                    if isinstance(idx, int) and 0 <= idx < len(topics):
                        topic_name = topics[idx].get("name")
                        if is_valid:
                            logger.info(f"   ✅ [通过] {topic_name}: {reason}")
                            valid_topics.append(topics[idx])
                        else:
                            logger.info(f"   ❌ [拒绝] {topic_name}: {reason}")
            else:
                # 如果结构错误则降级处理
                logger.warning("   ⚠️ 质量评估返回格式异常，解析失败，保留所有")
                return topics

            if len(valid_topics) < len(topics):
                removed_count = len(topics) - len(valid_topics)
                logger.info(f"🗑️ 过滤掉了 {removed_count} 个宽泛/低质量专题")
                
            return valid_topics
            
        except AIConfigurationError:
            raise
        except Exception as e:
            logger.error(f"❌ 解析质量评估结果失败: {e}\nRaw: {res}")
            return topics

    async def propose_topics_from_titles(self, titles: List[str]) -> List[Dict[str, str]]:
        """
        输入:
        - `titles`: 新闻标题列表
        
        输出:
        - 提炼出的专题列表 [{"name": "...", "description": "..."}, ...]
        
        作用:
        - 从大量标题中聚合出核心专题
        """
        if not titles:
            return []
            
        system_prompt = prompt_manager.get_system_prompt("topic_propose")
        
        # 限制数量以防超长
        limit_n = settings.TOPIC_AGGREGATION_TOP_N
        titles_subset = titles[:limit_n]
        titles_str = "\n".join([f"- {t}" for t in titles_subset])
        
        # 动态获取专题数量范围
        count_range = settings.TOPIC_GENERATION_COUNT or "1-5"
        min_count, max_count = 1, 5
        try:
            parts = count_range.split("-")
            if len(parts) == 2:
                min_count = int(parts[0].strip())
                max_count = int(parts[1].strip())
        except Exception:
            pass

        # 获取质量等级，默认为 3
        quality_level = getattr(settings, "TOPIC_QUALITY_LEVEL", 3)
        logger.info(f"🔍 专题生成质量等级: {quality_level}")

        criteria_desc = prompt_manager.get_user_prompt(f"topic_propose_criteria_{quality_level}")
        if not criteria_desc:
            criteria_desc = prompt_manager.get_user_prompt("topic_propose_criteria_3")

        prompt = prompt_manager.get_user_prompt(
            "topic_propose_user",
            min_count=min_count,
            max_count=max_count,
            criteria_desc=criteria_desc,
            titles_str=titles_str
        )
        logger.info(f"🤖 正在从 {len(titles_subset)} 条标题中提炼专题...")
        prefer_backup = self._get_prefer_backup("TOPIC_NAME")
        res = await self._call_llm_with_routes(prompt, system_prompt, prefer_backup=prefer_backup)
        
        try:
            if not res:
                return []
            cleaned = res.replace("```json", "").replace("```", "").strip()
            
            # 尝试修复常见的 JSON 格式错误（如尾部逗号、未转义引号等）
            # 这里先简单尝试直接解析
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # 假如解析失败，尝试用正则提取
                import re
                pattern = r'\{\s*"name"\s*:\s*"(.*?)"\s*,\s*"description"\s*:\s*"(.*?)"\s*\}'
                matches = re.findall(pattern, cleaned, re.DOTALL)
                if matches:
                    data = [{"name": m[0], "description": m[1]} for m in matches]
                else:
                    # 再次尝试，可能是单引号或其他格式
                    raise 

            if isinstance(data, list):
                valid_data = []
                for item in data:
                    if isinstance(item, dict) and "name" in item and "description" in item:
                        valid_data.append(item)
                
                # 再次强制截断，防止 AI 返回过多
                if len(valid_data) > max_count:
                    logger.warning(f"⚠️ AI 返回专题数量 ({len(valid_data)}) 超过限制 ({max_count})，已强制截断")
                    valid_data = valid_data[:max_count]
                
                logger.info(f"✅ AI 提炼出 {len(valid_data)} 个潜在专题")
                return valid_data
            return []
        except AIConfigurationError:
            raise
        except Exception as e:
            logger.error(f"❌ 解析专题提炼结果失败: {e}\nRaw: {res}")
            return []

    async def extract_news_info(self, content: str) -> List[Dict[str, Any]]:
        """
        输入:
        - `content`: 原始页面内容（HTML/XML/文本）

        输出:
        - 新闻条目列表（title/link/summary）

        作用:
        - 当常规 RSS/API 解析失败时，使用大模型从内容中抽取新闻条目
        """

        system_prompt = prompt_manager.get_system_prompt("news_extract_info")
        user_prompt = prompt_manager.get_user_prompt("news_extract_info", content=content[:20000])

        async def try_extract(client, model):
            res = await self._call_llm(client, model, user_prompt, system_prompt)
            if not res:
                return None
            try:
                clean_res = res.strip()
                if "```" in clean_res:
                    start = clean_res.find("{")
                    end = clean_res.rfind("}")
                    if start != -1 and end != -1:
                        clean_res = clean_res[start : end + 1]
                data = json.loads(clean_res)
                return data.get("items", [])
            except AIConfigurationError:
                raise
            except Exception as e:
                logger.warning(f"AI提取结果解析失败: {e}")
                return None

        if not (self._has_main_llm() or self._has_backup_llm()):
            return []

        routes = self._iter_llm_routes(self._get_prefer_backup("SUMMARY"))
        for r_idx, route in enumerate(routes):
            max_attempts = 3 if r_idx == 0 else 1
            for attempt in range(max_attempts):
                # 使用 async with 确保资源释放
                async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                    res = await try_extract(client, route["model"])
                if res is not None:
                    return res
                await asyncio.sleep(1)

        return []

    def _normalize_category(self, raw_category: str) -> str:
        """
        输入:
        - `raw_category`: 模型输出的原始分类

        输出:
        - 规范化后的分类名称（落到 `settings.NEWS_CATEGORIES` 之一）

        作用:
        - 将模型可能出现的近似分类映射为系统内置分类，减少脏数据
        """

        if not raw_category:
            return "其他"
        if raw_category in settings.NEWS_CATEGORIES:
            return raw_category
        for cat in settings.NEWS_CATEGORIES:
            if raw_category in cat or cat in raw_category:
                return cat
        return "其他"

    async def analyze_sentiment(self, title: str, content: str = "") -> Dict:
        """
        输入:
        - `title`: 新闻标题
        - `content`: 新闻摘要或正文（可选）

        输出:
        - 情感分析结果（score/label/category/region/keywords/entities）

        作用:
        - 对单条新闻进行深度舆情分析，主通道失败时降级到备用通道
        """

        categories_str = "、".join(settings.NEWS_CATEGORIES)
        system_prompt = prompt_manager.get_system_prompt("sentiment_analysis_single", categories_str=categories_str)
        user_prompt = prompt_manager.get_user_prompt("sentiment_analysis_single", title=title, content=content[:1000])

        async def try_analyze(client, model):
            res = await self._call_llm(client, model, user_prompt, system_prompt)
            if not res:
                return None
            try:
                clean_res = res.strip()
                if "```" in clean_res:
                    start = clean_res.find("{")
                    end = clean_res.rfind("}")
                    if start != -1 and end != -1:
                        clean_res = clean_res[start : end + 1]
                data = json.loads(clean_res)
                if "score" in data and "label" in data:
                    data["category"] = self._normalize_category(data.get("category", ""))
                    data["region"] = normalize_regions_to_countries(data.get("region"))
                    if not data.get("region") or data.get("region") in ["其他", "未知"]:
                        data["region"] = "全球"
                    
                    # 补全可能缺失的字段，防止 KeyError
                    if "keywords" not in data:
                        data["keywords"] = []
                    if "entities" not in data:
                        data["entities"] = []
                        
                    return data
            except AIConfigurationError:
                raise
            except Exception:
                pass
            return None

        routes = self._iter_llm_routes(self._get_prefer_backup("SENTIMENT"))
        for route in routes:
            # 使用 async with 确保资源释放
            async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                res = await try_analyze(client, route["model"])
            if res:
                return res

        return {
            "score": 50,
            "label": "中立",
            "category": "其他",
            "region": "其他",
            "keywords": [],
            "entities": [],
        }

    async def batch_analyze_sentiment(self, news_items: List[Dict]) -> Dict[int, Dict]:
        """
        输入:
        - `news_items`: 待分析新闻列表（至少包含 id/title）

        输出:
        - `id -> 分析结果` 的映射

        作用:
        - 对多条新闻进行批量快速分析，用于提升吞吐与降低成本
        """

        if not news_items:
            return {}

        categories_str = "、".join(settings.NEWS_CATEGORIES)
        system_prompt = prompt_manager.get_system_prompt("sentiment_analysis_batch", categories_str=categories_str)

        items_text = ""
        for item in news_items:
            items_text += f"[ID:{item['id']}] {item['title']}\n"
        user_prompt = prompt_manager.get_user_prompt("sentiment_analysis_batch", items_text=items_text)

        try:
            routes = self._iter_llm_routes(self._get_prefer_backup("SENTIMENT"))
            res: Optional[str] = None
            for route in routes:
                # 使用 async with 确保资源释放
                async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                    res = await self._call_llm(client, route["model"], user_prompt, system_prompt)
                if res:
                    break

            if not res:
                return {}

            clean_res = res.strip()
            # 无论是否包含 markdown 标记，都优先尝试提取 JSON 数组
            start = clean_res.find("[")
            end = clean_res.rfind("]")
            if start != -1 and end != -1:
                clean_res = clean_res[start : end + 1]
            else:
                # 兜底清理
                clean_res = clean_res.replace("```json", "").replace("```", "").strip()

            results_list = json.loads(clean_res)

            result_map: Dict[int, Dict] = {}
            if isinstance(results_list, list):
                for item in results_list:
                    if "id" in item:
                        if "category" in item:
                            item["category"] = self._normalize_category(item["category"])
                        item["region"] = normalize_regions_to_countries(item.get("region"))
                        result_map[item["id"]] = item
            return result_map

        except AIConfigurationError:
            raise
        except Exception as e:
            logger.error(f"批量情感分析失败: {e}")
            return {}

    def _get_prefer_backup(self, route_key: str) -> bool:
        """根据配置键获取是否优先使用备用 AI"""
        route_value = settings.AI_ROUTE.get(route_key, "main").lower()
        return route_value == "backup"

    async def generate_summary(self, title: str, content: str, max_words: Optional[int] = None) -> Optional[str]:
        """
        输入:
        - `title`: 新闻标题
        - `content`: 新闻正文
        - `max_words`: 最大字数限制 (可选，默认使用配置)

        输出:
        - 摘要文本 (如果失败返回 None)

        作用:
        - 使用 LLM 生成高质量新闻摘要，支持主备切换
        """
        if not content:
            return None
            
        # 使用配置的默认长度
        if max_words is None:
            max_words = getattr(settings, "SUMMARY_OUTPUT_LENGTH", 300)
            
        # 截取输入内容
        input_limit = getattr(settings, "SUMMARY_INPUT_MAX_LENGTH", 5000)
        if input_limit > 0 and len(content) > input_limit:
            content = content[:input_limit] + "\n...(已截断)"

        system_prompt = prompt_manager.get_system_prompt("summary_generation", max_words=max_words)
        user_prompt = prompt_manager.get_user_prompt("summary_generation", title=title, content=content)

        prefer_backup = self._get_prefer_backup("SUMMARY")
        routes = self._iter_llm_routes(prefer_backup)
        for route in routes:
            # 使用 async with 确保资源释放
            async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                res = await self._call_llm(
                    client,
                    route["model"],
                    user_prompt,
                    system_prompt,
                )
            if res:
                return res
        return None

    async def _call_llm_with_routes(
        self,
        user_prompt: str,
        system_prompt: str,
        prefer_backup: Optional[bool] = None,
    ) -> Optional[str]:
        prefer = False if prefer_backup is None else prefer_backup
        routes = self._iter_llm_routes(prefer)
        for i, route in enumerate(routes):
            # 使用 async with 确保 client 资源释放
            async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                res = await self._call_llm(
                    client,
                    route["model"],
                    user_prompt,
                    system_prompt,
                )
            
            if res:
                return res
            
            # 如果运行到这里，说明当前路由失败或返回空内容
            if i < len(routes) - 1:
                next_route = routes[i+1]
                logger.warning(f"⚠️ 路由 {route['model']} ({route['type']}) 调用失败或返回空，尝试切换到 -> {next_route['model']} ({next_route['type']})")
            else:
                logger.error(f"❌ 所有可用 AI 路由均调用失败")
        return None



    async def verify_topic_match_batch(self, tasks: List[Dict[str, str]]) -> List[Tuple[bool, str]]:
        """
        输入:
        - tasks: 任务列表，包含 {"topic_name": ..., "topic_summary": ..., "news_title": ..., "news_summary": ...}

        输出:
        - 结果列表，包含 (是否匹配, 理由)
        """
        if not tasks:
            return []

        # 记录请求内容
        logger.info(f"🤖 批量核验专题匹配: {len(tasks)} 组")
        for i, t in enumerate(tasks[:3]):  # 记录前 3 条用于预览
            logger.info(f"   [{i}] 专题: {t['topic_name']} <-> 新闻: {t['news_title']}")

        system_prompt = prompt_manager.get_system_prompt("topic_match_batch")

        tasks_text = ""
        for idx, task in enumerate(tasks):
            tasks_text += (
                f"--- 第 {idx+1} 组 ---\n"
                f"【专题】{task['topic_name']}\n"
                f"【专题概览】{(task['topic_summary'] or '')[:300]}\n"
                f"【新闻标题】{task['news_title']}\n"
                f"【新闻摘要】{(task['news_summary'] or '')[:300]}\n"
            )
            
        user_prompt = prompt_manager.get_user_prompt("topic_match_batch", tasks_text=tasks_text)

        prefer_backup = self._get_prefer_backup("TOPIC_MATCH")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        if not res:
            return [(False, "AI调用失败")] * len(tasks)

        try:
            clean = res.strip()
            if "```" in clean:
                start = clean.find("[")
                end = clean.rfind("]")
                if start != -1 and end != -1:
                    clean = clean[start : end + 1]
            results = json.loads(clean)
            
            output = []
            if isinstance(results, list):
                for item in results:
                    is_match = bool(item.get("match", False))
                    reason = item.get("reason", "无理由")
                    output.append((is_match, reason))
                
                # 确保长度匹配
                if len(output) < len(tasks):
                    output.extend([(False, "返回数量不足")] * (len(tasks) - len(output)))
                
                return output[:len(tasks)]
        except AIConfigurationError:
            raise
        except Exception as e:
            logger.error(f"批量专题核验解析失败: {e}")

        return [(False, "解析异常")] * len(tasks)




    async def generate_daily_timeline_events(self, date_str: str, news_items: List[Dict[str, Any]], topic_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        输入:
        - date_str: "YYYY-MM-DD"
        - news_items: List of {"id": ..., "title": ..., "summary": ...}
        - topic_name: Optional topic name to focus on

        输出:
        - List of events: [{"content": "...", "source_ids": [id1, id2...]}, ...]
        """
        if not news_items:
            return []

        logger.info(f"🤖 正在为 {date_str} 合成时间轴事件 (共 {len(news_items)} 条新闻)...")
        
        focus_instruction = ""
        if topic_name:
            focus_instruction = prompt_manager.get_user_prompt("topic_timeline_focus", topic_name=topic_name)

        system_prompt = prompt_manager.get_system_prompt("topic_timeline_generation", focus_instruction=focus_instruction)

        topic_focus = ""
        if topic_name:
            topic_focus = f"专题名称：{topic_name}\n"
        
        news_list = ""
        for item in news_items:
            news_list += f"[ID: {item['id']}] {item['title']}\n摘要: {(item['summary'] or '')[:100]}\n\n"
            
        user_prompt = prompt_manager.get_user_prompt("topic_timeline_generation", date_str=date_str, topic_focus=topic_focus, news_list=news_list)

        prefer_backup = self._get_prefer_backup("TOPIC_TIMELINE")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        
        if not res:
            return []
            
        try:
            clean = res.strip()
            if "```" in clean:
                start = clean.find("[")
                end = clean.rfind("]")
                if start != -1 and end != -1:
                    clean = clean[start : end + 1]
            events = json.loads(clean)
            return events
        except AIConfigurationError:
            raise
        except Exception as e:
            logger.error(f"❌ 解析时间轴合成结果失败: {e}\nRaw: {res}")
            return []

    async def check_topic_duplicate(
        self,
        new_name: str,
        new_desc: str,
        existing_name: str,
        existing_desc: str
    ) -> Tuple[bool, str]:
        """
        判断两个专题是否实质上是同一个事件。
        """
        system_prompt = prompt_manager.get_system_prompt("topic_duplicate_check")
        user_prompt = prompt_manager.get_user_prompt(
            "topic_duplicate_check",
            new_name=new_name,
            new_desc=new_desc[:500],
            existing_name=existing_name,
            existing_desc=existing_desc[:500]
        )
        
        prefer_backup = self._get_prefer_backup("TOPIC_MATCH")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        if not res:
            return False, "AI调用失败"
            
        clean = res.strip()
        try:
            if "```" in clean:
                start = clean.find("{")
                end = clean.rfind("}")
                if start != -1 and end != -1:
                    clean = clean[start : end + 1]
            data = json.loads(clean)
            return bool(data.get("duplicate", False)), data.get("reason", "无理由")
        except Exception:
            lowered = clean.lower()
            if "true" in lowered and "false" not in lowered:
                return True, "解析失败(fallback: true)"
            return False, "解析失败"

    async def stream_chat(self, query: str, context: str, model_type: str = "main") -> AsyncIterator[str]:
        """
        输入:
        - `query`: 用户问题
        - `context`: 相关新闻上下文
        - `model_type`: 使用主/备通道（main/backup）

        输出:
        - SSE 可迭代的增量文本片段

        作用:
        - 以流式方式与模型对话，适配前端实时展示
        """
        # 如果请求使用 main 通道，则进一步检查配置路由是否指定了优先备用
        if model_type == "main":
            if self._get_prefer_backup("CHAT"):
                model_type = "backup"

        api_key = settings.MAIN_AI_API_KEY
        base_url = settings.MAIN_AI_BASE_URL
        model = settings.MAIN_AI_MODEL

        if model_type == "backup":
            api_key = settings.BACKUP_AI_API_KEY
            base_url = settings.BACKUP_AI_BASE_URL
            model = settings.BACKUP_AI_MODEL

        system_prompt = prompt_manager.get_system_prompt("chat_stream")
        user_prompt = prompt_manager.get_user_prompt("chat_stream", context=context, query=query)

        try:
            # 使用 async with 确保资源释放
            async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
                extra_body = {}
                if "modelscope" in str(client.base_url):
                    extra_body["enable_thinking"] = False

                logger.info(f"开始流式对话请求: model={model}, stream=True")
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                    temperature=0.7,
                    timeout=60,
                    extra_body=extra_body if extra_body else None,
                )
                logger.info("流式请求已建立，开始读取 chunks")
                chunk_count = 0
                async for chunk in stream:
                    logger.debug(f"收到原始块: {chunk}")
                    if not chunk.choices:
                        logger.debug(f"块中无选项: {chunk}")
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        chunk_count += 1
                        # logger.debug(f"生成内容: {content!r}")
                        yield content
                    else:
                        logger.debug(f"块内容为空: {chunk}")
                logger.info(f"流式传输结束, 共发送 {chunk_count} 个 chunks")
        except Exception as e:
            logger.error(f"聊天流错误: {e}", exc_info=True)
            yield f"错误: {str(e)}"

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        输入:
        - `texts`: 待向量化文本列表

        输出:
        - 向量列表（与输入一一对应；失败时返回空向量）

        作用:
        - 调用 embedding 服务生成向量，用于语义检索与聚类
        """

        if not texts:
            return []
        if not self._has_embedding():
            return [[] for _ in texts]
        cleaned_texts = [t.replace("\n", " ").strip()[:1000] for t in texts]

        url = f"{settings.SILICONFLOW_BASE_URL.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        }

        all_embeddings: List[List[float]] = []
        batch_size = 20

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            payload = {
                "model": settings.EMBEDDING_MODEL,
                "input": batch,
                "encoding_format": "float",
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            batch_res = sorted(data["data"], key=lambda x: x["index"])
                            all_embeddings.extend([x["embedding"] for x in batch_res])
                        elif resp.status == 401:
                             error_text = await resp.text()
                             logger.error(f"❌ 向量 API 认证失败 (401): {error_text}")
                             raise AIConfigurationError("Embedding API Key 无效")
                        else:
                            logger.error(f"❌ 向量 API 错误: {await resp.text()}")
                            all_embeddings.extend([[] for _ in batch])
            except AIConfigurationError:
                raise
            except Exception as e:
                logger.error(f"❌ 向量网络错误: {e}")
                all_embeddings.extend([[] for _ in batch])
        return all_embeddings

    async def verify_cluster_batch(self, pairs: List[Dict[str, str]]) -> List[bool]:
        """
        输入:
        - `pairs`: 待核验新闻对列表（leader/candidate 标题）

        输出:
        - 布尔列表（与输入顺序一致，表示是否同一事件）

        作用:
        - 使用大模型对相似候选进行批量核验，减少误合并
        """

        if not pairs:
            return []
        if not (self._has_main_llm() or self._has_backup_llm()):
            return [False] * len(pairs)

        system_prompt = prompt_manager.get_system_prompt("cluster_verification_batch")

        pairs_text = ""
        for i, p in enumerate(pairs):
            pairs_text += f"{i + 1}. [{p['leader']}] vs [{p['candidate']}]\n"
            
        user_content = prompt_manager.get_user_prompt("cluster_verification_batch", pairs_text=pairs_text)

        async def try_verify(client, model):
            try:
                res = await self._call_llm(client, model, user_content, system_prompt)
                if not res:
                    return None

                clean_res = res.strip()
                if clean_res.startswith("```"):
                    start = clean_res.find("[")
                    end = clean_res.rfind("]")
                    if start != -1 and end != -1:
                        clean_res = clean_res[start : end + 1]
                else:
                    start = clean_res.find("[")
                    end = clean_res.rfind("]")
                    if start != -1 and end != -1:
                        clean_res = clean_res[start : end + 1]

                # 尝试修复 Python 风格的布尔值
                clean_res = clean_res.replace("True", "true").replace("False", "false")

                try:
                    results = json.loads(clean_res)
                except json.JSONDecodeError:
                    # 尝试使用正则提取布尔值
                    import re
                    bool_matches = re.findall(r'\b(true|false)\b', clean_res, re.IGNORECASE)
                    if len(bool_matches) == len(pairs):
                        results = [b.lower() == 'true' for b in bool_matches]
                    else:
                        raise

                if isinstance(results, list) and len(results) == len(pairs):
                    return [bool(x) for x in results]

                logger.error(f"❌ 批量核验返回格式错误: {results} (预期长度: {len(pairs)})")
                return None
            except AIConfigurationError:
                raise
            except Exception as e:
                logger.error(f"❌ 批量核验异常 ({model}): {e}")
                return None

        routes = self._iter_llm_routes(self._get_prefer_backup("CLUSTERING"))
        # 将 routes 转换为列表以便索引
        route_list = list(routes)
        
        for i, route in enumerate(route_list):
            # 如果是 backup 通道，尝试 3 次；如果是 main 通道，尝试 1 次
            is_backup = (route["base_url"] == settings.BACKUP_AI_BASE_URL)
            max_attempts = 3 if is_backup else 1
            
            for attempt in range(max_attempts):
                if attempt > 0:
                    await asyncio.sleep(2 if attempt == 1 else 10)
                
                # 使用 async with 确保资源释放
                async with AsyncOpenAI(api_key=route["api_key"], base_url=route["base_url"]) as client:
                    try:
                        res = await try_verify(client, route["model"])
                    except AIConfigurationError:
                        raise
                
                if res is not None:
                    return res
                
                # 如果运行到这里，说明尝试失败（返回了 None）
                if is_backup and attempt < max_attempts - 1:
                    logger.warning(f"⚠️ 备用AI核验失败 (第{attempt + 1}次)，准备重试...")

            # 当前路由的所有尝试都失败了
            if i < len(route_list) - 1:
                next_route = route_list[i+1]
                logger.warning(f"⚠️ 路由 {route['model']} (CLUSTERING) 调用失败，尝试切换到 -> {next_route['model']}")

        logger.error("❌ 所有通道核验均失败，跳过本批次")
        return [False] * len(pairs)


    async def generate_topic_initial_summary(self, topic_name: str, news_list: List[Dict[str, str]]) -> str:
        """
        输入:
        - `topic_name`: 专题名称
        - `news_list`: 新闻列表 [{"title": "...", "content": "..."}, ...]

        输出:
        - 专题初始事实摘要（纯文本）

        作用:
        - 生成基于事实的快速摘要，用于专题审核与去重
        """
        if not news_list:
            return ""

        logger.info(f"🤖 生成专题初始摘要: {topic_name} ({len(news_list)} 条新闻)")
        
        # 限制输入长度，避免Token溢出
        input_text = ""
        # 初始摘要不需要太多新闻，取 Top 10-15 即可
        for i, item in enumerate(news_list[:15]): 
            t = (item.get("title") or "").replace("\n", " ")
            c = (item.get("summary") or item.get("content") or "")[:200].replace("\n", " ")
            input_text += f"[{i+1}] {t}\n   内容: {c}\n\n"
        
        system_prompt = prompt_manager.get_system_prompt("topic_initial_summary")
        user_prompt = prompt_manager.get_user_prompt("topic_initial_summary", topic_name=topic_name, input_text=input_text)
        
        # 使用单独配置的路由，默认 main
        prefer_backup = self._get_prefer_backup("TOPIC_INITIAL_SUMMARY")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        if res:
            return res.strip()
        return ""

    async def generate_topic_overview(self, topic_name: str, news_list: List[Dict[str, str]]) -> str:
        """
        输入:
        - `topic_name`: 专题名称
        - `news_list`: 新闻列表 [{"title": "...", "content": "..."}, ...]

        输出:
        - 专题多维度综述（纯文本）
        
        作用:
        - 从“事件背景”、“发展过程”、“各方观点”、“未来展望”等维度对专题进行深度总结
        """
        if not news_list:
            return ""
            
        logger.info(f"🤖 生成专题综述: {topic_name} ({len(news_list)} 条新闻)")
        
        # 限制输入长度，避免Token溢出
        # 优先取最新的新闻，和最早的新闻，以覆盖全貌
        # 假设 news_list 已经包含了一定数量的代表性新闻
        input_text = ""
        for i, item in enumerate(news_list[:30]): # 最多取30条作为上下文
            t = (item.get("title") or "").replace("\n", " ")
            c = (item.get("summary") or item.get("content") or "")[:200].replace("\n", " ")
            input_text += f"[{i+1}] {t}\n   摘要: {c}\n\n"
            
        system_prompt = prompt_manager.get_system_prompt("topic_overview")
        
        user_prompt = prompt_manager.get_user_prompt("topic_overview", topic_name=topic_name, input_text=input_text)
        
        prefer_backup = self._get_prefer_backup("TOPIC_OVERVIEW")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        if res:
            return res.strip()
        return ""


ai_service = AIService() 
