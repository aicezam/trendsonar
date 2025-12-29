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
            max_retries = 3
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
                        return None # 故障转移到下一个路由
                    
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

        criteria_map = {
            1: (
                "1. 【基础事件标准】（满足即可）：\n"
                "   - **真实性**：必须是真实发生的新闻事件，而非虚构或纯粹的观点。\n"
                "   - **具体性**：有明确的主体和动作。\n"
                "   - **排除项**：排除极其琐碎的个人日常（如“某人吃了饭”）。允许地方性小事件。\n"
            ),
            2: (
                "1. 【一般事件标准】（满足即可）：\n"
                "   - **公共价值**：具有显著的公共信息价值，非纯粹的私人琐事。\n"
                "   - **具体性**：事件描述清晰，非模糊的行业概念。\n"
                "   - **讨论度**：引起了一定范围内的讨论或关注，非无人问津的信息。\n"
                "   - **排除项**：严格排除商业广告、极小范围的无关紧要事故、常规的日常例行报道。\n"
            ),
            3: (
                "1. 【重大事件评估标准】（必须全部满足）：\n"
                "   - **影响范围**：具有省级、全国级别的影响力，或在所属行业内具有重大影响。排除仅影响个别公司或小区的事件。\n"
                "   - **持续性**：具有持续发酵的潜力，不仅仅是昙花一现的瞬间消息。\n"
                "   - **社会热度**：属于当前社会热点，具有广泛的公众讨论度。\n"
                "   - **特殊放行**：对于涉及邻国冲突、外交纷争等事件（如“泰柬冲突”、“朝韩局势”），属于**国际政治**范畴，无论热度如何，都应视为具有国际影响力的潜力事件予以通过。\n"
            ),
            4: (
                "1. 【高影响力事件标准】（必须全部满足）：\n"
                "   - **全国/国际影响**：必须是全国性或国际性的重大新闻。地方性新闻（除非引发全国关注）一律不予通过。\n"
                "   - **高热度**：必须是当前公众讨论的焦点。\n"
                "   - **深度**：事件具有深远的社会或行业影响，非简单的日常通报。\n"
            ),
            5: (
                "1. 【顶级重大事件标准】（必须全部满足）：\n"
                "   - **历史级/战略级**：涉及国家政策重大调整、国际关系重大变化、或改变行业格局的历史性事件。\n"
                "   - **极高热度**：全网刷屏级的超级热点。\n"
                "   - **极严筛选**：宁缺毋滥，非此类特大事件一律不予通过。\n"
            )
        }
        
        # 默认使用等级 3
        criteria_text = criteria_map.get(quality_level, criteria_map[3])

        # 构建禁止项
        forbidden_common = (
            "   - 不能由多个非强关联主体组成的两个事件合并后的短语（如“日方官员拥核言论及靖国神社争议”）。\n"
            "3. 【查重与延伸】：\n"
            "   - 如果待创建专题是现有专题的子集或重复，标记为无效（或建议合并，此处直接设为 invalid）。\n"
        )
        
        forbidden_strict = (
            "2. 【禁止生成的情况】：\n"
            "   - **地方性日常事务**：如某地居民纠纷、普通治安案件、小范围天气预报。\n"
            "   - **非公共利益的商业行为**：如某公司常规促销、产品小版本更新、普通人事变动。\n"
            "   - **过于宽泛**：如“近期经济形势”、“科技新闻汇总”等非具体事件。\n"
        )
        
        forbidden_loose = (
            "2. 【禁止生成的情况】：\n"
            "   - **过于宽泛**：如“今日新闻”、“财经快讯”等完全无具体内容的标题。\n"
            "   - **纯粹广告**：无新闻价值的纯推销内容。\n"
        )

        if quality_level >= 3:
            forbidden_text = forbidden_strict + forbidden_common
        else:
            forbidden_text = forbidden_loose + forbidden_common

        system_prompt = (
            f"你是专题质量审核员（当前审核等级：{quality_level}/5）。请严格评估以下待创建的专题是否符合标准。\n"
            f"判定标准：\n"
            f"{criteria_text}"
            f"{forbidden_text}"
            "返回格式：必须是一个 JSON 数组，每个元素包含 'index' (整数), 'valid' (布尔值), 'reason' (简短理由)。\n"
            "例如：[{\"index\": 0, \"valid\": true, \"reason\": \"符合标准...\"}, {\"index\": 1, \"valid\": false, \"reason\": \"不符合...\"}]"
        )
        
        user_prompt = f"{existing_info}待审核专题列表：\n"
        for i, t in enumerate(topics):
            user_prompt += f"[{i}] 名称：{t.get('name')}\n    描述：{t.get('description')}\n\n"
            
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
            
        system_prompt = "你是一个专业的新闻分析师。请根据提供的新闻标题列表，聚合出近期发生的具体、具有一定持续性的专题事件。"
        
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

        criteria_desc = ""
        if quality_level <= 1:
            criteria_desc = (
                "**生成标准（宽松）**：\n"
                "1. **全面覆盖**：识别所有真实的、具体的、有一定价值的新闻事件，**包括地方性事件**。\n"
                "2. **具体性**：必须指向具体的事件，而非宽泛的概念。\n"
            )
        elif quality_level == 2:
            criteria_desc = (
                "**生成标准（较宽松）**：\n"
                "1. **公共价值**：识别具有一定公共信息价值的新闻事件。\n"
                "2. **排除琐碎**：忽略纯粹的个人琐事或无关紧要的小事故。\n"
                "3. **允许地方性**：允许重要的地方性社会新闻。\n"
            )
        elif quality_level == 3:
            criteria_desc = (
                "**生成标准（标准）**：\n"
                "1. **必须生成**：涉及**国际冲突、外交纷争、战争**的事件（无论热度如何，如“泰柬冲突”）。\n"
                "2. **重大社会热点**：具有全国性影响且讨论激烈的社会事件。\n"
                "3. **严格排除（即使有热度也不生成）**：\n"
                "   - **常规经济/金融波动**（如汇率涨跌、股市波动）。\n"
                "   - **基础设施建设/通车**（如高铁开通、大桥合龙）。\n"
                "   - **一般性自然灾害**（未造成重大人员伤亡或次生灾害的地震/天气）。\n"
                "   - **政策征求意见/常规发布**（非正式落地或引发巨大争议的政策）。\n"
            )
        elif quality_level == 4:
            criteria_desc = (
                "**生成标准（严格）**：\n"
                "1. **高影响力**：必须是**全国性或国际性**的重大新闻。\n"
                "2. **高热度**：必须是当前公众讨论的焦点。\n"
                "3. **严格排除**：忽略所有地方性新闻（除非引发全国关注）和一般性行业动态。\n"
            )
        else: # >= 5
            criteria_desc = (
                "**生成标准（极严）**：\n"
                "1. **顶级事件**：仅聚合**历史级、战略级或全网刷屏级**的超级事件。\n"
                "2. **宁缺毋滥**：非此类特大事件一律不生成。\n"
                "3. **深度影响**：关注涉及国家政策、国际关系巨变或行业格局改变的事件。\n"
            )

        prompt = f"""
请分析以下新闻标题（已标注热度），识别出 {min_count} 至 {max_count} 个（严格限制数量）具体的、具有主题性的热门专题事件。

{criteria_desc}

**专题命名标准化指南（必须严格执行）**：

1. **标题结构公式**：
   - **事件型**：`[核心主体] + [事件性质] + [聚合后缀]`
      * **聚合后缀**（必须包含）：**全纪录、进程、始末、风波、争议、后续影响、最新进展、灾情及救援**
      * ❌ **错误示例**（禁止像单条新闻标题）：
        - 泰柬签署停火联合声明
      * ✅ **正确示例**（必须体现专题性）：
        - 泰柬边境冲突始末
      
    - **会议/活动型**：`[年份] + [活动全称/简称] + [核心看点]`
      * 例：2024苹果秋季发布会：iPhone16系列发布
    
    - **负面/争议型**：`[主体] + [争议行为] + [及影响]`
     * 例：高市早苗错误言论及影响 / 日本核污水排海全记录
 
 2. **核心原则**：
    - **拒绝动作描述**：不要使用“某人做了某事”或“某事发生”这种陈述句作为专题名。
    - **必须体现时间跨度**：专题名应暗示这是一系列事件的集合，而不仅仅是一个瞬间动作。
    - **字数限制**：8-20 个汉字。
    - **客观中立**：避免情绪化形容词。

 3. **关键要求**：
    - **严格遵守数量限制**：输出的专题数量必须在 {min_count} 到 {max_count} 之间。
    - **优先高热度**：优先聚合热度高、报道量大的事件。
    - **拒绝宏大叙事**：不要生成“国际地缘政治”、“科技新闻汇总”等宽泛名称。
    - **拒绝单条新闻标题**：如果生成的名称看起来像一条新闻标题，请立即重写，加上“全纪录”、“进程”、“始末”等后缀。
    - **实体限定**：名称中必须包含核心实体（人名、地名、机构名）。
    - **时间范围**：专题事件必须从最近的时间窗口内获取。
    - **不要生成**：非重大、非持续性事件专题。

新闻标题列表：
{titles_str}

请仅返回一个 JSON 数组，格式如下：
[
  {{"name": "专题名称", "description": "专题描述"}},
  ...
]
不要包含任何 Markdown 代码块标记或其他文字。
"""
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

        system_prompt = (
            "你是一个数据提取助手。请从以下内容中提取新闻条目。\n"
            "返回一个JSON对象，包含 'items' 键，对应一个列表。列表中的每个对象包含：\n"
            "- 'title': 新闻标题\n"
            "- 'link': 新闻链接\n"
            "- 'summary': 新闻摘要（如果内容中包含摘要则直接使用，否则基于标题生成简短说明）\n"
            "如果无法提取，请返回 {'items': []}。"
        )
        user_prompt = f"内容：\n{content[:20000]}"

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
        system_prompt = (
            "你是一个专业的新闻舆情分析师。请分析给定的新闻标题和内容（摘要），提取以下信息：\n"
            "1. 情感倾向标签 (label): 只能是 '正面'、'中立' 或 '负面'。\n"
            "2. 情感分数 (score): 0到100之间的整数。0代表极度负面，50代表中立，100代表极度正面。\n"
            f"3. 所属领域 (category): 必须严格从[{categories_str}]中选择最合适的1个领域，禁止创造新分类、禁止使用“其他”作为所属领域。\n"
            "4. 涉及国家 (region): 只能输出国家名称（一个或多个），禁止输出省/市/区/县等行政区划，禁止输出“东亚/欧洲/中东”等大区。允许示例：'中国'、'美国'、'日本'、'韩国'、'俄罗斯'、'英国'、'法国'、'德国'、'印度'、'加拿大'、'澳大利亚'等。如果涉及多个国家，请用逗号分隔（如'中国,美国'）。如果确实无法判断，请输出'全球'。\n"
            "5. 关键词 (keywords): 提取3-5个核心关键词（实词），排除'的'、'了'等停用词。\n"
            "6. 涉及实体 (entities): 提取新闻中涉及的人名、公司名、组织机构名等。\n"
            "返回格式必须是合法的 JSON 对象，例如：\n"
            "{\n"
            '  "score": 85,\n'
            '  "label": "正面",\n'
            '  "category": "科技/科学",\n'
            '  "region": "中国",\n'
            '  "keywords": ["人工智能", "创新", "发布"],\n'
            '  "entities": ["OpenAI", "Sam Altman"]\n'
            "}"
        )
        user_prompt = f"标题：{title}\n内容：{content[:1000]}"

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
        system_prompt = (
            "你是一个专业的新闻舆情分析师。请分析给定的新闻标题，快速判断其情感倾向和所属领域。\n"
            "判定标准：\n"
            "1. 情感倾向 (label): '正面'、'中立' 或 '负面'。\n"
            "2. 情感分数 (score): 0-100 (负面<40, 中立40-60, 正面>60)。\n"
            f"3. 所属领域 (category): 必须严格从[{categories_str}]中选择最合适的1个，禁止创造新分类、禁止使用“其他”作为所属领域。\n"
            "4. 涉及国家 (region): 只能输出国家名称（一个或多个），禁止输出省/市/区/县等行政区划，禁止输出“东亚/欧洲/中东”等大区。允许示例：'中国'、'美国'、'日本'、'韩国'、'俄罗斯'、'英国'、'法国'、'德国'、'印度'、'加拿大'、'澳大利亚'等。如果涉及多个国家，请用逗号分隔（如'中国,美国'）。如果确实无法判断，请输出'全球'。\n"
            "返回格式：必须是合法的 JSON 数组，每个元素包含 id, label, score, category, region。例如：\n"
            '[{"id": 101, "label": "正面", "score": 80, "category": "政治军事", "region": "中国"}, ...]'
        )

        user_prompt = "请分析以下新闻：\n"
        for item in news_items:
            user_prompt += f"[ID:{item['id']}] {item['title']}\n"

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

    async def generate_summary(self, title: str, content: str, max_words: int = 300) -> Optional[str]:
        """
        输入:
        - `title`: 新闻标题
        - `content`: 新闻正文
        - `max_words`: 最大字数限制 (默认 300)

        输出:
        - 摘要文本 (如果失败返回 None)

        作用:
        - 使用 LLM 生成高质量新闻摘要，支持主备切换
        """
        if not content:
            return None
        system_prompt = (
            f"你是一个专业新闻编辑。请将与新闻标题相关的信息总结为{max_words}字左右的纯文字内容。\n"
            "要求：\n"
            "1. **内容详实**：拒绝简陋概括，必须保留 **具体人名、地名、数据、物品名称** 等关键实体信息，避免过度抽象。\n"
            "2. **去噪**：去除广告、链接等无关信息。\n"
            "3. **纯文本**：不要使用 HTML 标签。\n"
            "4. **直接输出**：直接开始输出摘要内容，不要包含任何“好的”、“根据您的要求”、“摘要如下”等客套话或前缀。"
        )
        user_prompt = f"标题：{title}\n\n正文：{content[:100000]}"

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

        system_prompt = (
            "你是事件一致性判定助手。请批量判断以下每组新闻是否属于对应专题追踪的同一新闻事件（或其直接后续进展）。\n"
            "判定标准：\n"
            "1) 视为同一事件：核心主体/地点/关键事实一致，或明显是同一事件的后续进展（通报、调查、进展、回应、二次影响）。\n"
            "2) 视为不同事件：主体无关、不同地区不同主体的相似话题、仅同类泛话题但无共同事实。\n"
            "3) 当信息不足时，倾向于返回 false。\n"
            "返回格式：必须是一个 JSON 数组，每个元素包含 'match' (布尔值) 和 'reason' (简短理由)。\n"
            "例如：[{\"match\": true, \"reason\": \"核心事实一致\"}, {\"match\": false, \"reason\": \"主体不同\"}]"
        )

        user_prompt = "请判断以下各组匹配情况：\n"
        for idx, task in enumerate(tasks):
            user_prompt += (
                f"--- 第 {idx+1} 组 ---\n"
                f"【专题】{task['topic_name']}\n"
                f"【专题概览】{(task['topic_summary'] or '')[:300]}\n"
                f"【新闻标题】{task['news_title']}\n"
                f"【新闻摘要】{(task['news_summary'] or '')[:300]}\n"
            )

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
            focus_instruction = f"7. **专题聚焦**：当前专题为“{topic_name}”，请**严格专注于与该专题直接相关的信息**。每个节点描述必须与专题名称高度相关。如果是无关的噪音新闻，请直接忽略。如果所有新闻都与该专题无关，返回空数组。\n"

        system_prompt = (
            "你是时间轴事件合成助手。请根据提供的某一天的新闻列表，将其合成为 1-2 个关键的时间轴节点事件。\n"
            "严格遵守以下规则：\n"
            "1. **合并同类项**：必须将报道同一事件的不同新闻合并为一个事件节点。\n"
            "2. **内容精炼**：每个事件描述控制在 100 字以内，概括核心事实。\n"
            "3. **关联来源**：对于每个生成的事件，必须列出支持该事件的所有新闻 ID (source_ids)。\n"
            "4. **数量限制**：**每天最多生成 2 个事件节点**。必须严格执行。如果有多件大事，取最重要的 2 件；如果是同一件事的多个方面，必须合并为 1 件。\n"
            "5. **严格时间轴合并**：\n"
            "   - 对于发布时间在 2 小时内的新闻，必须合并到同一个时间轴节点中，禁止单独创建新节点。\n"
            "   - 绝对禁止跨天合并新闻事件（输入已限定为同一天）。\n"
            "6. **直接输出**：不要包含“好的”、“根据提供的新闻”等客套话，直接返回结果。\n"
            "7. **返回格式**：仅返回 JSON 数组，例如：[{\"content\": \"事件描述...\", \"source_ids\": [1, 3]}, ...]\n"
            f"{focus_instruction}"
        )

        user_prompt = f"日期：{date_str}\n"
        if topic_name:
            user_prompt += f"专题名称：{topic_name}\n"
        user_prompt += "新闻列表：\n"
        for item in news_items:
            user_prompt += f"[ID: {item['id']}] {item['title']}\n摘要: {(item['summary'] or '')[:100]}\n\n"

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
        system_prompt = (
            "你是专题去重助手。请判断以下两个专题是否指的是同一个具体的新闻事件。\n"
            "判定标准：\n"
            "1. 【视为相同】：\n"
            "   - 指向同一个核心突发事件（如“某地地震”与“某地发生6.0级地震”）。\n"
            "   - 仅仅是命名角度不同（如“SpaceX星舰发射”与“星舰第五次试飞”）。\n"
            "   - 一个是另一个的子集或初期阶段，且核心事实完全重合。\n"
            "2. 【视为不同】：\n"
            "   - 不同的独立事件（如“俄乌冲突”与“巴以冲突”）。\n"
            "   - 同一类别的不同个体（如“某公司发布财报”与“另一公司发布财报”）。\n"
            "返回格式：仅返回一个JSON对象，例如：{\"duplicate\": true, \"reason\": \"...\"}"
        )
        user_prompt = (
            f"【专题A】名称：{new_name}\n描述：{new_desc[:500]}\n\n"
            f"【专题B】名称：{existing_name}\n描述：{existing_desc[:500]}"
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

        system_prompt = (
            "你是一个专业的新闻助手。请根据提供的新闻上下文回答用户的问题。\n"
            "如果答案不在上下文中，请使用你自己的知识回答，但请注明“根据已有新闻未找到相关信息，以下是我的补充...”。\n"
            "回答要简洁、客观。"
        )
        user_prompt = f"【新闻上下文】:\n{context}\n\n【用户问题】: {query}"

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

        system_prompt = (
            "请判断列表中每对新闻标题是否属于 **同一个新闻事件的报道** , 标准适当宽松，尽可能将相关关键词新闻判断为同一新闻事件。\n"
            "判定标准：\n"
            "1. 【视为相同】：\n"
            "   - **核心事实重合**：描述同一个具体的突发事件、政策发布、科技发现等。数字略有差异（如“1178名”与“逾千名”）或表述不同（如“禁产”与“禁止生产”）视为相同。\n"
            "   - **事件延续与关联**：同一大事件背景下的直接后续或反应（如“地震发生”与“地震后引发海啸”、“某人对地震的反应”）。\n"
            "   - **技术/专业话题**：同一漏洞、同一产品的不同解读（如“React漏洞”）。\n"
            "2. 【视为不同】：\n"
            "   - **主体完全无关**：如“黄金”与“白银”，“苹果”与“香蕉”。\n"
            "   - **明确的不同期数**：如“1月1日日报”与“1月2日日报”。\n"
            "返回格式：仅返回一个JSON数组，包含对应顺序的布尔值，例如：[true, false, true]"
        )

        user_content = "请判断以下新闻对：\n"
        for i, p in enumerate(pairs):
            user_content += f"{i + 1}. [{p['leader']}] vs [{p['candidate']}]\n"

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
        for route in routes:
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
                if is_backup:
                    logger.warning(f"⚠️ 备用AI核验失败 (第{attempt + 1}次)，准备重试...")

        logger.error("❌ 所有通道核验均失败，跳过本批次")
        return [False] * len(pairs)


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
            
        system_prompt = (
            "你是一个资深新闻评论员。请根据提供的多条新闻报道，对该专题事件进行全方位的深度综述。\n"
            "要求：\n"
            "1. **严格一致性**：综述内容必须与专题名称保持严格一致，禁止包含与专题无关的内容。\n"
            "2. **信息详实**：在综述中必须保留关键的**事实细节**，拒绝空洞的宏大叙事。请务必包含具体人名、关键物品名称、具体金额/估值、机构名称、具体的指控或争议点、各方具体回应和核心证据。\n"
            "3. **结构清晰**：请包含以下几个维度：\n"
            "   - **【事件背景】**：详细阐述事件起因。\n"
            "   - **【发展脉络】**：按逻辑梳理事件的升级过程。\n"
            "   - **【争议焦点】**：核心矛盾是什么。\n"
            "   - **【核心影响】**：具体的社会/行业影响。\n"
            "   - **【最新进展与展望】**：当前的调查状态及可能的走向。\n"
            "4. **深度分析**：不要只做表面拼凑，要分析事件背后的逻辑关联。\n"
            "5. **纯文本格式**：不要使用 Markdown 标题符号（如 #, ##, **），使用中文方括号【】作为小标题即可。分段换行要清晰。\n"
            "6. 字数控制在 600-1200 字之间。\n"
            "7. **直接输出**：直接开始正文，不要有“好的”、“综述如下”等开场白。"
        )
        
        user_prompt = f"专题名称：{topic_name}\n\n相关新闻报道：\n{input_text}"
        
        prefer_backup = self._get_prefer_backup("TOPIC_OVERVIEW")
        res = await self._call_llm_with_routes(user_prompt, system_prompt, prefer_backup=prefer_backup)
        if res:
            return res.strip()
        return "暂无法生成综述。"







ai_service = AIService()