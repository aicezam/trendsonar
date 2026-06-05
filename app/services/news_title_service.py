# 本文件用于封装新闻标题精简的业务逻辑，供摘要生成流程复用。

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.core.logger import logger
from app.models.news import News
from app.utils.title_tools import should_refine_title

if TYPE_CHECKING:
    from app.services.ai_service import AIService


async def refine_news_title_if_needed(
    news: News,
    summary: str = "",
    content: str = "",
    ai: Optional["AIService"] = None,
) -> bool:
    """
    输入:
    - `news`: 待处理新闻对象
    - `summary`: 当前新闻摘要
    - `content`: 当前新闻正文素材
    - `ai`: AI 服务实例，可选注入

    输出:
    - 标题是否被成功精简并写回对象

    作用:
    - 在摘要生成后，如果标题超过 30 字，则调用 AI 将标题精简为 20 字以内。
    """

    if not should_refine_title(news.title or ""):
        return False

    if ai is None:
        from app.services.ai_service import ai_service

        ai = ai_service

    try:
        refined_title = await ai.refine_title(news.title or "", summary=summary, content=content, max_chars=20)
    except Exception as e:
        logger.warning(f"标题精简失败，保留原标题: {e}")
        return False

    if not refined_title or refined_title == news.title:
        return False

    news.title = refined_title
    return True
