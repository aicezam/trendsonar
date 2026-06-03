"""
本文件用于统一处理新闻摘要素材选择逻辑，优先复用 RSS 等来源自带摘要，减少不必要的正文抓取。
主要函数：
- `get_existing_summary_material`: 判断现有摘要是否可直接作为分析素材
- `build_summary_generation_input`: 生成用于摘要模型的输入文本
"""

from __future__ import annotations

from typing import Optional

from app.core.config import get_settings
from app.utils.tools import clean_html_tags

settings = get_settings()


def get_existing_summary_material(summary: Optional[str], min_length: int = 4) -> Optional[str]:
    """
    输入:
    - `summary`: 新闻当前已有摘要，可能来自 RSS 或历史生成结果
    - `min_length`: 视为有效摘要的最小长度

    输出:
    - 清洗后的可用摘要文本；不可用时返回 `None`

    作用:
    - 统一判断已有摘要是否足以直接用于 AI 分析或作为正文抓取失败前的优先素材
    """

    cleaned_summary = clean_html_tags(summary or "").strip()
    if len(cleaned_summary) < min_length:
        return None
    return cleaned_summary


def build_summary_generation_input(
    *,
    content: Optional[str],
    original_summary: Optional[str],
) -> Optional[str]:
    """
    输入:
    - `content`: 已抓取或已存在的正文内容
    - `original_summary`: 来源侧自带的原始摘要

    输出:
    - 传给摘要模型的输入文本；无可用素材时返回 `None`

    作用:
    - 统一控制摘要生成时的输入优先级：
      1. 同时有正文和原始摘要时，优先把两者拼接给模型
      2. 只有原始摘要时，直接使用原始摘要，避免额外抓正文
      3. 只有正文时，直接使用正文
    """

    cleaned_content = clean_html_tags(content or "").strip()
    cleaned_summary = get_existing_summary_material(original_summary)

    content_limit = max(0, int(getattr(settings, "SUMMARY_INPUT_MAX_LENGTH", 5000) or 0))
    summary_limit = max(0, int(getattr(settings, "SUMMARY_ORIGIN_MAX_LENGTH", 300) or 0))

    if content_limit > 0 and len(cleaned_content) > content_limit:
        cleaned_content = cleaned_content[:content_limit]
    if summary_limit > 0 and cleaned_summary and len(cleaned_summary) > summary_limit:
        cleaned_summary = cleaned_summary[:summary_limit]

    if cleaned_summary and cleaned_content and cleaned_summary != cleaned_content:
        return f"原始摘要：{cleaned_summary}\n\n正文内容：{cleaned_content}"
    if cleaned_summary:
        return cleaned_summary
    if cleaned_content:
        return cleaned_content
    return None
