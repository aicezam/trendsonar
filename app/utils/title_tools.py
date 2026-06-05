# 本文件用于新闻标题长度判断、AI 精简结果清洗和本地兜底裁剪。

from __future__ import annotations

import re
from typing import Optional


def normalize_title_text(title: str) -> str:
    """
    输入:
    - `title`: 原始标题文本

    输出:
    - 去除多余空白后的标题文本

    作用:
    - 统一标题长度判断和保存前的基础清洗逻辑。
    """

    return re.sub(r"\s+", " ", str(title or "")).strip()


def should_refine_title(title: str, threshold: int = 30) -> bool:
    """
    输入:
    - `title`: 新闻标题
    - `threshold`: 触发精简的标题长度阈值

    输出:
    - 是否需要精简标题

    作用:
    - 仅在标题明显过长时触发 AI 精简，避免无谓调用和标题频繁变化。
    """

    return len(normalize_title_text(title)) > threshold


def normalize_refined_title(raw_title: str, max_chars: int = 20) -> Optional[str]:
    """
    输入:
    - `raw_title`: AI 返回的候选短标题
    - `max_chars`: 短标题最大字符数

    输出:
    - 清洗后的短标题；无法得到有效标题时返回 None

    作用:
    - 移除客套话、代码块和引号，并在模型未严格遵守长度时提供本地兜底。
    """

    title = normalize_title_text(raw_title)
    if not title:
        return None

    title = title.replace("```json", "").replace("```", "").strip()
    title = re.sub(r"^(短标题|标题|精简标题|新标题)\s*[:：]\s*", "", title).strip()
    title = title.strip("\"'“”‘’`，。；;：:、 ")
    title = re.sub(r"\s+", "", title)
    if not title:
        return None

    if len(title) <= max_chars:
        return title

    for sep in ("，", "。", "；", "：", "、", "-", "—", "|", "｜", " "):
        if sep in title:
            candidate = title.split(sep, 1)[0].strip()
            if 4 <= len(candidate) <= max_chars:
                return candidate

    return title[:max_chars].rstrip("，。；;：:、-—|｜ ")
