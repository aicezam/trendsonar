# 本文件用于将新闻文字渲染为可访问的 PNG 图片卡片。

from __future__ import annotations

import hashlib
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[2]
GENERATED_IMAGE_DIR = BASE_DIR / "static" / "generated" / "agent_images"
GENERATED_IMAGE_URL_PREFIX = "/static/generated/agent_images"
MAX_IMAGE_TEXT_CHARS = 1200
THEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "default": {
        "bg": (246, 247, 249),
        "panel": (255, 255, 255),
        "title": (22, 28, 36),
        "text": (58, 67, 80),
        "muted": (120, 130, 145),
        "accent": (31, 111, 235),
        "line": (224, 229, 236),
    },
    "dark": {
        "bg": (24, 28, 35),
        "panel": (34, 40, 50),
        "title": (242, 246, 250),
        "text": (207, 216, 228),
        "muted": (145, 156, 172),
        "accent": (98, 169, 255),
        "line": (64, 73, 88),
    },
    "warm": {
        "bg": (248, 246, 240),
        "panel": (255, 255, 252),
        "title": (31, 32, 34),
        "text": (72, 70, 66),
        "muted": (135, 124, 108),
        "accent": (195, 75, 45),
        "line": (230, 224, 214),
    },
}


def _font_candidates() -> list[Path]:
    """
    输入:
    - 无

    输出:
    - 可能存在的中文字体路径

    作用:
    - 兼容 Windows 与 Debian 容器，优先选择能显示中文的字体。
    """

    names = [
        "NotoSansSC-VF.ttf",
        "Noto Sans SC (TrueType).otf",
        "Noto Sans CJK SC Regular.otf",
        "SourceHanSansSC-Regular.otf",
        "msyh.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "DejaVuSans.ttf",
    ]
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]
    candidates: list[Path] = []
    project_font = BASE_DIR / "font.ttf"
    if project_font.exists():
        candidates.append(project_font)
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.exists():
                candidates.append(direct)
        for path in root.rglob("*"):
            if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and any(
                key in path.name.lower()
                for key in ("noto", "sourcehan", "msyh", "simhei", "simsun", "dejavu")
            ):
                candidates.append(path)
    return list(dict.fromkeys(candidates))


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    输入:
    - `size`: 字号
    - `bold`: 是否优先使用粗体

    输出:
    - Pillow 字体对象

    作用:
    - 为图片卡片加载中文字体，缺失时回退 Pillow 默认字体。
    """

    candidates = _font_candidates()
    if bold:
        candidates = sorted(candidates, key=lambda path: 0 if any(key in path.name.lower() for key in ("bold", "bd", "medium")) else 1)
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    """
    输入:
    - `draw`: 绘制对象
    - `text`: 文本
    - `font`: 字体

    输出:
    - 文本像素宽度

    作用:
    - 支持按像素宽度对中英文混排文本换行。
    """

    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0])


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    """
    输入:
    - `draw`: 绘制对象
    - `text`: 原始文本
    - `font`: 字体
    - `max_width`: 每行最大像素宽度
    - `max_lines`: 最大行数

    输出:
    - 换行后的文本行

    作用:
    - 避免新闻标题和摘要溢出图片边界。
    """

    normalized = " ".join(str(text or "").split())
    lines: list[str] = []
    for paragraph in normalized.split("\n"):
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and _text_width(draw, candidate, font) > max_width:
                lines.append(current)
                current = char
                if len(lines) >= max_lines:
                    break
            else:
                current = candidate
        if current and len(lines) < max_lines:
            lines.append(current)
        if len(lines) >= max_lines:
            break
    if lines and len(lines) == max_lines and _text_width(draw, lines[-1] + "...", font) > max_width:
        lines[-1] = textwrap.shorten(lines[-1], width=max(8, len(lines[-1]) - 2), placeholder="...")
    elif lines and len(lines) == max_lines:
        lines[-1] = lines[-1].rstrip("。；，、 ") + "..."
    return lines


def generate_news_text_image(
    *,
    title: str,
    body: str = "",
    source: str = "",
    time_label: str = "",
    footer: str = "TrendSonar",
    theme: str = "default",
    width: int = 1200,
    height: int = 675,
) -> dict[str, Any]:
    """
    输入:
    - `title`: 图片主标题
    - `body`: 正文或摘要
    - `source`/`time_label`: 来源与时间
    - `footer`: 页脚文字
    - `theme`: 主题名称
    - `width`/`height`: 图片尺寸

    输出:
    - 图片文件路径、URL 和渲染元信息

    作用:
    - 将新闻文字生成适合分享或插入回复的 PNG 图片。
    """

    safe_width = max(800, min(int(width or 1200), 1800))
    safe_height = max(450, min(int(height or 675), 1400))
    palette = THEMES.get(str(theme or "default").lower(), THEMES["default"])
    clean_title = " ".join(str(title or "").split())[:220] or "新闻卡片"
    clean_body = " ".join(str(body or "").split())[:MAX_IMAGE_TEXT_CHARS]
    clean_source = " ".join(str(source or "").split())[:80]
    clean_time = " ".join(str(time_label or "").split())[:80]
    clean_footer = " ".join(str(footer or "TrendSonar").split())[:80]

    image = Image.new("RGB", (safe_width, safe_height), palette["bg"])
    draw = ImageDraw.Draw(image)

    margin = int(safe_width * 0.07)
    panel_box = (margin, margin, safe_width - margin, safe_height - margin)
    draw.rounded_rectangle(panel_box, radius=20, fill=palette["panel"], outline=palette["line"], width=2)

    accent_x = panel_box[0] + 34
    accent_y = panel_box[1] + 38
    draw.rounded_rectangle((accent_x, accent_y, accent_x + 96, accent_y + 8), radius=4, fill=palette["accent"])

    title_font = _load_font(max(38, int(safe_width * 0.044)), bold=True)
    body_font = _load_font(max(23, int(safe_width * 0.025)))
    meta_font = _load_font(max(18, int(safe_width * 0.018)))
    footer_font = _load_font(max(17, int(safe_width * 0.017)))

    content_width = panel_box[2] - panel_box[0] - 72
    cursor_y = accent_y + 36
    title_lines = _wrap_text(draw, clean_title, title_font, content_width, 3)
    for line in title_lines:
        draw.text((panel_box[0] + 36, cursor_y), line, font=title_font, fill=palette["title"])
        cursor_y += int(title_font.size * 1.24)

    meta_parts = [part for part in (clean_source, clean_time) if part]
    if meta_parts:
        cursor_y += 12
        draw.text((panel_box[0] + 36, cursor_y), " / ".join(meta_parts), font=meta_font, fill=palette["muted"])
        cursor_y += int(meta_font.size * 1.8)
    else:
        cursor_y += 28

    if clean_body:
        draw.line((panel_box[0] + 36, cursor_y, panel_box[2] - 36, cursor_y), fill=palette["line"], width=2)
        cursor_y += 28
        body_lines = _wrap_text(draw, clean_body, body_font, content_width, 8)
        for line in body_lines:
            draw.text((panel_box[0] + 36, cursor_y), line, font=body_font, fill=palette["text"])
            cursor_y += int(body_font.size * 1.55)

    footer_y = panel_box[3] - 54
    draw.line((panel_box[0] + 36, footer_y - 20, panel_box[2] - 36, footer_y - 20), fill=palette["line"], width=1)
    draw.text((panel_box[0] + 36, footer_y), clean_footer, font=footer_font, fill=palette["muted"])

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha1(f"{clean_title}|{clean_body}|{stamp}".encode("utf-8")).hexdigest()[:10]
    filename = f"news_card_{stamp}_{digest}.png"
    GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_IMAGE_DIR / filename
    image.save(output_path, format="PNG", optimize=True)

    return {
        "ok": True,
        "url": f"{GENERATED_IMAGE_URL_PREFIX}/{filename}",
        "path": str(output_path),
        "width": safe_width,
        "height": safe_height,
        "title": clean_title,
    }
