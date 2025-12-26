"""
本文件用于提供新闻数据相关 API：筛选项、列表查询、摘要生成与图片导出等。
主要函数:
- `get_sources`: 获取来源列表
- `get_categories`: 获取分类列表
- `get_regions`: 获取地区列表
"""

import io
import os
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.news import News
from app.services.ai_service import ai_service
from app.services.crawler_service import crawler_service
from app.utils.tools import normalize_regions_to_countries

router = APIRouter(prefix="/api", tags=["news"])


@router.get("/sources")
async def get_sources(db: AsyncSession = Depends(get_db)):
    """
    输入:
    - `db`: 数据库会话（依赖注入）

    输出:
    - 来源列表（去重并排序）

    作用:
    - 为前端筛选器提供可用新闻来源集合
    """

    result = await db.execute(select(News.source).distinct())
    sources = result.scalars().all()
    return sorted([s for s in sources if s])


@router.get("/categories")
async def get_categories(db: AsyncSession = Depends(get_db)):
    """
    输入:
    - `db`: 数据库会话（依赖注入）

    输出:
    - 分类列表（去重并排序）

    作用:
    - 为前端筛选器提供可用新闻分类集合
    """

    result = await db.execute(select(News.category).distinct())
    categories = result.scalars().all()
    return sorted([c for c in categories if c])


@router.get("/regions")
async def get_regions(db: AsyncSession = Depends(get_db)):
    """
    输入:
    - `db`: 数据库会话（依赖注入）

    输出:
    - 地区列表（按逗号拆分、去重并排序）

    作用:
    - 为前端筛选器提供可用新闻地区集合
    """

    result = await db.execute(select(News.region).distinct())
    raw_regions = result.scalars().all()

    unique_regions = set()
    for r in raw_regions:
        if r:
            parts = [p.strip() for p in r.split(",")]
            for p in parts:
                normalized = normalize_regions_to_countries(p)
                if normalized in {"其他", "全球"}:
                    continue
                for c in normalized.split(","):
                    if c:
                        unique_regions.add(c)

    valid_regions = [r for r in unique_regions if r and r.lower() not in ["", "null", "其他", "全球"]]
    return sorted(valid_regions)


@router.get("/news")
async def get_news(
    q: str = "",
    page: int = 1,
    date: str = "24h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: str = "heat",
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - `q`: 搜索关键词（支持标题包含与向量相似度加权）
    - `page`: 页码
    - `date`: 快捷时间范围（24h/3d/7d/week/month/year/all 或 YYYY-MM-DD）
    - `start_date`/`end_date`: 自定义起止日期（YYYY-MM-DD）
    - `sort_by`: 排序方式（heat/date）
    - `category`/`region`/`source`: 筛选条件
    - `db`: 数据库会话（依赖注入）

    输出:
    - 新闻列表与页码信息

    作用:
    - 提供新闻列表查询接口，支持筛选、排序与语义检索
    """

    page_size = 20
    offset = (page - 1) * page_size

    stmt = select(News)

    now = datetime.now()
    if start_date or end_date:
        try:
            if start_date:
                s_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                start = datetime.combine(s_date, time.min)
                stmt = stmt.where(News.publish_date >= start)
            if end_date:
                e_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                end = datetime.combine(e_date, time.max)
                stmt = stmt.where(News.publish_date <= end)
        except ValueError:
            pass
    else:
        try:
            if date == "24h":
                start = now - timedelta(hours=24)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "3d":
                start = now - timedelta(days=3)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "7d":
                start = now - timedelta(days=7)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "week":
                start = now - timedelta(days=now.weekday())
                start = datetime.combine(start.date(), time.min)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "month":
                start = now.replace(day=1)
                start = datetime.combine(start.date(), time.min)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "year":
                start = now.replace(month=1, day=1)
                start = datetime.combine(start.date(), time.min)
                stmt = stmt.where(News.publish_date >= start)
            elif date == "all":
                pass
            elif date:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                start = datetime.combine(target_date, time.min)
                end = datetime.combine(target_date, time.max)
                stmt = stmt.where(News.publish_date >= start, News.publish_date <= end)
        except ValueError:
            pass

    if category and category != "all":
        if "," in category:
            stmt = stmt.where(News.category.in_(category.split(",")))
        else:
            stmt = stmt.where(News.category == category)

    normalized_region = normalize_regions_to_countries(region) if region and region != "all" else ""
    if normalized_region and normalized_region not in {"其他", "全球"}:
        selected_regions = normalized_region.split(",")
        conditions = [News.region.ilike(f"%{r}%") for r in selected_regions]
        stmt = stmt.where(or_(*conditions))

    if source and source != "all":
        if "," in source:
            stmt = stmt.where(News.source.in_(source.split(",")))
        else:
            stmt = stmt.where(News.source == source)

    if q:
        result = await db.execute(stmt)
        candidates = result.scalars().all()

        q_emb_list = await ai_service.get_embeddings([q])
        q_vec = np.array(q_emb_list[0]) if q_emb_list and q_emb_list[0] else None

        scored_news = []
        for n in candidates:
            score = 0.0
            if n.title and q.lower() in n.title.lower():
                score += 0.5

            if q_vec is not None and n.embedding:
                n_vec = np.array(n.embedding)
                norm_q = np.linalg.norm(q_vec)
                norm_n = np.linalg.norm(n_vec)
                if norm_q > 0 and norm_n > 0:
                    sim = np.dot(q_vec, n_vec) / (norm_q * norm_n)
                    score += float(sim)

            if score > 0.2:
                scored_news.append((score, n))

        scored_news.sort(key=lambda x: x[0], reverse=True)
        sliced = scored_news[offset : offset + page_size]

        data = []
        for score, n in sliced:
            data.append(
                {
                    "id": n.id,
                    "title": n.title,
                    "url": n.url,
                    "source": n.source,
                    "heat": n.heat_score,
                    "time": n.publish_date.isoformat(),
                    "summary": n.summary,
                    "sources": n.sources,
                    "category": n.category,
                    "region": normalize_regions_to_countries(n.region),
                    "sentiment_label": n.sentiment_label,
                    "sentiment_score": n.sentiment_score,
                }
            )

        return {"data": data, "page": page}

    if sort_by == "heat":
        stmt = stmt.order_by(desc(News.heat_score), desc(News.publish_date))
    else:
        stmt = stmt.order_by(desc(News.publish_date))

    result = await db.execute(stmt.offset(offset).limit(page_size))
    data = []
    for n in result.scalars().all():
        data.append(
            {
                "id": n.id,
                "title": n.title,
                "url": n.url,
                "source": n.source,
                "heat": n.heat_score,
                "time": n.publish_date.isoformat(),
                "summary": n.summary,
                "sources": n.sources,
                "category": n.category,
                "region": normalize_regions_to_countries(n.region),
                "sentiment_label": n.sentiment_label,
                "sentiment_score": n.sentiment_score,
            }
        )
    return {"data": data, "page": page}


@router.post("/generate_summary/{news_id}")
async def api_generate_summary(news_id: int, db: AsyncSession = Depends(get_db)):
    """
    输入:
    - `news_id`: 新闻 ID
    - `db`: 数据库会话（依赖注入）

    输出:
    - 生成的摘要文本

    作用:
    - 为指定新闻补抓正文并生成 AI 摘要，同时更新向量用于检索
    """

    news = await db.get(News, news_id)
    if not news:
        raise HTTPException(status_code=404)

    full_content = ""

    main_content = news.content
    if not main_content or len(main_content) < 50:
        main_content = await crawler_service.crawl_content(news.url) or ""
        if main_content:
            news.content = main_content

    full_content += f"【主报道】{news.title}\n{main_content}\n\n"

    if news.sources and len(news.sources) > 1:
        for src in news.sources:
            if src.get("url") == news.url:
                continue
            sub_content = await crawler_service.crawl_content(src["url"])
            if sub_content:
                full_content += f"【关联报道: {src.get('name')}】\n{sub_content}\n\n"
                if len(full_content) > 100000:
                    full_content = full_content[:100000] + "\n...(截断)..."
                    break

    if len(full_content) < 50:
        return {"summary": "抓取失败，内容过短"}

    summary = await ai_service.generate_summary(news.title, full_content)
    news.summary = summary

    try:
        txt_to_embed = f"{news.title} {summary} {full_content[:1000]}"
        embs = await ai_service.get_embeddings([txt_to_embed])
        if embs and embs[0]:
            news.embedding = embs[0]
    except Exception:
        pass

    db.add(news)
    await db.commit()
    return {"summary": summary}


@router.get("/news_image")
async def generate_news_image(date: str, db: AsyncSession = Depends(get_db)):
    """
    输入:
    - `date`: 时间范围（24h/3d/7d/week/month/year/all 或 YYYY-MM-DD）
    - `db`: 数据库会话（依赖注入）

    输出:
    - PNG 图片二进制流

    作用:
    - 生成 Top 热点新闻海报图片，便于分享与外部展示
    """

    try:
        now = datetime.now()
        title_suffix = ""
        start = None
        end = None

        if date == "24h":
            start = now - timedelta(hours=24)
            end = now
            title_suffix = "24小时热点"
        elif date == "3d":
            start = now - timedelta(days=3)
            end = now
            title_suffix = "3天内热点"
        elif date == "7d":
            start = now - timedelta(days=7)
            end = now
            title_suffix = "7天内热点"
        elif date == "week":
            start = now - timedelta(days=now.weekday())
            start = datetime.combine(start.date(), time.min)
            end = now
            title_suffix = "本周热点"
        elif date == "month":
            start = now.replace(day=1)
            start = datetime.combine(start.date(), time.min)
            end = now
            title_suffix = "本月热点"
        elif date == "year":
            start = now.replace(month=1, day=1)
            start = datetime.combine(start.date(), time.min)
            end = now
            title_suffix = "今年热点"
        elif date == "all":
            start = datetime.min
            end = now
            title_suffix = "历史热点"
        else:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            start = datetime.combine(target_date, time.min)
            end = datetime.combine(target_date, time.max)
            title_suffix = f"{date} 热点速览"
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD or 24h/3d/7d/week/month/year/all") from e

    stmt = (
        select(News)
        .where(News.publish_date >= start, News.publish_date <= end)
        .order_by(desc(News.heat_score))
        .limit(20)
    )
    result = await db.execute(stmt)
    news_list = result.scalars().all()

    if not news_list:
        raise HTTPException(status_code=404, detail="No news found for this date")

    try:
        font_path = "simhei.ttf" if os.name == "nt" else "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        if not os.path.exists(font_path) and os.name != "nt":
            possible_fonts = ["./font.ttf"]
            for p in possible_fonts:
                if os.path.exists(p):
                    font_path = p
                    break

        if os.path.exists("font.ttf"):
            font_path = "font.ttf"

        title_font = ImageFont.truetype(font_path, 40)
        news_title_font = ImageFont.truetype(font_path, 30)
        summary_font = ImageFont.truetype(font_path, 24)
        meta_font = ImageFont.truetype(font_path, 20)
    except Exception:
        title_font = ImageFont.load_default()
        news_title_font = ImageFont.load_default()
        summary_font = ImageFont.load_default()
        meta_font = ImageFont.load_default()

    width = 900
    padding = 50
    header_height = 140

    def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int):
        lines = []
        if not text:
            return lines
        current_line = ""
        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
        return lines

    layout_items = []
    total_height = header_height + 20

    for idx, news in enumerate(news_list):
        item_h = 0

        clean_title = news.title.replace("\n", "。").replace("\r", "") if news.title else ""
        clean_summary = news.summary.replace("\n", "。").replace("\r", "") if news.summary else "暂无摘要"

        title_lines = wrap_text(clean_title, news_title_font, width - padding * 2 - 40)
        item_h += 30 + len(title_lines) * 40

        summary_lines = wrap_text(clean_summary, summary_font, width - padding * 2 - 40)
        item_h += 20 + len(summary_lines) * 32

        item_h += 40 + 20

        layout_items.append({"news": news, "title_lines": title_lines, "summary_lines": summary_lines, "height": item_h})
        total_height += item_h + 20

    image = Image.new("RGB", (width, total_height), color="#f3f4f6")
    draw = ImageDraw.Draw(image)

    draw.rectangle([(0, 0), (width, header_height)], fill="#ffffff")
    draw.text((padding, 50), f"TrendSonar - {title_suffix}", font=title_font, fill="#1f2937")

    y = header_height + 20

    for idx, item in enumerate(layout_items):
        news = item["news"]
        card_h = item["height"]

        draw.rectangle([(padding, y), (width - padding, y + card_h)], fill="#ffffff", outline="#e5e7eb", width=1)

        ty = y + 30
        for line in item["title_lines"]:
            draw.text((padding + 20, ty), line, font=news_title_font, fill="#111827")
            ty += 40

        sy = ty + 10
        for line in item["summary_lines"]:
            draw.text((padding + 20, sy), line, font=summary_font, fill="#4b5563")
            sy += 32

        meta_y = y + card_h - 40
        pub_time = news.publish_date.strftime("%m-%d %H:%M")
        related_count = len(news.sources)
        meta_text = f"时间: {pub_time}   |   热度: {news.heat_score:.1f}   |   相关报道: {related_count}"

        draw.text((padding + 20, meta_y), meta_text, font=meta_font, fill="#ef4444")

        y += card_h + 20

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    return Response(content=img_bytes, media_type="image/png")
