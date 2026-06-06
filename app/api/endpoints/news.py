"""
本文件用于提供新闻数据相关 API：筛选项、列表查询、摘要生成与图片导出等。
主要函数:
- `get_sources`: 获取来源列表
- `get_categories`: 获取分类列表
- `get_regions`: 获取地区列表
"""

import io
import os
import time as perf_time
from datetime import datetime, time, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.core.database import get_db
from app.core.logger import logger
from app.models.news import News
from app.services.ai_service import ai_service
from app.services.crawler_service import crawler_service
from app.services.news_title_service import refine_news_title_if_needed
from app.utils.news_query import build_news_query_filters, serialize_news_item
from app.utils.news_search import normalize_similarity_terms, semantic_news_search
from app.utils.summary_material import build_summary_generation_input, get_existing_summary_material
from app.utils.tools import normalize_regions_to_countries

router = APIRouter(prefix="/api", tags=["news"])
_FILTER_CACHE_TTL_SECONDS = 300
_FILTER_CACHE: dict[str, dict[str, Any]] = {}


def _get_filter_cache(key: str) -> Optional[list[str]]:
    """
    输入:
    - `key`: 缓存键名

    输出:
    - 命中的字符串列表或 None

    作用:
    - 为新闻筛选项接口提供短 TTL 内存缓存，减少重复 distinct 查询
    """

    item = _FILTER_CACHE.get(key)
    if not item:
        return None
    if perf_time.monotonic() - item["created_at"] > _FILTER_CACHE_TTL_SECONDS:
        _FILTER_CACHE.pop(key, None)
        return None
    return item["value"]


def _set_filter_cache(key: str, value: list[str]) -> list[str]:
    """
    输入:
    - `key`: 缓存键名
    - `value`: 需要缓存的字符串列表

    输出:
    - 原始字符串列表

    作用:
    - 写入新闻筛选项短 TTL 内存缓存
    """

    _FILTER_CACHE[key] = {"created_at": perf_time.monotonic(), "value": value}
    return value


def _normalize_source_item(source: dict[str, Any]) -> dict[str, Any]:
    """
    输入:
    - `source`: 聚合来源中的原始字典

    输出:
    - 面向详情页展示的来源对象

    作用:
    - 兼容历史来源字段差异，避免前端反复判断 name/title/url/id。
    """

    return {
        "id": source.get("id"),
        "name": source.get("name") or source.get("source") or "未知来源",
        "title": source.get("title") or "",
        "url": source.get("url") or "",
    }


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

    cached = _get_filter_cache("sources")
    if cached is not None:
        return cached

    result = await db.execute(select(News.source).distinct())
    sources = result.scalars().all()
    return _set_filter_cache("sources", sorted([s for s in sources if s]))


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

    cached = _get_filter_cache("categories")
    if cached is not None:
        return cached

    result = await db.execute(select(News.category).distinct())
    categories = result.scalars().all()
    return _set_filter_cache("categories", sorted([c for c in categories if c]))


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

    cached = _get_filter_cache("regions")
    if cached is not None:
        return cached

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
    return _set_filter_cache("regions", sorted(valid_regions))


@router.get("/news")
async def get_news(
    q: str = "",
    page: int = 1,
    page_size: Optional[int] = None,
    date: Optional[str] = None,
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
    - `date`: 快捷时间范围（24h/3d/7d/week/month/year/all 或 YYYY-MM-DD）；关键词搜索未传时默认 all
    - `start_date`/`end_date`: 自定义起止日期（YYYY-MM-DD）
    - `sort_by`: 排序方式（heat/date）
    - `category`/`region`/`source`: 筛选条件
    - `db`: 数据库会话（依赖注入）

    输出:
    - 新闻列表与页码信息

    作用:
    - 提供新闻列表查询接口，支持筛选、排序与语义检索
    """

    has_query = bool((q or "").strip())
    effective_date = date or ("all" if has_query else "24h")
    page_size = page_size or (10 if has_query else 20)
    page_size = max(1, min(page_size, 50))
    max_page = 500 if effective_date == "all" else 1000
    page = max(1, min(page, max_page))
    offset = (page - 1) * page_size

    stmt = build_news_query_filters(
        select(News),
        date=effective_date,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
    )

    if has_query:
        search_result = await semantic_news_search(
            db,
            stmt,
            q,
            offset=offset,
            limit=page_size,
            candidate_limit=2000,
            min_score=0.2,
            log_prefix="/api/news",
        )
        data = [serialize_news_item(n) for _, n in search_result.items]
        logger.info(
            f"/api/news 搜索完成: q={q}, matched={search_result.total}, "
            f"embedding={search_result.used_embedding}, elapsed={search_result.elapsed_ms:.1f}ms"
        )
        return {"data": data, "page": page}

    if sort_by == "heat":
        stmt = stmt.options(defer(News.content), defer(News.embedding)).order_by(
            desc(News.heat_score), desc(News.publish_date)
        )
    else:
        stmt = stmt.options(defer(News.content), defer(News.embedding)).order_by(desc(News.publish_date))

    result = await db.execute(stmt.offset(offset).limit(page_size))
    data = [serialize_news_item(n) for n in result.scalars().all()]
    return {"data": data, "page": page}


@router.get("/news/top")
async def get_top_news(
    limit: int = Query(10, ge=1, le=200),
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
    - `limit`: 返回条数上限
    - `date`: 快捷时间范围（24h/3d/7d/week/month/year/all 或 YYYY-MM-DD）
    - `start_date`/`end_date`: 自定义起止日期（YYYY-MM-DD）
    - `sort_by`: 排序方式（heat/date）
    - `category`/`region`/`source`: 筛选条件
    - `db`: 数据库会话（依赖注入）

    输出:
    - TopN 新闻列表

    作用:
    - 获取指定时间范围内热度最高的新闻 TopN
    """

    stmt = build_news_query_filters(
        select(News).options(defer(News.content), defer(News.embedding)),
        date=date,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
    )

    if sort_by == "date":
        stmt = stmt.order_by(desc(News.publish_date))
    else:
        stmt = stmt.order_by(desc(News.heat_score), desc(News.publish_date))

    result = await db.execute(stmt.limit(limit))
    data = [serialize_news_item(n) for n in result.scalars().all()]
    return {"data": data, "limit": limit}


@router.get("/news/{news_id}")
async def get_news_detail(news_id: int, db: AsyncSession = Depends(get_db)):
    """
    输入:
    - `news_id`: 新闻 ID
    - `db`: 数据库会话

    输出:
    - 新闻详情、关联报道、相似新闻、相关专题和内容状态

    作用:
    - 为首页新闻详情弹层提供可复核的完整上下文。
    """

    news = await db.get(
        News,
        news_id,
        options=[defer(News.content), defer(News.embedding)],
    )
    if not news:
        raise HTTPException(status_code=404, detail="新闻不存在")

    related_sources = [_normalize_source_item(s) for s in (news.sources or []) if isinstance(s, dict)]
    if not any(s.get("url") == news.url for s in related_sources):
        related_sources.insert(
            0,
            {
                "id": news.id,
                "name": news.source or "主报道",
                "title": news.title or "",
                "url": news.url or "",
            },
        )

    return {
        "news": {
            **serialize_news_item(news),
            "is_ai_summary": bool(news.is_ai_summary),
            "crawled_at": news.crawled_at.isoformat() if news.crawled_at else None,
            "keywords": news.keywords or [],
            "entities": news.entities or [],
        },
        "related_sources": related_sources,
        "similar_news": [],
        "similar_news_deferred": True,
        "topics": [],
        "content_status": {
            "has_summary": bool(news.summary),
            "related_source_count": len(related_sources),
        },
    }


@router.get("/news/{news_id}/similar")
async def get_similar_news(news_id: int, db: AsyncSession = Depends(get_db)):
    news = await db.get(
        News,
        news_id,
        options=[defer(News.content), defer(News.embedding)],
    )
    if not news:
        raise HTTPException(status_code=404, detail="新闻不存在")

    keywords = normalize_similarity_terms(news.keywords, limit=10)
    entities = normalize_similarity_terms(news.entities, limit=8)
    keyword_keys = {kw.lower() for kw in keywords}
    query_terms = keywords + [entity for entity in entities if entity.lower() not in keyword_keys]
    if not query_terms:
        return {"data": []}

    stmt = (
        select(News)
        .options(defer(News.content))
        .where(News.id != news_id)
    )
    query_text = " ".join(query_terms)
    search_result = await semantic_news_search(
        db,
        stmt,
        query_text,
        offset=0,
        limit=10,
        candidate_limit=2000,
        min_score=0.5,
        text_terms=query_terms,
        log_prefix=f"/api/news/{news_id}/similar",
    )
    logger.info(
        f"/api/news/{news_id}/similar 语义召回完成: terms={len(query_terms)}, "
        f"matched={search_result.total}, embedding={search_result.used_embedding}, elapsed={search_result.elapsed_ms:.1f}ms"
    )

    similar_items = []
    for score, item in search_result.items:
        payload = serialize_news_item(item)
        payload["similarity_score"] = round(float(score), 4)
        similar_items.append(payload)

    return {"data": similar_items}


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

    original_summary = (news.summary or "").strip()
    main_content = (news.content or "").strip()
    summary_material = get_existing_summary_material(original_summary)
    if not main_content and not summary_material:
        main_content = await crawler_service.crawl_content(news.url) or ""
        if main_content:
            news.content = main_content

    main_input = build_summary_generation_input(
        content=main_content,
        original_summary=original_summary,
    )
    if not main_input:
        return {"summary": "抓取失败，内容过短"}

    full_content += f"【主报道】{news.title}\n{main_input}\n\n"

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

    summary = await ai_service.generate_summary(news.title, full_content)
    news.summary = summary
    news.is_ai_summary = True
    await refine_news_title_if_needed(news, summary=summary or "", content=full_content, ai=ai_service)

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
