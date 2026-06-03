"""
本文件用于实现全网抓取与解析逻辑，包括 RSS/HTML 解析、内容抓取与入库等流程。
主要类/对象:
- `CrawlerService`: 抓取服务实现
- `crawler_service`: 全局服务单例
"""

import asyncio
import email.utils
import inspect
import json
import logging
import contextlib
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import BASE_DIR, get_settings
from app.core.database import AsyncSessionLocal
from app.core.exceptions import AIConfigurationError, AIServiceUnavailableError
from app.core.logger import setup_logger
from app.models.news import News
from app.services.ai_service import ai_service
from app.services.concurrency_service import concurrency_service
from app.services.source_health_service import source_health_service
from app.utils.tools import clean_html_tags

settings = get_settings()
logger = setup_logger("CrawlerService")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

ARTICLE_SELECTORS = (
    "article",
    "main",
    "[role='main']",
    ".article",
    ".article-content",
    ".article_content",
    ".post-content",
    ".post_content",
    ".entry-content",
    ".content",
    ".main-content",
    ".news-content",
    ".rich_media_content",
    "#article",
    "#content",
)

CONTENT_DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "form",
    "input",
    "button",
    "nav",
    "header",
    "footer",
    "aside",
    "menu",
    ".nav",
    ".navbar",
    ".footer",
    ".header",
    ".sidebar",
    ".advertisement",
    ".ads",
    ".share",
    ".related",
    ".recommend",
    ".comment",
)

CRAWL4AI_EXCLUDED_TAGS = [
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "form",
    "input",
    "button",
    "nav",
    "header",
    "footer",
    "aside",
]

MAX_LIGHT_HTML_BYTES = 2 * 1024 * 1024
MIN_CONTENT_LENGTH = 50


class CrawlerService:
    """
    输入:
    - `news_sources.json` 中的新闻源配置

    输出:
    - 抓取到的新闻元信息列表，并入库为 `News` 记录

    作用:
    - 负责从 RSS/API/网页等来源抓取新闻，并将结果写入数据库
    """

    def __init__(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 初始化新闻源配置文件名
        """

        self.sources_file = "news_sources.json"

    def _default_headers(self, **overrides: str) -> Dict[str, str]:
        headers = dict(DEFAULT_HEADERS)
        headers.update({k: v for k, v in overrides.items() if v is not None})
        return headers

    def _decode_response_body(self, body: bytes, content_type: str = "") -> str:
        charset_match = re.search(r"charset=([\w.-]+)", content_type or "", flags=re.I)
        encodings = []
        if charset_match:
            encodings.append(charset_match.group(1))
        encodings.extend(["utf-8", "gb18030"])

        for encoding in encodings:
            try:
                return body.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return body.decode("utf-8", errors="replace")

    def _content_min_length(self) -> int:
        """
        输入:
        - 无

        输出:
        - 当前配置下可接受的正文最小长度

        作用:
        - 允许线上按站点特性调低动态页短文本的有效阈值，避免过早判空。
        """

        return max(10, int(getattr(settings, "CRAWLER_CONTENT_MIN_LENGTH", MIN_CONTENT_LENGTH) or MIN_CONTENT_LENGTH))

    def _normalize_content_text(
        self,
        text: str,
        max_length: int = 60000,
        min_length: Optional[int] = None,
    ) -> Optional[str]:
        if min_length is None:
            min_length = self._content_min_length()
        cleaned = clean_html_tags(text)
        cleaned = re.sub(r"(\S)\s+([，。！？；：、）】》])", r"\1\2", cleaned)
        cleaned = re.sub(r"([（【《])\s+(\S)", r"\1\2", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if len(cleaned) < min_length:
            return None
        return cleaned[:max_length]

    def _extract_article_text(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")

        for selector in CONTENT_DROP_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        candidates = []
        seen = set()
        for selector in ARTICLE_SELECTORS:
            for node in soup.select(selector):
                node_id = id(node)
                if node_id in seen:
                    continue
                seen.add(node_id)
                text = node.get_text(separator="\n", strip=True)
                text_len = len(clean_html_tags(text))
                if text_len >= MIN_CONTENT_LENGTH:
                    candidates.append((text_len, text))

        if not candidates and soup.body:
            text = soup.body.get_text(separator="\n", strip=True)
            text_len = len(clean_html_tags(text))
            if text_len >= MIN_CONTENT_LENGTH:
                candidates.append((text_len, text))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return self._normalize_content_text(candidates[0][1])

    async def crawl_content_light(self, target_url: str) -> Optional[str]:
        """
        以 HTTP + HTML 解析方式优先抓正文，避免大多数静态新闻页启动浏览器。
        """
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(headers=self._default_headers(), timeout=timeout) as session:
                async with session.get(target_url, allow_redirects=True) as resp:
                    if resp.status != 200:
                        return None

                    content_type = resp.headers.get("Content-Type", "")
                    if content_type and not any(t in content_type.lower() for t in ("html", "xml", "text")):
                        return None

                    body = await resp.content.read(MAX_LIGHT_HTML_BYTES + 1)
                    if len(body) > MAX_LIGHT_HTML_BYTES:
                        logger.debug(f"   ⚠️ [轻量抓取] 页面过大，回退浏览器: {target_url}")
                        return None

                    html = self._decode_response_body(body, content_type)
                    return self._extract_article_text(html)
        except Exception as e:
            logger.debug(f"   ⚠️ [轻量抓取] 失败，回退浏览器: {target_url} ({e})")
            return None

    def _make_browser_config(self):
        from crawl4ai import BrowserConfig

        log_level = getattr(logging, (settings.LOG_LEVEL or "").upper(), logging.INFO)
        is_verbose = log_level == logging.DEBUG
        return self._build_supported_config(
            BrowserConfig,
            {
                "headless": True,
                "verbose": is_verbose,
                "text_mode": True,
                "light_mode": True,
            },
        )

    def _browser_page_timeout_ms(self, wait_seconds: float = 0.0) -> int:
        """
        输入:
        - `wait_seconds`: 页面导航后还需要额外等待的秒数

        输出:
        - 浏览器页面导航超时时间，单位毫秒

        作用:
        - 让浏览器内部导航超时早于外层正文补抓硬超时，避免外层取消时页面仍在 goto 导致 TargetClosedError 噪音。
        """

        configured_timeout = max(10000, int(getattr(settings, "CRAWLER_PAGE_TIMEOUT_MS", 60000) or 60000))
        fetch_timeout = float(getattr(settings, "CRAWLER_FETCH_TIMEOUT_SECONDS", 45.0) or 45.0)
        if fetch_timeout <= 0:
            return configured_timeout

        reserved_seconds = max(2.0, wait_seconds + 2.0)
        capped_seconds = max(5.0, fetch_timeout - reserved_seconds)
        return min(configured_timeout, int(capped_seconds * 1000))

    def _make_run_config(self, *, dynamic_wait: bool = False):
        """
        输入:
        - `dynamic_wait`: 是否启用动态页慢速等待配置

        输出:
        - 当前 crawl4ai 版本支持的运行配置

        作用:
        - 快速抓取失败后，可用更长等待时间重新读取动态渲染页面，降低空壳页面误判率。
        """

        from crawl4ai import CacheMode, CrawlerRunConfig

        log_level = getattr(logging, (settings.LOG_LEVEL or "").upper(), logging.INFO)
        is_verbose = log_level == logging.DEBUG
        fast_wait = max(0.0, float(getattr(settings, "CRAWLER_FAST_WAIT_SECONDS", 1.5) or 0.0))
        dynamic_wait_seconds = max(fast_wait, float(getattr(settings, "CRAWLER_DYNAMIC_WAIT_SECONDS", 6.0) or 6.0))
        wait_seconds = dynamic_wait_seconds if dynamic_wait else fast_wait
        page_timeout = self._browser_page_timeout_ms(wait_seconds)
        return self._build_supported_config(
            CrawlerRunConfig,
            {
                "cache_mode": CacheMode.BYPASS,
                "verbose": is_verbose,
                "only_text": True,
                "excluded_tags": CRAWL4AI_EXCLUDED_TAGS,
                "remove_forms": True,
                "exclude_external_images": True,
                "exclude_all_images": True,
                "exclude_external_links": True,
                "exclude_social_media_links": True,
                "wait_until": "domcontentloaded",
                "page_timeout": page_timeout,
                "delay_before_return_html": wait_seconds,
                "scan_full_page": dynamic_wait,
                "wait_for_images": False,
                "screenshot": False,
                "pdf": False,
            },
        )

    def _extract_crawl4ai_markdown(self, result, *, min_length: Optional[int] = None) -> Optional[str]:
        if not result or not result.markdown:
            return None
        if hasattr(result.markdown, "raw_markdown"):
            return self._normalize_content_text(result.markdown.raw_markdown, min_length=min_length)
        return self._normalize_content_text(str(result.markdown), min_length=min_length)

    async def _crawl_content_with_playwright(self, target_url: str) -> Optional[str]:
        """
        输入:
        - `target_url`: 目标新闻页面 URL

        输出:
        - Playwright 直接渲染后提取到的正文；失败返回 None

        作用:
        - 作为 crawl4ai 失败后的兜底抓取路径，处理部分站点在 crawl4ai 中导航超时或正文为空的问题。
        """

        try:
            from playwright.async_api import async_playwright

            wait_seconds = max(
                1.0,
                float(getattr(settings, "CRAWLER_DYNAMIC_WAIT_SECONDS", 6.0) or 6.0),
            )
            timeout_ms = self._browser_page_timeout_ms(wait_seconds)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=DEFAULT_HEADERS["User-Agent"],
                    locale="zh-CN",
                    extra_http_headers={
                        "Accept": DEFAULT_HEADERS["Accept"],
                        "Accept-Language": DEFAULT_HEADERS["Accept-Language"],
                    },
                )
                page = await context.new_page()
                try:
                    await page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_ms)
                    await page.wait_for_timeout(int(wait_seconds * 1000))
                    html = await page.content()
                    return self._extract_article_text(html)
                finally:
                    await context.close()
                    await browser.close()
        except Exception as e:
            logger.debug(f"   ⚠️ [Playwright兜底] 抓取失败: {target_url} ({e})")
            return None

    def _build_supported_config(self, config_cls, values: Dict[str, Any]):
        """
        Crawl4AI 配置项在不同版本间会变化，只传当前构造器支持的参数。
        """
        try:
            signature = inspect.signature(config_cls)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            if not accepts_kwargs:
                values = {k: v for k, v in values.items() if k in signature.parameters}
        except (TypeError, ValueError):
            pass
        return config_cls(**values)

    def _get_sources_path_candidates(self) -> List[Path]:
        """
        输入:
        - 无

        输出:
        - 可能的新闻源配置文件路径列表（按优先级）

        作用:
        - 兼容 `data/` 与项目根目录两种放置方式
        """

        return [
            BASE_DIR / "data" / self.sources_file,
            BASE_DIR / self.sources_file,
        ]

    def load_sources(self) -> List[Dict[str, Any]]:
        """
        输入:
        - 无

        输出:
        - 新闻源配置列表（每项包含 name/weight/address）

        作用:
        - 从配置文件读取新闻源，为抓取流程提供输入
        """

        try:
            path = None
            for candidate in self._get_sources_path_candidates():
                if candidate.exists():
                    path = candidate
                    break

            if path is None:
                logger.warning("未找到新闻源文件")
                return []

            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载新闻源失败: {e}")
            return []

    def _clean_summary(self, summary: Optional[str]) -> Optional[str]:
        """
        输入:
        - `summary`: 原始摘要（可能包含 HTML）

        输出:
        - 清洗后的摘要

        作用:
        - 移除无法访问的内部图片链接
        """
        if not summary:
            return summary
            
        # 优先使用 clean_html_tags 进行彻底清洗
        return clean_html_tags(summary)


    def _process_meta(
        self,
        source_name: str,
        weight: float,
        title: Optional[str],
        url: Optional[str],
        pub_date: datetime,
        summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        输入:
        - `source_name`: 来源名称
        - `weight`: 来源权重
        - `title`: 标题
        - `url`: 链接
        - `pub_date`: 发布时间
        - `summary`: 摘要（可选）

        输出:
        - 清洗后的新闻元信息字典；不合规时返回 None

        作用:
        - 统一做空值、域名黑名单过滤与字段标准化
        """

        if not url or not title:
            return None
        for domain in settings.IGNORED_DOMAINS:
            if domain in url:
                return None
        
        # 清洗摘要中的坏链
        cleaned_summary = self._clean_summary(summary)

        return {
            "title": title.strip(),
            "url": url.strip(),
            "source": source_name,
            "publish_date": pub_date,
            "heat": weight,
            "summary": cleaned_summary,
        }

    async def fetch_and_parse(
        self, session: aiohttp.ClientSession, source: Dict[str, Any], prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        输入:
        - `session`: 复用的 HTTP 会话
        - `source`: 单个新闻源配置
        - `prefix`: 日志前缀（用于并发进度展示）

        输出:
        - 当前新闻源抓取到的新闻元信息列表

        作用:
        - 兼容 JSON API、RSS/XML，并在失败时尝试用 AI 从页面内容中抽取条目
        """

        url = source.get("address")
        name = source.get("name")
        weight = source.get("weight", 1.0)

        log_prefix = f"   {prefix}" if prefix else ""
        logger.debug(f"{log_prefix} 正在抓取: {name} ({url})")

        try:
            async with session.get(url, timeout=20) as resp:
                if resp.status != 200:
                    logger.error(f"抓取失败 {name}: HTTP {resp.status}")
                    return []

                content_type = resp.headers.get("Content-Type", "").lower()
                try:
                    text = await resp.text()
                except UnicodeDecodeError:
                    raw = await resp.read()
                    text = raw.decode("utf-8", errors="ignore")

                items = []

                if "json" in content_type:
                    try:
                        data = json.loads(text)
                        raw_items = []
                        if isinstance(data, list):
                            raw_items = data
                        elif isinstance(data, dict):
                            raw_items = data.get("data", []) or data.get("items", []) or data.get("stories", [])

                        for item in raw_items:
                            title = item.get("title")
                            link = item.get("url") or item.get("link") or item.get("share_url")
                            summary = item.get("summary") or item.get("description") or item.get("digest")
                            pub_date = datetime.now()

                            p = self._process_meta(name, weight, title, link, pub_date, summary)
                            if p:
                                items.append(p)

                        if items:
                            pass
                    except Exception as e:
                        logger.warning(f"JSON解析失败 {name}: {e}")

                if not items:
                    try:
                        features = "xml"
                        try:
                            import lxml  # noqa: F401

                            features = "lxml-xml"
                        except ImportError:
                            pass

                        soup = BeautifulSoup(text, features)
                        rss_items = soup.find_all("item") or soup.find_all("entry")
                        if rss_items:
                            for item in rss_items:
                                title_elem = item.find("title")
                                title = title_elem.text if title_elem else ""

                                link_elem = item.find("link")
                                link = link_elem.text if link_elem else ""
                                if not link and link_elem and link_elem.get("href"):
                                    link = link_elem.get("href")

                                desc_elem = item.find("content:encoded") or item.find("content")
                                if not desc_elem:
                                    desc_elem = item.find("description") or item.find("summary")

                                summary = desc_elem.text if desc_elem else ""

                                pub_date = datetime.now()
                                date_elem = item.find("pubDate") or item.find("dc:date") or item.find("updated")
                                if date_elem and date_elem.text:
                                    try:
                                        parsed_tuple = email.utils.parsedate_tz(date_elem.text)
                                        if parsed_tuple:
                                            ts = email.utils.mktime_tz(parsed_tuple)
                                            pub_date = datetime.fromtimestamp(ts)
                                        else:
                                            dt_str = date_elem.text.replace("Z", "+00:00")
                                            pub_date = datetime.fromisoformat(dt_str)
                                    except Exception:
                                        pass

                                p = self._process_meta(name, weight, title, link, pub_date, summary)
                                if p:
                                    items.append(p)
                            if items:
                                source_health_service.record_fetch_result(
                                    str(url or name or ""),
                                    name=str(name or ""),
                                    count=len(items),
                                    ok=True,
                                    error=None,
                                )
                                return items
                    except Exception as e:
                        logger.warning(f"RSS/XML解析失败 {name}: {e}")

                if not items:
                    logger.info(f"常规解析失败，尝试 AI 提取: {name}")
                    extracted_items = await ai_service.extract_news_info(text)
                    for item in extracted_items:
                        p = self._process_meta(
                            name, weight, item.get("title"), item.get("link"), datetime.now(), item.get("summary")
                        )
                        if p:
                            items.append(p)

                source_health_service.record_fetch_result(
                    str(url or name or ""),
                    name=str(name or ""),
                    count=len(items),
                    ok=bool(items),
                    error=None if items else "未提取到新闻条目",
                )
                return items

        except (AIConfigurationError, AIServiceUnavailableError):
            raise
        except Exception as e:
            logger.error(f"抓取异常 {name}: {e}")
            source_health_service.record_fetch_result(
                str(url or name or ""),
                name=str(name or ""),
                count=0,
                ok=False,
                error=str(e),
            )
            return []

    async def fetch_all_sources(self) -> List[Dict[str, Any]]:
        """
        输入:
        - 无

        输出:
        - 全部新闻源抓取到的新闻元信息列表

        作用:
        - 并发抓取多个新闻源并汇总结果
        """

        logger.info("🕷️ 开始全网抓取 (基于配置源)...")
        sources = self.load_sources()
        if not sources:
            logger.warning("没有加载到任何新闻源，请检查 news_sources.json")
            return []

        all_news = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        }

        connector = aiohttp.TCPConnector(limit=50)
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = []
            total_sources = len(sources)

            async def fetch_wrapper(idx, src):
                return await self.fetch_and_parse(session, src, prefix=f"({idx}/{total_sources})")

            for i, src in enumerate(sources, 1):
                if src.get("enabled", True):
                    tasks.append(fetch_wrapper(i, src))

            results = await asyncio.gather(*tasks)
            for res in results:
                all_news.extend(res)

        # 显式 GC
        import gc
        gc.collect()
        
        return all_news

    async def save_raw_news(self, news_list: List[Dict[str, Any]]) -> None:
        """
        输入:
        - `news_list`: 抓取到的新闻元信息列表

        输出:
        - 无

        作用:
        - 按关键词过滤/去重后，将新闻写入数据库，并保证 URL 唯一
        """

        if not news_list:
            return

        follow_keywords = settings.FOLLOW_KEYWORDS
        final_list = news_list

        if follow_keywords:
            keywords = [k.strip() for k in follow_keywords.split(",") if k.strip()]
            if keywords:
                logger.info(f"🔍 开始关键词过滤 (关键词: {keywords})")

                titles = [item["title"] for item in news_list]
                try:
                    title_embeddings = await ai_service.get_embeddings(titles)
                    keyword_embeddings = await ai_service.get_embeddings(keywords)

                    if title_embeddings and keyword_embeddings:
                        filtered_list = []
                        import numpy as np

                        kw_vecs = np.array(keyword_embeddings)
                        kw_norms = np.linalg.norm(kw_vecs, axis=1, keepdims=True)
                        kw_vecs_norm = kw_vecs / (kw_norms + 1e-9)

                        for i, item in enumerate(news_list):
                            if not title_embeddings[i]:
                                continue

                            t_vec = np.array(title_embeddings[i])
                            t_norm = np.linalg.norm(t_vec)
                            if t_norm == 0:
                                continue
                            t_vec_norm = t_vec / t_norm

                            sims = np.dot(t_vec_norm, kw_vecs_norm.T)
                            max_sim = float(np.max(sims))

                            if max_sim >= settings.FOLLOW_KEYWORDS_THRESHOLD:
                                filtered_list.append(item)

                        logger.info(f"✅ 过滤完成: {len(news_list)} -> {len(filtered_list)}")
                        final_list = filtered_list
                    else:
                        logger.warning("⚠️ 向量计算失败，跳过关键词过滤")
                except Exception as e:
                    logger.error(f"❌ 关键词过滤异常: {e}")

        urls = [item["url"] for item in final_list]
        existing_urls = set()
        async with AsyncSessionLocal() as db:
            if urls:
                result = await db.execute(select(News.url).where(News.url.in_(urls)))
                existing_urls = set(r for r in result.scalars())

        final_list = [item for item in final_list if item["url"] not in existing_urls]

        titles = [item["title"] for item in final_list]
        existing_pairs = set()
        if titles:
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(News.title, News.source).where(News.title.in_(titles)))
                for row in result:
                    existing_pairs.add((row.title, row.source))

        original_count = len(final_list)
        final_list = [item for item in final_list if (item["title"], item["source"]) not in existing_pairs]
        skipped_by_title = original_count - len(final_list)
        if skipped_by_title > 0:
            logger.debug(f"   🚫 因标题和来源重复跳过 {skipped_by_title} 条")

        async with AsyncSessionLocal() as db:
            count = 0
            for item in final_list:
                stmt = insert(News).values(
                    url=item["url"],
                    title=item["title"],
                    source=item["source"],
                    publish_date=item["publish_date"],
                    heat_score=item["heat"],
                    summary=item.get("summary", ""),
                    sources=[{"name": item["source"], "url": item["url"]}],
                    sentiment_score=item.get("sentiment_score", 50.0),
                    sentiment_label=item.get("sentiment_label", "中立"),
                    keywords=item.get("keywords", []),
                    entities=item.get("entities", []),
                )

                stmt = stmt.on_conflict_do_nothing(index_elements=["url"])
                res = await db.execute(stmt)
                if res.rowcount > 0:
                    count += 1
            await db.commit()
            logger.info(f"📥 入库新增 {count} 条")

    def _is_weibo_url(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(domain in host for domain in ("weibo.com", "weibo.cn"))

    def _parse_cookie_header(self, cookie_str: str) -> Dict[str, str]:
        cookies: Dict[str, str] = {}
        for part in (cookie_str or "").split(";"):
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            name = name.strip()
            value = value.strip()
            if name:
                cookies[name] = value
        return cookies

    def _format_cookie_header(self, cookies: Dict[str, str]) -> str:
        return "; ".join(f"{name}={value}" for name, value in cookies.items() if name and value)

    def _save_weibo_cookie(self, cookie_str: str) -> None:
        from app.core.config import CONFIG_PATH

        raw_text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
        cookie_line = f"WEIBO_COOKIE: {json.dumps(cookie_str, ensure_ascii=False)}"

        if raw_text.lstrip().startswith("{"):
            data = json.loads(raw_text or "{}")
            data["WEIBO_COOKIE"] = cookie_str
            CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return

        if re.search(r"(?m)^WEIBO_COOKIE\s*:", raw_text):
            raw_text = re.sub(r"(?m)^WEIBO_COOKIE\s*:.*$", cookie_line, raw_text, count=1)
        else:
            raw_text = raw_text.rstrip() + "\n\n" + cookie_line + "\n"
        CONFIG_PATH.write_text(raw_text, encoding="utf-8")

    def _merge_and_save_weibo_cookies(self, response_cookies) -> None:
        current_cookie = (settings.WEIBO_COOKIE or "").strip()
        if not current_cookie or current_cookie.startswith("Example:"):
            return

        merged = self._parse_cookie_header(current_cookie)
        changed = False
        for cookie in response_cookies.values():
            name = getattr(cookie, "key", None)
            value = getattr(cookie, "value", None)
            if not name or not value:
                continue
            if merged.get(name) != value:
                merged[name] = value
                changed = True

        if not changed:
            return

        new_cookie = self._format_cookie_header(merged)
        if not new_cookie:
            return

        try:
            self._save_weibo_cookie(new_cookie)
            settings.WEIBO_COOKIE = new_cookie
            logger.info("💾 [微博抓取] 已根据成功响应更新 WEIBO_COOKIE")
        except Exception as e:
            logger.warning(f"   ⚠️ [微博抓取] 更新 WEIBO_COOKIE 失败: {e}")

    async def crawl_weibo_content(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        return await self.crawl_weibo_simple(session, url)

    async def crawl_weibo_simple(self, session: aiohttp.ClientSession, url: str, retry: bool = True) -> Optional[str]:
        """
        输入:
        - `session`: HTTP 会话
        - `url`: 微博详情页链接
        - `retry`: 是否在失败时尝试刷新 Cookie 并重试

        输出:
        - 抓取到的正文文本；失败返回 None

        作用:
        - 以轻量方式抓取微博内容，避免重型渲染带来的开销
        """

        cookie = (settings.WEIBO_COOKIE or "").strip()
        if not cookie or cookie.startswith("Example:"):
            logger.warning("   ⚠️ [微博抓取] 未配置有效 WEIBO_COOKIE，请在管理页填写登录后的完整 Cookie")
            return None

        headers = self._default_headers(
            Accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            Cookie=cookie,
        )

        logger.debug(f"   🔍 [微博抓取] 正在抓取: {url}")
        try:
            async with session.get(url, headers=headers, timeout=30) as resp:
                final_url = str(resp.url)
                if resp.status != 200:
                    logger.error(f"   ❌ [微博抓取] HTTP {resp.status}")
                    return None

                if "passport.weibo.com" in final_url or "/newlogin" in final_url:
                    logger.warning("   ⚠️ [微博抓取] 跳转到微博登录入口，请更新 WEIBO_COOKIE")
                    return None

                content_bytes = await resp.read()
                html = self._decode_response_body(content_bytes, resp.headers.get("Content-Type", ""))

                if "Sina Visitor System" in html or "访问受限" in html:
                    logger.warning("   ⚠️ [微博抓取] 触发访客/访问限制页面，请更新 WEIBO_COOKIE")
                    return None

                soup = BeautifulSoup(html, "html.parser")

                cards = soup.select("div.card-wrap")
                content_list = []

                for card in cards:
                    txt_p = card.select_one("p.txt")
                    if txt_p:
                        text = txt_p.get_text(separator=" ", strip=True)
                        content_list.append(text)

                if not content_list:
                    logger.warning("   ⚠️ [微博抓取] 未找到微博卡片，请检查 WEIBO_COOKIE 是否有效")
                    body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
                    if body_text:
                        logger.debug(f"   ⚠️ [微博抓取] 无卡片页面文本长度: {len(body_text)}")
                    return None

                self._merge_and_save_weibo_cookies(resp.cookies)
                logger.debug(f"   ✅ [微博抓取] 抓取到 {len(content_list)} 条微博")
                return "\n\n".join(content_list)

        except Exception as e:
            logger.error(f"   ❌ [微博抓取] 异常: {e}")
            return None

    @contextlib.asynccontextmanager
    async def make_crawler(self):
        """
        输入:
        - 无

        输出:
        - AsyncWebCrawler 实例上下文

        作用:
        - 创建并管理爬虫实例的生命周期，支持批量复用
        """
        try:
            from crawl4ai import AsyncWebCrawler

            browser_conf = self._make_browser_config()
            
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                yield crawler

        except Exception as e:
            logger.error(f"❌ 爬虫初始化失败: {e}")
            yield None

    async def crawl_content_with_instance(self, target_url: str, crawler) -> Optional[str]:
        """
        输入:
        - `target_url`: 目标页面 URL
        - `crawler`: 复用的爬虫实例

        输出:
        - 页面正文/Markdown 文本；失败返回 None

        作用:
        - 使用复用的爬虫实例抓取内容，减少浏览器启动开销
        """
        if not crawler:
            return await self.crawl_content(target_url)

        async def do_crawl() -> Optional[str]:
            logger.debug(f"抓取新闻 (复用实例): {target_url}")

            if self._is_weibo_url(target_url):
                try:
                    async with aiohttp.ClientSession() as session:
                        return await self.crawl_weibo_content(session, target_url)
                except Exception as e:
                    logger.error(f"❌ 微博抓取失败: {e}")
                    return None

            light_content = await self.crawl_content_light(target_url)
            if light_content:
                return light_content

            try:
                run_conf = self._make_run_config(dynamic_wait=False)
                result = await crawler.arun(url=target_url, config=run_conf)
                content = self._extract_crawl4ai_markdown(result)
                if content:
                    return content

                logger.debug(f"   ⚠️ [浏览器抓取] 快速模式未获得正文，切换动态等待: {target_url}")
                dynamic_run_conf = self._make_run_config(dynamic_wait=True)
                dynamic_result = await crawler.arun(url=target_url, config=dynamic_run_conf)
                content = self._extract_crawl4ai_markdown(dynamic_result)
                if content:
                    return content

                logger.debug(f"   ⚠️ [浏览器抓取] crawl4ai 未获得正文，切换 Playwright 兜底: {target_url}")
                return await self._crawl_content_with_playwright(target_url)

            except Exception as e:
                logger.error(f"❌ 抓取失败: {e}")
                logger.debug(f"   ⚠️ [浏览器抓取] crawl4ai 异常，切换 Playwright 兜底: {target_url}")
                return await self._crawl_content_with_playwright(target_url)

        return await concurrency_service.run_crawler(do_crawl)

    async def crawl_content(self, target_url: str) -> Optional[str]:
        """
        输入:
        - `target_url`: 目标页面 URL

        输出:
        - 页面正文/Markdown 文本；失败返回 None

        作用:
        - 抓取新闻正文，用于摘要生成与深度分析
        """
        async def do_crawl() -> Optional[str]:
            logger.debug(f"抓取新闻: {target_url}")

            if self._is_weibo_url(target_url):
                try:
                    async with aiohttp.ClientSession() as session:
                        return await self.crawl_weibo_content(session, target_url)
                except Exception as e:
                    logger.error(f"❌ 微博抓取失败: {e}")
                    return None

            light_content = await self.crawl_content_light(target_url)
            if light_content:
                return light_content

            try:
                from crawl4ai import AsyncWebCrawler

                browser_conf = self._make_browser_config()
                run_conf = self._make_run_config(dynamic_wait=False)

                async with AsyncWebCrawler(config=browser_conf) as crawler:
                    result = await crawler.arun(url=target_url, config=run_conf)
                    content = self._extract_crawl4ai_markdown(result)
                    if content:
                        return content

                    logger.debug(f"   ⚠️ [浏览器抓取] 快速模式未获得正文，切换动态等待: {target_url}")
                    dynamic_run_conf = self._make_run_config(dynamic_wait=True)
                    dynamic_result = await crawler.arun(url=target_url, config=dynamic_run_conf)
                    content = self._extract_crawl4ai_markdown(dynamic_result)
                    if content:
                        return content

                    logger.debug(f"   ⚠️ [浏览器抓取] crawl4ai 未获得正文，切换 Playwright 兜底: {target_url}")
                    return await self._crawl_content_with_playwright(target_url)

            except Exception as e:
                logger.error(f"❌ 抓取失败: {e}")
                logger.debug(f"   ⚠️ [浏览器抓取] crawl4ai 异常，切换 Playwright 兜底: {target_url}")
                return await self._crawl_content_with_playwright(target_url)

        return await concurrency_service.run_crawler(do_crawl)


crawler_service = CrawlerService()
