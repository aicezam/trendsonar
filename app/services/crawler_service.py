"""
本文件用于实现全网抓取与解析逻辑，包括 RSS/HTML 解析、内容抓取与入库等流程。
主要类/对象:
- `CrawlerService`: 抓取服务实现
- `crawler_service`: 全局服务单例
"""

import asyncio
import email.utils
import json
import logging
import contextlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import BASE_DIR, get_settings
from app.core.database import AsyncSessionLocal
from app.core.logger import setup_logger
from app.models.news import News
from app.services.ai_service import ai_service
from app.utils.tools import clean_html_tags

settings = get_settings()
logger = setup_logger("CrawlerService")


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

                return items

        except Exception as e:
            logger.error(f"抓取异常 {name}: {e}")
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

    async def _refresh_weibo_cookie(self) -> Optional[str]:
        """
        自动刷新微博访客 Cookie
        """
        logger.info("🔄 正在尝试自动刷新微博 Cookie...")
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                
                # 访问微博搜索页面，触发访客认证
                try:
                    await page.goto("https://s.weibo.com/weibo?q=Python", timeout=30000)
                    await page.wait_for_load_state("networkidle")
                except Exception as e:
                    logger.warning(f"页面加载超时或出错，尝试直接获取Cookie: {e}")

                cookies = await context.cookies()
                await browser.close()
                
                # 提取并拼接 Cookie
                cookie_list = [f"{c['name']}={c['value']}" for c in cookies]
                cookie_str = "; ".join(cookie_list)
                
                if "SUB=" in cookie_str:
                    logger.info("✅ 微博 Cookie 刷新成功")
                    # 更新内存中的配置
                    settings.WEIBO_COOKIE = cookie_str

                    # 尝试持久化到 config.yaml
                    try:
                        from app.utils.config_io import load_yaml_dict, dump_yaml_text, save_yaml_text
                        from app.core.config import CONFIG_PATH
                        
                        config_data = load_yaml_dict(CONFIG_PATH)
                        config_data["WEIBO_COOKIE"] = cookie_str
                        save_yaml_text(CONFIG_PATH, dump_yaml_text(config_data))
                        logger.info("💾 微博 Cookie 已保存到 config.yaml")
                    except Exception as e:
                        logger.error(f"❌ 保存 Cookie 到配置文件失败: {e}")

                    return cookie_str
                else:
                    logger.warning("⚠️ 微博 Cookie 刷新失败: 未找到 SUB 字段")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ 微博 Cookie 刷新异常: {e}")
            return None

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

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cookie": settings.WEIBO_COOKIE,
        }

        logger.debug(f"   🔍 [微博抓取] 正在抓取: {url}")
        try:
            async with session.get(url, headers=headers, timeout=30) as resp:
                if resp.status != 200:
                    logger.error(f"   ❌ [微博抓取] HTTP {resp.status}")
                    return None

                content_bytes = await resp.read()
                try:
                    html = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        html = content_bytes.decode("gb18030")
                    except UnicodeDecodeError:
                        html = content_bytes.decode("utf-8", errors="replace")

                soup = BeautifulSoup(html, "html.parser")

                cards = soup.select("div.card-wrap")
                content_list = []

                for card in cards:
                    txt_p = card.select_one("p.txt")
                    if txt_p:
                        text = txt_p.get_text(separator=" ", strip=True)
                        content_list.append(text)

                if not content_list:
                    logger.warning("   ⚠️ [微博抓取] 未找到微博卡片，尝试提取全文")
                    body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
                    if "Sina Visitor System" in body_text or "访问受限" in body_text:
                        logger.error("   ❌ [微博抓取] 触发反爬验证")
                        
                        if retry:
                            new_cookie = await self._refresh_weibo_cookie()
                            if new_cookie:
                                return await self.crawl_weibo_simple(session, url, retry=False)

                        return None
                    return body_text[:5000]

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
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

            log_level = getattr(logging, (settings.LOG_LEVEL or "").upper(), logging.INFO)
            is_verbose = log_level == logging.DEBUG
            
            browser_conf = BrowserConfig(headless=True, verbose=is_verbose)
            
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

        logger.debug(f"抓取新闻 (复用实例): {target_url}")

        if "weibo.com" in target_url or "weibo.cn" in target_url:
            try:
                async with aiohttp.ClientSession() as session:
                    return await self.crawl_weibo_simple(session, target_url)
            except Exception as e:
                logger.error(f"❌ 微博抓取失败: {e}")
                return None

        try:
            from crawl4ai import CacheMode, CrawlerRunConfig

            log_level = getattr(logging, (settings.LOG_LEVEL or "").upper(), logging.INFO)
            is_verbose = log_level == logging.DEBUG

            run_conf = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                verbose=is_verbose
            )

            result = await crawler.arun(url=target_url, config=run_conf)

            if result and result.markdown:
                if hasattr(result.markdown, "raw_markdown"):
                    return clean_html_tags(result.markdown.raw_markdown)
                return clean_html_tags(str(result.markdown))

        except Exception as e:
            logger.error(f"❌ 抓取失败: {e}")
        return None

    async def crawl_content(self, target_url: str) -> Optional[str]:
        """
        输入:
        - `target_url`: 目标页面 URL

        输出:
        - 页面正文/Markdown 文本；失败返回 None

        作用:
        - 抓取新闻正文，用于摘要生成与深度分析
        """

        logger.debug(f"抓取新闻: {target_url}")

        if "weibo.com" in target_url or "weibo.cn" in target_url:
            try:
                async with aiohttp.ClientSession() as session:
                    return await self.crawl_weibo_simple(session, target_url)
            except Exception as e:
                logger.error(f"❌ 微博抓取失败: {e}")
                return None

        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

            log_level = getattr(logging, (settings.LOG_LEVEL or "").upper(), logging.INFO)
            is_verbose = log_level == logging.DEBUG
            
            browser_conf = BrowserConfig(headless=True, verbose=is_verbose)
            run_conf = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                verbose=is_verbose
            )

            async with AsyncWebCrawler(config=browser_conf) as crawler:
                result = await crawler.arun(url=target_url, config=run_conf)

            if result and result.markdown:
                if hasattr(result.markdown, "raw_markdown"):
                    return clean_html_tags(result.markdown.raw_markdown)
                return clean_html_tags(str(result.markdown))

        except Exception as e:
            logger.error(f"❌ 抓取失败: {e}")
        return None


crawler_service = CrawlerService()