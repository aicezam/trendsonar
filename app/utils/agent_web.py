# 本文件用于提供智能体内置网页查询工具的 URL 安全校验与轻量搜索能力。

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup


SEARCH_ENDPOINTS = (
    {"name": "duckduckgo", "url": "https://html.duckduckgo.com/html/", "params": {"kl": "wt-wt"}},
    {"name": "bing", "url": "https://www.bing.com/search", "params": {}},
)
SEARCH_TIMEOUT_SECONDS = 15
BLOCKED_WEB_HOSTS = {"localhost", "metadata.google.internal"}
DEFAULT_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
}


def _is_blocked_ip(value: str) -> bool:
    """
    输入:
    - `value`: IP 字符串

    输出:
    - 是否属于不应由智能体访问的地址

    作用:
    - 阻止网页抓取工具访问本机、内网、链路本地和保留地址，降低 SSRF 风险。
    """

    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _decode_duckduckgo_href(href: str) -> str:
    """
    输入:
    - `href`: DuckDuckGo 搜索结果链接

    输出:
    - 尽量还原后的真实 URL

    作用:
    - DuckDuckGo HTML 结果可能使用 `/l/?uddg=...` 跳转链接，这里解出原始地址。
    """

    clean = str(href or "").strip()
    if not clean:
        return ""
    parsed = urlparse(clean)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return clean


def compact_web_text(value: Any, max_chars: int = 8000) -> str:
    """
    输入:
    - `value`: 任意网页文本
    - `max_chars`: 最大字符数

    输出:
    - 清理空白并截断后的文本

    作用:
    - 控制网页工具返回给智能体的体积。
    """

    text = " ".join(str(value or "").split())
    return text[:max_chars]


async def ensure_public_web_url(url: str) -> str:
    """
    输入:
    - `url`: 待访问网页地址

    输出:
    - 校验后的 URL

    作用:
    - 仅允许智能体访问公开 http/https 地址，避免读取容器内网或本机服务。
    """

    clean_url = str(url or "").strip()
    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("网页工具 URL 必须是合法的 http/https 地址")

    host = parsed.hostname.lower()
    if host in BLOCKED_WEB_HOSTS or _is_blocked_ip(host):
        raise ValueError("网页工具禁止访问本机、内网或元数据地址")

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"网页工具无法解析目标域名: {host}") from exc
    for info in infos:
        address = info[4][0]
        if _is_blocked_ip(address):
            raise ValueError("网页工具禁止访问解析到本机或内网的地址")
    return clean_url


async def simple_web_search(q: str, *, limit: int = 5, timeout_seconds: int = SEARCH_TIMEOUT_SECONDS) -> dict[str, Any]:
    """
    输入:
    - `q`: 搜索关键词
    - `limit`: 返回数量上限
    - `timeout_seconds`: 请求超时时间

    输出:
    - 搜索结果列表，包含标题、URL 和摘要

    作用:
    - 为智能体提供无需额外配置的轻量网页查询能力；只返回搜索结果，不抓取详情页。
    """

    query = str(q or "").strip()
    if not query:
        return {"ok": False, "message": "搜索关键词不能为空", "items": []}

    safe_limit = max(1, min(int(limit or 5), 10))
    timeout = aiohttp.ClientTimeout(total=max(5, min(int(timeout_seconds or SEARCH_TIMEOUT_SECONDS), 30)))
    errors: list[str] = []
    async with aiohttp.ClientSession(headers=DEFAULT_WEB_HEADERS, timeout=timeout) as session:
        for endpoint in SEARCH_ENDPOINTS:
            source = str(endpoint["name"])
            params = {"q": query, **dict(endpoint.get("params") or {})}
            try:
                async with session.get(str(endpoint["url"]), params=params) as resp:
                    html = await resp.text(errors="ignore")
                    if resp.status >= 400:
                        errors.append(f"{source}: HTTP {resp.status}")
                        continue
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                errors.append(f"{source}: {message}")
                continue

            items = _parse_search_results(source, html, safe_limit)
            if items:
                return {"ok": True, "query": query, "source": source, "items": items, "message": ""}
            errors.append(f"{source}: 没有解析到搜索结果")

    return {
        "ok": False,
        "query": query,
        "items": [],
        "message": "；".join(errors) or "搜索请求失败",
    }


def _parse_search_results(source: str, html: str, limit: int) -> list[dict[str, str]]:
    """
    输入:
    - `source`: 搜索源名称
    - `html`: 搜索页 HTML
    - `limit`: 返回数量上限

    输出:
    - 解析后的搜索结果列表

    作用:
    - 兼容 DuckDuckGo 与 Bing 的 HTML 结果结构。
    """

    soup = BeautifulSoup(html or "", "html.parser")
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    if source == "bing":
        candidates = [(item.select_one("h2 a"), item.select_one("p")) for item in soup.select("li.b_algo")]
    else:
        candidates = [(item.select_one(".result__a"), item.select_one(".result__snippet")) for item in soup.select(".result")]

    for link, snippet_el in candidates:
        if not link:
            continue
        title = compact_web_text(link.get_text(" ", strip=True), 200)
        href = _decode_duckduckgo_href(link.get("href") or "")
        if not title or not href.startswith(("http://", "https://")) or href in seen_urls:
            continue
        snippet = compact_web_text(snippet_el.get_text(" ", strip=True) if snippet_el else "", 500)
        items.append({"title": title, "url": href, "snippet": snippet})
        seen_urls.add(href)
        if len(items) >= limit:
            break
    return items
