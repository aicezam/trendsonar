"""
本文件用于提供统一的爬虫、Embedding 与 LLM 并发预算，避免后台任务和接口同时打满外部资源。
主要类/对象:
- `ConcurrencyService`: 全局并发预算管理器
- `concurrency_service`: 服务单例
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from app.core.config import get_settings

T = TypeVar("T")


class ConcurrencyService:
    """
    输入:
    - 系统配置中的并发上限

    输出:
    - 受统一信号量保护的异步调用结果

    作用:
    - 为爬虫、Embedding 和 LLM 调用提供全局并发预算，降低外部 API、网络与数据库压力峰值
    """

    def __init__(self) -> None:
        settings = get_settings()
        crawler_limit = max(1, getattr(settings, "CRAWLER_CONCURRENCY", 2))
        embedding_limit = max(1, getattr(settings, "EMBEDDING_CONCURRENCY", 5))
        llm_limit = max(1, getattr(settings, "LLM_CONCURRENCY", 5))
        self.crawler_sem = asyncio.Semaphore(crawler_limit)
        self.embedding_sem = asyncio.Semaphore(embedding_limit)
        self.llm_sem = asyncio.Semaphore(llm_limit)

    async def run_crawler(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        输入:
        - `func`: 需要执行的爬虫异步函数

        输出:
        - 爬虫函数执行结果

        作用:
        - 统一限制正文补抓和新闻源抓取类任务的并发数量
        """

        async with self.crawler_sem:
            return await func()

    async def run_embedding(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        输入:
        - `func`: 需要执行的 Embedding 异步函数

        输出:
        - Embedding 函数执行结果

        作用:
        - 统一限制向量化请求并发数量，减少外部 embedding 服务压力
        """

        async with self.embedding_sem:
            return await func()

    async def run_llm(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        输入:
        - `func`: 需要执行的 LLM 异步函数

        输出:
        - LLM 函数执行结果

        作用:
        - 统一限制摘要、情感分析、报告和专题等 LLM 调用并发数量
        """

        async with self.llm_sem:
            return await func()


concurrency_service = ConcurrencyService()
