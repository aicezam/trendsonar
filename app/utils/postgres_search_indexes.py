"""
本文件用于集中维护 PostgreSQL 新闻搜索索引 SQL 和执行工具。
主要函数:
- `ensure_postgres_search_indexes`: 在已有连接上创建搜索索引
- `optimize_postgres_search_indexes`: 通过数据库连接串执行并发建索引
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

POSTGRES_SEARCH_INDEX_SQL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE INDEX IF NOT EXISTS ix_news_title_trgm ON news USING gin (title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_news_summary_trgm ON news USING gin (summary gin_trgm_ops) WHERE summary IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_news_source_trgm ON news USING gin (source gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_news_keywords_text_trgm ON news USING gin ((keywords::text) gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_news_entities_text_trgm ON news USING gin ((entities::text) gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_news_region_trgm ON news USING gin (region gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_news_category_trgm ON news USING gin (category gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS ix_news_heat_publish_date ON news (heat_score DESC, publish_date DESC)",
    "CREATE INDEX IF NOT EXISTS ix_news_publish_date_heat_desc ON news (publish_date DESC, heat_score DESC)",
)

POSTGRES_SEARCH_INDEX_CONCURRENTLY_SQL: tuple[str, ...] = tuple(
    sql.replace("CREATE INDEX IF NOT EXISTS", "CREATE INDEX CONCURRENTLY IF NOT EXISTS")
    if sql.startswith("CREATE INDEX IF NOT EXISTS") else sql
    for sql in POSTGRES_SEARCH_INDEX_SQL
)


async def execute_postgres_index_statements(conn: Any, statements: Iterable[str]) -> None:
    """
    输入:
    - `conn`: SQLAlchemy 异步连接
    - `statements`: 需要执行的 SQL 语句集合

    输出:
    - 无

    作用:
    - 顺序执行 PostgreSQL 搜索索引 SQL，供启动初始化和维护脚本复用。
    """

    for sql in statements:
        await conn.execute(text(sql))


async def ensure_postgres_search_indexes(conn: Any) -> None:
    """
    输入:
    - `conn`: SQLAlchemy 异步连接

    输出:
    - 无

    作用:
    - 在当前事务连接上创建搜索索引，适合显式开启后的新库初始化兜底。
    """

    await execute_postgres_index_statements(conn, POSTGRES_SEARCH_INDEX_SQL)


async def optimize_postgres_search_indexes(database_url: str) -> None:
    """
    输入:
    - `database_url`: PostgreSQL 异步连接串

    输出:
    - 无

    作用:
    - 使用 autocommit 和并发建索引语句优化线上搜索索引，降低对业务查询的阻塞。
    """

    if "postgresql" not in (database_url or "").lower():
        raise RuntimeError("当前索引优化仅支持 PostgreSQL")

    engine = create_async_engine(database_url, echo=False, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await execute_postgres_index_statements(conn, POSTGRES_SEARCH_INDEX_CONCURRENTLY_SQL)
    finally:
        await engine.dispose()


async def main() -> None:
    """
    输入:
    - 环境变量 `REMOTE_DATABASE_URL` 或 `DATABASE_URL`

    输出:
    - 控制台执行进度

    作用:
    - 作为可追踪的命令入口，便于线上部署后手动执行 PostgreSQL 并发建索引。
    """

    database_url = os.environ.get("REMOTE_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    if not database_url.strip():
        raise RuntimeError("请先设置 REMOTE_DATABASE_URL 或 DATABASE_URL")
    await optimize_postgres_search_indexes(database_url)
    print("PostgreSQL 搜索索引优化完成")


if __name__ == "__main__":
    asyncio.run(main())
