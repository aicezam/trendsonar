"""
本文件用于提供异步数据库引擎与会话的惰性初始化，并为路由提供依赖注入会话。
主要函数:
- `get_engine`: 懒加载创建 `AsyncEngine`
- `get_sessionmaker`: 懒加载创建 `sessionmaker`
- `AsyncSessionLocal`: 获取新的 `AsyncSession`
- `init_db`: 创建数据库表结构
- `get_db`: FastAPI 依赖注入会话生成器
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings
from app.core.logger import logger
from app.utils.postgres_search_indexes import ensure_postgres_search_indexes
from app.utils.schema_migration import run_schema_migrations

settings = get_settings()

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[sessionmaker] = None


async def _ensure_postgres_search_indexes(conn: Any) -> None:
    """
    输入:
    - `conn`: SQLAlchemy 异步连接

    输出:
    - 无

    作用:
    - PostgreSQL 环境下按显式开关创建搜索所需 trigram 扩展和 GIN 索引，新库建议优先使用并发建索引脚本。
    """

    if "postgresql" not in (settings.DATABASE_URL or "").lower():
        return
    enabled = os.environ.get("TRENDSONAR_AUTO_SEARCH_INDEXES", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    try:
        await ensure_postgres_search_indexes(conn)
    except Exception as exc:
        logger.warning(f"⚠️ PostgreSQL 搜索索引初始化跳过: {exc}")


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is not None:
        return _engine
    if not (settings.DATABASE_URL or "").strip():
        raise RuntimeError("未配置 DATABASE_URL，数据库功能不可用")

    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=20,
        max_overflow=10,
    )

    # 针对 SQLite 启用 WAL 模式以提高并发稳定性
    if "sqlite" in settings.DATABASE_URL:
        from sqlalchemy import event
        
        @event.listens_for(_engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
            except Exception as e:
                logger.warning(f"⚠️ SQLite PRAGMA 初始化失败，继续使用当前连接模式: {e}")
            finally:
                cursor.close()

    return _engine


def get_sessionmaker() -> sessionmaker:
    global _sessionmaker
    if _sessionmaker is not None:
        return _sessionmaker

    engine = get_engine()
    _sessionmaker = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return _sessionmaker


def AsyncSessionLocal() -> AsyncSession:
    return get_sessionmaker()()

Base = declarative_base()


async def init_db() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 初始化数据库表结构（根据 ORM 模型创建表）
    """

    from app.models.news import News  # noqa: F401
    from app.models.report import ReportCache  # noqa: F401
    from app.models.topic import Topic, TopicTimelineItem  # noqa: F401
    from app.models.clustering_history import ClusteringHistory  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await run_schema_migrations(conn, Base.metadata)
        await _ensure_postgres_search_indexes(conn)
        logger.info("✅ 数据库表结构和关键索引初始化完成")


async def check_db_connection(verbose: bool = True) -> bool:
    """
    检查数据库连接是否可用
    """
    try:
        if not (settings.DATABASE_URL or "").strip():
            if verbose:
                logger.warning("⚠️ 未配置 DATABASE_URL")
            return False
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        if verbose:
            logger.warning(f"⚠️ 数据库连接检查失败: {e}")
        return False



async def dispose_engine() -> None:
    """
    输入:
    - 无
    
    输出:
    - 无
    
    作用:
    - 强制释放数据库引擎的连接池，用于释放空闲连接占用
    """
    global _engine
    if _engine:
        await _engine.dispose()
        # logger.info("🔌 数据库连接池已释放")


async def get_db():
    """
    输入:
    - 无

    输出:
    - 依赖注入可用的 `AsyncSession` 生成器

    作用:
    - 为 FastAPI 路由提供数据库会话，并保证请求结束后自动释放
    """
    
    # 移除冗余的连接检查，避免每次请求都进行 SELECT 1 导致性能损耗或锁竞争
    # if not await check_db_connection(verbose=False):
    #     raise HTTPException(status_code=503, detail="系统配置错误或数据库连接失败，请检查配置。")

    try:
        async with AsyncSessionLocal() as session:
            yield session
    except Exception as e:
        import traceback
        logger.error(f"❌ 数据库会话异常: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=f"数据库连接异常: {str(e)}")
