"""
本文件用于检测 ORM 模型与实际数据库结构差异，并执行安全的自动迁移。
主要函数:
- `run_schema_migrations`: 在应用启动或容器内脚本中补齐缺失表、列和声明索引
- `main`: 容器内命令入口
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.core.config import get_settings
from app.core.logger import setup_logger

logger = setup_logger("SchemaMigration")
SCHEMA_MIGRATION_TABLE = "schema_migrations"
SCHEMA_MIGRATION_KEY = "orm_metadata"


def _quote_identifier(conn: Any, name: str) -> str:
    """
    输入:
    - `conn`: 同步数据库连接
    - `name`: 表名或列名

    输出:
    - 当前数据库方言下安全引用后的标识符

    作用:
    - 生成 ALTER TABLE 语句时避免关键字或大小写导致 SQL 解析问题。
    """

    return conn.dialect.identifier_preparer.quote(name)


def _compile_column_type(conn: Any, column: Any) -> str:
    """
    输入:
    - `conn`: 同步数据库连接
    - `column`: SQLAlchemy Column 对象

    输出:
    - 当前数据库方言下的列类型 SQL

    作用:
    - 将 ORM 字段类型转换成数据库可执行的 ADD COLUMN 类型声明。
    """

    return column.type.compile(dialect=conn.dialect)


def _metadata_signature(metadata: MetaData) -> str:
    """
    输入:
    - `metadata`: ORM 元数据

    输出:
    - 当前模型结构签名

    作用:
    - 用表、列、类型和索引声明生成稳定签名，避免容器每次启动都重复做结构差异扫描。
    """

    payload: list[dict[str, Any]] = []
    for table in metadata.sorted_tables:
        payload.append(
            {
                "table": table.name,
                "columns": [
                    {
                        "name": column.name,
                        "type": str(column.type),
                        "primary_key": bool(column.primary_key),
                        "nullable": bool(column.nullable),
                    }
                    for column in table.columns
                ],
                "indexes": sorted(index.name or "" for index in table.indexes),
            }
        )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_migration_table_sync(conn: Any) -> None:
    """
    输入:
    - `conn`: 同步数据库连接

    输出:
    - 无

    作用:
    - 创建轻量迁移状态表，用于记录当前 ORM 结构签名。
    """

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name VARCHAR(120) PRIMARY KEY,
                signature VARCHAR(80) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    )


def _get_recorded_signature_sync(conn: Any) -> str:
    """
    输入:
    - `conn`: 同步数据库连接

    输出:
    - 已记录的模型结构签名，不存在时返回空字符串

    作用:
    - 判断本次容器启动是否需要重新执行结构差异检测。
    """

    _ensure_migration_table_sync(conn)
    row = conn.execute(
        text(f"SELECT signature FROM {SCHEMA_MIGRATION_TABLE} WHERE name = :name"),
        {"name": SCHEMA_MIGRATION_KEY},
    ).first()
    return str(row[0]) if row else ""


def _record_signature_sync(conn: Any, signature: str) -> None:
    """
    输入:
    - `conn`: 同步数据库连接
    - `signature`: 当前模型结构签名

    输出:
    - 无

    作用:
    - 迁移检测完成后记录签名，下一次启动可跳过无变化检测。
    """

    dialect_name = str(getattr(conn.dialect, "name", "") or "").lower()
    if "postgresql" in dialect_name:
        sql = text(
            f"""
            INSERT INTO {SCHEMA_MIGRATION_TABLE} (name, signature, applied_at)
            VALUES (:name, :signature, CURRENT_TIMESTAMP)
            ON CONFLICT (name) DO UPDATE
            SET signature = EXCLUDED.signature, applied_at = CURRENT_TIMESTAMP
            """
        )
    else:
        sql = text(
            f"""
            INSERT OR REPLACE INTO {SCHEMA_MIGRATION_TABLE} (name, signature, applied_at)
            VALUES (:name, :signature, CURRENT_TIMESTAMP)
            """
        )
    conn.execute(sql, {"name": SCHEMA_MIGRATION_KEY, "signature": signature})


def _add_missing_columns_sync(conn: Any, metadata: MetaData) -> list[str]:
    """
    输入:
    - `conn`: 同步数据库连接
    - `metadata`: ORM 元数据

    输出:
    - 已执行的迁移动作描述列表

    作用:
    - 对已存在的表补齐 ORM 中新增的列；历史数据存在时，新列统一先按可空列添加，避免启动迁移失败。
    """

    inspector = inspect(conn)
    actions: list[str] = []
    for table in metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing_columns = {item["name"] for item in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            if column.primary_key:
                logger.warning(f"跳过主键列自动补齐: {table.name}.{column.name}")
                continue

            table_name = _quote_identifier(conn, table.name)
            column_name = _quote_identifier(conn, column.name)
            column_type = _compile_column_type(conn, column)
            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            conn.execute(text(sql))
            actions.append(f"add_column:{table.name}.{column.name}")
    return actions


def _create_declared_indexes_sync(conn: Any, metadata: MetaData) -> list[str]:
    """
    输入:
    - `conn`: 同步数据库连接
    - `metadata`: ORM 元数据

    输出:
    - 已检测的索引名称列表

    作用:
    - 为已存在表补齐模型声明的普通索引；`checkfirst=True` 保证索引存在时不会重复创建。
    """

    actions: list[str] = []
    for table in metadata.sorted_tables:
        for index in table.indexes:
            index.create(bind=conn, checkfirst=True)
            if index.name:
                actions.append(f"ensure_index:{index.name}")
    return actions


async def run_schema_migrations(conn: AsyncConnection, metadata: MetaData) -> list[str]:
    """
    输入:
    - `conn`: SQLAlchemy 异步连接
    - `metadata`: ORM 元数据

    输出:
    - 已执行或检测的迁移动作描述列表

    作用:
    - 容器启动时检测结构差异并自动迁移：先创建缺失表，再补齐缺失列和普通索引。
    """

    signature = _metadata_signature(metadata)
    recorded_signature = await conn.run_sync(_get_recorded_signature_sync)
    if recorded_signature == signature:
        return []

    await conn.run_sync(metadata.create_all)
    column_actions = await conn.run_sync(_add_missing_columns_sync, metadata)
    index_actions = await conn.run_sync(_create_declared_indexes_sync, metadata)
    await conn.run_sync(_record_signature_sync, signature)
    actions = column_actions + index_actions
    if column_actions:
        logger.info(f"数据库结构自动迁移完成: {column_actions}")
    return actions


async def main() -> None:
    """
    输入:
    - 运行时配置中的 `DATABASE_URL`

    输出:
    - 控制台迁移结果

    作用:
    - 提供容器内手动执行入口：`python -m app.utils.schema_migration`。
    """

    from app.core.database import Base, get_engine
    from app.models.news import News  # noqa: F401
    from app.models.report import ReportCache  # noqa: F401
    from app.models.topic import Topic, TopicTimelineItem  # noqa: F401
    from app.models.clustering_history import ClusteringHistory  # noqa: F401

    settings = get_settings()
    if not (settings.DATABASE_URL or "").strip():
        raise RuntimeError("未配置 DATABASE_URL，无法执行数据库迁移")
    engine = get_engine()
    async with engine.begin() as conn:
        actions = await run_schema_migrations(conn, Base.metadata)
    print(f"数据库迁移检测完成，动作数: {len(actions)}")


if __name__ == "__main__":
    asyncio.run(main())
