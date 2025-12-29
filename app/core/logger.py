"""
本文件用于初始化并提供项目统一日志能力（根 logger 配置与命名 logger 获取）。
主要函数:
- `configure_logging`: 初始化根日志格式与等级
- `setup_logger`: 获取具备统一格式的命名 logger
"""

import logging
import sys
from collections import deque
from threading import Lock
from typing import Deque, Optional

from app.core.config import get_settings

settings = get_settings()

_log_lock: Lock = Lock()
_log_buffer: Deque[str] = deque(maxlen=1000)
_memory_handler: Optional[logging.Handler] = None


def _resolve_level(level: str) -> int:
    """
    输入:
    - `level`: 日志等级字符串（如 INFO/DEBUG）

    输出:
    - `logging` 对应的等级整数

    作用:
    - 将字符串日志等级转换为 `logging` 可用的等级值
    """

    return getattr(logging, (level or "").upper(), logging.INFO)


class _InMemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        with _log_lock:
            _log_buffer.append(msg)


def get_cached_log_text() -> str:
    with _log_lock:
        return "\n".join(_log_buffer)


def clear_cached_logs() -> None:
    with _log_lock:
        _log_buffer.clear()


def configure_logging() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 初始化根 logger 的输出格式与等级，并同步常见库的日志等级
    """

    log_level = _resolve_level(settings.LOG_LEVEL)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root.setLevel(log_level)

    global _memory_handler
    if _memory_handler is None:
        _memory_handler = _InMemoryLogHandler()
        _memory_handler.setLevel(log_level)
        _memory_handler.setFormatter(formatter)
        root.addHandler(_memory_handler)
    else:
        _memory_handler.setLevel(log_level)
        _memory_handler.setFormatter(formatter)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        if _memory_handler not in uv_logger.handlers:
            uv_logger.addHandler(_memory_handler)

    noisy_level = log_level
    if log_level == logging.INFO:
        noisy_level = logging.WARNING

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "asyncio",
        "crawl4ai",
        "playwright",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(noisy_level)


def setup_logger(name: str) -> logging.Logger:
    """
    输入:
    - `name`: logger 名称

    输出:
    - `logging.Logger` 实例

    作用:
    - 创建并返回指定名称的 logger（若已存在 handler 则复用）
    """

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    configure_logging()
    log_level = _resolve_level(settings.LOG_LEVEL)
    logger.setLevel(log_level)

    return logger


logger = setup_logger(settings.APP_NAME)
