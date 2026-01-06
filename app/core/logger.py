"""
本文件用于初始化并提供项目统一日志能力（根 logger 配置与命名 logger 获取）。
主要函数:
- `configure_logging`: 初始化根日志格式与等级
- `setup_logger`: 获取具备统一格式的命名 logger
"""

import logging
import sys
from collections import deque
from datetime import date, timedelta
from pathlib import Path
import re
from threading import Lock
from typing import Deque, Optional

from app.core.config import BASE_DIR, get_settings

settings = get_settings()

_log_lock: Lock = Lock()
_log_buffer: Deque[str] = deque(maxlen=1000)
_memory_handler: Optional[logging.Handler] = None
_date_file_handler: Optional[logging.Handler] = None

_LOG_FILE_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\.log$")


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


def _get_logs_dir() -> Path:
    return BASE_DIR / "logs"


def _get_retention_days() -> int:
    try:
        days = int(getattr(settings, "LOG_RETENTION_DAYS", 3))
    except Exception:
        days = 3
    return max(1, days)


def _cleanup_old_log_files(logs_dir: Path, retention_days: int) -> None:
    if retention_days < 1:
        return
    if not logs_dir.exists():
        return

    cutoff = date.today() - timedelta(days=retention_days - 1)
    for p in logs_dir.iterdir():
        if not p.is_file():
            continue
        m = _LOG_FILE_RE.match(p.name)
        if not m:
            continue
        try:
            d = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        except Exception:
            continue
        if d < cutoff:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                continue


class _DateFileLogHandler(logging.Handler):
    def __init__(self, logs_dir: Path, retention_days: int) -> None:
        super().__init__()
        self._logs_dir = logs_dir
        self._retention_days = max(1, retention_days)
        self._handler: Optional[logging.FileHandler] = None
        self._current_date: Optional[date] = None
        self._fh_lock: Lock = Lock()

    def _ensure_file_handler(self) -> None:
        today = date.today()
        if self._handler is not None and self._current_date == today:
            return

        if self._handler is not None:
            try:
                self._handler.close()
            except Exception:
                pass

        self._logs_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._logs_dir / f"{today.strftime('%Y-%m-%d')}.log"
        self._handler = logging.FileHandler(file_path, encoding="utf-8")
        self._handler.setLevel(self.level)
        if self.formatter is not None:
            self._handler.setFormatter(self.formatter)
        self._current_date = today

        _cleanup_old_log_files(self._logs_dir, self._retention_days)

    def setLevel(self, level: int) -> None:
        super().setLevel(level)
        with self._fh_lock:
            if self._handler is not None:
                self._handler.setLevel(level)

    def setFormatter(self, fmt: logging.Formatter) -> None:
        super().setFormatter(fmt)
        with self._fh_lock:
            if self._handler is not None:
                self._handler.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        with self._fh_lock:
            self._ensure_file_handler()
            if self._handler is None:
                return
            self._handler.emit(record)


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

    global _date_file_handler
    if _date_file_handler is None:
        _date_file_handler = _DateFileLogHandler(_get_logs_dir(), _get_retention_days())
        _date_file_handler.setLevel(log_level)
        _date_file_handler.setFormatter(formatter)
        root.addHandler(_date_file_handler)
        _cleanup_old_log_files(_get_logs_dir(), _get_retention_days())
    else:
        _date_file_handler.setLevel(log_level)
        _date_file_handler.setFormatter(formatter)

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
        if _date_file_handler is not None and _date_file_handler not in uv_logger.handlers:
            uv_logger.addHandler(_date_file_handler)
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
        "httpx",
        "httpcore",
        "openai",
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
