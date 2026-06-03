# 本文件用于提供异步重试工具，集中处理外部服务短暂不可用的补救逻辑。

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, TypeVar

from app.core.logger import setup_logger

logger = setup_logger("RetryUtils")

T = TypeVar("T")


async def retry_async_result(
    func: Callable[[], Awaitable[Optional[T]]],
    *,
    attempts: int = 2,
    delay_seconds: float = 3.0,
    per_attempt_timeout_seconds: Optional[float] = None,
    min_valid_length: int = 1,
    label: str = "异步任务",
    before_retry: Optional[Callable[[int], Awaitable[None]]] = None,
) -> Optional[T]:
    """
    输入:
    - `func`: 返回可选结果的异步函数
    - `attempts`: 最大尝试次数
    - `delay_seconds`: 失败后等待秒数
    - `per_attempt_timeout_seconds`: 单次尝试硬超时秒数
    - `min_valid_length`: 字符串结果的最小有效长度
    - `label`: 日志标签
    - `before_retry`: 每次重试前执行的异步钩子，参数为下一次尝试序号

    输出:
    - 第一个有效结果；全部失败时返回 None

    作用:
    - 对正文抓取等外部 I/O 做轻量补救重试，避免单次网络抖动导致直接跳过。
    """

    total_attempts = max(1, attempts)
    for attempt in range(total_attempts):
        try:
            if per_attempt_timeout_seconds and per_attempt_timeout_seconds > 0:
                result = await asyncio.wait_for(func(), timeout=per_attempt_timeout_seconds)
            else:
                result = await func()
            if isinstance(result, str):
                if len(result.strip()) >= min_valid_length:
                    return result
            elif result is not None:
                return result
        except asyncio.TimeoutError:
            logger.warning(f"{label}第 {attempt + 1}/{total_attempts} 次超时: 超过 {per_attempt_timeout_seconds:g} 秒")
        except Exception as exc:
            logger.warning(f"{label}第 {attempt + 1}/{total_attempts} 次失败: {exc}")

        if attempt < total_attempts - 1:
            logger.info(f"{label}将在 {delay_seconds:g} 秒后重试一次")
            if before_retry is not None:
                try:
                    await before_retry(attempt + 2)
                except Exception as exc:
                    logger.warning(f"{label}重试前钩子执行失败: {exc}")
            await asyncio.sleep(delay_seconds)

    logger.warning(f"{label}已尝试 {total_attempts} 次仍未获得有效结果，跳过")
    return None
