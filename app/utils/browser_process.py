# 本文件用于清理容器内浏览器与 crawl4ai 残留进程，帮助正文补抓失败后的重试恢复。

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Iterable

from app.core.logger import setup_logger

logger = setup_logger("BrowserProcess")

_LAST_CLEANUP_AT = 0.0
_CLEANUP_LOCK = asyncio.Lock()
_BROWSER_PROCESS_PATTERNS = (
    "crawl4ai",
    "playwright",
    "chrome_crashpad",
    "chrome-linux",
    "chromium-browser",
    "chromium --",
    "google-chrome",
    "headless_shell",
)


def is_container_environment() -> bool:
    """
    输入:
    - 无

    输出:
    - 是否处于 Linux 容器环境

    作用:
    - 将进程清理限定在线上容器内，避免本地 Windows/macOS 开发环境误杀浏览器。
    """

    if os.name != "posix":
        return False
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(marker in cgroup for marker in ("docker", "containerd", "kubepods"))


async def _run_pkill(pattern: str) -> int:
    """
    输入:
    - `pattern`: pkill -f 的命令行匹配片段

    输出:
    - pkill 退出码

    作用:
    - 异步执行单个进程匹配清理命令。
    """

    try:
        process = await asyncio.create_subprocess_exec(
            "pkill",
            "-f",
            pattern,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.debug("容器内未安装 pkill，将使用 /proc 扫描方式清理浏览器进程")
        return -1

    stdout, stderr = await process.communicate()
    if process.returncode not in (0, 1):
        detail = (stderr or stdout or b"").decode("utf-8", errors="ignore").strip()
        logger.warning(f"清理浏览器进程失败: pattern={pattern}, code={process.returncode}, detail={detail[:200]}")
    return int(process.returncode or 0)


def _iter_matched_pids(patterns: Iterable[str]) -> list[int]:
    """
    输入:
    - `patterns`: 需要匹配的命令行片段

    输出:
    - 匹配到的进程 PID 列表

    作用:
    - 在没有 pkill 的精简容器中，通过读取 /proc/<pid>/cmdline 查找浏览器残留进程。
    """

    current_pid = os.getpid()
    normalized_patterns = [item.lower() for item in patterns if item]
    matched: list[int] = []

    try:
        proc_entries = list(Path("/proc").iterdir())
    except Exception as exc:
        logger.warning(f"无法扫描 /proc 进程列表: {exc}")
        return []

    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (0, 1, current_pid):
            continue

        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue

        cmdline = raw_cmdline.replace(b"\x00", b" ").decode("utf-8", errors="ignore").lower()
        if not cmdline:
            continue
        if any(pattern in cmdline for pattern in normalized_patterns):
            matched.append(pid)

    return matched


async def _cleanup_processes_by_proc(patterns: Iterable[str]) -> int:
    """
    输入:
    - `patterns`: 需要匹配的命令行片段

    输出:
    - 成功发送清理信号的进程数量

    作用:
    - 作为 pkill 缺失时的纯 Python 回退清理方案。
    """

    matched_pids = _iter_matched_pids(patterns)
    if not matched_pids:
        return 0

    killed = 0
    for pid in matched_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            logger.warning(f"无权限终止浏览器进程: pid={pid}, error={exc}")

    await asyncio.sleep(1)

    for pid in matched_pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            logger.warning(f"无权限强制终止浏览器进程: pid={pid}, error={exc}")

    return killed


async def cleanup_browser_processes(
    *,
    reason: str = "正文补抓重试",
    cooldown_seconds: float = 10.0,
    patterns: Iterable[str] = _BROWSER_PROCESS_PATTERNS,
) -> bool:
    """
    输入:
    - `reason`: 清理原因
    - `cooldown_seconds`: 最小清理间隔
    - `patterns`: 需要匹配的进程命令行片段

    输出:
    - 是否实际执行了清理

    作用:
    - 在容器内杀掉浏览器/crawl4ai 残留进程，避免上一次导航超时污染下一次重试。
    """

    global _LAST_CLEANUP_AT

    if not is_container_environment():
        logger.debug(f"非容器环境，跳过浏览器进程清理: {reason}")
        return False

    async with _CLEANUP_LOCK:
        now = time.monotonic()
        if now - _LAST_CLEANUP_AT < cooldown_seconds:
            logger.debug(f"浏览器进程清理处于冷却期，跳过: {reason}")
            return False

        _LAST_CLEANUP_AT = now
        logger.warning(f"准备清理容器内浏览器/crawl4ai 残留进程: {reason}")
        killed_any = False
        pkill_available = True
        for pattern in patterns:
            code = await _run_pkill(pattern)
            if code == -1:
                pkill_available = False
                break
            if code == 0:
                killed_any = True

        if not pkill_available:
            killed_count = await _cleanup_processes_by_proc(patterns)
            killed_any = killed_count > 0
            logger.info(f"/proc 回退清理完成，命中 {killed_count} 个浏览器/crawl4ai 进程")

        await asyncio.sleep(1)
        logger.info("浏览器/crawl4ai 残留进程清理完成" if killed_any else "未发现需要清理的浏览器/crawl4ai 残留进程")
        return True
