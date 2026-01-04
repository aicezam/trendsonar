"""
本文件用于实现管理后台的认证会话、配置读取/写入与触发重启等能力。
主要函数:
- `create_admin_session_token`: 登录成功后生成会话 token
- `is_admin_request`: 校验请求是否已登录
- `load_config_yaml_text`: 读取 `config.yaml` 原始文本
- `save_config_yaml_text`: 校验并写入 `config.yaml`
- `schedule_restart`: 安排进程重启以加载新配置
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import Request

from app.core.config import CONFIG_PATH, get_settings

_ADMIN_COOKIE_NAME = "trendsonar_admin_token"
_TOKEN_TTL_SECONDS = 12 * 60 * 60
_TOKENS: Dict[str, float] = {}


def create_admin_session_token() -> str:
    token = secrets.token_urlsafe(32)
    _TOKENS[token] = time.time() + _TOKEN_TTL_SECONDS
    return token


def _is_token_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    exp = _TOKENS.get(token)
    if not exp:
        return False
    if exp < time.time():
        _TOKENS.pop(token, None)
        return False
    return True


def is_admin_request(request: Request) -> bool:
    token = request.cookies.get(_ADMIN_COOKIE_NAME)
    return _is_token_valid(token)


def get_admin_cookie_name() -> str:
    return _ADMIN_COOKIE_NAME


def load_config_yaml_text() -> str:
    path: Path = CONFIG_PATH
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_config_yaml_text(yaml_text: str) -> None:
    normalized_text = yaml_text or ""
    data = yaml.safe_load(normalized_text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml 顶层必须为映射（key-value）结构")

    CONFIG_PATH.write_text(normalized_text, encoding="utf-8")


def schedule_restart(delay_seconds: float = 1.0) -> None:
    def _safe_kill(pid: int, sig: int) -> None:
        try:
            os.kill(pid, sig)
        except Exception:
            return

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _collect_descendant_pids(root_pid: int) -> list[int]:
        if os.name != "posix" or not Path("/proc").exists():
            return []

        ppid_to_children: Dict[int, list[int]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            stat_path = entry / "stat"
            try:
                stat = stat_path.read_text(encoding="utf-8", errors="ignore")
                end = stat.rfind(") ")
                if end == -1:
                    continue
                rest = stat[end + 2 :].split()
                if len(rest) < 2:
                    continue
                ppid = int(rest[1])
                pid = int(entry.name)
            except Exception:
                continue
            ppid_to_children.setdefault(ppid, []).append(pid)

        descendants: list[int] = []
        stack = [root_pid]
        seen = {root_pid}
        while stack:
            current = stack.pop()
            for child in ppid_to_children.get(current, []):
                if child in seen:
                    continue
                seen.add(child)
                descendants.append(child)
                stack.append(child)

        return descendants

    def _terminate_descendants(root_pid: int) -> None:
        pids = _collect_descendant_pids(root_pid)
        if not pids:
            return

        for pid in reversed(pids):
            _safe_kill(pid, signal.SIGTERM)

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if not any(_pid_alive(pid) for pid in pids):
                return
            time.sleep(0.1)

        for pid in reversed(pids):
            if _pid_alive(pid):
                _safe_kill(pid, signal.SIGKILL)

    async def _shutdown() -> None:
        await asyncio.sleep(delay_seconds)
        _terminate_descendants(os.getpid())
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            os._exit(0)

    asyncio.create_task(_shutdown())


def verify_admin_password(password: str) -> bool:
    settings = get_settings()
    return bool(settings.ADMIN_PASSWORD) and password == settings.ADMIN_PASSWORD
