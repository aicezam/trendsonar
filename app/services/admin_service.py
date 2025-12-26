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
    async def _shutdown() -> None:
        await asyncio.sleep(delay_seconds)
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            os._exit(0)

    asyncio.create_task(_shutdown())


def verify_admin_password(password: str) -> bool:
    settings = get_settings()
    return bool(settings.ADMIN_PASSWORD) and password == settings.ADMIN_PASSWORD
