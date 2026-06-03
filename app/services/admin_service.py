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
import hashlib
import hmac
import os
import secrets
import signal
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import yaml
from fastapi import Request

from app.core.config import CONFIG_PATH, get_settings

_ADMIN_COOKIE_NAME = "trendsonar_admin_token"
_TOKEN_TTL_SECONDS = 12 * 60 * 60
_TOKENS: Dict[str, float] = {}
_LOGIN_FAILURE_WINDOW_SECONDS = 5 * 60
_LOGIN_FAILURE_LIMIT = 5
_LOGIN_FAILURES: Dict[str, Deque[float]] = {}


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


def is_secure_cookie_request(request: Request) -> bool:
    """
    输入:
    - `request`: FastAPI 请求对象

    输出:
    - 当前请求是否应写入 secure Cookie

    作用:
    - 根据直连协议或反向代理转发头判断 HTTPS 部署场景
    """

    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded_proto == "https"


def get_admin_client_key(request: Request) -> str:
    """
    输入:
    - `request`: FastAPI 请求对象

    输出:
    - 用于登录失败限流的客户端标识

    作用:
    - 优先使用反向代理传递的客户端 IP，回退到直连地址
    """

    forwarded_for = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded_for:
        return forwarded_for
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def is_admin_login_locked(client_key: str) -> bool:
    """
    输入:
    - `client_key`: 客户端标识

    输出:
    - 是否处于登录失败锁定窗口

    作用:
    - 限制短时间内的管理员登录爆破尝试
    """

    now = time.time()
    failures = _LOGIN_FAILURES.get(client_key)
    if not failures:
        return False
    while failures and now - failures[0] > _LOGIN_FAILURE_WINDOW_SECONDS:
        failures.popleft()
    return len(failures) >= _LOGIN_FAILURE_LIMIT


def record_admin_login_failure(client_key: str) -> None:
    """
    输入:
    - `client_key`: 客户端标识

    输出:
    - 无

    作用:
    - 记录一次管理员登录失败，并清理过期失败记录
    """

    now = time.time()
    failures = _LOGIN_FAILURES.setdefault(client_key, deque())
    while failures and now - failures[0] > _LOGIN_FAILURE_WINDOW_SECONDS:
        failures.popleft()
    failures.append(now)


def clear_admin_login_failures(client_key: str) -> None:
    """
    输入:
    - `client_key`: 客户端标识

    输出:
    - 无

    作用:
    - 登录成功后清除当前客户端的失败记录
    """

    _LOGIN_FAILURES.pop(client_key, None)


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

    config_dir = CONFIG_PATH.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_PATH.with_name(f".{CONFIG_PATH.name}.{os.getpid()}.{time.time_ns()}.tmp")
    backup_path = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".bak")

    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.write(normalized_text)
            f.flush()
            os.fsync(f.fileno())

        if CONFIG_PATH.exists():
            shutil.copy2(CONFIG_PATH, backup_path)
        os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        if not CONFIG_PATH.exists() and backup_path.exists():
            try:
                os.replace(backup_path, CONFIG_PATH)
            except Exception:
                pass
        raise


def schedule_restart(delay_seconds: float = 1.0) -> None:
    def _in_docker() -> bool:
        if os.name != "posix":
            return False
        if Path("/.dockerenv").exists():
            return True
        try:
            cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
            if "docker" in cgroup or "containerd" in cgroup or "kubepods" in cgroup:
                return True
        except Exception:
            pass
        return os.environ.get("RUNNING_IN_DOCKER") == "1"

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
            if _in_docker():
                os._exit(0)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            os._exit(0)

    asyncio.create_task(_shutdown())


def verify_admin_password(password: str) -> bool:
    """
    输入:
    - `password`: 用户提交的管理员密码

    输出:
    - 密码是否匹配

    作用:
    - 支持明文兼容配置，并兼容 sha256/pbkdf2_sha256 哈希格式
    """

    settings = get_settings()
    stored_password = settings.ADMIN_PASSWORD or ""
    if not stored_password:
        return False

    submitted = password or ""
    if stored_password.startswith("sha256$"):
        parts = stored_password.split("$", 2)
        if len(parts) != 3:
            return False
        _, salt, expected = parts
        digest = hashlib.sha256(f"{salt}{submitted}".encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, expected)

    if stored_password.startswith("pbkdf2_sha256$"):
        parts = stored_password.split("$", 3)
        if len(parts) != 4:
            return False
        _, rounds_text, salt, expected = parts
        try:
            rounds = max(100_000, int(rounds_text))
        except Exception:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            submitted.encode("utf-8"),
            salt.encode("utf-8"),
            rounds,
        ).hex()
        return hmac.compare_digest(digest, expected)

    return hmac.compare_digest(submitted, stored_password)
