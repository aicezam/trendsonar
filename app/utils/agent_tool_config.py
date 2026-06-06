# 本文件用于读写智能体自定义工具配置，供管理端维护工具元数据。

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
CUSTOM_AGENT_TOOLS_FILE = BASE_DIR / "data" / "agent_tools.json"
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,63}$")


def _normalize_custom_tool(raw: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """
    输入:
    - `raw`: 管理端提交或文件中读取的工具配置
    - `strict`: 是否按保存场景校验执行器完整性

    输出:
    - 字段完整、名称合法的自定义工具配置

    作用:
    - 统一自定义工具元数据结构，避免前端展示和后续扩展时字段缺失。
    """

    name = str(raw.get("name") or "").strip()
    if not _TOOL_NAME_RE.match(name):
        raise ValueError("工具名称只能包含字母、数字和下划线，并且必须以字母开头")

    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    executor = raw.get("executor") if isinstance(raw.get("executor"), dict) else {}
    enabled = bool(raw.get("enabled", True))
    if strict and enabled and not executor:
        raise ValueError("启用自定义工具必须配置 executor 执行器")
    if strict and executor:
        executor_type = str(executor.get("type") or "http").strip().lower()
        if executor_type != "http":
            raise ValueError("当前自定义工具只支持 HTTP 执行器")
        method = str(executor.get("method") or "GET").strip().upper()
        if method not in {"GET", "POST"}:
            raise ValueError("自定义 HTTP 工具只支持 GET/POST")
        if not str(executor.get("url") or "").strip():
            raise ValueError("自定义 HTTP 工具必须配置 url")
        if "query" in executor and not isinstance(executor.get("query"), dict):
            raise ValueError("自定义 HTTP 工具的 query 必须是 JSON 对象")
        if "headers" in executor and not isinstance(executor.get("headers"), dict):
            raise ValueError("自定义 HTTP 工具的 headers 必须是 JSON 对象")
        executor = {**executor, "type": executor_type, "method": method}
    return {
        "name": name,
        "title": str(raw.get("title") or name).strip(),
        "description": str(raw.get("description") or "").strip(),
        "parameters": parameters,
        "executor": executor,
        "prompt_hint": str(raw.get("prompt_hint") or "").strip(),
        "enabled": enabled,
        "kind": "custom",
    }


def load_custom_agent_tools() -> list[dict[str, Any]]:
    """
    输入:
    - 无

    输出:
    - 自定义工具配置列表

    作用:
    - 从 `data/agent_tools.json` 读取管理端新增的工具元数据。
    """

    if not CUSTOM_AGENT_TOOLS_FILE.exists():
        return []
    try:
        data = json.loads(CUSTOM_AGENT_TOOLS_FILE.read_text(encoding="utf-8") or "[]")
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    tools: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            tools.append(_normalize_custom_tool(item))
        except ValueError:
            continue
    return tools


def save_custom_agent_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """
    输入:
    - `tool`: 需要新增或更新的自定义工具配置

    输出:
    - 保存后的标准化工具配置

    作用:
    - 按工具名称进行 upsert 保存，供管理端维护可执行自定义工具。
    """

    normalized = _normalize_custom_tool(tool, strict=True)
    tools = load_custom_agent_tools()
    replaced = False
    for index, item in enumerate(tools):
        if item.get("name") == normalized["name"]:
            tools[index] = normalized
            replaced = True
            break
    if not replaced:
        tools.append(normalized)

    CUSTOM_AGENT_TOOLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_AGENT_TOOLS_FILE.write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def delete_custom_agent_tool(name: str) -> bool:
    """
    输入:
    - `name`: 自定义工具名称

    输出:
    - 是否删除成功

    作用:
    - 删除管理端维护的自定义工具配置。
    """

    clean_name = str(name or "").strip()
    tools = load_custom_agent_tools()
    next_tools = [item for item in tools if item.get("name") != clean_name]
    if len(next_tools) == len(tools):
        return False

    CUSTOM_AGENT_TOOLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_AGENT_TOOLS_FILE.write_text(json.dumps(next_tools, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
