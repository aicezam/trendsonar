"""
本文件用于读取/写入项目根目录的 `config.yaml`，并提供 YAML 文本与字典之间的转换工具。
主要函数:
- `load_yaml_dict`: 从 YAML 文件读取为字典（不存在则返回空字典）
- `dump_yaml_text`: 将字典序列化为 YAML 文本
- `save_yaml_text`: 将 YAML 文本写入文件（自动创建父目录）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_dict(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}

    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return {}

    data = yaml.safe_load(raw_text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml 顶层必须为映射（key-value）结构")
    return data


def dump_yaml_text(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def save_yaml_text(file_path: Path, yaml_text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(yaml_text, encoding="utf-8")

