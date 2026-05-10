"""
本文件用于记录新闻源抓取和测试健康状态，便于管理后台展示最近一次加载、失败原因和测试结果。
主要类/对象:
- `SourceHealthService`: 新闻源健康状态读写服务
- `source_health_service`: 服务单例
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import BASE_DIR
from app.core.logger import logger


class SourceHealthService:
    """
    输入:
    - 新闻源状态文件路径

    输出:
    - 新闻源健康状态快照

    作用:
    - 用 JSON 文件轻量记录新闻源最近抓取、测试、失败和启用状态，降低排查成本
    """

    def __init__(self) -> None:
        self.path = BASE_DIR / "data" / "source_health.json"

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except Exception as e:
            logger.error(f"读取新闻源健康状态失败: {e}")
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"保存新闻源健康状态失败: {e}")

    def get_status(self, key: str) -> Dict[str, Any]:
        """
        输入:
        - `key`: 新闻源唯一键

        输出:
        - 指定新闻源状态

        作用:
        - 查询新闻源最近加载、测试和失败信息
        """

        return self._load().get(key, {})

    def get_all_statuses(self) -> Dict[str, Any]:
        """
        输入:
        - 无

        输出:
        - 所有新闻源状态字典

        作用:
        - 为管理后台批量展示新闻源健康状态
        """

        return self._load()

    def record_fetch_result(
        self,
        key: str,
        *,
        name: str,
        count: int,
        ok: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        输入:
        - `key`: 新闻源唯一键
        - `name`: 新闻源名称
        - `count`: 本次抓取数量
        - `ok`: 是否成功
        - `error`: 失败原因

        输出:
        - 无

        作用:
        - 记录自动或手动抓取时的新闻源加载状态
        """

        data = self._load()
        item = data.get(key, {})
        now = datetime.now().isoformat()
        failure_count = int(item.get("failure_count") or 0)
        if ok:
            failure_count = 0
        else:
            failure_count += 1
        item.update(
            {
                "name": name,
                "last_fetch_at": now,
                "last_fetch_count": count,
                "last_fetch_status": "success" if ok else "failed",
                "failure_count": failure_count,
                "last_error": error,
            }
        )
        data[key] = item
        self._save(data)

    def record_test_result(
        self,
        key: str,
        *,
        name: str,
        count: int,
        ok: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        输入:
        - `key`: 新闻源唯一键
        - `name`: 新闻源名称
        - `count`: 测试提取数量
        - `ok`: 是否测试成功
        - `error`: 失败原因

        输出:
        - 无

        作用:
        - 记录管理后台新闻源测试结果
        """

        data = self._load()
        item = data.get(key, {})
        item.update(
            {
                "name": name,
                "last_test_at": datetime.now().isoformat(),
                "last_test_count": count,
                "last_test_status": "success" if ok else "failed",
                "last_test_error": error,
            }
        )
        data[key] = item
        self._save(data)


source_health_service = SourceHealthService()
