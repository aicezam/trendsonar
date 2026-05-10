"""
本文件用于管理后台任务的运行状态、互斥锁和统一启动入口。
主要类:
- `TaskManager`: 提供任务去重、状态记录、异常捕获和后台启动能力
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from app.core.logger import logger

TaskCallable = Callable[[], Awaitable[Any]]


class TaskManager:
    """
    输入:
    - 后台任务名称和异步任务函数

    输出:
    - 当前任务状态快照

    作用:
    - 使用内存锁避免同类后台任务重复运行，并统一记录开始、结束、异常与进度信息
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def _ensure_task(self, name: str) -> Dict[str, Any]:
        if name not in self._tasks:
            self._tasks[name] = {
                "name": name,
                "status": "idle",
                "running": False,
                "started_at": None,
                "finished_at": None,
                "progress": "未启动",
                "last_error": None,
            }
        return self._tasks[name]

    def _serialize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": item["name"],
            "status": item["status"],
            "running": item["running"],
            "started_at": item["started_at"].isoformat() if item.get("started_at") else None,
            "finished_at": item["finished_at"].isoformat() if item.get("finished_at") else None,
            "progress": item.get("progress"),
            "last_error": item.get("last_error"),
        }

    async def start(self, name: str, task_func: TaskCallable, *, progress: str = "运行中") -> Dict[str, Any]:
        """
        输入:
        - `name`: 任务名称
        - `task_func`: 无参数异步任务函数
        - `progress`: 任务启动后的初始进度描述

        输出:
        - 任务状态快照；如果任务已经运行则返回现有运行状态

        作用:
        - 在当前协程中执行任务，并确保同名任务同一时间只运行一个实例
        """

        async with self._lock:
            item = self._ensure_task(name)
            if item["running"]:
                logger.warning(f"⏳ 后台任务已在运行，跳过重复启动: {name}")
                return self._serialize(item)

            now = datetime.now()
            item.update(
                {
                    "status": "running",
                    "running": True,
                    "started_at": now,
                    "finished_at": None,
                    "progress": progress,
                    "last_error": None,
                }
            )

        try:
            logger.info(f"🚀 后台任务启动: {name}")
            await task_func()
        except Exception as e:
            async with self._lock:
                item = self._ensure_task(name)
                item.update(
                    {
                        "status": "failed",
                        "running": False,
                        "finished_at": datetime.now(),
                        "progress": "执行失败",
                        "last_error": str(e),
                    }
                )
            logger.error(f"❌ 后台任务失败: {name}: {e}\n{traceback.format_exc()}")
            return self._serialize(item)

        async with self._lock:
            item = self._ensure_task(name)
            item.update(
                {
                    "status": "success",
                    "running": False,
                    "finished_at": datetime.now(),
                    "progress": "执行完成",
                    "last_error": None,
                }
            )
            logger.info(f"✅ 后台任务完成: {name}")
            return self._serialize(item)

    async def start_background(self, name: str, task_func: TaskCallable, *, progress: str = "运行中") -> Dict[str, Any]:
        """
        输入:
        - `name`: 任务名称
        - `task_func`: 无参数异步任务函数
        - `progress`: 任务启动后的初始进度描述

        输出:
        - 启动后的任务状态快照；如果任务已运行则不重复创建后台任务

        作用:
        - 用 `asyncio.create_task` 后台运行任务，同时复用 `start` 的去重和状态记录逻辑
        """

        async with self._lock:
            item = self._ensure_task(name)
            if item["running"]:
                logger.warning(f"⏳ 后台任务已在运行，跳过重复启动: {name}")
                return self._serialize(item)

            now = datetime.now()
            item.update(
                {
                    "status": "running",
                    "running": True,
                    "started_at": now,
                    "finished_at": None,
                    "progress": progress,
                    "last_error": None,
                }
            )

        async def runner() -> None:
            try:
                logger.info(f"🚀 后台任务启动: {name}")
                await task_func()
            except Exception as e:
                async with self._lock:
                    current = self._ensure_task(name)
                    current.update(
                        {
                            "status": "failed",
                            "running": False,
                            "finished_at": datetime.now(),
                            "progress": "执行失败",
                            "last_error": str(e),
                        }
                    )
                logger.error(f"❌ 后台任务失败: {name}: {e}\n{traceback.format_exc()}")
                return

            async with self._lock:
                current = self._ensure_task(name)
                current.update(
                    {
                        "status": "success",
                        "running": False,
                        "finished_at": datetime.now(),
                        "progress": "执行完成",
                        "last_error": None,
                    }
                )
            logger.info(f"✅ 后台任务完成: {name}")

        asyncio.create_task(runner())
        return await self.get_status(name)

    async def update_progress(self, name: str, progress: str) -> None:
        """
        输入:
        - `name`: 任务名称
        - `progress`: 最新进度描述

        输出:
        - 无

        作用:
        - 为管理端展示后台任务执行阶段提供轻量状态更新
        """

        async with self._lock:
            item = self._ensure_task(name)
            item["progress"] = progress

    async def get_status(self, name: str) -> Dict[str, Any]:
        """
        输入:
        - `name`: 任务名称

        输出:
        - 指定任务的状态快照

        作用:
        - 查询单个后台任务的当前状态
        """

        async with self._lock:
            return self._serialize(self._ensure_task(name))

    async def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        输入:
        - 无

        输出:
        - 所有已注册任务的状态快照

        作用:
        - 为管理后台提供统一任务状态查询入口
        """

        async with self._lock:
            for name in ("pipeline", "analyze_all"):
                self._ensure_task(name)
            return {name: self._serialize(item) for name, item in self._tasks.items()}


task_manager = TaskManager()
