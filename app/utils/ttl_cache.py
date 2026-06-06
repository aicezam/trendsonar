"""
本文件用于提供轻量 TTL 内存缓存工具，供高频接口缓存短期可复用的序列化结果。
主要类:
- `TtlMemoryCache`: 支持过期淘汰和最大容量控制的进程内缓存
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    """
    输入:
    - `created_at`: 写入时的单调时钟时间
    - `value`: 缓存值

    输出:
    - 缓存条目对象

    作用:
    - 将缓存值和写入时间绑定，便于统一判断过期。
    """

    created_at: float
    value: T


class TtlMemoryCache(Generic[T]):
    """
    输入:
    - `ttl_seconds`: 缓存有效期，单位秒
    - `max_size`: 最大缓存条目数

    输出:
    - 可按键读写的短 TTL 内存缓存

    作用:
    - 为慢查询接口提供进程内短缓存，减少重复数据库查询和 Python 计算。
    """

    def __init__(self, ttl_seconds: float, max_size: int = 128) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_size = max(1, int(max_size))
        self._items: dict[Hashable, _CacheEntry[T]] = {}

    def get(self, key: Hashable) -> Optional[T]:
        """
        输入:
        - `key`: 缓存键

        输出:
        - 命中的缓存值，未命中或已过期时返回 None

        作用:
        - 读取缓存并惰性清理过期条目。
        """

        entry = self._items.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._items.pop(key, None)
            return None
        return entry.value

    def set(self, key: Hashable, value: T) -> T:
        """
        输入:
        - `key`: 缓存键
        - `value`: 需要缓存的值

        输出:
        - 原始缓存值

        作用:
        - 写入缓存，并在超过容量时优先淘汰最早写入的条目。
        """

        self._items[key] = _CacheEntry(created_at=time.monotonic(), value=value)
        self._evict_if_needed()
        return value

    def delete(self, key: Hashable) -> None:
        """
        输入:
        - `key`: 缓存键

        输出:
        - 无

        作用:
        - 主动删除单个缓存条目，用于数据被更新后的精确失效。
        """

        self._items.pop(key, None)

    def clear_expired(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 主动清理所有过期缓存条目。
        """

        expired_keys = [key for key, entry in self._items.items() if self._is_expired(entry)]
        for key in expired_keys:
            self._items.pop(key, None)

    def _is_expired(self, entry: _CacheEntry[T]) -> bool:
        """
        输入:
        - `entry`: 缓存条目

        输出:
        - 是否已经超过 TTL

        作用:
        - 统一缓存过期判断口径。
        """

        return time.monotonic() - entry.created_at > self.ttl_seconds

    def _evict_if_needed(self) -> None:
        """
        输入:
        - 无

        输出:
        - 无

        作用:
        - 控制缓存容量，避免高频不同查询导致进程内存持续增长。
        """

        if len(self._items) <= self.max_size:
            return
        self.clear_expired()
        while len(self._items) > self.max_size:
            oldest_key = min(self._items, key=lambda key: self._items[key].created_at)
            self._items.pop(oldest_key, None)
