"""
本文件用于聚合各模块路由，并提供统一的 `api_router` 给主应用注册。
主要对象:
- `api_router`: API 路由聚合器
"""

from fastapi import APIRouter

from app.api.endpoints import news, reports, system, topics

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(news.router)
api_router.include_router(reports.router)
api_router.include_router(topics.router)
