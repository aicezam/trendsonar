"""
本文件用于提供网络图谱相关 API，包括图谱总览、节点展开、节点详情和相关新闻分页。
主要函数:
- `get_graph_overview`: 获取首屏图谱
- `expand_graph_node`: 展开指定词项邻域
- `get_graph_node_detail`: 获取词项分析
- `get_graph_node_news`: 获取词项相关新闻
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.graph_service import graph_service

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/overview")
async def get_graph_overview(
    range: str = Query("24h", max_length=20),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    sort_by: str = Query("heat", pattern="^(heat|date)$"),
    limit: int = Query(120, ge=20, le=500),
    edge_limit: int = Query(420, ge=50, le=2000),
    min_edge_weight: int = Query(2, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - 时间、分类、地区、来源和图谱规模限制

    输出:
    - 核心词项节点和共现边

    作用:
    - 为图谱页面首屏加载提供受控规模的网络数据。
    """

    return await graph_service.get_overview(
        db,
        range_key=range,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
        sort_by=sort_by,
        limit=limit,
        edge_limit=edge_limit,
        min_edge_weight=min_edge_weight,
    )


@router.get("/expand")
async def expand_graph_node(
    term: str = Query(..., min_length=1, max_length=80),
    range: str = Query("24h", max_length=20),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    sort_by: str = Query("heat", pattern="^(heat|date)$"),
    limit: int = Query(90, ge=10, le=300),
    edge_limit: int = Query(360, ge=20, le=1200),
    min_edge_weight: int = Query(1, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - `term`: 中心词项
    - 时间、分类、地区、来源和图谱规模限制

    输出:
    - 中心词项邻域节点和边

    作用:
    - 支持前端按点击或缩放渐进加载隐藏节点。
    """

    return await graph_service.expand_node(
        db,
        term=term,
        range_key=range,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
        sort_by=sort_by,
        limit=limit,
        edge_limit=edge_limit,
        min_edge_weight=min_edge_weight,
    )


@router.get("/node/{term}")
async def get_graph_node_detail(
    term: str,
    range: str = Query("24h", max_length=20),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - `term`: 词项文本
    - 时间、分类、地区、来源筛选条件

    输出:
    - 词项概览、趋势、相邻词、来源分布和相关新闻

    作用:
    - 为图谱侧边详情面板提供分析数据。
    """

    return await graph_service.get_node_detail(
        db,
        term=term,
        range_key=range,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
    )


@router.get("/node/{term}/news")
async def get_graph_node_news(
    term: str,
    range: str = Query("24h", max_length=20),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    sort_by: str = Query("heat", pattern="^(heat|date)$"),
    page: int = Query(1, ge=1, le=500),
    page_size: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - `term`: 词项文本
    - 分页和筛选条件

    输出:
    - 相关新闻分页列表

    作用:
    - 支持图谱详情面板继续加载相关新闻。
    """

    return await graph_service.get_term_news(
        db,
        term=term,
        range_key=range,
        start_date=start_date,
        end_date=end_date,
        category=category,
        region=region,
        source=source,
        sort_by=sort_by,
        page=page,
        page_size=page_size,
    )
