"""
本文件用于提供报表相关 API：生成全局报表、生成关键词报表、读取历史与图表数据等。
主要函数:
- `get_recent_reports`: 获取最近生成的报表
- `generate_global_report`: 生成并缓存全局报表
- `get_report_history`: 获取报表历史
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.report_service import report_service

router = APIRouter(prefix="/api/report", tags=["report"])


@router.get("/recent")
async def get_recent_reports():
    """
    输入:
    - 无

    输出:
    - 最近关键词报表列表

    作用:
    - 为报表页提供“最近生成”入口
    """

    return await report_service.get_recent_reports()


@router.post("/generate-global")
async def generate_global_report(period: str = Query("weekly", pattern="^(daily|weekly|monthly)$")):
    """
    输入:
    - `period`: 报表周期（daily/weekly/monthly）

    输出:
    - 任务执行结果

    作用:
    - 生成并缓存指定周期的全局大盘报表
    """

    await report_service.generate_and_cache_global_report(period)
    return {"status": "ok", "message": f"全局 {period} 报表已生成"}


@router.get("/history")
async def get_report_history(keyword: Optional[str] = None):
    """
    输入:
    - `keyword`: 关键词（可选）

    输出:
    - 对应关键词的历史列表；若不传则返回全局历史

    作用:
    - 提供报表历史记录查询，支持关键词与全局两种模式
    """

    if keyword:
        return await report_service.get_report_history(keyword)
    return await report_service.get_global_history()


@router.get("/load/{report_id}")
async def load_report(report_id: int):
    """
    输入:
    - `report_id`: 报表缓存 ID

    输出:
    - 报表结构化数据

    作用:
    - 读取指定历史报表，用于前端回溯查看
    """

    data = await report_service.load_report(report_id)
    if not data:
        raise HTTPException(status_code=404, detail="Report not found")
    return data


@router.get("/analysis")
async def get_report_analysis(
    q: Optional[str] = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    limit: Optional[int] = None,
    generate_ai: Optional[bool] = False,
):
    """
    输入:
    - `q`: 关键词（可选）
    - `start_date`/`end_date`: 起止日期（可选）
    - `category`/`region`/`source`: 过滤条件（可选）
    - `limit`: 取样上限（可选）
    - `generate_ai`: 是否生成 AI 分析文字结论

    输出:
    - 报表分析数据（摘要、图表数据、Top 新闻、AI 分析）

    作用:
    - 按条件生成报表数据，供前端图表渲染与下载
    """

    return await report_service.get_analysis_data(q, start_date, end_date, category, region, source, limit, generate_ai)


@router.get("/chart-data")
async def get_chart_data(
    type: str,
    category: Optional[str] = None,
    region: Optional[str] = None,
    q: Optional[str] = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    输入:
    - `type`: 图表类型（word_cloud/source/sentiment/list 等）
    - `category`/`region`: 过滤条件（可选）
    - `q`: 关键词（可选）
    - `start_date`/`end_date`: 起止日期（可选）

    输出:
    - 对应图表所需数据结构

    作用:
    - 为前端按需刷新单个图表提供轻量接口
    """

    return await report_service.get_chart_data(type, category, region, q, start_date, end_date)

