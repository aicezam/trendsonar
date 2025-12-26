"""
本文件用于提供系统相关 API：聊天、任务触发，以及管理端配置读取/写入接口。
主要函数/类:
- `api_trigger_crawl`: 手动触发抓取与分析
- `chat_api`: RAG 问答接口（支持流式）
- `api_get_admin_config`: 读取 `config.yaml`
- `api_update_admin_config`: 写入 `config.yaml` 并触发重启
- `UpdateConfigPayload`: 配置更新请求体
"""

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import settings, verify_admin_access
from app.core.database import get_db
from app.core.config import get_missing_config_keys, BASE_DIR
from app.core.logger import clear_cached_logs, get_cached_log_text, logger
from app.models.news import News
from app.schemas.system import AdminAuth
from app.services.ai_service import ai_service
from app.services.admin_service import is_admin_request, load_config_yaml_text, save_config_yaml_text, schedule_restart
from app.services.pipeline_service import background_analyze_all, reanalyze_all_categories, run_manual
from app.utils.tools import parse_query_time_range

router = APIRouter(prefix="/api", tags=["system"])


@router.post("/admin/reanalyze_all_categories")
async def api_reanalyze_all_categories(auth: AdminAuth):
    """
    输入:
    - `auth`: 管理员鉴权请求体

    输出:
    - 全量重分析任务结果（状态与更新条数）

    作用:
    - 触发对全量新闻进行重新分析（情感/分类/关键词/实体）
    """

    if auth.password != settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="密码错误")
    return await reanalyze_all_categories()


@router.post("/admin/analyze_all_sentiment", dependencies=[Depends(verify_admin_access)])
async def api_analyze_all_sentiment():
    """
    输入:
    - 无

    输出:
    - 启动结果

    作用:
    - 以后台任务方式触发对历史数据的情感与关键词补全
    """

    asyncio.create_task(background_analyze_all())
    return {"status": "started"}


class UpdateConfigPayload(BaseModel):
    yaml_text: str


class UpdateSourcesPayload(BaseModel):
    json_text: str


def _get_news_sources_path() -> Path:
    candidates = [
        BASE_DIR / "data" / "news_sources.json",
        BASE_DIR / "news_sources.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Default to data/news_sources.json
    return candidates[0]


@router.get("/app_info")
async def api_get_app_info():
    return {"app_name": settings.APP_NAME, "version": settings.VERSION}


@router.get("/admin/config")
async def api_get_admin_config(request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    return {"yaml": load_config_yaml_text(), "missing_keys": get_missing_config_keys(settings)}


@router.put("/admin/config")
async def api_update_admin_config(payload: UpdateConfigPayload, request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    try:
        save_config_yaml_text(payload.yaml_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    schedule_restart()
    return {"ok": True, "restarting": True}


@router.get("/admin/logs")
async def api_get_admin_logs(request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    return {"logs": get_cached_log_text()}


@router.delete("/admin/logs")
async def api_clear_admin_logs(request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    clear_cached_logs()
    return {"ok": True}


@router.post("/trigger_crawl", dependencies=[Depends(verify_admin_access)])
async def api_trigger_crawl():
    """
    输入:
    - 无

    输出:
    - 启动结果

    作用:
    - 手动触发一次抓取与分析全流程任务
    """

    asyncio.create_task(run_manual())
    return {"status": "started"}


@router.get("/chat", dependencies=[Depends(verify_admin_access)])
async def chat_api(
    query: str,
    stream: bool = True,
    use_backup: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    输入:
    - `query`: 用户问题
    - `stream`: 是否以 SSE 流式返回
    - `use_backup`: 是否强制使用备用模型
    - `db`: 数据库会话（依赖注入）

    输出:
    - SSE 流式响应或完整回答

    作用:
    - 基于新闻向量检索构造上下文，实现 RAG 问答
    """

    logger.info(f"收到聊天请求: query={query}, stream={stream}, use_backup={use_backup}")
    q_emb_list = await ai_service.get_embeddings([query])
    q_vec = np.array(q_emb_list[0]) if q_emb_list and q_emb_list[0] else None

    context_text = ""
    if q_vec is not None:
        start_date, end_date = parse_query_time_range(query)

        stmt = select(News).where(News.embedding.is_not(None))
        if start_date:
            stmt = stmt.where(News.publish_date >= start_date)
        if end_date:
            stmt = stmt.where(News.publish_date < end_date)

        result = await db.execute(stmt)
        candidates = result.scalars().all()

        all_scored = []
        high_relevance_count = 0

        for n in candidates:
            if not n.embedding:
                continue
            n_vec = np.array(n.embedding)
            sim = 0.0
            norm_q = np.linalg.norm(q_vec)
            norm_n = np.linalg.norm(n_vec)
            if norm_q > 0 and norm_n > 0:
                sim = float(np.dot(q_vec, n_vec) / (norm_q * norm_n))

            all_scored.append((sim, n))
            if sim > 0.65:
                high_relevance_count += 1

        if high_relevance_count >= 1:
            all_scored.sort(key=lambda x: x[0], reverse=True)
            top_news = all_scored[:50]
        else:
            all_scored.sort(key=lambda x: x[1].heat_score, reverse=True)
            top_news = all_scored[:100]

        for i, (score, n) in enumerate(top_news):
            context_text += (
                f"{i+1}. [{n.title}] (来源:{n.source}, 时间:{n.publish_date}, 热度:{n.heat_score})\n摘要: {n.summary}\n\n"
            )

    if not context_text:
        context_text = "未找到相关新闻。"

    model_type = "backup" if use_backup else "main"
    if stream:
        logger.info(f"开始流式返回: model_type={model_type}")
        async def stream_wrapper():
            try:
                async for chunk in ai_service.stream_chat(query, context_text, model_type):
                    yield chunk
            except Exception as e:
                logger.error(f"Stream wrapper error: {e}")
                yield f"Error: {e}"
            finally:
                logger.info("Stream wrapper finished")

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    full_response = ""
    async for chunk in ai_service.stream_chat(query, context_text, model_type):
        full_response += chunk
    return {"response": full_response}


@router.get("/admin/news_sources")
async def api_get_news_sources(request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    path = _get_news_sources_path()
    return {"json": path.read_text(encoding="utf-8") if path.exists() else "[]", "path": str(path)}


@router.put("/admin/news_sources")
async def api_update_news_sources(payload: UpdateSourcesPayload, request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    try:
        data = json.loads(payload.json_text)
        if not isinstance(data, list):
            raise ValueError("新闻源配置必须是 JSON 列表")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 格式错误: {e}")
    path = _get_news_sources_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.json_text, encoding="utf-8")
    return {"ok": True}