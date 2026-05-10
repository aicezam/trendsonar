"""
本文件用于提供系统相关 API：聊天、任务触发，以及管理端配置读取/写入接口。
主要函数/类:
- `api_trigger_crawl`: 手动触发抓取与分析
- `chat_api`: RAG 问答接口（支持流式）
- `api_get_admin_config`: 读取 `config.yaml`
- `api_update_admin_config`: 写入 `config.yaml` 并触发重启
- `UpdateConfigPayload`: 配置更新请求体
"""

import gc
import json
import re
import time as perf_time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import defer
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
from app.services.source_health_service import source_health_service
from app.services.task_manager import task_manager
from app.utils.tools import parse_query_time_range

router = APIRouter(prefix="/api", tags=["system"])

_LOG_FILE_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\.log$")


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
    - 以后台任务方式触发对历史数据的情感与关键词补全，并防止重复启动
    """

    status = await task_manager.start_background(
        "analyze_all",
        background_analyze_all,
        progress="全量情感与关键词补全执行中",
    )
    return {"status": "running" if status.get("running") else status.get("status"), "task": status}


class UpdateConfigPayload(BaseModel):
    yaml_text: str


class UpdateSourcesPayload(BaseModel):
    json_text: str


class NewsSourcePayload(BaseModel):
    name: str
    weight: float = 1.0
    address: str
    enabled: bool = True


class UpdateSourcesStructuredPayload(BaseModel):
    sources: list[NewsSourcePayload]


def _get_news_sources_path() -> Path:
    candidates = [
        BASE_DIR / "data" / "news_sources.json",
        BASE_DIR / "news_sources.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 默认为 data/news_sources.json
    return candidates[0]


def _normalize_news_source(raw: dict) -> dict:
    """
    输入:
    - `raw`: 原始新闻源配置项

    输出:
    - 标准化后的新闻源配置

    作用:
    - 兼容历史配置中缺少 enabled 字段的情况，并保证卡片编辑需要的字段完整
    """

    return {
        "name": str(raw.get("name") or "").strip(),
        "weight": float(raw.get("weight", 1.0) or 1.0),
        "address": str(raw.get("address") or "").strip(),
        "enabled": bool(raw.get("enabled", True)),
    }


def _load_news_sources_data() -> list[dict]:
    """
    输入:
    - 无

    输出:
    - 标准化后的新闻源列表

    作用:
    - 从新闻源配置文件读取数据，并兼容旧格式
    """

    path = _get_news_sources_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8") or "[]")
    if not isinstance(data, list):
        raise ValueError("新闻源配置必须是 JSON 列表")
    return [_normalize_news_source(item) for item in data if isinstance(item, dict)]


def _save_news_sources_data(sources: list[dict]) -> None:
    """
    输入:
    - `sources`: 新闻源配置列表

    输出:
    - 无

    作用:
    - 保存标准化新闻源配置到配置文件
    """

    normalized = [_normalize_news_source(item) for item in sources]
    path = _get_news_sources_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=4), encoding="utf-8")


def _source_key(source: dict) -> str:
    """
    输入:
    - `source`: 新闻源配置项

    输出:
    - 用于健康状态索引的稳定键

    作用:
    - 优先使用地址作为唯一键，避免新闻源改名后丢失状态
    """

    return str(source.get("address") or source.get("name") or "").strip()


def _get_logs_dir() -> Path:
    return BASE_DIR / "logs"


@router.get("/app_info")
async def api_get_app_info():
    return {"app_name": settings.APP_NAME, "version": settings.VERSION}


@router.get("/admin/config")
async def api_get_admin_config(request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    yaml_text = load_config_yaml_text()
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception:
        data = {}
    return {"yaml": yaml_text, "config": data, "missing_keys": get_missing_config_keys(settings)}


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


@router.get("/admin/log_files")
async def api_list_log_files(request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")

    logs_dir = _get_logs_dir()
    if not logs_dir.exists():
        return {"files": [], "retention_days": int(getattr(settings, "LOG_RETENTION_DAYS", 3) or 3)}

    items = []
    for p in logs_dir.iterdir():
        if not p.is_file():
            continue
        if not _LOG_FILE_RE.match(p.name):
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        items.append(
            {
                "name": p.name,
                "size": int(st.st_size),
                "mtime": float(st.st_mtime),
            }
        )
    items.sort(key=lambda x: x["name"], reverse=True)
    return {"files": items, "retention_days": int(getattr(settings, "LOG_RETENTION_DAYS", 3) or 3)}


@router.get("/admin/log_file")
async def api_get_log_file(name: str, request: Request, tail_lines: int = 5000):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")

    if not _LOG_FILE_RE.match(name or ""):
        raise HTTPException(status_code=400, detail="日志文件名不合法")
    tail_lines = max(1, min(int(tail_lines or 5000), 20000))

    path = _get_logs_dir() / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="日志文件不存在")

    lines = deque(maxlen=tail_lines)
    total = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total += 1
                lines.append(line.rstrip("\n"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {e}")

    return {
        "name": name,
        "text": "\n".join(lines),
        "total_lines": total,
        "returned_lines": len(lines),
        "truncated": total > len(lines),
        "retention_days": int(getattr(settings, "LOG_RETENTION_DAYS", 3) or 3),
    }


@router.post("/trigger_crawl", dependencies=[Depends(verify_admin_access)])
async def api_trigger_crawl():
    """
    输入:
    - 无

    输出:
    - 启动结果

    作用:
    - 手动触发一次抓取与分析全流程任务，并与定时流水线共享运行锁
    """

    status = await task_manager.start_background(
        "pipeline",
        run_manual,
        progress="手动全流程执行中",
    )
    return {"status": "running" if status.get("running") else status.get("status"), "task": status}


@router.get("/admin/tasks", dependencies=[Depends(verify_admin_access)])
async def api_get_task_statuses():
    """
    输入:
    - 无

    输出:
    - 所有后台任务状态

    作用:
    - 为管理端提供后台任务运行状态、时间、进度和失败原因
    """

    return {"tasks": await task_manager.get_all_statuses()}


@router.get("/chat")
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
        rag_start = perf_time.perf_counter()
        start_date, end_date = parse_query_time_range(query)

        # 优化：RAG 只需要 embedding 和摘要，不需要加载正文
        stmt = select(News).options(defer(News.content)).where(News.embedding.is_not(None))
        if start_date:
            stmt = stmt.where(News.publish_date >= start_date)
        if end_date:
            stmt = stmt.where(News.publish_date < end_date)
        if not start_date and not end_date:
            stmt = stmt.where(News.publish_date >= datetime.now() - timedelta(days=7))
        stmt = stmt.order_by(desc(News.heat_score), desc(News.publish_date)).limit(2000)

        result = await db.execute(stmt)
        candidates = result.scalars().all()

        all_scored = []
        high_relevance_count = 0
        norm_q = float(np.linalg.norm(q_vec))
        q_dim = len(q_vec)

        for n in candidates:
            if not n.embedding or len(n.embedding) != q_dim:
                continue
            n_vec = np.array(n.embedding)
            sim = 0.0
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

        elapsed_ms = (perf_time.perf_counter() - rag_start) * 1000
        logger.info(
            f"/api/chat RAG 检索完成: candidates={len(candidates)}, scored={len(all_scored)}, "
            f"context={len(top_news)}, elapsed={elapsed_ms:.1f}ms"
        )

        # 显式清理 RAG 中间变量，释放内存
        del candidates, all_scored, top_news
        gc.collect()

    if not context_text:
        context_text = "未找到相关新闻。"

    model_type = "backup" if use_backup else "main"
    if stream:
        logger.info(f"开始流式返回: model_type={model_type}")
        async def stream_wrapper():
            count = 0
            try:
                async for chunk in ai_service.stream_chat(query, context_text, model_type):
                    count += 1
                    yield chunk
                
                if count == 0:
                    logger.warning("流式包装器返回 0 个块，发送兜底消息")
                    yield "AI 未返回任何内容，请查看后台日志，可能有敏感信息，请切换模型重试"
            except Exception as e:
                logger.error(f"流式包装器错误: {e}")
                yield f"Error: {e}"
            finally:
                logger.info(f"流式包装器结束，总块数: {count}")

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
    try:
        sources = _load_news_sources_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"新闻源配置读取失败: {e}")
    health = source_health_service.get_all_statuses()
    items = []
    for source in sources:
        key = _source_key(source)
        items.append({**source, "key": key, "health": health.get(key, {})})
    return {
        "sources": items,
        "json": json.dumps(sources, ensure_ascii=False, indent=4),
        "path": str(path),
    }


@router.put("/admin/news_sources")
async def api_update_news_sources(payload: UpdateSourcesPayload, request: Request):
    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    try:
        data = json.loads(payload.json_text)
        if not isinstance(data, list):
            raise ValueError("新闻源配置必须是 JSON 列表")
        _save_news_sources_data(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 格式错误: {e}")
    return {"ok": True}


@router.put("/admin/news_sources/structured")
async def api_update_news_sources_structured(payload: UpdateSourcesStructuredPayload, request: Request):
    """
    输入:
    - `payload`: 结构化新闻源列表

    输出:
    - 保存结果

    作用:
    - 支持管理端卡片式编辑新闻源，保存时写回 news_sources.json
    """

    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    sources = [item.model_dump() for item in payload.sources]
    if any(not item.get("name") or not item.get("address") for item in sources):
        raise HTTPException(status_code=400, detail="新闻源名称和地址不能为空")
    _save_news_sources_data(sources)
    return {"ok": True}


@router.post("/admin/news_sources/test")
async def api_test_news_source(source: NewsSourcePayload, request: Request):
    """
    输入:
    - `source`: 单个新闻源配置

    输出:
    - 测试抓取状态和最多 10 条预览结果

    作用:
    - 管理端测试单个新闻源是否可用，不写入数据库
    """

    if not is_admin_request(request):
        raise HTTPException(status_code=401, detail="未登录")
    from app.services.crawler_service import crawler_service
    import aiohttp

    normalized = _normalize_news_source(source.model_dump())
    key = _source_key(normalized)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            items = await crawler_service.fetch_and_parse(session, normalized, prefix="测试")
        preview = items[:10]
        source_health_service.record_test_result(
            key,
            name=normalized["name"],
            count=len(items),
            ok=bool(items),
            error=None if items else "未提取到新闻条目",
        )
        return {
            "ok": bool(items),
            "count": len(items),
            "preview": preview,
            "message": "测试成功" if items else "未提取到新闻条目",
            "health": source_health_service.get_status(key),
        }
    except Exception as e:
        source_health_service.record_test_result(
            key,
            name=normalized["name"],
            count=0,
            ok=False,
            error=str(e),
        )
        return {
            "ok": False,
            "count": 0,
            "preview": [],
            "message": str(e),
            "health": source_health_service.get_status(key),
        }
