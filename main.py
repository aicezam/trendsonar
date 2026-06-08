"""
本文件用于启动 FastAPI 应用并注册页面路由、静态资源与生命周期任务。
主要函数/类:
- `lifespan`: 应用生命周期管理（初始化数据库、启动定时任务）
- `page_index`: 首页渲染
- `page_report`: 报告页渲染
- `page_admin`: 管理页渲染
- `admin_login`: 管理登录（写入 Cookie 会话）
- `admin_logout`: 管理退出（清理 Cookie 会话）
- `AdminLoginPayload`: 管理登录请求体
"""

# 在导入业务模块前，先解析 --config 参数并设置环境变量
import argparse as _argparse
import os as _os

_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", "-c", type=str, default=None)
_args, _ = _parser.parse_known_args()
if _args.config:
    _os.environ["TRENDSONAR_CONFIG"] = _args.config

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.api import api_router
from app.api.deps import settings, templates
from app.core.config import BASE_DIR, CONFIG_PATH, get_missing_config_keys
from app.core.database import dispose_engine, init_db
from app.core.logger import configure_logging, setup_logger
from app.services.admin_service import (
    clear_admin_login_failures,
    create_admin_session_token,
    get_admin_client_key,
    get_admin_cookie_name,
    is_admin_login_locked,
    is_admin_request,
    is_secure_cookie_request,
    record_admin_login_failure,
    verify_admin_password,
)
from app.services.pipeline_service import scheduled_task
from app.services.topic_service import topic_service



DB_INIT_ERROR: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    输入:
    - `app`: FastAPI 应用实例

    输出:
    - 生命周期上下文（启动后进入、退出时清理）

    作用:
    - 应用启动时初始化数据库并启动定时任务
    """

    global DB_INIT_ERROR
    lifespan_logger = setup_logger("lifespan")
    db_initialized = False
    try:
        await init_db()
        DB_INIT_ERROR = None
        db_initialized = True
    except Exception as e:
        DB_INIT_ERROR = str(e)
        lifespan_logger.error(f"❌ 初始化数据库失败: {e}")
        lifespan_logger.warning("=" * 60)
        lifespan_logger.warning("⚠️  系统配置缺失或数据库连接失败！")
        lifespan_logger.warning("⚠️  定时抓取任务将暂停执行，直到配置修正。")
        lifespan_logger.warning(f"⚠️  请访问管理后台修改配置：http://localhost:{settings.PORT}/admin")
        lifespan_logger.warning("=" * 60)
        
    if db_initialized:
        # 统一由 pipeline_service 的 scheduled_task 进行调度，避免重复运行
        # asyncio.create_task(topic_service.scheduled_topic_task())
        asyncio.create_task(scheduled_task())
    else:
        lifespan_logger.warning("⚠️ 由于数据库初始化失败，定时任务已跳过启动。")
    try:
        yield
    finally:
        await dispose_engine()


configure_logging()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    debug=settings.DEBUG
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    """
    输入:
    - `request`: FastAPI 请求对象

    输出:
    - 首页 HTML 响应

    作用:
    - 渲染并返回仪表盘页面
    """

    missing_keys = get_missing_config_keys(settings)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "settings": settings, "active_page": "index", "missing_keys": missing_keys, "db_error": DB_INIT_ERROR},
    )


@app.get("/report", response_class=HTMLResponse)
async def page_report(request: Request):
    """
    输入:
    - `request`: FastAPI 请求对象

    输出:
    - 报告页面 HTML 响应

    作用:
    - 渲染并返回舆情分析报告页面
    """

    missing_keys = get_missing_config_keys(settings)
    authed = is_admin_request(request)
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "settings": settings,
            "active_page": "report",
            "missing_keys": missing_keys,
            "authed": authed,
            "db_error": DB_INIT_ERROR,
        },
    )


@app.get("/topics", response_class=HTMLResponse)
async def page_topics(request: Request):
    missing_keys = get_missing_config_keys(settings)
    authed = is_admin_request(request)
    return templates.TemplateResponse(
        "topics.html",
        {"request": request, "settings": settings, "active_page": "topics", "missing_keys": missing_keys, "authed": authed, "db_error": DB_INIT_ERROR},
    )


@app.get("/topics/{topic_id}", response_class=HTMLResponse)
async def page_topic_detail(topic_id: int, request: Request):
    missing_keys = get_missing_config_keys(settings)
    return templates.TemplateResponse(
        "topic_detail.html",
        {
            "request": request,
            "settings": settings,
            "active_page": "topics",
            "missing_keys": missing_keys,
            "topic_id": topic_id,
            "db_error": DB_INIT_ERROR,
        },
    )


@app.get("/graph", response_class=HTMLResponse)
async def page_graph(request: Request):
    """
    输入:
    - `request`: FastAPI 请求对象

    输出:
    - 知识图谱页面 HTML 响应

    作用:
    - 渲染语义关系画布页面，用于查看新闻词项之间的实体、议题和风险关联。
    """

    missing_keys = get_missing_config_keys(settings)
    return templates.TemplateResponse(
        "graph.html",
        {
            "request": request,
            "settings": settings,
            "active_page": "graph",
            "missing_keys": missing_keys,
            "db_error": DB_INIT_ERROR,
        },
    )


class AdminLoginPayload(BaseModel):
    password: str


@app.get("/admin", response_class=HTMLResponse)
async def page_admin(request: Request):
    missing_keys = get_missing_config_keys(settings)
    authed = is_admin_request(request)
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "settings": settings, "active_page": "admin", "missing_keys": missing_keys, "authed": authed, "db_error": DB_INIT_ERROR},
    )


@app.post("/admin/login")
async def admin_login(payload: AdminLoginPayload, request: Request):
    client_key = get_admin_client_key(request)
    if is_admin_login_locked(client_key):
        return JSONResponse(status_code=429, content={"ok": False, "message": "登录失败次数过多，请稍后再试"})

    if not verify_admin_password(payload.password):
        record_admin_login_failure(client_key)
        return JSONResponse(status_code=403, content={"ok": False, "message": "密码错误"})

    clear_admin_login_failures(client_key)
    token = create_admin_session_token()
    resp = JSONResponse(content={"ok": True})
    resp.set_cookie(
        key=get_admin_cookie_name(),
        value=token,
        httponly=True,
        samesite="lax",
        secure=is_secure_cookie_request(request),
    )
    return resp


@app.post("/admin/logout")
async def admin_logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(get_admin_cookie_name())
    return resp


if __name__ == "__main__":
    log_level = (settings.LOG_LEVEL or "info").lower()
    print(f"📄 配置文件: {CONFIG_PATH}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=log_level,
        access_log=log_level in {"debug", "info"},
    )
