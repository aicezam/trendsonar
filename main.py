"""
本文件用于启动 FastAPI 应用并注册页面路由、静态资源与生命周期任务。
主要函数/类:
- `lifespan`: 应用生命周期管理（初始化数据库、启动定时任务）
- `page_index`: 首页渲染
- `page_report`: 报表页渲染
- `page_admin`: 管理页渲染
- `admin_login`: 管理登录（写入 Cookie 会话）
- `admin_logout`: 管理退出（清理 Cookie 会话）
- `AdminLoginPayload`: 管理登录请求体
"""

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.api import api_router
from app.api.deps import settings, templates
from app.core.config import BASE_DIR, get_missing_config_keys
from app.core.database import init_db
from app.core.logger import configure_logging, setup_logger
from app.services.admin_service import create_admin_session_token, get_admin_cookie_name, is_admin_request, verify_admin_password
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
        asyncio.create_task(topic_service.scheduled_topic_task())
        asyncio.create_task(scheduled_task())
    else:
        lifespan_logger.warning("⚠️ 由于数据库初始化失败，定时任务已跳过启动。")
    yield


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
    - 报表页面 HTML 响应

    作用:
    - 渲染并返回舆情分析报表页面
    """

    missing_keys = get_missing_config_keys(settings)
    return templates.TemplateResponse(
        "report.html",
        {"request": request, "settings": settings, "active_page": "report", "missing_keys": missing_keys, "db_error": DB_INIT_ERROR},
    )


@app.get("/topics", response_class=HTMLResponse)
async def page_topics(request: Request):
    missing_keys = get_missing_config_keys(settings)
    return templates.TemplateResponse(
        "topics.html",
        {"request": request, "settings": settings, "active_page": "topics", "missing_keys": missing_keys, "db_error": DB_INIT_ERROR},
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
async def admin_login(payload: AdminLoginPayload):
    if not verify_admin_password(payload.password):
        return JSONResponse(status_code=403, content={"ok": False, "message": "密码错误"})

    token = create_admin_session_token()
    resp = JSONResponse(content={"ok": True})
    resp.set_cookie(
        key=get_admin_cookie_name(),
        value=token,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.post("/admin/logout")
async def admin_logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(get_admin_cookie_name())
    return resp


if __name__ == "__main__":
    log_level = (settings.LOG_LEVEL or "info").lower()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=log_level,
        access_log=log_level in {"debug", "info"},
    )
