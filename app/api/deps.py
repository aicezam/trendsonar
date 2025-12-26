"""
本文件用于提供 FastAPI 依赖注入与模板引擎实例的集中出口。
主要对象:
- `settings`: 全局配置对象
- `templates`: Jinja2 模板渲染器
- `get_db`: 数据库会话依赖注入生成器
"""

from fastapi import Request, HTTPException, status
from fastapi.templating import Jinja2Templates

from app.core.config import BASE_DIR, get_settings
from app.core.database import get_db
from app.services.admin_service import is_admin_request

settings = get_settings()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


async def verify_admin_access(request: Request):
    """
    依赖项：校验当前请求是否包含有效的管理员 Token (Cookie)。
    若未通过校验，抛出 401 异常。
    """
    if not is_admin_request(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权访问，请先登录",
        )


__all__ = ["get_db", "settings", "templates", "verify_admin_access"]
