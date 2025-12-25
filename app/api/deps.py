"""
本文件用于提供 FastAPI 依赖注入与模板引擎实例的集中出口。
主要对象:
- `settings`: 全局配置对象
- `templates`: Jinja2 模板渲染器
- `get_db`: 数据库会话依赖注入生成器
"""

from fastapi.templating import Jinja2Templates

from app.core.config import BASE_DIR, get_settings
from app.core.database import get_db

settings = get_settings()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

__all__ = ["get_db", "settings", "templates"]
