"""
本包用于集中导出服务层单例，便于上层直接引用。
主要导出:
- `ai_service`
- `cluster_service`
- `crawler_service`
- `report_service`
"""

from app.services.ai_service import ai_service
from app.services.cluster_service import cluster_service
from app.services.crawler_service import crawler_service
from app.services.report_service import report_service

__all__ = ["ai_service", "cluster_service", "crawler_service", "report_service"]

