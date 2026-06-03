"""
本文件用于加载项目运行配置：`.env` 仅保存管理员密码，其余配置读取 `config.yaml`。
主要函数/类:
- `Settings`: 运行时配置模型（支持类型校验与默认值）
- `get_settings`: 获取配置单例（带缓存）
- `get_missing_config_keys`: 计算关键配置缺失项（用于页面提示）
- `_normalize_yaml_config`: 将 YAML 配置键标准化为大写
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.config_io import load_yaml_dict

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.yaml"


def _normalize_yaml_config(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        if isinstance(k, str):
            normalized[k.upper()] = v
        else:
            normalized[str(k).upper()] = v
    return normalized


class Settings(BaseSettings):
    """
    输入:
    - 环境变量与 `.env` 文件中的配置项

    输出:
    - 统一的运行时配置对象

    作用:
    - 集中管理项目运行所需的配置，并提供默认值与类型校验
    """

    APP_NAME: str = "TrendSonar"
    VERSION: str = "0.2.5"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_RETENTION_DAYS: int = 3
    PORT: int = 8193

    DATABASE_URL: Optional[str] = None

    SILICONFLOW_API_KEY: Optional[str] = None
    SILICONFLOW_BASE_URL: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    MAIN_AI_API_KEY: Optional[str] = None
    MAIN_AI_BASE_URL: Optional[str] = None
    MAIN_AI_MODEL: Optional[str] = None
    MAIN_AI_CONCURRENCY: int = 10
    BACKUP_AI_API_KEY: Optional[str] = None
    BACKUP_AI_BASE_URL: Optional[str] = None
    BACKUP_AI_MODEL: Optional[str] = None
    BACKUP_AI_CONCURRENCY: int = 10

    CRAWLER_CONCURRENCY: int = 2
    CRAWLER_CONTENT_MIN_LENGTH: int = 30
    CRAWLER_FAST_WAIT_SECONDS: float = 1.5
    CRAWLER_DYNAMIC_WAIT_SECONDS: float = 6.0
    CRAWLER_FETCH_TIMEOUT_SECONDS: float = 45.0
    CRAWLER_PAGE_TIMEOUT_MS: int = 60000
    CRAWLER_RETRY_ATTEMPTS: int = 2
    CRAWLER_RETRY_DELAY_SECONDS: float = 8.0
    EMBEDDING_CONCURRENCY: int = 5
    LLM_CONCURRENCY: int = 5

    DATA_CLEANUP_ENABLED: bool = False
    DATA_CLEANUP_MIN_HEAT: float = 1.0
    DATA_CLEANUP_PROTECT_DAYS: int = 3

    CLUSTERING_THRESHOLD: float = 0.78
    CLUSTERING_TIME_WINDOW_HOURS: int = 24
    SCHEDULE_INTERVAL_MINUTES: int = 30
    AUTO_SUMMARY_TOP_N: int = 20
    REGENERATE_SUMMARY_TOP_N: int = 100
    AUTO_ANALYSIS_TOP_N: int = 5
    ANALYSIS_BATCH_SIZE: int = 10

    # Token 优化配置
    ANALYSIS_INPUT_MAX_LENGTH: int = 1000
    SUMMARY_INPUT_MAX_LENGTH: int = 5000
    SUMMARY_ORIGIN_MAX_LENGTH: int = 300
    SUMMARY_OUTPUT_LENGTH: int = 300

    WEIBO_COOKIE: str = ""

    FOLLOW_KEYWORDS: str = ""
    FOLLOW_KEYWORDS_THRESHOLD: float = 0.65

    NEWS_CATEGORIES: List[str] = ["时政军事", "财经商业", "科技科学", "社会民生", "文娱体育"]

    ADMIN_PASSWORD: Optional[str] = None

    IGNORED_DOMAINS: List[str] = []

    TOPIC_LOOKBACK_DAYS: int = 3
    TOPIC_AGGREGATION_TOP_N: int = 300
    TOPIC_GENERATION_COUNT: str = "3-10"
    TOPIC_RECALL_POOL_SIZE: int = 2000
    TOPIC_MATCH_THRESHOLD: float = 0.5
    TOPIC_MATCH_MAX_CANDIDATES: int = 20
    TOPIC_MIN_NEWS_COUNT: int = 3
    TOPIC_UPDATE_LOOKBACK_DAYS: int = 3
    TOPIC_SCHEDULE_INTERVAL_HOURS: int = 4
    TOPIC_NEWS_MIN_HEAT: float = 1.0  # 参与专题生成的最低新闻热度
    TOPIC_QUALITY_LEVEL: int = 3  # 专题质量审核等级 (1-5)
    TOPIC_DISCOVERY_MAX_NEWS_FOR_CLUSTER: int = 800  # 每轮进入程序聚类的最高热度新闻数量
    TOPIC_DISCOVERY_MAX_CLUSTERS: int = 30  # 每轮最多送 AI 审核的候选事件簇数量
    TOPIC_DISCOVERY_AI_BATCH_SIZE: int = 15  # 候选事件簇批量审核大小
    TOPIC_CLUSTER_MIN_SCORE: float = 55.0  # 程序聚类候选进入 AI 审核的最低分
    TOPIC_CLUSTER_SIM_THRESHOLD: float = 0.62  # 常规语义合并阈值
    TOPIC_CLUSTER_STRONG_SIM_THRESHOLD: float = 0.78  # 强语义合并阈值
    TOPIC_CLUSTER_MAX_DAYS_SPAN: int = 7  # 同一事件簇允许的最大日期跨度
    TOPIC_EVIDENCE_MAX_TITLES: int = 6  # 单个候选证据包最多保留标题数
    TOPIC_EVIDENCE_MAX_CHARS: int = 900  # 单个候选证据包事实摘要最大字符数
    TOPIC_TIMELINE_MAX_EVENTS_PER_DAY: int = 4  # 每天时间轴节点上限
    TOPIC_OVERVIEW_MAX_NEWS: int = 30  # 专题综述最多使用的新闻素材数
    TOPIC_CREATE_MAX_PER_RUN: int = 5  # 每轮最多新建专题数
    TOPIC_MIN_SOURCE_COUNT: int = 2  # 自动生成专题要求的最少来源数
    TOPIC_EXISTING_MERGE_HINT_THRESHOLD: float = 0.68  # 候选簇提示合并现有专题的相似度阈值

    # AI 路由配置
    AI_ROUTE: Dict[str, str] = {
        "SUMMARY": "main",
        "SENTIMENT": "backup",
        "KEYWORDS": "backup",
        "CLUSTERING": "backup",
        "TOPIC_NAME": "backup",
        "TOPIC_EVAL": "backup",
        "TOPIC_MATCH": "backup",
        "TOPIC_TIMELINE": "backup",
        "TOPIC_OVERVIEW": "backup",
        "TOPIC_INITIAL_SUMMARY": "main",
        "REPORT": "backup",
        "CHAT": "backup"
    }

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        def yaml_settings():
            return _normalize_yaml_config(load_yaml_dict(CONFIG_PATH))

        return (
            init_settings,
            yaml_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def get_missing_config_keys(settings: Settings) -> List[str]:
    required_keys = [
        "DATABASE_URL",
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_BASE_URL",
        "EMBEDDING_MODEL",
        "MAIN_AI_API_KEY",
        "MAIN_AI_BASE_URL",
        "MAIN_AI_MODEL",
    ]

    missing: List[str] = []
    for k in required_keys:
        v = getattr(settings, k, None)
        if v is None:
            missing.append(k)
            continue
        if isinstance(v, str) and not v.strip():
            missing.append(k)
            continue
    if not (settings.ADMIN_PASSWORD or "").strip():
        missing.append("ADMIN_PASSWORD")
    return missing


@lru_cache()
def get_settings() -> Settings:
    """
    输入:
    - 无

    输出:
    - `Settings` 单例实例

    作用:
    - 通过缓存避免重复解析环境变量，提高配置读取性能
    """

    return Settings()


def reload_settings() -> Settings:
    """
    输入:
    - 无
    
    输出:
    - 重新加载后的 Settings 实例
    
    作用:
    - 清除 get_settings 的缓存并重新加载配置，用于运行时配置更新
    """
    get_settings.cache_clear()
    return get_settings()
