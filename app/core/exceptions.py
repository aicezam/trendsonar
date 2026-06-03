"""
本文件用于定义项目统一的业务异常基类，便于上层统一处理。
主要类:
- `TrendSonarError`: 业务异常基类
- `AIConfigurationError`: AI 配置相关异常
- `AIServiceUnavailableError`: AI 服务暂时不可用异常
"""

class TrendSonarError(Exception):
    """
    输入:
    - 业务错误信息

    输出:
    - 异常对象

    作用:
    - 作为项目统一的业务异常基类，便于 API 层集中捕获与转换
    """

    pass


class AIConfigurationError(TrendSonarError):
    """
    输入:
    - AI 配置错误信息

    输出:
    - 异常对象

    作用:
    - 标识 AI 服务配置错误（如 API Key 无效），用于触发服务降级或暂停
    """
    pass


class AIServiceUnavailableError(TrendSonarError):
    """
    输入:
    - AI 服务不可用的错误信息

    输出:
    - 异常对象

    作用:
    - 标识 AI 服务连接、限流或服务端异常在重试后仍不可用，用于终止当前后台任务。
    """

    pass
