"""
本文件用于定义项目统一的业务异常基类，便于上层统一处理。
主要类:
- `TrendSonarError`: 业务异常基类
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
