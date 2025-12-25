"""
本文件用于定义系统相关的请求体/响应体数据模型。
主要类:
- `AdminAuth`: 管理接口鉴权请求体
"""

from pydantic import BaseModel


class AdminAuth(BaseModel):
    """
    输入:
    - `password`: 管理员口令

    输出:
    - 管理接口鉴权请求体模型

    作用:
    - 为管理端接口提供简单鉴权参数结构
    """

    password: str
