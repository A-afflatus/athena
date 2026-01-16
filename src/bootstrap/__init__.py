"""
配置模块

提供应用配置和日志管理功能。

使用示例:
    >>> # 获取日志器
    >>> logger = get_logger(__name__)
    >>> logger.info("Hello!")
"""

from src.bootstrap.bootstrap import AppContext, bootstrap

__all__ = [
    "AppContext",
    "bootstrap"
]
