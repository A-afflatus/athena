"""
配置模块

提供应用配置和日志管理功能。

使用示例:
    >>> from config import settings, get_logger
    >>> 
    >>> # 获取配置
    >>> log_level = settings.get("log.level")
    >>> 
    >>> # 获取日志器
    >>> logger = get_logger(__name__)
    >>> logger.info("Hello!")
"""

from config.logger import get_logger, setup_logger
from config.settings import Settings, get_settings, settings
from config.bootstrap import AppContext, bootstrap

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "setup_logger",
    "get_logger",
    "AppContext",
    "bootstrap"
]
