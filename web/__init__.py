"""
Web 模块初始化
自动注册所有路由和 WebSocket 端点
"""
from __future__ import annotations

from web.web import app
import web.robot  # pyright: ignore[reportUnusedImport]
import web.asr  # pyright: ignore[reportUnusedImport]
import web.tts  # pyright: ignore[reportUnusedImport]

__all__ = ["app"]