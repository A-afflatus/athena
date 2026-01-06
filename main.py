#!/usr/bin/env python3
"""
Athena - 智能个人助理

主程序入口，负责：
1. 应用引导和初始化
2. 信号处理和优雅退出
3. 主业务逻辑执行
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import TYPE_CHECKING

import uvicorn
from bootstrap.bootstrap import bootstrap
from bootstrap.logger import get_logger
from middleware.graphiti import close_graphiti
from web import app

if TYPE_CHECKING:
    from bootstrap.bootstrap import AppContext

logger = get_logger(__name__)


class Application:
    """
    应用程序主类

    封装应用的生命周期管理，包括启动、运行、停止等阶段。

    Attributes:
        ctx: 应用上下文
    """

    def __init__(self) -> None:
        self.ctx: AppContext | None = None
        self._logger: logging.Logger | None = None
        self._shutdown_event: asyncio.Event | None = None
        # web服务
        self._uvicorn_server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    def start(self) -> None:
        """同步启动入口，内部调用异步方法"""
        asyncio.run(self._start())

    async def _run(self) -> None:
        # _logger 在此方法调用前已初始化
        assert self._logger is not None
        assert self._shutdown_event is not None

        profile = self.ctx.profile if self.ctx else "unknown"
        logger.info(f"当前运行环境: {profile}")
        # 启动 FastAPI WebSocket 服务
        await self._start_web_server()

    async def _start_web_server(self) -> None:
        """启动 FastAPI WebSocket 服务"""
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8080,
            log_config=None,  # 使用我们自己的日志系统
        )
        self._uvicorn_server = uvicorn.Server(config)
        # 在后台任务中启动 uvicorn 服务器
        self._server_task = asyncio.create_task(self._uvicorn_server.serve())

    async def _start(self) -> None:
        """异步启动方法"""
        try:
            # 在异步上下文中创建事件
            self._shutdown_event = asyncio.Event()

            # 引导应用
            self.ctx = await bootstrap()
            self._logger = get_logger(__name__)

            # 使用 asyncio 的原生信号处理（更简洁且线程安全）
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._handle_signal, sig)

            await self._run()
            # 等待关闭事件
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            if self._logger is not None:
                self._logger.info("收到中断信号，正在退出...")
        except Exception as e:
            if self._logger:
                self._logger.exception(f"应用运行异常: {e}")
            else:
                print(f"启动失败: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """异步关闭方法"""
        if self._logger is not None:
            self._logger.info("应用正在关闭...")

        # 关闭 Graphiti
        logger.info("正在关闭 Graphiti...")
        await close_graphiti()
        logger.info("Graphiti 已关闭")
        # 停止 uvicorn 服务器
        logger.info("正在关闭 FastAPI WebSocket 服务...")
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            await self._server_task if self._server_task else None
        logger.info("FastAPI WebSocket 服务已关闭")
        logger.info("应用已安全退出")

    def _handle_signal(self, signum: int) -> None:
        """处理信号"""
        signal_name = signal.Signals(signum).name
        if self._logger is not None:
            self._logger.info(f"收到信号: {signal_name}")
        # 直接设置事件（因为已经在事件循环的线程中了）
        if self._shutdown_event is not None:
            self._shutdown_event.set()


def main() -> int:
    app = Application()
    app.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
