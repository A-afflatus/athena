#!/usr/bin/env python3
"""
Athena - 智能个人助理

主程序入口，负责：
1. 应用引导和初始化
2. 信号处理和优雅退出
3. 主业务逻辑执行
"""

from __future__ import annotations

import logging
import signal
import sys
from types import FrameType
from typing import TYPE_CHECKING

from bootstrap import bootstrap
from config.logger import get_logger

if TYPE_CHECKING:
    from bootstrap import AppContext


class Application:
    """
    应用程序主类
    
    封装应用的生命周期管理，包括启动、运行、停止等阶段。
    
    Attributes:
        ctx: 应用上下文
        _running: 运行状态标志
    """
    
    def __init__(self) -> None:
        self.ctx: AppContext | None = None
        self._logger: logging.Logger | None = None
    
    def start(self) -> None:
        try:
            # 引导应用
            self.ctx = bootstrap()
            self._logger = get_logger(__name__)
            
            # 注册信号处理
            self._setup_signal_handlers()
            self._run()
            
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
            self._shutdown()
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """处理信号"""
        signal_name = signal.Signals(signum).name
        if self._logger is not None:
            self._logger.info(f"收到信号: {signal_name}")
        self._running = False
    
    def _run(self) -> None:
        # _logger 在此方法调用前已初始化
        assert self._logger is not None
        profile = self.ctx.profile if self.ctx else "unknown"
        self._logger.info(f"当前运行环境: {profile}")

        # TODO: 在这里添加你的业务逻辑
        # 示例：启动 API 服务器、任务调度器等


        self._logger.info("Athena 已启动！")

    
    def _shutdown(self) -> None:
        self._running = False
        
        if self._logger is not None:
            self._logger.info("应用正在关闭...")
            # TODO: 在这里添加清理逻辑
            # 例如：关闭数据库连接、停止后台任务等
            self._logger.info("应用已安全退出")


def main() -> int:
    app = Application()
    app.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
