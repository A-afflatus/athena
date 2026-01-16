"""
日志配置模块

提供统一的日志管理，支持：
- 自定义格式（包含 trace_id）
- 文件输出（带日志轮转）
- 控制台输出（可选彩色）
- 多种日志级别
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# 默认配置
DEFAULT_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] "
    "[%(machine_id)s.%(thread)d.%(trace_id)-16s] >>> %(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILE = "logs/athena.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5


class TraceIdFilter(logging.Filter):
    """
    日志过滤器：为日志记录添加 trace_id 和 machine_id
    
    确保每条日志记录都有这些字段，便于分布式追踪。
    """
    
    def __init__(self, machine_id: str = "000000"):
        super().__init__()
        self.machine_id = machine_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 添加默认的 trace_id（如果不存在）
        if not hasattr(record, "trace_id"):
            record.trace_id = "________________"
        
        # 添加 machine_id
        record.machine_id = self.machine_id
        
        return True


class ColoredFormatter(logging.Formatter):
    """
    带颜色的日志格式化器（仅用于控制台输出）
    
    不同日志级别使用不同颜色，提高可读性。
    """
    
    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",     # 青色
        "INFO": "\033[32m",      # 绿色
        "WARNING": "\033[33m",   # 黄色
        "ERROR": "\033[31m",     # 红色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        # 保存原始级别名称
        original_levelname = record.levelname
        
        # 添加颜色
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # 格式化
        result = super().format(record)
        
        # 恢复原始级别名称
        record.levelname = original_levelname
        
        return result


class LoggerConfig:
    """
    日志配置管理器
    
    负责配置和管理应用日志系统。使用单例模式确保全局只有一个配置实例。
    
    Attributes:
        log_level: 日志级别
        log_file: 日志文件路径
        machine_id: 机器标识
        console_output: 是否输出到控制台
        colored_output: 控制台是否使用彩色输出
    """
    
    _instance: LoggerConfig | None = None
    _configured: bool = False
    
    def __new__(cls) -> LoggerConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if LoggerConfig._configured:
            return
        
        self.log_level: str = "INFO"
        self.log_file: str = DEFAULT_LOG_FILE
        self.machine_id: str = "000000"
        self.console_output: bool = True
        self.colored_output: bool = True
        self.max_bytes: int = DEFAULT_MAX_BYTES
        self.backup_count: int = DEFAULT_BACKUP_COUNT
    
    def configure(
        self,
        log_level: str = "INFO",
        log_file: str = DEFAULT_LOG_FILE,
        machine_id: str = "000000",
        console_output: bool = True,
        colored_output: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
    ) -> None:
        """
        配置日志系统
        
        Args:
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: 日志文件路径
            machine_id: 机器标识，用于分布式环境
            console_output: 是否输出到控制台
            colored_output: 控制台是否使用彩色
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的日志文件数量
        """
        self.log_level = log_level.upper()
        self.log_file = log_file
        self.machine_id = machine_id
        self.console_output = console_output
        self.colored_output = colored_output
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置根日志器
        self._setup_root_logger()
        
        LoggerConfig._configured = True
    
    def _setup_root_logger(self) -> None:
        """配置根日志器"""
        # 获取应用的根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVELS.get(self.log_level, logging.INFO))
        
        # 清除已有的处理器
        root_logger.handlers.clear()
        
        # 创建过滤器
        trace_filter = TraceIdFilter(self.machine_id)
        
        # 文件处理器（带轮转）
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(LOG_LEVELS.get(self.log_level, logging.INFO))
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT))
        file_handler.addFilter(trace_filter)
        root_logger.addHandler(file_handler)
        
        # 控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(LOG_LEVELS.get(self.log_level, logging.INFO))
            
            # 根据配置选择是否使用彩色输出
            if self.colored_output and sys.stdout.isatty():
                formatter = ColoredFormatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            else:
                formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            
            console_handler.setFormatter(formatter)
            console_handler.addFilter(trace_filter)
            root_logger.addHandler(console_handler)
        
        # 阻止日志向上传播到根日志器
        root_logger.propagate = False
        
        # 设置 graphiti_core 相关日志级别为 ERROR（减少索引相关的 INFO 日志）
        graphiti_logger = logging.getLogger("graphiti_core")
        graphiti_logger.setLevel(logging.ERROR)
        
        # 设置 Neo4j 驱动日志级别为 ERROR（减少数据库相关的 INFO 日志）
        neo4j_logger = logging.getLogger("neo4j")
        neo4j_logger.setLevel(logging.ERROR)


# 全局日志配置实例
_logger_config = LoggerConfig()


def setup_logger(
    log_level: str = "INFO",
    log_file: str = DEFAULT_LOG_FILE,
    machine_id: str = "000000",
    console_output: bool = True,
    colored_output: bool = True,
) -> None:
    """
    设置日志系统
    
    这是配置日志的主入口函数，应在应用启动时调用一次。
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        machine_id: 机器标识
        console_output: 是否输出到控制台
        colored_output: 是否使用彩色输出
    """
    _logger_config.configure(
        log_level=log_level,
        log_file=log_file,
        machine_id=machine_id,
        console_output=console_output,
        colored_output=colored_output,
    )


def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)
