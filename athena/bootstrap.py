"""
应用引导模块

负责应用的初始化流程，包括：
1. 加载环境变量
2. 解析命令行参数
3. 加载配置文件
4. 初始化日志系统
5. 获取机器标识

该模块提供了一个干净的启动接口，将所有初始化逻辑封装在一起。
"""

from __future__ import annotations

import argparse
import platform
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from athena.config.logger import setup_logger, get_logger
from athena.config.settings import settings

if TYPE_CHECKING:
    from argparse import Namespace


@dataclass
class AppContext:
    """
    应用上下文
    
    存储应用运行时的关键信息，作为应用状态的统一载体。
    
    Attributes:
        profile: 运行环境 (dev, prod, test)
        machine_id: 机器唯一标识
        debug: 是否为调试模式
    """
    profile: str = "dev"
    machine_id: str = field(default_factory=lambda: _generate_machine_id())
    debug: bool = False
    
    def __post_init__(self) -> None:
        """初始化后的验证和处理"""
        # 确保 profile 是有效的
        valid_profiles = {"dev", "prod", "test", "staging"}
        if self.profile not in valid_profiles:
            logger = get_logger(__name__)
            logger.warning(f"未知的环境配置: {self.profile}，使用 dev 作为默认值")
            self.profile = "dev"


def _generate_machine_id() -> str:
    """结合主机名和 MAC 地址生成一个 6 位的机器 ID。"""
    try:
        # 使用 UUID 的节点部分（基于 MAC 地址）
        node = uuid.getnode()
        hostname = platform.node()
        # 组合并取哈希的前 6 位
        combined = f"{hostname}-{node}"
        return format(hash(combined) & 0xFFFFFF, "06x")
    except Exception:
        return "000000"


def parse_arguments() -> Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="athena",
        description="Athena - 智能个人助理",
        epilog="Copyright © 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--profile", "-p",
        default="dev",
        choices=["dev", "prod", "test"],
        help="运行环境配置 (default: dev)",
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试模式",
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="配置文件目录路径",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="覆盖配置文件中的日志级别",
    )
    
    return parser.parse_args()


def bootstrap(args: Namespace | None = None) -> AppContext:
    """
    引导应用启动
    
    执行完整的初始化流程，返回应用上下文。
    
    Args:
        args: 命令行参数，如果为 None 则自动解析
        
    Returns:
        初始化完成的应用上下文
        
    Raises:
        FileNotFoundError: 配置文件不存在时抛出
        
    Example:
        >>> ctx = bootstrap()
        >>> print(f"运行环境: {ctx.profile}")
    """
    # 1. 加载环境变量（.env 文件）
    load_dotenv()
    
    # 2. 解析命令行参数
    if args is None:
        args = parse_arguments()
    
    # 3. 创建应用上下文
    ctx = AppContext(
        profile=args.profile,
        debug=args.debug,
    )
    
    # 4. 加载配置文件
    settings.load(
        profile=ctx.profile,
        config_dir=args.config_dir,
    )
    
    # 5. 确定日志级别（命令行参数优先）
    log_level = args.log_level or settings.get("log.level", "INFO")
    if ctx.debug:
        log_level = "DEBUG"
    
    # 6. 初始化日志系统
    setup_logger(
        log_level=log_level,
        log_file=settings.get("log.file", "logs/athena.log"),
        machine_id=ctx.machine_id,
        console_output=settings.get("log.console-output", True),
    )
    
    # 7. 记录启动信息
    logger = get_logger(__name__)
    logger.info(f"应用启动 | 环境: {ctx.profile} | 机器ID: {ctx.machine_id}")
    logger.debug(f"配置加载完成: {settings}")
    
    return ctx
