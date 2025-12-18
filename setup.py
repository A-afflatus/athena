# 加载配置
import argparse

from dotenv import load_dotenv

from config.logger import setup_logger
from config.settings import mount_config


def load_config():
    # 加载环境变量
    load_dotenv()
    # 加载参数
    parser = argparse.ArgumentParser(
        prog="athena",
        description="Athena personal assistant.",
        epilog="Copyright(r), 2025",
    )
    parser.add_argument("--profile", default="dev", help="Environment profile to use")
    # 解析参数
    args = parser.parse_args()
    # 机器id
    machine_id = "000000"
    # 环境
    profile = args.profile
    # 挂载配置
    settings = mount_config(profile)
    # 设置日志
    setup_logger(
        log_level=settings.get("log.level"),
        log_file=settings.get("log.file"),
        machine_id=machine_id,
        console_output=settings.get("log.console_output"),
    )
