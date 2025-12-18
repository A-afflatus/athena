"""
Athena 主程序入口
"""
from config.logger import get_logger
from setup import load_config
import logging

def main():
    # 加载配置
    load_config()

    logger = get_logger()

    logger.info("Hello from athena!", extra={"traceid": "66f8a9d2e7b3c4a0"})


if __name__ == "__main__":
    main()
