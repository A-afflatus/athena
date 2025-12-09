"""
Athena 主程序入口
"""

from dotenv import load_dotenv
from athena.utils.logger import setup_logger

load_dotenv()
logger = setup_logger()

def main():
    """主函数"""
    # logger.info("Hello from athena!", extra={"traceid": "66f8a9d2e7b3c4a0"})
    logger.info("Hello from athena!")


if __name__ == "__main__":
    main()
