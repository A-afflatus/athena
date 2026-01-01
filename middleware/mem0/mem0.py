import os

from mem0 import MemoryClient

from config.logger import get_logger

logger = get_logger(__name__)

mem0 = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
def get_mem0() -> MemoryClient:
    return mem0
