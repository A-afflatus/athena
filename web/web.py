from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动时执行
    print("Athena Web程序启动中...")
    yield
    # 关闭时执行
    print("Athena Web程序关闭中...")


app = FastAPI(title="Athena WebSocket API", version="0.1.0", lifespan=lifespan)
