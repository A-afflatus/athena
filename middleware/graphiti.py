"""
Graphiti 配置模块

提供统一的 Graphiti 管理，支持：
- 延迟初始化
- 单例模式管理 Graphiti 实例
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from graphiti_core import Graphiti
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GraphitiConfig:
    """
    Graphiti 配置管理器

    负责配置和管理 Graphiti 实例。使用单例模式确保全局只有一个配置实例。

    Attributes:
        neo4j_uri: Neo4j 连接 URI
        neo4j_user: Neo4j 用户名
        neo4j_password: Neo4j 密码
        auto_init_indices: 是否自动初始化索引和约束
        graphiti: Graphiti 实例
    """

    _instance: GraphitiConfig | None = None
    _configured: bool = False

    def __new__(cls) -> GraphitiConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if GraphitiConfig._configured:
            return
        self.graphiti: Graphiti | None = None

    async def configure(self) -> None:
        self.neo4j_uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        # 初始化 Graphiti 实例
        self.graphiti = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=OpenAIGenericClient(config=LLMConfig(
                api_key=os.getenv("LLM_QWEN_API_KEY") or "",
                model="qwen-flash",
                small_model="qwen-flash",
                base_url=os.getenv("LLM_QWEN_BASE_URL"),
            )),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=os.getenv("LLM_QWEN_API_KEY") or "",
                    embedding_model="qwen2.5-vl-embedding",
                    base_url=os.getenv("LLM_QWEN_BASE_URL"),
                )
            )
            # todo trace 集成
        )

        GraphitiConfig._configured = True

        logger.info("正在初始化 Graphiti 索引和约束...")
        await self.graphiti.build_indices_and_constraints()
        logger.info("Graphiti 索引和约束初始化完成")

        logger.info(f"Graphiti 配置完成 | URI: {self.neo4j_uri} | User: {self.neo4j_user}")

    async def close(self) -> None:
        """关闭 Graphiti 连接"""
        if self.graphiti is not None:
            await self.graphiti.close()
            logger.info("Graphiti 连接已关闭")


# 全局 Graphiti 配置实例
_graphiti_config = GraphitiConfig()


async def setup_graphiti() -> None:
    await _graphiti_config.configure()


def get_graphiti() -> Graphiti:
    if _graphiti_config.graphiti is None:
        raise RuntimeError("Graphiti 尚未配置，请先调用 setup_graphiti()")
    return _graphiti_config.graphiti


async def close_graphiti() -> None:
    """关闭 Graphiti 连接"""
    if _graphiti_config.graphiti is not None:
        await _graphiti_config.close()
