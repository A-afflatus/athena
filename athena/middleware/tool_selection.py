"""
工具选择中间件

根据用户意图动态调整agent携带的工具列表，提高效率和准确性。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast, override

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

from athena.context import AthenaState, ChatEventData, DialogueContext, IntentionType
from athena.tools.base import ToolType, Tools
from bootstrap.logger import get_logger

logger = get_logger(__name__)


class ToolSelectionMiddleware(AgentMiddleware[AthenaState, DialogueContext]):
    """根据意图动态选择工具的中间件"""

    def __init__(self, tools: Tools):
        self._tools_collection: Tools = tools

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 从上下文中获取用户意图列表
        runtime = request.runtime
        context = cast(DialogueContext, runtime.context)
        intentions = (
            context.user_intention
            if context.user_intention
            else [IntentionType.GENERAL]
        )

        # 根据意图列表选择工具（合并所有意图需要的工具）
        selected_tools = self._tools_collection.query_tools(
            type=ToolType.GENERAL, intentions=intentions
        )

        logger.info(f"选择工具: 【{[tool.name for tool in selected_tools]}】")
        await adispatch_custom_event(
            "tool_selection",
            ChatEventData(
                chunk_type="text",
                content=f"选择工具: 【{[tool.name for tool in selected_tools]}】",
            ),
        )
        # override工具列表
        request = request.override(tools=[tool.get_tool() for tool in selected_tools])
        # 调用处理器执行模型请求
        return await handler(request)
