"""
工具选择中间件

根据用户意图动态调整agent携带的工具列表，提高效率和准确性。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.tools import BaseTool

from athena.config.logger import get_logger
from athena.core.entity.entity import DialogueContext, DialogueState, IntentionType

logger = get_logger(__name__)

_tools_group = {
    IntentionType.WEATHER: ["maps_weather"],
}


class ToolSelectionMiddleware(AgentMiddleware[DialogueState, DialogueContext]):
    """根据意图动态选择工具的中间件"""

    def __init__(self, all_tools: list[BaseTool]):
        self.all_tools = all_tools
        # 构建工具名称到工具的映射
        self.tool_map = {tool.name: tool for tool in all_tools}

    def _get_tools_for_intention(self, intentions: list[str] | None) -> list[BaseTool]:
        if not intentions or len(intentions) == 0:
            # 如果没有识别到意图，返回所有工具
            return self.all_tools

        # 将字符串意图转换为枚举类型
        intention_types = [IntentionType(intention) for intention in intentions]

        # 如果包含退出意图，不需要任何工具
        if IntentionType.EXIT in intention_types:
            return []

        # 收集所有需要的工具名称（去重）
        selected_tool_names = set()

        for intention_type in intention_types:
            if intention_type == IntentionType.WEATHER:
                # 天气意图需要天气相关工具
                weather_tools = [
                    tool
                    for tool in self.all_tools
                    if tool.name.lower() in _tools_group[IntentionType.WEATHER]
                ]
                if weather_tools:
                    selected_tool_names.update(tool.name for tool in weather_tools)
                else:
                    # 如果没有专门的天气工具，添加所有工具
                    selected_tool_names.update(tool.name for tool in self.all_tools)

            elif intention_type == IntentionType.SEARCH:
                # 搜索意图可能需要搜索相关工具（如果有的话）
                #todo 这里现在还没有对应的搜索工具
                search_tools = [
                    tool
                    for tool in self.all_tools
                    if "search" in tool.name.lower() or "搜索" in getattr(tool, "description", "").lower()
                ]
                if search_tools:
                    selected_tool_names.update(tool.name for tool in search_tools)
                else:
                    # 如果没有专门的搜索工具，添加所有工具（让模型使用内置搜索）
                    selected_tool_names.update(tool.name for tool in self.all_tools)

            elif intention_type == IntentionType.GENERAL:
                # 通用意图，添加所有工具
                selected_tool_names.update(tool.name for tool in self.all_tools)

        # 转换为列表并保持顺序（按原始工具列表的顺序）
        selected_tools = [
            tool for tool in self.all_tools if tool.name in selected_tool_names
        ]

        logger.info(
            f"意图 {intentions}，选择工具: {[t.name for t in selected_tools]} "
            f"({len(selected_tools)}/{len(self.all_tools)})"
        )

        # 如果没有任何工具被选中，返回所有工具作为默认
        return selected_tools if selected_tools else self.all_tools

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 从状态中获取用户意图列表
        state = cast(DialogueState, request.state)
        intentions = state.get("user_intention")

        # 根据意图列表选择工具（合并所有意图需要的工具）
        selected_tools = self._get_tools_for_intention(intentions)

        # 如果工具列表有变化，使用 override 创建新的请求
        if len(selected_tools) != len(request.tools) or {t.name for t in selected_tools} != {t.name for t in request.tools}:
            logger.debug(
                f"根据意图 {intentions} 调整工具: "
                f"从 {len(request.tools)} 个工具调整为 {len(selected_tools)} 个工具"
            )
            request = request.override(tools=selected_tools)

        # 调用处理器执行模型请求
        return await handler(request)

