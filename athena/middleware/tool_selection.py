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

from athena.context import AthenaState, DialogueContext, IntentionType
from athena.tools.base import BaseToolEntity
from bootstrap.logger import get_logger

logger = get_logger(__name__)

_tools_group = {
    IntentionType.WEATHER: ["maps_weather", "save_user_info"],
}

# todo 这个工具选择器感觉可以做一层加工，类似于skills那样的模式，先解析提示词，然后做意图归纳，最后生成对应提示词

class ToolSelectionMiddleware(AgentMiddleware[AthenaState, DialogueContext]):
    """根据意图动态选择工具的中间件"""

    def __init__(self, all_tools: list[BaseToolEntity]):
        self.all_tools = all_tools
        # 构建工具名称到工具的映射
        self.tool_map = {tool.name: tool for tool in all_tools}

    def _get_tools_for_intention(
        self, intentions: list[IntentionType] | None
    ) -> list[BaseTool]:
        if not intentions or len(intentions) == 0:
            # 如果没有识别到意图，返回所有工具
            return [tool.get_tool() for tool in self.all_tools]

        # 将字符串意图转换为枚举类型

        # 如果包含退出意图，不需要任何工具
        if IntentionType.EXIT in intentions:
            return []

        # 收集所有需要的工具名称（去重）
        selected_tool_names = set()

        for intention_type in intentions:
            if intention_type == IntentionType.WEATHER:
                # 天气意图需要天气相关工具
                weather_tools = [
                    tool.get_tool()
                    for tool in self.all_tools
                    if tool.name.lower() in _tools_group[IntentionType.WEATHER]
                ]
                if weather_tools:
                    selected_tool_names.update(tool.name for tool in weather_tools)
                else:
                    # 如果没有专门的天气工具，添加所有工具
                    selected_tool_names.update(tool.name for tool in self.all_tools)

            elif intention_type == IntentionType.SEARCH:
                # todo 这里现在还没有对应的搜索工具
                continue
            elif intention_type == IntentionType.GENERAL:
                # 通用意图，添加所有工具
                selected_tool_names.update(tool.name for tool in self.all_tools)

        # 转换为列表并保持顺序（按原始工具列表的顺序）
        selected_tools = [
            tool for tool in self.all_tools if tool.name in selected_tool_names
        ]

        logger.info(
            f"意图 {[intention.value for intention in intentions]}，选择工具: {[t.name for t in selected_tools]} "
            f"({len(selected_tools)}/{len(self.all_tools)})"
        )
        if len(selected_tools) == 0:
            return [tool.get_tool() for tool in self.all_tools]

        # 如果没有任何工具被选中，返回所有工具作为默认
        return [tool.get_tool() for tool in selected_tools]

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 从上下文中获取用户意图列表
        runtime = request.runtime
        context = cast(DialogueContext, runtime.context)
        intentions = context.user_intention

        # 根据意图列表选择工具（合并所有意图需要的工具）
        selected_tools = self._get_tools_for_intention(intentions)

        # override工具列表
        request = request.override(tools=selected_tools)  # type: ignore
        # 调用处理器执行模型请求
        return await handler(request)
