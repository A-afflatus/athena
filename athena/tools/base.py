from enum import Enum
from re import I
from langchain_core.tools import BaseTool

from athena.context import IntentionType


class ToolType(Enum):
    """工具类型"""

    SYSTEM = "SYSTEM"  # 系统工具,只有任务Agent能触发的工具
    GENERAL = "GENERAL"  # 通用工具


class BaseToolEntity:
    """基础工具实体"""

    type: ToolType
    intentions: list[IntentionType]
    tool: BaseTool
    name: str
    description: str

    def __init__(
        self,
        tool: BaseTool,
        type: ToolType = ToolType.GENERAL,
        intentions: list[IntentionType] = [], # 工具支持的意图列表，当为空时，表示支持所有意图
        description: str | None = None,
    ):
        # 覆盖工具的描述
        if description:
            tool.description = description
        self.tool = tool
        self.type = type
        self.intentions = intentions
        self.name = tool.name
        self.description = tool.description

    def is_available(self, intentions: list[IntentionType]) -> bool:
        return len(self.intentions) == 0 or any(intention in self.intentions for intention in intentions)

    def get_tool(self) -> BaseTool:
        return self.tool


class Tools:
    """工具集合"""

    tools: list[BaseToolEntity]

    def __init__(self, tools: list[BaseToolEntity]):
        self.tools = tools

    def get_tools(self) -> list[BaseTool]:
        return [tool.get_tool() for tool in self.tools]

    def add_wrap_tool(self) -> list[BaseToolEntity]:
        return self.tools

    def query_tools(
        self,
        name: str | None = None,
        type: ToolType | None = None,
        intentions: list[IntentionType] = [],
    ) -> list[BaseToolEntity]:
        return [
            tool
            for tool in self.tools
            if (name is None or tool.name == name)
            and (type is None or tool.type == type)
            and tool.is_available(intentions)
        ]
