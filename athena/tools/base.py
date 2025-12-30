from enum import Enum
from langchain_core.tools import BaseTool

from athena.context import IntentionType


class ToolType(Enum):
    """工具类型"""

    USER = "user"  # 按用户限定的工具
    SYSTEM = "system"  # 系统工具,只有任务Agent能触发的工具
    GENERAL = "general"  # 通用工具


class BaseToolEntity:
    """基础工具实体"""

    type: ToolType
    intentions: list[IntentionType]
    tool: BaseTool
    name: str

    def __init__(
        self,
        tool: BaseTool,
        type: ToolType,
        intentions: list[IntentionType] = [],
    ):
        self.tool = tool
        self.type = type
        self.intentions = intentions
        self.name = tool.name

    def is_available(self, intentions: list[IntentionType]) -> bool:
        return any(intention in self.intentions for intention in intentions)

    def get_tool(self) -> BaseTool:
        return self.tool


class UserTool(BaseToolEntity):
    """按用户限定的工具"""

    user_ids: list[str]

    def __init__(
        self,
        tool: BaseTool,
        intentions: list[IntentionType] = [],
        user_ids: list[str] = [],
    ):
        super().__init__(tool=tool, type=ToolType.USER, intentions=intentions)
        self.user_ids = user_ids

    def is_user_available(self, user_id: str) -> bool:
        return any(user_id == uid for uid in self.user_ids)


class SystemTool(BaseToolEntity):
    """只有任务Agent能触发的工具"""

    def __init__(self, tool: BaseTool):
        super().__init__(tool=tool, type=ToolType.SYSTEM)


class GeneralTool(BaseToolEntity):
    """通用工具"""

    def __init__(self, tool: BaseTool, intentions: list[IntentionType] = []):
        super().__init__(tool=tool, type=ToolType.GENERAL, intentions=intentions)


class Tools:
    """工具集合"""

    tools: list[BaseToolEntity]

    def __init__(self, tools: list[BaseToolEntity]):
        self.tools = tools

    def get_tools(self) -> list[BaseTool]:
        return [tool.get_tool() for tool in self.tools]

    def add_wrap_tool(self) -> list[BaseToolEntity]:
        return self.tools
