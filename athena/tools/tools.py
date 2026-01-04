from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from openai import BaseModel
from pydantic import Field

from athena.context import DialogueContext, UserGender
from athena.tools.base import GeneralTool, Tools
from athena.tools.mcp import init_mcp_tools


async def init_tools() -> Tools:
    """初始化工具"""
    tools = await init_mcp_tools()
    return Tools(tools=[GeneralTool(tool=tool) for tool in tools] + [GeneralTool(tool=save_user_info)])


class SaveUserInfoInput(BaseModel):
    """保存用户信息输入"""
    user_name: str | None = Field(description="用户名称")
    user_gender: UserGender | None = Field(description="用户性别")
    user_location: str | None = Field(description="用户当前位置")

@tool
def save_user_info(runtime: ToolRuntime[DialogueContext], input: SaveUserInfoInput) -> Command:
    """
    保存用户信息，用于第一次与用户沟通，和用户交流获得用户的基本信息
    Example:
        save_user_info(input=SaveUserInfoInput(user_name="张三", user_gender="男", user_location="北京"))
    """
    runtime.context.user_name = input.user_name if input.user_name else None
    runtime.context.user_gender = input.user_gender if input.user_gender else None
    runtime.context.user_location = input.user_location if input.user_location else None

    return Command(
        update={
            "context": runtime.context,
            "messages": [
                ToolMessage(
                    content="用户信息已成功保存",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
