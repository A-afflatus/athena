from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from openai import BaseModel
from pydantic import Field

from athena.context import DialogueContext, UserGender
from athena.tools.base import BaseToolEntity, Tools
from athena.tools.mcp import init_mcp_tools
from playwright.async_api import async_playwright


class SaveUserInfoInput(BaseModel):
    """保存用户信息输入"""

    user_name: str | None = Field(description="用户名称")
    user_gender: UserGender | None = Field(description="用户性别")
    user_location: str | None = Field(description="用户当前位置")


@tool
def save_user_info(
    runtime: ToolRuntime[DialogueContext], input: SaveUserInfoInput
) -> Command:
    """
    在对话过程中，当 AI 识别到用户提供了其基本信息（姓名、性别、所在地）且当前对话上下文中缺失这些信息时，调用此工具进行保存。
    注意：此工具仅用于记录先前未知的用户信息。如果上下文已包含这些信息，则不应调用。
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


async def init_tools() -> Tools:
    """初始化工具"""
    # mcp工具
    mcp_tools = await init_mcp_tools()
    # 声明式工具
    declaration_tools = [
        BaseToolEntity(tool=save_user_info),
    ]
    return Tools(tools=mcp_tools + declaration_tools)
