from athena.tools.base import Tools, GeneralTool
from athena.tools.mcp import init_mcp_tools


async def init_tools() -> Tools:
    """初始化工具"""
    tools = await init_mcp_tools()
    return Tools(tools=[GeneralTool(tool=tool) for tool in tools])