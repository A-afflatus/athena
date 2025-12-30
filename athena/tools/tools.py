from athena.tools.mcp import init_mcp_tools

async def init_tools():
    """初始化工具"""
    tools = await init_mcp_tools()
    return tools