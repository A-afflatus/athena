import os
from langchain_mcp_adapters.client import MultiServerMCPClient


async def init_mcp_tools():
    """初始化mcp工具"""
    mcp_client = MultiServerMCPClient(
        {
            # 高德mcp工具
            "amap": {
                "transport": "streamable_http",
                "url": f"https://mcp.amap.com/mcp?key={os.getenv('MCP_AMAP_KEY')}",
            },
            # * 经测试，这个websearch机器不准确 千问模型自带联网搜索，就不在这里设置mcp工具了
            # "bing-web-search": {
            #     "transport": "stdio",
            #     "command": "bunx",
            #     "args": ["bing-cn-mcp"],
            # }
        }
    )
    tools_all = await mcp_client.get_tools()
    tools = [
        tool
        for tool in tools_all
        if getattr(tool, "name", None) == "maps_weather"
        or not getattr(tool, "name", "").startswith("maps")
    ]
    # 强化 maps_weather 工具的描述，使其更清晰和详细
    for tool in tools:
        if getattr(tool, "name", None) == "maps_weather":
            tool.description = "根据省市名称(如北京、安徽、安庆等,最低到市级别不能再细分)查询指定城市未来几天的天气情况"
    return tools
