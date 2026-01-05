import os
from typing import Callable
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from athena.context import IntentionType
from athena.tools.base import BaseToolEntity

# 工具包装映射
wrap_map: dict[str, Callable[[BaseTool], BaseToolEntity]] = {
    "maps_weather": lambda tool: (
        BaseToolEntity(
            tool=tool,
            intentions=[IntentionType.WEATHER],
            # 强化 maps_weather 工具的描述，使其更清晰和详细
            description="根据省市名称(如北京、安徽、安庆等,最低到市级别不能再细分)查询指定城市未来几天的天气情况",
        )
    )
}


async def init_mcp_tools():
    """初始化mcp工具"""
    mcp_client = MultiServerMCPClient(
        {
            # 高德mcp工具
            "amap": {
                "transport": "streamable_http",
                "url": f"https://mcp.amap.com/mcp?key={os.getenv('MCP_AMAP_KEY')}",
            }
        }
    )
    tools_all = await mcp_client.get_tools()
    # 筛选包装工具
    return [wrap_map[tool.name](tool) for tool in tools_all if tool.name in wrap_map]
