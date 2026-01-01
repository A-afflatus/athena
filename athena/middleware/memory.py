from datetime import datetime
from typing import Any, cast, override

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime

from athena.context import AthenaState, DialogueContext
from config.logger import get_logger
from middleware.graphiti.graphiti import get_graphiti
from middleware.mem0.mem0 import get_mem0

logger = get_logger(__name__)


class UserMemoryMiddleware(AgentMiddleware[AthenaState, DialogueContext]):
    def __init__(self):
        # 对话记忆
        self.memory = get_mem0()
        # todo 这个应该是另一个中间件图知识库
        self.graphiti = get_graphiti()

    @override
    async def abefore_agent(
        self, state: AthenaState, runtime: Runtime[DialogueContext]
    ) -> dict[str, Any] | None:
        """在 Agent 运行前：检索记忆"""
        messages = state.get("messages", [])
        if not messages:
            return None

        user_query = cast(str, messages[-1].content)

        user_id = runtime.context.user_id
        filters = {"OR": [{"user_id": user_id}]}

        # 检索相关事实
        relevant_memories = self.memory.search(user_query, top_k=10, filters=filters)  # type: ignore
        # 将记忆格式化后存入 context，后续可以通过 dynamic_prompt 注入到系统提示词
        if relevant_memories:
            facts = [
                f"时间:{m['created_at']}-内容:{m['memory']}"
                for m in relevant_memories["results"]
            ]
            # 保证一次多轮会话只检索一次长期记忆
            runtime.context.long_term_memory = (
                [
                    f"时间:{m['created_at']}-内容:{m['memory']}"
                    for m in relevant_memories["results"]
                ]
                if len(facts) > 0
                else [
                    f"时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-内容:开始本次对话"
                ]
            )
            return {
                "context": runtime.context,
            }
        return None

