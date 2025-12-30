import os
from typing import Any, override
from langgraph.runtime import Runtime
from mem0 import MemoryClient
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from athena.entity.entity import DialogueState, DialogueContext
from config.logger import get_logger

logger = get_logger(__name__)
class MemoryMiddleware(AgentMiddleware[DialogueState, DialogueContext]):
    def __init__(self):
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY 环境变量未设置")
        self.memory = MemoryClient(api_key=api_key)

    @override
    def before_agent(self, state: DialogueState, runtime: Runtime[DialogueContext]) -> dict[str, Any] | None:
        """在 Agent 运行前：检索记忆"""
        messages = state.get("messages", [])
        if not messages:
            return None
            
        user_query = messages[-1].content
        context = runtime.context
        user_id = context.user_id
        
        filters = {
            "OR":[
                {
                    "user_id": user_id
                }
            ]
        }
        # 检索相关事实
        relevant_memories = self.memory.search(user_query,version="v2",filters=filters) # type: ignore
        # 将记忆格式化后存入 context，后续可以通过 dynamic_prompt 注入到系统提示词
        if relevant_memories:
            facts = [f"记录时间:{m['created_at']}\n记忆内容:{m['memory']}" for m in relevant_memories["results"]]
            # 更新 context，保留原有的 context 属性
            context.long_term_memory = facts
            return {"context": context}
        return None

    @override
    def after_agent(self, state: DialogueState, runtime: Any) -> None:
        """在 Agent 结束后：存储记忆"""
        messages = state.get("messages", [])
        user_id = runtime.context.user_id
        # 至少要有用户消息和 AI 回复
        if len(messages) < 2:
            return

        filtered_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                filtered_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content != "":
                filtered_messages.append({"role": "assistant", "content": msg.content})
        self.memory.add(filtered_messages,user_id=user_id)