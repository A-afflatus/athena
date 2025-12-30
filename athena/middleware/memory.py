from datetime import datetime
import os
import threading
from typing import Any, override
from langgraph.runtime import Runtime
from mem0 import MemoryClient
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from athena.context import DialogueState, DialogueContext
from config.logger import get_logger

logger = get_logger(__name__)


class UserMemoryMiddleware(AgentMiddleware[DialogueState, DialogueContext]):
    def __init__(self):
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY 环境变量未设置")
        self.memory = MemoryClient(api_key=api_key)

    @override
    def before_agent(
        self, state: DialogueState, runtime: Runtime[DialogueContext]
    ) -> dict[str, Any] | None:
        """在 Agent 运行前：检索记忆"""
        long_term_memory = state.get("long_term_memory", [])
        # 一次多轮会话只检索一次长期记忆
        if long_term_memory is not None and len(long_term_memory) > 0:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        user_query = messages[-1].content

        user_id = runtime.context.user_id
        filters = {"OR": [{"user_id": user_id}]}

        # 检索相关事实
        relevant_memories = self.memory.search(user_query, top_k=100, filters=filters)  # type: ignore
        # 将记忆格式化后存入 context，后续可以通过 dynamic_prompt 注入到系统提示词
        if relevant_memories:
            facts = [
                f"时间:{m['created_at']}-内容:{m['memory']}"
                for m in relevant_memories["results"]
            ]
            # 保证一次多轮会话只检索一次长期记忆
            return {
                "long_term_memory": (
                    [
                        f"时间:{m['created_at']}-内容:{m['memory']}"
                        for m in relevant_memories["results"]
                    ]
                    if len(facts) > 0
                    else [
                        f"时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-内容:开始本次对话"
                    ]
                )
            }
        return None

    @override
    def after_agent(self, state: DialogueState, runtime: Any) -> None:
        """在 Agent 结束后：存储记忆"""
        messages = state.get("messages", [])
        user_id = runtime.context.user_id
        # 至少要有用户消息和 AI 回复
        if len(messages) < 2:
            return

        # 在单独线程中执行，不阻塞主流程
        threading.Thread(
            target=self._add_memory_async, args=(messages, user_id), daemon=True
        ).start()

    def _add_memory_async(self, messages: list, user_id: str) -> None:
        """异步添加记忆到存储"""
        try:
            filtered_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    filtered_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content != "":
                    filtered_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
            self.memory.add(filtered_messages, user_id=user_id)
        except Exception as e:
            logger.error(f"异步添加记忆失败: {e}", exc_info=True)
