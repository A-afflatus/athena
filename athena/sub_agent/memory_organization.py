from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from config.logger import get_logger
from middleware.mem0.mem0 import get_mem0
from model.models import init_model

logger = get_logger(__name__)

memory_organization_agent = create_agent(
    model=init_model("qwen3-max-preview"),
    tools=[],
    system_prompt="你是一个记忆组织者，你的任务是组织用户的记忆，并将其存储到记忆库中。",
)

async def save_memory(messages: list[AnyMessage], user_id: str) -> None:
    await _add_memory_async(messages, user_id)


async def _add_memory_async(messages: list, user_id: str) -> None:
    """添加记忆到mem0"""
    mem0 = get_mem0()
    try:
        filtered_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                filtered_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content != "":
                filtered_messages.append(
                    {"role": "assistant", "content": msg.content}
                )
        mem0.add(filtered_messages, user_id=user_id)
    except Exception as e:
        logger.error(f"添加记忆到mem0失败: {e}", exc_info=True)
