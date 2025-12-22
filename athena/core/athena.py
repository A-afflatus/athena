
"""
Athena 核心类

实现智能对话功能，基于 LangChain 构建。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from athena.config.settings import get_settings
from athena.config.logger import get_logger
from langchain_qwq import ChatQwen
from langchain.agents import create_agent



class Athena:
    logger = get_logger(__name__)
    messages = []
    def __init__(self):
        settings = get_settings()
        # 对话agent的系统提示词
        # 统一对话agent
        self.llm_qwen = ChatQwen(
            model_name="qwen-flash", 
            temperature=0.5, 
            enable_thinking=False,
            api_key=SecretStr(settings.get("llm.qwen.api-key")),
            api_base=settings.get("llm.qwen.base-url"),
            verbose=settings.get("is-debug"),
        )
        self.agent = create_agent(
            model=self.llm_qwen,
            tools=[],
            system_prompt="你是一个类似钢铁侠中贾维斯的智能ai助手，回答直击问题根源，不要给出任何解释，直接回答问题。",
            debug=settings.get("is-debug"),
        )
        

    def chat(self, message: str) -> str:
        self.messages.append(HumanMessage(content=message))
        response = self.llm_qwen.invoke(self.messages)
        self.messages.append(AIMessage(content=response.content))
        return response.content