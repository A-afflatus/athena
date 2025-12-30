"""
Athena 核心类

实现智能对话功能，基于 LangChain 构建。
"""

from __future__ import annotations

import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from athena.middleware.dichotomy_prompts import dynamic_system_prompt
from athena.middleware.memory import UserMemoryMiddleware
from athena.tools.base import BaseToolEntity
from config.logger import get_logger
from athena.context import DialogueContext, DialogueState, UserGender, UserType
from athena.middleware.intention_recognition import IntentionRecognitionMiddleware
from athena.middleware.tool_selection import ToolSelectionMiddleware
from athena.tools.tools import init_tools
from model.models import init_model

logger = get_logger(__name__)

# ! 通过声色识别用户，识别到了返回识别的部分结果(用于称呼对方)，并更新用户上下文，识别不到提示ai第一次与用户沟通，和用户交流获得用户的基本信息，调用工具新建用户并更新用户上下文#

class Athena:
    def __init__(self):
        super().__init__()

    def init_llm(self):
        """初始化模型"""
        # 核心llm - 用于对话，需要高质量理解和自然表达
        self.admin_llm = init_model(
            model="qwen3-max", temperature=0.7, extra_body={"enable_search": True}
        )
        # 意图识别llm
        self.intention_llm = init_model(model="qwen-flash", temperature=0.1)
        # 总结llm - 用于上下文摘要，需要快速且准确
        self.summarization_llm = init_model(
            model="qwen-plus", enable_thinking=True, temperature=0.1
        )
    async def init_middleware(self, tools: list[BaseToolEntity]):
        """初始化中间件"""

        # 意图识别中间件
        intention_middleware = IntentionRecognitionMiddleware(llm=self.intention_llm)

        # 工具选择中间件（根据意图动态调整工具）
        tool_selection_middleware = ToolSelectionMiddleware(all_tools=tools)
        # 用户记忆中间件
        user_memory_middleware = UserMemoryMiddleware()
        return [
            user_memory_middleware,  # 用户级别记忆
            intention_middleware,  # 意图识别
            tool_selection_middleware,  # 工具选择（必须在意图识别之后）
            dynamic_system_prompt, # 动态系统提示词
            SummarizationMiddleware(model=self.summarization_llm) # 上下文摘要
        ]

    async def init(self):
        """初始化agent"""
        # 初始化模型
        self.init_llm()
        # 工具
        tools = await init_tools()
        # 短期记忆
        memory = InMemorySaver()
        # 属性存储 命名空间+键值对的方式
        self.store = InMemoryStore()
        # 中间件
        middleware = await self.init_middleware(tools.add_wrap_tool())

        self.agent = create_agent(
            model=self.admin_llm,
            tools=tools.get_tools(),
            checkpointer=memory,
            store=self.store,
            context_schema=DialogueContext,  # 单伦上下文
            state_schema=DialogueState,  # 多伦对话状态
            middleware=middleware,
        )

    async def dialogue(self):
        """对话"""
        thread_id = str(uuid.uuid4())
        while True:
            user_input = input("你：")
            if user_input == "exit":
                break
            if not user_input.strip():
                continue
            config = {"configurable": {"thread_id": thread_id}}
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,  # type: ignore
                # context=DialogueContext(
                #     thread_id=thread_id,
                #     user_id="root",
                #     user_type=UserType.OWNER,
                #     user_name="汪京",
                #     user_gender=UserGender.MALE,
                #     user_location="北京市亦庄经济开发区",
                # ),
                context=DialogueContext(
                    thread_id=thread_id,
                    user_id="3333",
                    user_type=UserType.STRANGER,
                    user_gender=UserGender.MALE,
                ),
                version="v2",
            ):
                # logger.warning(f"event: {event}")
                if event.get("event") == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        print(f"\033[34m{chunk.content}\033[0m", end="", flush=True)
            print()

            # 检查退出标志
            state = self.agent.get_state(config)  # type: ignore
            if state.values.get("should_exit", False):
                logger.info("用户请求退出，结束对话")
                break

            # 流程流
            # on_chain_start
            # on_chain_stream
            # on_chain_end
            # 工具流
            # on_tool_start
            # on_tool_end
            # agent流
            # on_chat_model_start
            # on_chat_model_stream
            # on_chat_model_end

            # 还有个思考呢？

            # 模型流
            # on_llm_start
            # on_llm_stream
            # on_llm_end
            # 检索流
            # on_retriever_start
            # on_retriever_end
