"""
Athena 核心类

实现智能对话功能，基于 LangChain 构建。
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from athena.context import (
    AthenaState,
    ChatEventListener,
    ChatRequest,
    DialogueContext,
    of_chat_event,
)
from athena.middleware.dichotomy_prompts import dynamic_system_prompt
from athena.middleware.intention_recognition import IntentionRecognitionMiddleware
from athena.middleware.memory import UserMemoryMiddleware
from athena.middleware.tool_selection import ToolSelectionMiddleware
from athena.tools.base import Tools
from athena.tools import init_tools
from bootstrap.logger import get_logger
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
        self.intention_llm = init_model(model="qwen-flash-2025-07-28", temperature=0.1)
        # 总结llm - 用于上下文摘要，需要快速且准确
        self.summarization_llm = init_model(
            model="qwen-flash-2025-07-28", enable_thinking=True, temperature=0.1
        )

    async def init_middleware(self, tools: Tools):
        """初始化中间件"""

        # 意图识别中间件
        intention_middleware = IntentionRecognitionMiddleware(llm=self.intention_llm)
        # 工具选择中间件（根据意图动态调整工具）
        # tool_selection_middleware = ToolSelectionMiddleware(tools)
        # 用户记忆中间件
        user_memory_middleware = UserMemoryMiddleware()

        return [
            user_memory_middleware,  # 用户级别记忆
            intention_middleware,  # 意图识别
            # tool_selection_middleware,  # 工具选择（必须在意图识别之后）
            dynamic_system_prompt,  # 动态系统提示词
            SummarizationMiddleware(model=self.summarization_llm),  # 上下文摘要
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
        middleware = await self.init_middleware(tools)

        self.agent = create_agent(
            model=self.admin_llm,
            tools=tools.get_tools(),
            checkpointer=memory,
            store=self.store,
            context_schema=DialogueContext,  # 单伦上下文
            state_schema=AthenaState,  # 基于thread_id 的多伦对话的状态
            middleware=middleware,
        )

    async def chat(self, chat: ChatRequest, listener: ChatEventListener):
        """对话"""
        thread_id = chat.thread_id
        user_input = chat.user_input
        context = chat.context
        await listener.on_chat_start()
        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": context.user_id + "-" + thread_id}},
            context=context,
            version="v2",
        ):
            # logger.warning(f"event: {event}") # 调试事件
            await listener.on_chat_event_stream(of_chat_event(event))
        await listener.on_chat_end()
        # 检查退出标志
        if context.should_exit:
            logger.info("用户请求退出，结束对话")
            await listener.on_exit()

###
# | event                  | 含义                | 特点
# | ---------------------- | ------------------- | -------------------------------------------------------------------------------------
# | `on_chat_model_start`  | Chat Model 开始运行   | 输入包含 messages，metadata 包含模型配置 (如 temperature, model_name)
# | `on_chat_model_stream` | Chat Model 流式输出   | 包含 AIMessageChunk，可能是文本内容 (content) 或工具调用片段 (tool_call_chunks)
# | `on_chat_model_end`    | Chat Model 运行结束   | 输出完整的 AIMessage
# | `on_llm_start`         | 非聊天模型 开始运行    | 
# | `on_llm_stream`        | 非聊天模型 流式输出    | 
# | `on_llm_end`           | 非聊天模型 运行结束    | 
# | `on_chain_start`       | Chain 开始运行        | 输入是字典或对象，name 字段标识 Chain 名称 (如 UserMemoryMiddleware, LangGraph)
# | `on_chain_stream`      | Chain 流式输出        | 包含状态更新或中间结果 chunk
# | `on_chain_end`         | Chain 运行结束        | 输出最终执行结果
# | `on_tool_start`        | Tool 开始运行         | 输入是工具参数，name 字段标识工具名称 (如 maps_weather)
# | `on_tool_end`          | Tool 运行结束         | 输出工具执行结果
# | `on_retriever_start`   | Retriever 召回开始    | 
# | `on_retriever_end`     | Retriever 召回结束    | 
# | `on_prompt_start`      | ChatPromptTemplate 开始处理       | 
# | `on_prompt_end`        | ChatPromptTemplate 处理结束       | 
###