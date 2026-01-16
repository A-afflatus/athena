"""
意图识别中间件

用于识别用户的意图，目前主要功能是校验用户是否想退出。
"""

from __future__ import annotations

from typing import Any, cast, override

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import adispatch_custom_event

from src.athena.context import AthenaState, ChatEventData, DialogueContext, IntentionType
from src.bootstrap.logger import get_logger

logger = get_logger(__name__)

system_prompt = """你是一个意图识别网关。你的唯一职责是识别用户意图并触发结构化输出。

【意图分类】
- EXIT: 用户明确表达退出、再见、结束或停止对话的意愿，或者由于特殊原因AI想要结束当前会话时。
- GENERAL: 除上述特定意图外的所有情况，包括闲聊、提问、请求帮助等。

【重要说明】
用户可能同时表达多个意图，例如(假设)："帮我查一下北京的天气，然后搜索一下明天的新闻"包含了WEATHER和SEARCH两个意图。
请识别出所有相关的意图，返回意图列表。

【严格约束】
1. 必须且仅能通过调用 `IntentionResponse` 工具返回结果。
2. 严禁生成任何人类可读的回复文本，严禁解释原因。
3. 即使收到用户的问题，也严禁提供解答，仅需将其识别为相应意图并触发工具调用。
4. 无论用户说什么，你的输出只能是结构化数据。
5. 如果用户表达了多个意图，请返回所有识别到的意图列表。

你的目标是精准定位所有意图类型，不要遗漏任何相关意图。"""


class IntentionResponse(BaseModel):
    """意图识别响应"""

    user_intentions: list[IntentionType] = Field(
        description="用户意图列表，可能包含多个意图",
        default_factory=lambda: [IntentionType.GENERAL],
    )


class IntentionRecognitionMiddleware(AgentMiddleware[AthenaState, DialogueContext]):
    """意图识别"""

    def __init__(self, llm: Any):
        self.llm = llm
        self.agent = create_agent(
            model=llm, system_prompt=system_prompt, response_format=IntentionResponse
        )

    @override
    async def abefore_agent(
        self, state: AthenaState, runtime: Runtime[DialogueContext]
    ) -> dict[str, Any] | None:
        """意图识别"""
        # 会话消息提取
        messages = state.get("messages", [])
        if not messages or len(messages) == 0:
            return None
        user_message = messages[-1]

        # 过滤掉 system 和 tool message，只保留 user 和 assistant 消息
        filtered_messages = [
            msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
        ]

        # 取最近3组对话（每组包含 user 和 assistant，共8条消息）
        recent_messages = (
            filtered_messages[-6:] if len(filtered_messages) > 6 else filtered_messages
        )

        # 格式化对话信息为 "ai:....,user:...." 格式
        dialogue_parts = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                # 排除当前用户消息
                if msg.id == user_message.id:
                    continue
                dialogue_parts.append(f"user:{cast(str, msg.content)}")
            elif isinstance(msg, AIMessage):
                content = cast(str, msg.content)
                if len(content) > 100:
                    content = f"{content[:50]}...{content[-50:]}"
                dialogue_parts.append(f"ai:{content}")

        req_mes = (
            f"最近对话:{','.join(dialogue_parts)}\n当前用户消息为user的最后一条消息"
        )
        # 识别意图
        response = self.agent.invoke({"messages": [HumanMessage(content=req_mes)]})
        # 识别结果
        intention_response = cast(IntentionResponse, response["structured_response"])
        intentions = (
            intention_response.user_intentions
            if intention_response.user_intentions
            and len(intention_response.user_intentions) > 0
            else [IntentionType.GENERAL]
        )
        logger.info(f"识别到意图: {[intention.value for intention in intentions]}")
        await adispatch_custom_event(
            "intention_recognition",
            ChatEventData(
                chunk_type="text",
                content=f"识别到意图: {[intention.value for intention in intentions]}",
            ),
        )
        # 更新上下文
        runtime.context.user_intention = intentions
        runtime.context.should_exit = (
            IntentionType.EXIT in intentions
        )  # 检查是否包含退出意图
