"""
意图识别中间件

用于识别用户的意图，目前主要功能是校验用户是否想退出。
"""

from __future__ import annotations

from typing import Any, cast, override

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from config.logger import get_logger
from athena.entity.entity import DialogueContext, DialogueState, IntentionType

logger = get_logger(__name__)

system_prompt="""你是一个意图识别网关。你的唯一职责是识别用户意图并触发结构化输出。

【意图分类】
- EXIT: 用户明确表达退出、再见、结束或停止对话的意愿。
- WEATHER: 用户询问天气、气温、降雨、风力等天气相关信息。
- SEARCH: 用户需要搜索信息、查询资料、查找内容等。
- GENERAL: 除上述特定意图外的所有情况，包括闲聊、提问、请求帮助等。

【重要说明】
用户可能同时表达多个意图，例如："帮我查一下北京的天气，然后搜索一下明天的新闻"包含了WEATHER和SEARCH两个意图。
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
        default_factory=lambda: [IntentionType.GENERAL]
    )


class IntentionRecognitionMiddleware(AgentMiddleware[DialogueState, DialogueContext]):
    """意图识别"""

    def __init__(self, llm: Any):
        self.llm = llm
        self.agent = create_agent(
            model=llm,
            system_prompt=system_prompt,
            response_format=IntentionResponse
        )
    @override
    def before_agent(self,state: DialogueState, runtime: Runtime[DialogueContext]) -> dict[str, Any] | None:
        """意图识别"""
        # state 是字典类型，需要使用字典访问方式
        messages = state.get("messages", [])
        if not messages or len(messages) == 0:
            return None
        user_message = messages[-1].content
        req_mes = f"长期记忆:{runtime.context.long_term_memory}\n用户消息:{user_message}"
        response = self.agent.invoke({"messages": [HumanMessage(content=req_mes)]})
        intention_response = cast(IntentionResponse, response['structured_response'])
        intentions = intention_response.user_intentions
        
        # 将意图枚举列表转换为字符串列表
        intention_values = [intention.value for intention in intentions]
        
        # 检查是否包含退出意图
        has_exit = IntentionType.EXIT in intentions
        
        result = {
            "user_intention": intention_values,
            "should_exit": has_exit
        }
        logger.info(f"识别到意图: {intention_values}")
        return result