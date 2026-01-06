from enum import Enum
from typing import Awaitable, Callable

from langchain.agents import AgentState
from pydantic import BaseModel, Field


class UserType(Enum):
    """用户类型"""

    OWNER = "主人"
    ACQUAINTANCE = "熟人"
    GUEST = "客人"
    STRANGER = "陌生人"


class UserGender(Enum):
    """用户性别"""

    MALE = "男"
    FEMALE = "女"


class IntentionType(Enum):
    """用户意图类型"""

    EXIT = "退出"
    WEATHER = "天气"
    SEARCH = "搜索"
    GENERAL = "通用" # 兜底意图，当没有其他意图时，使用此意图


class DialogueContext(BaseModel):
    """对话上下文"""

    user_id: str = Field(description="用户ID")
    user_type: UserType = Field(description="用户身份类型")
    user_name: str | None = Field(description="用户名称", default=None)
    user_gender: UserGender | None = Field(description="用户性别", default=None)
    user_location: str | None = Field(description="用户当前位置", default=None)

    user_intention: list[IntentionType] | None = Field(
        description="用户意图类型列表，可能包含多个意图", default=[IntentionType.GENERAL]
    )
    should_exit: bool = Field(description="是否应该退出对话", default=False)
    long_term_memory: list[str] = Field(description="用户级长期记忆", default_factory=list)

    def user_info(self) -> str:
        return f"用户ID: {self.user_id},用户身份类型: {self.user_type.value},用户名称: {self.user_name},用户性别: {self.user_gender.value if self.user_gender else '未知'},用户当前位置: {self.user_location}"


class AthenaState(AgentState):
    """代理状态"""


class ChatRequest(BaseModel):
    """聊天请求"""

    thread_id: str = Field(description="会话ID")
    user_input: str = Field(description="用户输入")
    context: DialogueContext = Field(description="对话上下文")

class ChatEvent(BaseModel):
    """聊天事件"""

    thread_id: str = Field(description="会话ID")
    chunk: str = Field(description="流式响应的片段")


class ChatEventListener(BaseModel):
    """聊天事件监听器"""

    on_chat_start: Callable[[], Awaitable[None]] = Field(description="单轮对话开始事件")
    on_chat_end: Callable[[], Awaitable[None]] = Field(description="单轮对话结束事件")
    on_chat_model_stream: Callable[[ChatEvent], Awaitable[None]] = Field(description="聊天模型流式响应")
    on_exit: Callable[[], Awaitable[None]] = Field(description="退出事件")