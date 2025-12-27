from enum import Enum
from langchain.agents import AgentState
from pydantic import BaseModel, Field

class IntentionType(Enum):
    """用户意图类型"""
    EXIT = "EXIT"
    WEATHER = "WEATHER"
    SEARCH = "SEARCH"
    GENERAL = "GENERAL"

class DialogueContext(BaseModel):
    """用户上下文"""
    thread_id: str = Field(description="对话ID")
    user_id: str = Field(description="用户ID")
    user_type: str = Field(description="用户身份类型", default="陌生人")
    user_name: str | None = Field(description="用户名称", default=None)
    user_gender: str | None = Field(description="用户性别", default=None)
    user_location: str = Field(description="用户当前位置", default="未知")
    user_message: str | None = Field(description="用户消息", default=None)


class DialogueState(AgentState):
    """对话状态"""
    should_exit: bool = Field(description="是否应该退出对话",default=False)
    user_intention: list[str] | None = Field(description="用户意图类型列表，可能包含多个意图", default=None)