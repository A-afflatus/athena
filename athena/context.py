from enum import Enum

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

    EXIT = "EXIT"
    WEATHER = "WEATHER"
    SEARCH = "SEARCH"
    GENERAL = "GENERAL"


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
        return f"用户ID: {self.user_id},用户身份类型: {self.user_type.value},用户名称: {self.user_name},用户性别: {self.user_gender.value},用户当前位置: {self.user_location}"


class AthenaState(AgentState):
    """代理状态"""
