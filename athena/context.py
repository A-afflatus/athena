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
    """用户上下文"""

    thread_id: str = Field(description="对话ID")
    user_id: str = Field(description="用户ID")
    user_type: UserType = Field(description="用户身份类型")
    user_name: str | None = Field(description="用户名称", default=None)
    user_gender: UserGender | None = Field(description="用户性别", default=None)
    user_location: str | None = Field(description="用户当前位置", default=None)

    
    user_intention: list[IntentionType] | None  = Field(description="用户意图类型列表，可能包含多个意图", default=[IntentionType.GENERAL])


class DialogueState(AgentState):
    """对话状态"""

    should_exit: bool  # 是否应该退出对话
    long_term_memory: list[str] # 用户级长期记忆
    
