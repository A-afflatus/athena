from typing import cast

from langchain.agents.middleware import ModelRequest, dynamic_prompt

from athena.context import DialogueContext, UserType
from athena.prompt.service_standard import (
    acquaintance_service_standard,
    guest_service_standard,
    owner_service_standard,
    stranger_service_standard,
)
from athena.prompt.system import athena_system_prompt_template


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """提示词分化中间件，根据用户类型和用户信息动态生成系统提示词"""
    context = cast(DialogueContext, request.runtime.context)
    long_term_memory = context.long_term_memory
    if UserType.OWNER == context.user_type:
        return _build_prompt(owner_service_standard, context,long_term_memory)
    if UserType.ACQUAINTANCE == context.user_type:
        return _build_prompt(acquaintance_service_standard, context,long_term_memory)
    if UserType.GUEST == context.user_type:
        return _build_prompt(guest_service_standard, context,long_term_memory)
    return _build_prompt(stranger_service_standard, context,long_term_memory)


def _build_prompt(service_standard: str, context: DialogueContext,long_term_memory: list[str]) -> str:
    return athena_system_prompt_template.format(
        user_id=context.user_id,
        user_type=context.user_type.value,
        user_name=context.user_name or "未知,用到时需提前向用户确认",
        user_gender=context.user_gender.value if context.user_gender else "未知,用到时需提前向用户确认",
        user_location=context.user_location or "未知,用到时需提前向用户确认",
        service_standard=service_standard,
        long_term_memory="\n".join(long_term_memory),
    )
