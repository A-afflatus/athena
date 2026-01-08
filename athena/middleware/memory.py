import asyncio
from datetime import datetime
from typing import Any, cast, override

from graphiti_core.nodes import EpisodeType
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import adispatch_custom_event

from athena.context import AthenaState, ChatEventData, DialogueContext
from bootstrap.logger import get_logger
from middleware.graphiti import get_graphiti
from model.models import init_model

logger = get_logger(__name__)


class MemoryFactsResponse(BaseModel):
    """记忆事实提取响应"""

    facts: list[str] = Field(
        description="提取的用户事实列表，如果没有值得记录的事实则为空列表",
        default_factory=list,
    )


class UserMemoryMiddleware(AgentMiddleware[AthenaState, DialogueContext]):
    def __init__(self):
        # 对话记忆
        # todo 这个应该是另一个中间件图知识库
        self.graphiti = get_graphiti()

    @override
    async def abefore_agent(
        self, state: AthenaState, runtime: Runtime[DialogueContext]
    ) -> dict[str, Any] | None:
        """在 Agent 运行前：检索记忆"""
        messages = state.get("messages", [])
        if not messages:
            return None

        user_query = cast(str, messages[-1].content)

        user_id = runtime.context.user_id

        graphiti_results = await self.graphiti.search(user_query, group_ids=[user_id])

        # 格式化 Graphiti 记忆
        graphiti_facts = []
        for result in graphiti_results:
            fact_str = result.fact
            if hasattr(result, "valid_at") and result.valid_at:
                fact_str += f" (生效: {result.valid_at})"
            if hasattr(result, "invalid_at") and result.invalid_at:
                fact_str += f" (失效: {result.invalid_at})"
            graphiti_facts.append(fact_str)

        # 将图谱记忆存入 context，后续可以通过 dynamic_prompt 注入到系统提示词
        if graphiti_facts:
            runtime.context.long_term_memory = graphiti_facts
        else:
            runtime.context.long_term_memory = [
                f"时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-内容:开始本次对话"
            ]

    @override
    async def aafter_agent(
        self, state: AthenaState, runtime: Runtime[DialogueContext]
    ) -> dict[str, Any] | None:
        """在 Agent 运行后：存储记忆"""
        # 获取最后一组人机对话
        messages = state.get("messages", [])
        # 过滤出 AIMessage 和 HumanMessage
        messages = [
            m
            for m in messages
            if isinstance(m, (AIMessage, HumanMessage)) and m.content != ""
        ]

        await save_memory(
            cast(list[AnyMessage], messages[-2:]), runtime
        )  # pyright: ignore[reportArgumentType]
        return None


system_prompt = """
你是一个专业的个人记忆构建专家。你的核心任务是从人机对话中精准提取关于用户的**长期事实**，构建用户的个人画像。

### 核心原则
1. **信源严格控制**：
   - **只提取用户明确表达的内容**。
   - **严禁**提取 AI 的回复内容、建议或 AI 提供的知识。
   - **严禁**基于 AI 的建议反推用户的偏好（例如：AI 推荐 A，用户未明确表示喜欢 A，绝不可记录用户喜欢 A）。

2. **长期价值导向**：
   - ✅ **提取**：个人偏好（饮食、娱乐）、身份背景（职业、年龄、居住地）、社会关系（家庭、朋友）、重要经历、长期目标、身体状况、信仰价值观。
   - ❌ **忽略**：临时性指令（"帮我查天气"）、日常寒暄（"你好"）、针对当前任务的上下文补充（"把这个改成红色"）。

3. **记录规范**：
   - **独立完整**：每条事实必须是独立完整的陈述句，不依赖上下文即可理解。
   - **主语明确**：必须显式包含主语（使用用户真实姓名），严禁使用"他"、"她"、"我"、"用户"等代词。
   - **原子化**：一条记录只包含一个核心事实。
   - **时效性**：对于有时效性的状态（如"生病"、"正在旅行"），尽量包含时间状语。

4. **增量更新策略**：
   - 对比【当前已有的信息】，已存在且未变更的信息不要重复提取。
   - 只有当出现新信息或旧信息发生变更/修正时才进行记录。

### 示例分析

**场景 1：误判 AI 建议**
* 对话：
  User: "我最近想学点乐器。"
  AI: "尤克里里很简单，适合新手，要试试吗？"
* ❌ 错误提取：王五喜欢尤克里里。 (来源是 AI，用户未确认)
* ✅ 正确提取：王五最近有学习乐器的意愿。

**场景 2：事实修正**
* 对话：
  User: "我搬家了，现在住在杭州。"
* 假设已有记忆：王五住在北京。
* ✅ 正确提取：王五目前居住在杭州市。

**场景 3：非长期记忆**
* 对话：
  User: "帮我把这封邮件发给李四。"
* ✅ 正确提取：(无) - 这是一个临时指令。

如果没有发现任何值得记录的新事实，请返回空列表。
"""

memory_organization_agent = create_agent(
    model=init_model("qwen-flash", temperature=0.1),
    tools=[],
    system_prompt=system_prompt,
    response_format=MemoryFactsResponse,
)


async def save_memory(messages: list[AnyMessage], runtime: Runtime[DialogueContext]):
    if not messages or len(messages) == 0:
        return []
    # 图谱记忆-graphiti
    asyncio.create_task(_add_graphiti_memory_async(messages, runtime))


async def _add_graphiti_memory_async(
    messages: list[AnyMessage], runtime: Runtime[DialogueContext]
):
    context = runtime.context
    """添加图谱记忆到graphiti"""
    try:
        graphiti = get_graphiti()
        # 已有的事实信息
        long_term_memory = (
            context.long_term_memory
            if context.long_term_memory and len(context.long_term_memory) > 0
            else []
        )

        # 通过agent 简化 分析关系以及行为
        messages_str = "\n".join(
            [
                (
                    f"用户: {msg.content}"
                    if isinstance(msg, HumanMessage)
                    else f"AI: {msg.content}"
                )
                for msg in messages
            ]
        )

        existing_facts_str = "\n".join(long_term_memory) if long_term_memory else "无"

        response = memory_organization_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"请分析以下对话内容，结合用户当前已有的画像信息，提取新的记忆事实。\n\n"
                            f"### 输入信息\n"
                            f"<user_profile>\n"
                            f"用户姓名: {context.user_name}\n"
                            f"{context.user_info()}\n"
                            f"</user_profile>\n\n"
                            f"<existing_memories>\n"
                            f"{existing_facts_str}\n"
                            f"</existing_memories>\n\n"
                            f"<conversation_log>\n"
                            f"{messages_str}\n"
                            f"</conversation_log>\n\n"
                            f"### 执行要求\n"
                            f"1. 仔细比对 <conversation_log> 中的信息与 <existing_memories>。\n"
                            f"2. 仅提取 **新的** 或 **有变更** 的事实。\n"
                            f'3. **特别注意**：仔细区分 <conversation_log> 中 "用户" 和 "AI" 的发言。**绝对禁止**将 AI 的观点记为用户的事实。\n'
                            f"4. 提取结果中的主语请统一使用: **{context.user_name}**。\n\n"
                            f"请输出提取结果："
                        )
                    )
                ]
            }
        )
        memory_response = cast(MemoryFactsResponse, response["structured_response"])
        user_facts = memory_response.facts

        if not user_facts or len(user_facts) == 0:
            return
        await adispatch_custom_event(
            "save_memory",
            ChatEventData(
                chunk_type="text",
                content=f"提取到新的用户事实: {user_facts}，开始保存图谱记忆...",
            ),
        )

        # 生成episode名称，使用时间戳和用户ID
        episode_name = f"对话记录_{context.user_id}_{context.user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 添加episode到graphiti
        # 使用 user_id 作为 group_id 来分区不同用户的图谱
        await graphiti.add_episode(
            name=episode_name,
            episode_body="\n".join(user_facts),
            source=EpisodeType.message,
            source_description="用户对话记录",
            reference_time=datetime.now(),
            group_id=context.user_id,  # 使用 user_id 作为图谱分区标识
        )
        logger.info(f"成功添加图谱记忆到graphiti，用户ID: {context.user_id}")
    except Exception as e:
        logger.error(f"添加图谱记忆到graphiti失败: {e}", exc_info=True)
