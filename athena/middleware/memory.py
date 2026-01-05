from datetime import datetime
from typing import Any, cast, override

from graphiti_core.nodes import EpisodeType
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from athena.context import AthenaState, DialogueContext
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
            fact_str = f"图谱事实: {result.fact}"
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

        return {
            "context": runtime.context,
        }

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
            cast(list[AnyMessage], messages[-2:]), runtime.context
        )  # pyright: ignore[reportArgumentType]
        return None


memory_organization_agent = create_agent(
    model=init_model("qwen-flash", temperature=0.1),
    tools=[],
    system_prompt="""你是一个专业的记忆提取专家。你的任务是从用户与 AI 的对话中提取出**长期有价值**的用户事实。

### 提取准则：
1. **严格限制来源**：**只提取用户明确表达的信息**。绝对不要将 AI 在回复中提供的建议、描述或知识点误认为是用户的偏好或事实。
2. **关注长期记忆**：提取用户的个人偏好、生活习惯、职业背景、家庭成员、重要经历、长期目标、地理位置等。
3. **过滤无用信息**：忽略日常寒暄、简单的礼貌用语、以及仅与当前对话流程相关的临时性指令。
4. **事实化表达**：每个事实应是一个独立、完整的陈述句。确保主语明确，统一使用用户的姓名作为主语。不要使用"他/她"等模糊代词。
5. **严禁过度解读**：不要基于 AI 的回复进行推断。如果用户说"我喜欢吃面"，不要根据 AI 推荐的"油泼面"就提取出"用户喜欢吃油泼面"。
6. **增量提取**：如果信息在提供的"用户信息"中已经清晰记录且没有发生变更，则无需重复提取。
7. **原子性**：每条事实只包含一个核心信息点。

### 示例：
- 对话：用户："我最近在减肥"。AI："建议多吃鸡胸肉和西兰花"。
  提取：王五最近正在减肥。（不要提取王五喜欢吃鸡胸肉）
- 对话：用户："我住在上海"。AI："上海是个好地方"。
  提取：王五目前居住在上海市。

如果没有发现任何值得记录的新事实，请返回空列表。""",
    response_format=MemoryFactsResponse,
)


async def save_memory(messages: list[AnyMessage], context: DialogueContext) -> None:
    if not messages or len(messages) == 0:
        return
    # 图谱记忆-graphiti
    await _add_graphiti_memory_async(messages, context)


async def _add_graphiti_memory_async(
    messages: list[AnyMessage], context: DialogueContext
) -> None:
    """添加图谱记忆到graphiti"""
    try:
        graphiti = get_graphiti()

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

        response = memory_organization_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"请从以下对话中提取新的、值得记录的用户事实。注意：只能提取用户表达的信息，不要提取 AI 提供的知识或建议。如果没有发现新事实，请返回空列表。\n\n主语请统一使用: {context.user_name}。\n当前已有的用户信息: {context.user_info()}\n\n待分析的对话内容:\n{messages_str}"
                    )
                ]
            }
        )
        memory_response = cast(MemoryFactsResponse, response["structured_response"])
        user_facts = memory_response.facts

        if not user_facts or len(user_facts) == 0:
            logger.warning("没有有效的用户事实，跳过graphiti记忆保存")
            return

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
