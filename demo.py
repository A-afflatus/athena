from dataclasses import dataclass
import sys
from typing import Any, cast
from langchain.agents.middleware.types import AgentMiddleware, before_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentState, create_agent
from langchain.messages import RemoveMessage
from langchain.tools import ToolRuntime, tool
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt

from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore, Item
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import Field, SecretStr
from pydantic import BaseModel


load_dotenv()


llm = ChatQwen(
    api_key=SecretStr("sk-0622e4a2dc0a4c138009a4a796cae4e7"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-flash",
)
llm_deepseek = ChatDeepSeek(
    api_key=SecretStr("sk-c52af06c35504a448a134fb76a12a684"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
# llm = ChatOpenAI(
#     api_key=SecretStr("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"),
#     base_url="https://openrouter.ai/api/v1",
#     model="qwen/qwen3-32b"
# )

# ! 用户上下文
@dataclass
class UserContext:
    user_id: str = Field(description="用户ID")
    user_name: str = Field(description="用户名")
    city: str = Field(description="用户城市")

# ! 结构化响应
@dataclass
class WeatherResponse(BaseModel):
    city: str = Field(description="城市名称")
    weather: str = Field(description="天气情况")
    temperature: str = Field(description="温度")

# ! 记忆状态 除了messages之外，其他的状态信息都不会直接传递给模型，而是放到state中
# 上面的context一般只适用于单词对话的简单上下文内容，而state是多伦对话的内容
class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict


@tool
def get_weather(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前人的城市天气情况
    Returns:
        天气情况
    Example:
        get_weather() -> "晴,26℃"clear
    """
    writer = runtime.stream_writer
    writer("<<<开始调用工具get_weather>>>")
    # 从全局状态中取
    store = cast(BaseStore, runtime.store)
    weathers = cast(Item, store.get(("system",), "weather"))
    return weathers.value.get(runtime.context.city, f"没有相关天气信息")


# ! 使用工具来做graph的状态管理和流程控制
@tool
def get_and_set_city(runtime: ToolRuntime[UserContext]) -> Command:
    """获取当前人的城市，并设置到用户上下文中
    Example:
        get_and_set_city()
    """
    new_city = "北京"
    # ! 发送流式输出
    writer = runtime.stream_writer
    writer("<<<开始调用工具get_and_set_city>>>")
    # 更新 context 和添加 ToolMessage
    updated_context = UserContext(
        user_id=runtime.context.user_id,
        user_name=runtime.context.user_name,
        city=new_city,
    )
    return Command(
        update={
            "context": updated_context,
            "messages": [
                ToolMessage(
                    content=f"{new_city}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def exit_conversation(runtime: ToolRuntime[UserContext]) -> Command:
    """当用户表达退出意图时调用此工具（如说再见、结束对话、退出等）
    此工具用于判断用户是否想要结束对话。
    如果用户表达了退出意图，请调用此工具。
    """
    writer = runtime.stream_writer
    writer("<<<用户表达了退出意图>>>")
    # 在store中设置退出标志
    store = cast(BaseStore, runtime.store)
    store.put(("session", runtime.context.user_id), "should_exit", {"value": True})
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="已确认用户退出意图",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# ! 使用内存存储状态，可多实例、多会话共享存储
store = InMemoryStore()
# 这是 命名空间，key，value的存储组合
store.put(
    ("system",),
    "weather",
    {
        "北京": "晴,26℃",
        "上海": "多云,23℃",
        "广州": "小雨,28℃",
        "深圳": "阴,27℃",
        "杭州": "小雨,22℃",
        "成都": "多云,20℃",
        "重庆": "晴,30℃",
    },
)
# ! 短期记忆
memory = InMemorySaver()


# ! 修剪消息
@before_model
def trim_messages(state: CustomAgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
    """修剪消息"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None # 不需要修剪

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


# ! 动态系统提示词
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name # type: ignore
    system_prompt = cast(str, request.system_message.content)  # pyright: ignore[reportOptionalMemberAccess]
    if "汪京" in user_name:
        return f"你现在面临的是一个顶级领导，他是你的顶头上司，说什么话都要先想好了再回复，谨慎发言,{system_prompt}"
    return system_prompt

# ! 流式输出
agent = create_agent(
    model=llm,
    system_prompt="多功能助手"
                  "如果用户表达了退出意图（如说再见、结束对话、退出等），请调用exit_conversation工具。",
    tools=[get_weather, get_and_set_city, exit_conversation],
    context_schema=UserContext,  # 上下文模式
    store=store, # 存储
    checkpointer=memory, # 短期记忆
    state_schema=CustomAgentState, # 记忆状态
    # 中间件
    middleware=[dynamic_system_prompt,
    SummarizationMiddleware( # type: ignore # ! langchain自带的对话摘要中间件
        model=llm_deepseek,
        trigger=("tokens", 1000), # 触发条件，当上下文超过2000个token时，触发摘要
        keep=("messages", 5), # 保留条件，保留最后5条消息
    )], # 中间件
)
# 初始化上下文
user_context = UserContext(user_id="123", user_name="汪京", city="北京")

# 循环等待控制台输入
print("请输入您的问题（说再见或表达退出意图即可退出）：")
while True:
    try:
        # 等待用户输入
        user_input = input("\n> ").strip()
        
        # 如果输入为空，跳过
        if not user_input:
            continue
        
        # 重置退出标志
        store.put(("session", user_context.user_id), "should_exit", {"value": False})
        
        # 流式处理用户输入
        print("回答：", end="", flush=True)
        # ! 基本的流式输出接收方式
        # region 基本的流式输出接收方式

        # for model, chunk in agent.stream({
        #         "messages": [HumanMessage(content=user_input)],
        #         "user_id": user_context.user_id,   # type: ignore
        #         "preferences": {"theme": "dark"},   # type: ignore
        #     },
        #     {"configurable":{"thread_id":user_context.user_id}}, # 设置线程ID
        #     context=user_context,
        #     stream_mode=["custom", "messages"],
        # ):
        #     # ! 压缩上下文的输出也会在这里，注意单独处理或过滤
        #     if model == "custom":
        #         print("\n" + chunk, flush=True)  # type: ignore
        #     elif model == "messages":
        #         message = chunk[0]
        #         print(message.text, end="", flush=True)  # type: ignore
        # endregion
        
        
        for mode,chunk in agent.stream({
                "messages": [HumanMessage(content=user_input)]
            },
            {"configurable":{"thread_id":user_context.user_id}}, # 设置线程ID
            context=user_context,
            stream_mode=["updates", "custom", "messages"],
        ):
            if mode == "updates":
                if isinstance(chunk, dict) and "model" in chunk:
                    print()  # 换行
            elif mode == "custom":
                print(chunk)
            elif mode == "messages" and chunk[1]["langgraph_node"] == "model": # type: ignore
                message = chunk[0]
                print(message.text, end="", flush=True)  # type: ignore
            

        print()  # 换行


        
        # 检查是否调用了退出工具
        exit_item = store.get(("session", user_context.user_id), "should_exit")
        if exit_item and exit_item.value.get("value", False):
            break
        
    except (KeyboardInterrupt, EOFError):
        print("\n\n再见！")
        break

# ! 格式化输出
# agent = create_agent(
#     model=llm,
#     system_prompt="你是一个天气预报员，请根据用户输入的城市，返回该城市的天气情况。",
#     tools=[get_weather],
#     response_format=WeatherResponse,
# )
# result = agent.invoke({"messages": [HumanMessage(content="今天广州的天气怎么样?")]})
# print(result['structured_response'])

sys.exit(0)