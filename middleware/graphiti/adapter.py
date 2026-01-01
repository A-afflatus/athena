import json
import logging
from collections.abc import Iterable
from typing import Any, cast, override

import numpy as np
from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.prompts.models import Message
from langchain.chat_models import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LangchainGenericClient(LLMClient):
    """Langchain 通用模型适配器

    基于 OpenAIGenericClient 的实现，适配 langchain 的 BaseChatModel。
    支持将 graphiti 的 Message 格式转换为 langchain 的 BaseMessage 格式，
    并处理 JSON 响应格式。
    """

    def __init__(
        self,
        client: BaseChatModel,
    ):
        """
        初始化 LangchainGenericClient

        Args:
            client: Langchain 的 BaseChatModel 实例 ！！模型需要支持json format
        """
        self.client = client

    def _convert_messages(self, messages: list[Message]) -> list[BaseMessage]:
        """将 graphiti 的 Message 列表转换为 langchain 的 BaseMessage 列表"""
        langchain_messages: list[BaseMessage] = []
        for m in messages:
            cleaned_content = self._clean_input(m.content)
            if m.role == "user":
                langchain_messages.append(HumanMessage(content=cleaned_content))
            elif m.role == "system":
                langchain_messages.append(SystemMessage(content=cleaned_content))
        return langchain_messages

    @override
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """生成响应
        Args:
            messages: Graphiti Message 列表
            response_model: 可选的响应模型（用于结构化输出）
            max_tokens: 最大 token 数
            model_size: 模型大小

        Returns:
            解析后的 JSON 字典
        """
        # 转换消息格式
        langchain_messages = self._convert_messages(messages)

        try:
            # 如果有 response_model，需要在 prompt 中添加 JSON schema 说明
            if response_model is not None:
                schema_name = getattr(response_model, "__name__", "structured_response")
                json_schema = response_model.model_json_schema()
                # 将 JSON schema 说明添加到最后一条用户消息
                if langchain_messages and isinstance(
                    langchain_messages[-1], HumanMessage
                ):
                    schema_instruction = (
                        f"\n\nRespond with a JSON object matching this schema:\n"
                        f"{json.dumps(json_schema, indent=2, ensure_ascii=False)}"
                    )
                    langchain_messages[-1].content += schema_instruction

            # 调用 langchain 客户端
            response = await self.client.ainvoke(
                input=langchain_messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                    },
                },
            )

            # 提取响应内容
            result: str
            if isinstance(response, BaseMessage):
                content = response.content
                # 处理 content 可能是字符串或列表的情况
                if isinstance(content, str):
                    result = content
                elif isinstance(content, list):
                    # 如果是列表，尝试提取文本部分
                    text_parts = []
                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            text_parts.append(str(item["text"]))
                        else:
                            text_parts.append(str(item))
                    result = "".join(text_parts)
                else:
                    result = str(content)
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)

            # 尝试解析 JSON
            return json.loads(result)

        except Exception as e:
            logger.error(f"Graphiti LangchainGenericClient Error: {e}")
            raise


class LangchainEmbedder(EmbedderClient):
    """Langchain 嵌入模型适配器"""

    def __init__(self, client: Embeddings):
        self.client = client

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """创建单个嵌入向量"""
        text_input = ""
        if isinstance(input_data, str):
            text_input = input_data
        elif isinstance(input_data, list) and all(
            isinstance(item, str) for item in input_data
        ):
            # 检查列表中的元素是否都是字符串
            text_input = "\n".join(cast(list[str], input_data))
        else:
            text_input = "\n".join(map(str, input_data))
        return self.client.embed_query(text_input)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """批量创建嵌入向量"""
        if not input_data_list:
            return []
        return await self.client.aembed_documents(input_data_list)


class LangchainReranker(CrossEncoderClient):
    def __init__(self, client: BaseChatOpenAI):
        client.temperature = 0
        client.max_tokens = 1
        client.logit_bias = {6432: 1, 7983: 1}  # True 和 False 的 token ID
        client.logprobs = True
        client.top_logprobs = 2
        # 设置客户端
        self.client = client

    @override
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        对 passages 进行排序，根据它们与 query 的相关性。

        这个方法使用 BaseChatOpenAI 的 logprobs 功能来判断每个 passage 是否与 query 相关。
        通过 logit_bias 强制模型只输出 "True" 或 "False"，然后使用 logprobs 计算相关性分数。

        Args:
            query: 查询字符串
            passages: 待排序的文本段落列表

        Returns:
            排序后的 (passage, score) 元组列表，按分数降序排列
        """
        if not passages:
            return []

        # 为每个 passage 创建消息列表
        messages_list: list[list[BaseMessage]] = [
            [
                SystemMessage(
                    content="You are an expert tasked with determining whether the passage is relevant to the query"
                ),
                HumanMessage(
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """
                ),
            ]
            for passage in passages
        ]

        try:
            # 并发调用所有 passages
            responses = await semaphore_gather(
                *[self.client.agenerate([messages]) for messages in messages_list]
            )

            # 提取 logprobs 并计算分数
            scores: list[float] = []
            for response in responses:
                try:
                    # 从 ChatResult 中提取 logprobs
                    generation = response.generations[0][0]
                    generation_info = generation.generation_info or {}
                    logprobs_data = generation_info.get("logprobs")

                    if logprobs_data and "content" in logprobs_data:
                        top_logprobs = logprobs_data["content"][0].get(
                            "top_logprobs", []
                        )
                        if len(top_logprobs) > 0:
                            # 获取第一个 token 的 logprob
                            first_token_logprob = top_logprobs[0].get("logprob", 0.0)
                            first_token = (
                                top_logprobs[0]
                                .get("token", "")
                                .strip()
                                .split(" ")[0]
                                .lower()
                            )

                            # 计算归一化的概率
                            norm_logprob = np.exp(first_token_logprob)

                            # 如果第一个 token 是 "true"，使用其概率作为分数
                            # 如果是 "false"，使用 1 - 概率作为分数
                            if first_token == "true":
                                scores.append(norm_logprob)
                            else:
                                scores.append(1 - norm_logprob)
                        else:
                            logger.warning("无法获取 top_logprobs，使用默认分数")
                            scores.append(0.5)
                    else:
                        logger.warning("无法获取 logprobs，使用默认分数")
                        scores.append(0.5)
                except Exception as e:
                    logger.error(f"处理响应时出错: {e}")
                    scores.append(0.5)

            # 确保 scores 和 passages 长度一致
            if len(scores) != len(passages):
                logger.error(
                    f"分数数量 ({len(scores)}) 与 passages 数量 ({len(passages)}) 不匹配"
                )
                # 补齐缺失的分数
                while len(scores) < len(passages):
                    scores.append(0.5)

            # 组合结果并排序
            results = [
                (passage, score)
                for passage, score in zip(passages, scores, strict=True)
            ]
            results.sort(reverse=True, key=lambda x: x[1])
            return results

        except Exception as e:
            logger.error(f"Graphiti ReRanker Error: {e}")
            raise
