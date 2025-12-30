import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel
from langchain_qwq import ChatQwen
from pydantic import SecretStr

load_dotenv()

api_key = os.getenv("LLM_QWEN_API_KEY")
base_url = os.getenv("LLM_QWEN_BASE_URL")


def init_model(
    model: str = "qwen-flash",
    enable_thinking: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    match model:
        case "qwen3-max":
            return ChatQwen(
                model="qwen3-max",
                enable_thinking=enable_thinking,
                api_key=SecretStr(api_key), # type: ignore
                base_url=base_url,
                **kwargs,
            )
        case "qwen-flash":
            return ChatQwen(
                model="qwen-flash",
                enable_thinking=enable_thinking,
                api_key=SecretStr(api_key), # type: ignore
                base_url=base_url,
                **kwargs,
            )
        case "qwen-plus":
            return ChatQwen(
                model="qwen-plus",
                enable_thinking=enable_thinking,
                api_key=SecretStr(api_key), # type: ignore
                base_url=base_url,
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid model: {model}")
