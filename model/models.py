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
    model_kwargs: dict[str, Any] = {},  # noqa: B006
    extra_body: dict[str, Any] = {},  # noqa: B006
    **kwargs: Any,
) -> BaseChatModel:
    match model:
        case model if model.startswith("qwen"):
            return ChatQwen(
                model=model,
                enable_thinking=enable_thinking,
                api_key=SecretStr(api_key), # type: ignore
                base_url=base_url,
                model_kwargs=model_kwargs,
                extra_body=extra_body,
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid model: {model}")
