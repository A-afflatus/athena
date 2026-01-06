import os
from typing import Any

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from pydantic import SecretStr

from model.adapter.qwen_embedding import DashScopeMultiModalEmbeddings
from model.adapter.volcengine_embedding import VolcEngineMultimodalEmbedding

load_dotenv()

qwen_api_key = os.getenv("LLM_QWEN_API_KEY")
volcengine_api_key = os.getenv("VOLC_ENGINE_API_KEY")

def init_embedding_model(
    model: str = "qwen2.5-vl-embedding",
) -> Embeddings:
    model_lower = model.lower()
    if model_lower.startswith("qwen"):
        return DashScopeMultiModalEmbeddings(
            model=model,
            api_key=SecretStr(qwen_api_key), # type: ignore
        )
    elif model_lower.startswith("doubao"):
        return VolcEngineMultimodalEmbedding(
            model=model,
            api_key=volcengine_api_key, # type: ignore
        )
    else:
        raise ValueError(f"Invalid model: {model}")
