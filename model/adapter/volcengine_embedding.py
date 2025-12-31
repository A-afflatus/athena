from typing import override

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.multimodal_embedding import (
    MultimodalEmbeddingContentPartTextParam,
)


class VolcEngineMultimodalEmbedding(Embeddings):
    """火山引擎多模态嵌入模型适配器"""

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.model = model
        self.client = Ark(api_key=api_key)

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    @override
    def embed_query(self, text: str) -> list[float]:
        resp = self.client.multimodal_embeddings.create(
            model=self.model,
            input=[MultimodalEmbeddingContentPartTextParam(text=text, type="text")],
        )
        return resp.data.embedding
