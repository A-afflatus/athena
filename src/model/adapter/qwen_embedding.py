# 百炼多模态嵌入模型装饰器

from typing import override

from dashscope import (
    MultiModalEmbedding,
    MultiModalEmbeddingItemAudio,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
)
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from langchain.embeddings.base import Embeddings


def _check_response(resp: DashScopeAPIResponse) -> None:
    if resp.status_code != 200:
        raise ValueError(
            f"Embedding request failed: status_code={resp.status_code}, "
            f"code={resp.code}, message={resp.message}"
        )

class DashScopeMultiModalEmbeddings(Embeddings):
    """
    DashScope多模态嵌入模型[百炼平台]
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    @override
    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本"""
        resp = MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=[MultiModalEmbeddingItemText(text, factor=1.0)],
        )
        _check_response(resp)
        return list(resp.output["embeddings"][0]["embedding"])

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入文档列表，复用embed_query的逻辑"""
        if not texts:
            return []
        embeddings = []
        for text in texts:
            resp = self.embed_query(text)
            embeddings.append(resp)
        return embeddings

    def embed_images(self, image: str) -> list[list[float]]:
        """嵌入图片列表 支持url和base64 注意对应模型支持的图片格式"""
        resp = MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=[MultiModalEmbeddingItemImage(image, factor=1.0)],
        )
        _check_response(resp)
        return list(resp.output["embeddings"][0]["embedding"])

    def embed_audios(self, audio: str) -> list[list[float]]:
        """嵌入音频列表 仅支持url 注意对应模型支持的音频格式"""
        resp = MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=[MultiModalEmbeddingItemAudio(audio, factor=1.0)],
        )
        _check_response(resp)
        return list(resp.output["embeddings"][0]["embedding"])

    def embed_videos(self, video: str) -> list[list[float]]:
        """嵌入视频列表 仅支持url 注意对应模型支持的视频格式"""
        resp = MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=[{"video": video, "factor": 1.0}],# pyright: ignore[reportArgumentType]
        )
        _check_response(resp)
        return list(resp.output["embeddings"][0]["embedding"])
