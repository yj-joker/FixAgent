"""
文本向量化模块

使用阿里云百炼 qwen2.5-vl-embedding 将文本转为高维向量。
与 image_embedding.py 共用同一模型，确保文本和图片向量在同一语义空间。

【模型信息】
- 模型: qwen2.5-vl-embedding
- 默认维度: 1024（可选 2048/768/512）
- 输入格式: [{"text": "..."}]
"""

import asyncio
import hashlib
import logging
import redis
from typing import Optional, List

import dashscope
from config.settings import get_settings

logger = logging.getLogger(__name__)


class TextEmbedding:
    """文本向量化服务，使用 qwen2.5-vl-embedding 统一模型。"""

    def __init__(self):
        self.settings = get_settings()
        self.model = "qwen2.5-vl-embedding"
        self.dimensions = 1024
        dashscope.api_key = self.settings.dashscope_api_key
        self.redis = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
            decode_responses=False
        )
        self.cache_ttl = self.settings.redis_ttl

    def _get_cache_key(self, text: str) -> str:
        return f"emb:v2:{hashlib.md5(text.encode()).hexdigest()}"

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        data = self.redis.get(self._get_cache_key(text))
        if data:
            import pickle
            return pickle.loads(data)
        return None

    def _set_to_cache(self, text: str, embedding: List[float]) -> None:
        import pickle
        self.redis.setex(self._get_cache_key(text), self.cache_ttl, pickle.dumps(embedding))

    def _call_api_sync(self, inputs: List[dict]) -> List[List[float]]:
        """同步调用 dashscope MultiModalEmbedding API。"""
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=inputs
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API 返回错误 code={resp.status_code} message={resp.message}"
            )

        if resp.output and "embeddings" in resp.output:
            embeddings = sorted(resp.output["embeddings"], key=lambda x: x.get("index", 0))
            result = [e["embedding"] for e in embeddings]
            if result:
                logger.debug(f"文本向量化完成 模型={self.model} 维度={len(result[0])}")
            return result

        raise ValueError(f"Embedding API 响应格式异常: {resp}")

    async def embed(self, text: str) -> List[float]:
        """单条文本向量化。"""
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        embeddings = await asyncio.to_thread(
            self._call_api_sync, [{"text": text}]
        )
        result = embeddings[0]
        self._set_to_cache(text, result)
        return result

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化，单次 API 调用。"""
        results: List[Optional[List[float]]] = []
        uncached_indices: List[int] = []
        uncached_inputs: List[dict] = []

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_inputs.append({"text": text})

        if uncached_inputs:
            new_embeddings = await asyncio.to_thread(
                self._call_api_sync, uncached_inputs
            )
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                self._set_to_cache(texts[idx], emb)

        return results


_text_embedding: Optional[TextEmbedding] = None


def get_text_embedding() -> TextEmbedding:
    global _text_embedding
    if _text_embedding is None:
        _text_embedding = TextEmbedding()
    return _text_embedding
