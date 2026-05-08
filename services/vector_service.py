"""
向量数据库服务

基于 Redis Stack 实现向量存储和相似度搜索
"""

import json
import struct
import time
import redis
from typing import List, Dict, Any, Optional
from config.settings import get_settings


class VectorService:
    """
    Redis 向量数据库服务

    提供向量存储、检索、删除功能
    使用 Redis Stack 的向量搜索能力（KNN搜索）
    """

    INDEX_NAME = "knowledge_vectors"
    VECTOR_DIM = 1024  # text-embedding-v4 输出维度

    def __init__(self):
        self.settings = get_settings()
        self.redis = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
            decode_responses=False
        )
        self._ensure_index()

    def _ensure_index(self):
        """确保向量索引存在"""
        try:
            self.redis.execute_command("FT.INFO", self.INDEX_NAME)
        except redis.exceptions.ResponseError:
            # 索引不存在，创建它（Redis Stack 2.x / RediSearch 2.x 语法，需要参数数量）
            self.redis.execute_command(
                "FT.CREATE",
                self.INDEX_NAME,
                "SCHEMA",
                "id", "TEXT",
                "text", "TEXT",
                "vector", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(self.VECTOR_DIM), "DISTANCE_METRIC", "COSINE",
                "metadata", "TEXT",
                "created_at", "NUMERIC"
            )

    def _to_bytes(self, vector: List[float]) -> bytes:
        """将向量列表转为字节数组"""
        return struct.pack(f"{len(vector)}f", *vector)

    def add_vector(
        self,
        doc_id: str,
        text: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加向量到数据库

        Args:
            doc_id: 文档唯一ID
            text: 原始文本内容
            vector: 1024维向量列表
            metadata: 其他元数据（可选）

        Returns:
            是否添加成功
        """
        try:
            key = f"doc:{doc_id}"
            metadata_json = json.dumps(metadata) if metadata else "{}"

            # 存储文档数据并自动索引
            self.redis.hset(key, mapping={
                "id": doc_id,
                "text": text,
                "vector": self._to_bytes(vector),
                "metadata": metadata_json,
                "created_at": str(int(time.time()))
            })
            return True
        except Exception as e:
            print(f"[ERROR] add_vector failed: {e}")
            return False

    def add_vector_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        批量添加向量

        Args:
            documents: 文档列表，每个元素包含:
                - doc_id: 文档ID
                - text: 文本内容
                - vector: 向量
                - metadata: 元数据（可选）

        Returns:
            成功添加的数量
        """
        success_count = 0
        for doc in documents:
            if self.add_vector(
                doc["doc_id"],
                doc["text"],
                doc["vector"],
                doc.get("metadata")
            ):
                success_count += 1
        return success_count

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            vector: 查询向量（1024维）
            top_k: 返回前K个最相似结果
            include_metadata: 是否包含元数据

        Returns:
            相似文档列表，每个元素包含:
                - doc_id: 文档ID
                - text: 文本内容
                - score: 相似度分数
                - metadata: 元数据
        """
        try:
            query_vector = self._to_bytes(vector)

            # 执行 KNN 搜索
            results = self.redis.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                f"*=>[KNN {top_k} @vector $vector AS score]",
                "PARAMS", "2", "vector", query_vector,
                "RETURN", "3", "id", "text", "score",
                "SORTBY", "score",
                "LIMIT", "0", str(top_k),
                "DIALECT", "2"
            )

            # 解析结果
            docs = []
            if results and len(results) > 1:
                # results[0] 是总数量，之后每两项为一组：[key, [fields...]]
                for i in range(1, len(results), 2):
                    key = results[i]
                    fields = results[i + 1]
                    # fields 是 [field, value, field, value, ...]
                    field_dict = {}
                    for j in range(0, len(fields), 2):
                        field_dict[fields[j]] = fields[j + 1]

                    doc = {
                        "doc_id": field_dict.get(b"id", b"").decode() if isinstance(field_dict.get(b"id"), bytes) else field_dict.get(b"id", ""),
                        "score": float(field_dict.get(b"score", 0))
                    }
                    if include_metadata:
                        text_val = field_dict.get(b"text", b"")
                        doc["text"] = text_val.decode() if isinstance(text_val, bytes) else text_val
                    docs.append(doc)

            return docs

        except Exception as e:
            print(f"[ERROR] search failed: {e}")
            return []

    async def search_by_text(
        self,
        text: str,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        直接用文本搜索（内部自动转向量）

        Args:
            text: 查询文本
            top_k: 返回前K个最相似结果
            include_metadata: 是否包含元数据

        Returns:
            相似文档列表
        """
        from embeddings.text_embedding import get_text_embedding

        embedding = get_text_embedding()
        vector = await embedding.embed(text)
        return self.search(vector, top_k, include_metadata)

    def delete(self, doc_id: str) -> bool:
        """
        删除向量

        Args:
            doc_id: 文档ID

        Returns:
            是否删除成功
        """
        try:
            self.redis.execute_command("FT.DEL", self.INDEX_NAME, doc_id)
            return True
        except Exception as e:
            print(f"[ERROR] delete failed: {e}")
            return False

    def count(self) -> int:
        """
        获取向量总数

        Returns:
            向量数量
        """
        try:
            info = self.redis.execute_command("FT.INFO", self.INDEX_NAME)
            for i, field in enumerate(info):
                if field == "num_docs":
                    return int(info[i + 1])
            return 0
        except:
            return 0


# 单例模式
_vector_service: Optional[VectorService] = None


def get_vector_service() -> VectorService:
    """获取向量服务单例"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
