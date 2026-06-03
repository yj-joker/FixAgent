"""
向量数据库服务

基于 Redis Stack 实现向量存储和相似度搜索
"""

import json
import logging
import struct
import time
import redis
from typing import List, Dict, Any, Optional
from config.settings import get_settings

logger = logging.getLogger(__name__)


_REDIS_TAG_ESCAPE_CHARS = set(r',.<>{}[]"\'`:;!@#$%^&*()-+=~|/ ')


def escape_redis_tag_value(value: Any) -> str:
    """Escape punctuation that RediSearch treats as TAG query syntax."""
    escaped = []
    for char in str(value):
        if char == "\\" or char in _REDIS_TAG_ESCAPE_CHARS or char.isspace():
            escaped.append("\\")
        escaped.append(char)
    return "".join(escaped)


def build_redis_filter(
    category: str = None,
    tags: List[str] = None,
    document_id: str = None,
    chunk_type: str = None,
    device_type: str = None,
    document_version: str = None,
    manual_type: str = None,
) -> Optional[str]:
    """构建 RediSearch 过滤表达式，供 API 层和工具层复用。

    Args:
        category: 分类过滤，如 "motor"
        tags: 标签过滤，如 ["bearing", "overheat"]

    Returns:
        RediSearch 过滤表达式，如 "@category:{motor}" 或
        "(@category:{motor}) (@tags:{bearing|overheat})"，无过滤条件时返回 None
    """
    filter_parts = []
    if category:
        filter_parts.append(f"@category:{{{escape_redis_tag_value(category)}}}")
    if tags:
        tag_str = "|".join(escape_redis_tag_value(tag) for tag in tags)
        filter_parts.append(f"@tags:{{{tag_str}}}")
    if document_id:
        filter_parts.append(f"@document_id:{{{escape_redis_tag_value(document_id)}}}")
    if chunk_type:
        filter_parts.append(f"@chunk_type:{{{escape_redis_tag_value(chunk_type)}}}")
    if device_type:
        filter_parts.append(f"@device_type:{{{escape_redis_tag_value(device_type)}}}")
    if document_version:
        filter_parts.append(f"@document_version:{{{escape_redis_tag_value(document_version)}}}")
    if manual_type:
        filter_parts.append(f"@manual_type:{{{escape_redis_tag_value(manual_type)}}}")
    if not filter_parts:
        return None
    return " ".join(f"({p})" for p in filter_parts)


class VectorService:
    """
    Redis 向量数据库服务

    提供向量存储、检索、删除功能
    使用 Redis Stack 的向量搜索能力（KNN搜索）
    """

    INDEX_NAME = "knowledge_vectors_v2"
    VECTOR_KEY_PREFIX = "doc:"
    DOCUMENT_KEY_PREFIX = "document:"
    TEXT_CACHE_PATTERNS = ("cache:emb:text:*", "emb:*")
    IMAGE_CACHE_PATTERNS = ("cache:emb:image:*", "img_emb:*")
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
        """确保向量索引存在（含分类/标签过滤所需的 TAG 字段）"""
        try:
            self.redis.execute_command("FT.INFO", self.INDEX_NAME)
            # 索引已存在，尝试补充 TAG 字段（RediSearch 2.0+ 支持 FT.ALTER）
            self._migrate_index()
        except redis.exceptions.ResponseError:
            self.redis.execute_command(
                "FT.CREATE",
                self.INDEX_NAME,
                "ON", "HASH",
                "PREFIX", "1", self.VECTOR_KEY_PREFIX,
                "SCHEMA",
                "id", "TEXT",
                "text", "TEXT",
                "vector", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(self.VECTOR_DIM), "DISTANCE_METRIC", "COSINE",
                "metadata", "TEXT",
                "category", "TAG",
                "tags", "TAG",
                "document_id", "TAG",
                "chunk_type", "TAG",
                "device_type", "TAG",
                "document_version", "TAG",
                "manual_type", "TAG",
                "created_at", "NUMERIC"
            )

    def _migrate_index(self):
        """为已有索引追加 category/tags TAG 字段（字段已存在时静默跳过）"""
        for field_name in ("category", "tags", "document_id", "chunk_type", "device_type", "document_version", "manual_type"):
            try:
                self.redis.execute_command(
                    "FT.ALTER", self.INDEX_NAME, "SCHEMA", "ADD", field_name, "TAG"
                )
            except redis.exceptions.ResponseError:
                pass

    def _to_bytes(self, vector: List[float]) -> bytes:
        """将向量列表转为字节数组"""
        return struct.pack(f"{len(vector)}f", *vector)

    def add_vector(
        self,
        doc_id: str,
        text: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        category: str = None,
        tags: List[str] = None
    ) -> bool:
        """
        添加向量到数据库

        Args:
            doc_id: 文档唯一ID
            text: 原始文本内容
            vector: 1024维向量列表
            metadata: 其他元数据（可选）
            category: 分类标签（可选，如 "motor"），用于过滤检索
            tags: 标签列表（可选，如 ["bearing", "overheat"]），用于过滤检索

        Returns:
            是否添加成功
        """
        try:
            key = f"{self.VECTOR_KEY_PREFIX}{doc_id}"
            # ensure_ascii=False：保留中文原文，避免存入Redis后变成 \uXXXX 乱码
            metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"

            mapping = {
                "id": doc_id,
                "text": text,
                "vector": self._to_bytes(vector),
                "metadata": metadata_json,
                "created_at": str(int(time.time()))
            }
            if category:
                mapping["category"] = category
            if tags:
                mapping["tags"] = ",".join(tags) if isinstance(tags, list) else tags
            for field_name in ("document_id", "chunk_type", "device_type", "document_version", "manual_type"):
                value = (metadata or {}).get(field_name)
                if value:
                    mapping[field_name] = value

            self.redis.hset(key, mapping=mapping)
            return True
        except Exception as e:
            logger.error(f"向量添加失败: {e}")
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
                doc.get("metadata"),
                doc.get("category"),
                doc.get("tags")
            ):
                success_count += 1
        return success_count

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            vector: 查询向量（1024维）
            top_k: 返回前K个最相似结果
            include_metadata: 是否包含元数据和文本内容
            filter: RediSearch 过滤表达式（可选）。
                    例: "@category:{motor}" 按分类过滤
                        "@tags:{bearing|overheat}" 按标签过滤
                        "(@category:{motor} @tags:{bearing})" 组合过滤

        Returns:
            相似文档列表，每个元素包含:
                - doc_id: 文档ID
                - text: 文本内容（include_metadata=True 时）
                - score: 相似度分数
                - metadata: 元数据字典（include_metadata=True 时）
        """
        try:
            query_vector = self._to_bytes(vector)

            # 构建搜索语句：filter 为空时用 *（全量），否则用 filter 限定范围
            if filter:
                query = f"({filter})=>[KNN {top_k} @vector $vector AS score]"
            else:
                query = f"*=>[KNN {top_k} @vector $vector AS score]"

            results = self.redis.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                query,
                "PARAMS", "2", "vector", query_vector,
                "RETURN", "4", "id", "text", "score", "metadata",
                "SORTBY", "score",
                "LIMIT", "0", str(top_k),
                "DIALECT", "2"
            )

            # 解析结果
            docs = []
            if results and len(results) > 1:
                for i in range(1, len(results), 2):
                    key = results[i]
                    fields = results[i + 1]
                    field_dict = {}
                    for j in range(0, len(fields), 2):
                        field_dict[fields[j]] = fields[j + 1]

                    def _decode(field_name: bytes, default=""):
                        val = field_dict.get(field_name)
                        if val is None:
                            return default
                        return val.decode() if isinstance(val, bytes) else val

                    doc = {
                        "doc_id": _decode(b"id"),
                        "score": float(field_dict.get(b"score", 0))
                    }
                    from services.retrieval_policy import cosine_distance_to_relevance

                    doc["raw_score"] = doc["score"]
                    doc["raw_score_type"] = "cosine_distance"
                    doc["relevance_score"] = cosine_distance_to_relevance(doc["score"])
                    if include_metadata:
                        doc["text"] = _decode(b"text")

                        metadata_raw = _decode(b"metadata", "{}")
                        try:
                            doc["metadata"] = json.loads(metadata_raw)
                        except (json.JSONDecodeError, TypeError):
                            doc["metadata"] = {}
                    docs.append(doc)

            return docs

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        include_metadata: bool = True,
        filter: str = None
    ) -> List[Dict[str, Any]]:
        """Run lexical recall over stored chunk text."""
        if not query_text.strip():
            return []
        try:
            query_body = " ".join(part for part in query_text.replace("-", " ").split() if part)
            text_query = f"@text:({query_body})"
            query = f"({filter}) {text_query}" if filter else text_query
            results = self.redis.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                query,
                "RETURN", "3", "id", "text", "metadata",
                "LIMIT", "0", str(top_k),
                "DIALECT", "2"
            )
            docs = []
            if results and len(results) > 1:
                for rank, i in enumerate(range(1, len(results), 2), start=1):
                    fields = results[i + 1]
                    field_dict = {fields[j]: fields[j + 1] for j in range(0, len(fields), 2)}

                    def _decode(field_name: bytes, default=""):
                        val = field_dict.get(field_name)
                        if val is None:
                            return default
                        return val.decode() if isinstance(val, bytes) else val

                    doc = {
                        "doc_id": _decode(b"id"),
                        "score": float(rank),
                        "raw_score": float(rank),
                        "raw_score_type": "keyword_rank",
                        "relevance_score": round(1.0 / rank, 6)
                    }
                    if include_metadata:
                        doc["text"] = _decode(b"text")
                        try:
                            doc["metadata"] = json.loads(_decode(b"metadata", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            doc["metadata"] = {}
                    docs.append(doc)
            return docs
        except Exception as e:
            logger.warning(f"keyword_search failed: {e}")
            return []

    async def search_by_text(
        self,
        text: str,
        top_k: int = 5,
        include_metadata: bool = True,
        filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        直接用文本搜索（内部自动转向量）

        Args:
            text: 查询文本
            top_k: 返回前K个最相似结果
            include_metadata: 是否包含元数据
            filter: RediSearch 过滤表达式（可选）

        Returns:
            相似文档列表
        """
        from embeddings.text_embedding import get_text_embedding

        embedding = get_text_embedding()
        vector = await embedding.embed(text)
        return self.search(vector, top_k, include_metadata, filter)

    def delete(self, doc_id: str) -> bool:
        """
        删除向量

        通过删除 Redis Hash key 来移除向量。
        Redis Search 索引会自动感知 Hash 被删除并从索引中移除该文档。

        Args:
            doc_id: 文档ID（存储时 key 为 "doc:{doc_id}"）

        Returns:
            是否删除成功
        """
        try:
            key = f"{self.VECTOR_KEY_PREFIX}{doc_id}"
            result = self.redis.delete(key)
            if result:
                logger.info(f"向量删除成功: {key}")
            else:
                logger.warning(f"向量键不存在: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"向量删除失败: {e}")
            return False

    def delete_batch(self, doc_ids: List[str]) -> int:
        """批量删除向量，返回成功删除的数量"""
        deleted = 0
        for doc_id in doc_ids:
            if self.delete(doc_id):
                deleted += 1
        return deleted

    def delete_by_document(self, document_id: str) -> int:
        """Delete all vector records that belong to one imported document."""
        if not document_id:
            return 0
        try:
            # 确保索引存在（Redis 重启后 RediSearch 索引可能丢失）
            self._ensure_index()

            deleted = 0
            query = build_redis_filter(document_id=document_id)
            while True:
                results = self.redis.execute_command(
                    "FT.SEARCH",
                    self.INDEX_NAME,
                    query,
                    "RETURN", "1", "id",
                    "LIMIT", "0", "10000",
                    "DIALECT", "2"
                )
                keys = [results[i] for i in range(1, len(results), 2)]
                if not keys:
                    break
                for key in keys:
                    if self.redis.delete(key):
                        deleted += 1
            logger.info(f"delete_by_document 完成: document_id={document_id}, deleted={deleted}")
            return deleted
        except Exception as e:
            logger.error(f"delete_by_document failed: {e}", exc_info=True)
            return 0

    def put_document_manifest(self, document_id: str, manifest: Dict[str, Any]) -> bool:
        """Persist import lifecycle metadata outside individual chunk vectors."""
        if not document_id:
            return False
        try:
            self.redis.hset(
                f"{self.DOCUMENT_KEY_PREFIX}{document_id}",
                mapping={
                    "document_id": document_id,
                    "manifest": json.dumps(manifest, ensure_ascii=False),
                    "updated_at": str(int(time.time())),
                },
            )
            return True
        except Exception as e:
            logger.error(f"put_document_manifest failed: {e}")
            return False

    def get_document_manifest(self, document_id: str) -> Dict[str, Any]:
        """Read document import metadata for status or rebuild workflows."""
        try:
            raw = self.redis.hget(f"{self.DOCUMENT_KEY_PREFIX}{document_id}", "manifest")
            if raw is None:
                return {}
            text = raw.decode() if isinstance(raw, bytes) else raw
            return json.loads(text)
        except Exception as e:
            logger.error(f"get_document_manifest failed: {e}")
            return {}

    def list_documents_by_kg_status(self, status: str) -> List[str]:
        """扫描 document:* manifest，返回 kg_status 命中的 document_id 列表。"""
        ids = []
        for key in self.redis.scan_iter(match=f"{self.DOCUMENT_KEY_PREFIX}*", count=1000):
            raw = self.redis.hget(key, "manifest")
            if not raw:
                continue
            try:
                m = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                continue
            if m.get("kg_status") == status:
                ids.append(m.get("document_id"))
        return [i for i in ids if i]

    def _count_keys(self, patterns) -> int:
        keys = set()
        for pattern in patterns:
            keys.update(self.redis.scan_iter(match=pattern, count=1000))
        return len(keys)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Return separate counts for long-lived vectors, manifests and cache keys."""
        text_cache = self._count_keys(self.TEXT_CACHE_PATTERNS)
        image_cache = self._count_keys(self.IMAGE_CACHE_PATTERNS)
        return {
            "vector_records": self._count_keys((f"{self.VECTOR_KEY_PREFIX}*",)),
            "indexed_vector_records": self.count(),
            "document_manifests": self._count_keys((f"{self.DOCUMENT_KEY_PREFIX}*",)),
            "cache": {
                "text": text_cache,
                "image": image_cache,
                "total": text_cache + image_cache,
            },
        }

    def clear_embedding_cache(self) -> Dict[str, int]:
        """Delete only disposable embedding cache keys, never vectors or manifests."""
        deleted = {"text_deleted": 0, "image_deleted": 0}
        for pattern in self.TEXT_CACHE_PATTERNS:
            for key in self.redis.scan_iter(match=pattern, count=1000):
                deleted["text_deleted"] += int(bool(self.redis.delete(key)))
        for pattern in self.IMAGE_CACHE_PATTERNS:
            for key in self.redis.scan_iter(match=pattern, count=1000):
                deleted["image_deleted"] += int(bool(self.redis.delete(key)))
        deleted["total_deleted"] = deleted["text_deleted"] + deleted["image_deleted"]
        return deleted

    def count(self) -> int:
        """
        获取向量总数

        Returns:
            向量数量
        """
        try:
            info = self.redis.execute_command("FT.INFO", self.INDEX_NAME)
            for i, field in enumerate(info):
                field_name = field.decode() if isinstance(field, bytes) else field
                if field_name == "num_docs":
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
