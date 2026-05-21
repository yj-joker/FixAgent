"""
知识库检索工具

封装文本/多模态向量化 + Redis 向量检索流程，提供统一的知识检索能力。
支持纯文本查询和图文混合查询。

【调用链】
文本: query → TextEmbedding.embed() → 1024维向量 → VectorService.search()
图文: query + image_urls → MultimodalEmbedding.embed() → 融合向量 → VectorService.search()
"""

from typing import List, Optional
import logging

from tools.base_tool import BaseTool, ToolException
from embeddings.text_embedding import get_text_embedding
from embeddings.multimodal_embedding import get_multimodal_embedding
from services.vector_service import get_vector_service
from schemas.models import VectorSearchResult

logger = logging.getLogger(__name__)


class KnowledgeRetrievalTool(BaseTool):
    """知识库向量检索工具，支持纯文本和图文混合查询。"""

    @property
    def name(self) -> str:
        return "knowledge_retrieval"

    @property
    def description(self) -> str:
        return (
            "从向量知识库中检索与查询语义最相似的文档。"
            "支持纯文本查询和图文混合查询。"
            "支持按 category（分类）和 tags（标签）过滤。"
            "适用场景：用户询问设备知识、故障原因、维修方法等需要查资料的情况。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询文本（自然语言描述）"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回文档数量，默认5",
                    "default": 5
                },
                "category": {
                    "type": "string",
                    "description": "分类过滤，如 motor/pump/bearing 等"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签过滤，如 ['bearing', 'overheat']，OR语义"
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用户上传的图片URL列表，用于图文混合检索（可选）"
                }
            },
            "required": ["query"]
        }

    @staticmethod
    def _build_filter(category: str = None, tags: List[str] = None) -> Optional[str]:
        parts = []
        if category:
            parts.append(f"@category:{{{category}}}")
        if tags:
            tags_expr = "|".join(tags)
            parts.append(f"@tags:{{{tags_expr}}}")

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return f"({' '.join(parts)})"

    async def _execute(
        self,
        query: str,
        top_k: int = 5,
        category: str = None,
        tags: List[str] = None,
        image_urls: List[str] = None
    ) -> List[VectorSearchResult]:
        """
        执行知识检索，支持纯文本和图文混合。

        Args:
            query: 查询文本
            top_k: 返回文档数量
            category: 分类过滤
            tags: 标签过滤
            image_urls: 图片URL列表（可选），提供时启用图文混合检索

        Returns:
            List[VectorSearchResult]: 按 score 降序排列的检索结果

        Raises:
            ToolException: EMBEDDING_FAILED / SEARCH_FAILED
        """
        try:
            if image_urls:
                # 图文混合检索：同时生成文本向量和图片向量，取平均融合
                mm = get_multimodal_embedding()
                result = await mm.embed(text=query, image_urls=image_urls)

                text_vec = result.get("text_vector")
                image_vecs = result.get("image_vectors", [])

                all_vecs = []
                if text_vec:
                    all_vecs.append(text_vec)
                all_vecs.extend(image_vecs)

                if not all_vecs:
                    raise ToolException(
                        code="EMBEDDING_FAILED",
                        message="图文向量化均失败"
                    )

                dim = len(all_vecs[0])
                vector = [sum(col) / len(all_vecs) for col in zip(*all_vecs)]
                logger.info(
                    f"[knowledge_retrieval] 多模态检索: 文本 + {len(image_vecs)} 张图片 -> 融合为 {dim} 维向量"
                )
            else:
                embedding_service = get_text_embedding()
                vector = await embedding_service.embed(query)
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(
                code="EMBEDDING_FAILED",
                message=f"向量化失败: {e}"
            )

        filter_expr = self._build_filter(category, tags)

        try:
            vector_service = get_vector_service()
            docs = vector_service.search(
                vector, top_k=top_k, include_metadata=True, filter=filter_expr
            )
        except Exception as e:
            raise ToolException(
                code="SEARCH_FAILED",
                message=f"向量检索失败: {e}"
            )

        results: List[VectorSearchResult] = []
        for doc in docs:
            results.append(VectorSearchResult(
                id=doc.get("doc_id", ""),
                score=doc.get("score", 0.0),
                content=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            ))

        return results


_retrieval_tool: Optional[KnowledgeRetrievalTool] = None


def get_knowledge_retrieval_tool() -> KnowledgeRetrievalTool:
    global _retrieval_tool
    if _retrieval_tool is None:
        _retrieval_tool = KnowledgeRetrievalTool()
    return _retrieval_tool
