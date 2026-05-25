"""Knowledge retrieval tool for multimodal RAG evidence."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from embeddings.multimodal_embedding import get_multimodal_embedding
from embeddings.text_embedding import get_text_embedding
from schemas.models import VectorSearchResult
from services.retrieval_policy import (
    detect_query_intent,
    diversify_candidates,
    rerank_candidates,
    summarize_confidence,
)
from services.vector_service import build_redis_filter, escape_redis_tag_value, get_vector_service
from tools.base_tool import BaseTool, ToolException

logger = logging.getLogger(__name__)


class KnowledgeRetrievalTool(BaseTool):
    """Retrieve text, table, and image evidence from the knowledge store."""

    @property
    def name(self) -> str:
        return "knowledge_retrieval"

    @property
    def description(self) -> str:
        return (
            "Retrieve maintenance knowledge evidence from text, table, and image records. "
            "Use it for fault causes, repair steps, parameters, diagrams, and image evidence."
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User retrieval query."},
                "top_k": {"type": "integer", "default": 5, "description": "Final evidence count."},
                "category": {"type": "string", "description": "Existing category filter."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Existing tag filter."},
                "document_id": {"type": "string", "description": "Restrict retrieval to one imported document."},
                "chunk_type": {"type": "string", "description": "text/table/image/image_summary filter."},
                "device_type": {"type": "string", "description": "Device type metadata filter."},
                "document_version": {"type": "string", "description": "Document version metadata filter."},
                "manual_type": {"type": "string", "description": "Manual type metadata filter."},
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional user images for multimodal query embedding.",
                },
            },
            "required": ["query"],
        }

    @staticmethod
    def _build_filter(
        category: str = None,
        tags: List[str] = None,
        document_id: str = None,
        chunk_type: str = None,
        device_type: str = None,
        document_version: str = None,
        manual_type: str = None,
    ) -> Optional[str]:
        if not any((document_id, chunk_type, device_type, document_version, manual_type)):
            parts = []
            if category:
                parts.append(f"@category:{{{escape_redis_tag_value(category)}}}")
            if tags:
                parts.append(f"@tags:{{{'|'.join(escape_redis_tag_value(tag) for tag in tags)}}}")
            if not parts:
                return None
            return parts[0] if len(parts) == 1 else f"({' '.join(parts)})"
        return build_redis_filter(
            category=category,
            tags=tags,
            document_id=document_id,
            chunk_type=chunk_type,
            device_type=device_type,
            document_version=document_version,
            manual_type=manual_type,
        )

    @staticmethod
    def _canonical_id(doc: Dict) -> str:
        doc_id = doc.get("doc_id", "")
        metadata = doc.get("metadata") or {}
        if metadata.get("source_image_id"):
            return metadata["source_image_id"]
        if metadata.get("chunk_type") == "image_summary":
            return doc_id.replace(":ims:", ":img:")
        return doc_id

    @staticmethod
    def _mark_route(doc: Dict, route: str) -> Dict:
        item = dict(doc)
        item["metadata"] = dict(item.get("metadata") or {})
        if item.get("relevance_score") is None and item.get("score") is not None:
            item["relevance_score"] = item["score"]
        item["routes"] = sorted(set(item.get("routes") or []) | {route})
        item["retrieval_route"] = route
        return item

    @classmethod
    def _merge_candidates(cls, candidates: List[Dict]) -> List[Dict]:
        merged: Dict[str, Dict] = {}
        for candidate in candidates:
            key = cls._canonical_id(candidate)
            if key not in merged:
                item = dict(candidate)
                item["doc_id"] = key
                merged[key] = item
                continue
            current = merged[key]
            routes = sorted(set(current.get("routes") or []) | set(candidate.get("routes") or []))
            if candidate.get("relevance_score", 0.0) > current.get("relevance_score", 0.0):
                current.update(candidate)
                current["doc_id"] = key
            current["routes"] = routes
            current_meta = current.setdefault("metadata", {})
            current_meta.update(
                {name: value for name, value in (candidate.get("metadata") or {}).items() if value not in ("", None)}
            )
        return list(merged.values())

    async def _embed_query(self, query: str, image_urls: List[str] = None) -> List[float]:
        try:
            if image_urls:
                result = await get_multimodal_embedding().embed(text=query, image_urls=image_urls)
                vectors = []
                if result.get("text_vector"):
                    vectors.append(result["text_vector"])
                vectors.extend(result.get("image_vectors", []))
                if not vectors:
                    raise ToolException(code="EMBEDDING_FAILED", message="multimodal embedding returned no vectors")
                return [sum(values) / len(vectors) for values in zip(*vectors)]
            return await get_text_embedding().embed(query)
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(code="EMBEDDING_FAILED", message=f"embedding failed: {e}")

    async def _execute(
        self,
        query: str,
        top_k: int = 5,
        category: str = None,
        tags: List[str] = None,
        image_urls: List[str] = None,
        document_id: str = None,
        chunk_type: str = None,
        device_type: str = None,
        document_version: str = None,
        manual_type: str = None,
    ) -> List[VectorSearchResult]:
        vector = await self._embed_query(query, image_urls)
        intent = detect_query_intent(query)
        recall_k = max(top_k * 3, top_k)
        base_filter = self._build_filter(
            category=category,
            tags=tags,
            document_id=document_id,
            chunk_type=chunk_type,
            device_type=device_type,
            document_version=document_version,
            manual_type=manual_type,
        )

        try:
            vector_service = get_vector_service()
            candidates = []
            if chunk_type or intent != "image":
                candidates.extend(
                    self._mark_route(doc, "semantic")
                    for doc in vector_service.search(
                        vector, top_k=recall_k, include_metadata=True, filter=base_filter
                    )
                )
            if hasattr(vector_service, "keyword_search") and (chunk_type or intent != "image"):
                candidates.extend(
                    self._mark_route(doc, "keyword")
                    for doc in vector_service.keyword_search(
                        query, top_k=recall_k, include_metadata=True, filter=base_filter
                    )
                )
            if not chunk_type and intent in {"image", "mixed"}:
                image_filter = self._build_filter(
                    category, tags, document_id, "image", device_type, document_version, manual_type
                )
                summary_filter = self._build_filter(
                    category, tags, document_id, "image_summary", device_type, document_version, manual_type
                )
                candidates.extend(
                    self._mark_route(doc, "image_vector")
                    for doc in vector_service.search(vector, top_k=recall_k, include_metadata=True, filter=image_filter)
                )
                candidates.extend(
                    self._mark_route(doc, "image_summary")
                    for doc in vector_service.search(vector, top_k=recall_k, include_metadata=True, filter=summary_filter)
                )
            if not chunk_type and intent in {"table", "mixed"}:
                table_filter = self._build_filter(
                    category, tags, document_id, "table", device_type, document_version, manual_type
                )
                candidates.extend(
                    self._mark_route(doc, "table_vector")
                    for doc in vector_service.search(vector, top_k=recall_k, include_metadata=True, filter=table_filter)
                )
        except Exception as e:
            raise ToolException(code="SEARCH_FAILED", message=f"retrieval search failed: {e}")

        merged = self._merge_candidates(candidates)
        ranked = rerank_candidates(query, merged, intent=intent)
        selected = diversify_candidates(ranked, top_k=top_k, intent=intent)
        confidence = summarize_confidence(selected, intent=intent)

        results: List[VectorSearchResult] = []
        for doc in selected:
            metadata = dict(doc.get("metadata") or {})
            if metadata.get("chunk_type") == "image_summary":
                metadata["source_chunk_type"] = "image_summary"
                metadata["chunk_type"] = "image"
            routes = sorted(set(doc.get("routes") or []))
            metadata["retrieval_routes"] = routes
            metadata["matched_types"] = confidence["matched_types"]
            metadata["retrieval_confidence"] = confidence["confidence"]
            metadata["confidence_reason"] = {
                "best_relevance_score": confidence["best_relevance_score"],
                "candidate_count": confidence["candidate_count"],
                "dual_image_hit": confidence["dual_image_hit"],
                "intent": intent,
            }
            if confidence["confidence"] == "low":
                metadata["answer_policy"] = "insufficient_evidence"
            results.append(
                VectorSearchResult(
                    id=doc.get("doc_id", ""),
                    score=doc.get("relevance_score", doc.get("score", 0.0)),
                    content=doc.get("text", ""),
                    metadata=metadata,
                    raw_score=doc.get("raw_score"),
                    raw_score_type=doc.get("raw_score_type"),
                    relevance_score=doc.get("relevance_score"),
                    retrieval_route=routes[0] if routes else doc.get("retrieval_route"),
                    rerank_score=doc.get("rerank_score"),
                )
            )
        return results


_retrieval_tool: Optional[KnowledgeRetrievalTool] = None


def get_knowledge_retrieval_tool() -> KnowledgeRetrievalTool:
    global _retrieval_tool
    if _retrieval_tool is None:
        _retrieval_tool = KnowledgeRetrievalTool()
    return _retrieval_tool
