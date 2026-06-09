"""Knowledge retrieval tool for multimodal RAG evidence."""

from __future__ import annotations

import logging
import asyncio
from typing import Dict, List, Optional

from embeddings.multimodal_embedding import get_multimodal_embedding
from embeddings.text_embedding import get_text_embedding
from schemas.models import VectorSearchResult
from services.retrieval_planner import build_retrieval_plan, confidence_intent
from services.retrieval_ranker import rank_candidates, should_use_expensive_rerank
from services.retrieval_context_expander import expand_retrieval_context
from services.retrieval_policy import (
    diversify_candidates,
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
        record_type: str = None,
        status: str = None,
        chunk_label: str = None,
    ) -> Optional[str]:
        if not any((document_id, chunk_type, device_type, document_version, manual_type, record_type, status, chunk_label)):
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
            record_type=record_type,
            status=status,
            chunk_label=chunk_label,
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

    @staticmethod
    def _average_vectors(vectors: List[List[float]]) -> Optional[List[float]]:
        valid_vectors = [vector for vector in vectors if vector]
        if not valid_vectors:
            return None
        return [sum(values) / len(valid_vectors) for values in zip(*valid_vectors)]

    async def _embed_query_vectors(self, query: str, image_urls: List[str] = None) -> Dict[str, List[float] | List[List[float]]]:
        try:
            if image_urls:
                result = await get_multimodal_embedding().embed(text=query, image_urls=image_urls)
                text_vector = result.get("text_vector")
                image_vectors = result.get("image_vectors", [])
                if not text_vector and not image_vectors:
                    raise ToolException(code="EMBEDDING_FAILED", message="multimodal embedding returned no vectors")
                return {
                    "text_vector": text_vector,
                    "image_vectors": image_vectors,
                    "image_vector": self._average_vectors(image_vectors),
                }
            return {"text_vector": await get_text_embedding().embed(query), "image_vectors": [], "image_vector": None}
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(code="EMBEDDING_FAILED", message=f"embedding failed: {e}")

    async def _embed_query(self, query: str, image_urls: List[str] = None) -> List[float]:
        vectors = await self._embed_query_vectors(query, image_urls)
        text_vector = vectors.get("text_vector")
        image_vector = vectors.get("image_vector")
        if image_urls:
            fused = self._average_vectors([vector for vector in (text_vector, image_vector) if vector])
            if fused:
                return fused
        if text_vector:
            return text_vector
        if image_vector:
            return image_vector
        raise ToolException(code="EMBEDDING_FAILED", message="embedding returned no vectors")

    @staticmethod
    def _route_name(route: str, relaxed: bool = False) -> str:
        route_name = "table_vector" if route == "table" else route
        return f"{route_name}_relaxed" if relaxed else route_name

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
        query_vectors = await self._embed_query_vectors(query, image_urls)
        plan = build_retrieval_plan(query, has_images=bool(image_urls), explicit_chunk_type=chunk_type)
        confidence_type = confidence_intent(plan)
        recall_k = max(top_k * 3, top_k)
        optional_filter_used = any((category, tags, device_type, document_version, manual_type))

        def filter_for_route(route: str, relaxed: bool = False) -> Optional[str]:
            route_chunk_type = chunk_type
            if not route_chunk_type:
                if route == "table":
                    route_chunk_type = "table"
                elif route == "image_vector":
                    route_chunk_type = "image"
                elif route == "image_summary":
                    route_chunk_type = "image_summary"
            return self._build_filter(
                category=None if relaxed else category,
                tags=None if relaxed else tags,
                document_id=document_id,
                chunk_type=route_chunk_type,
                device_type=None if relaxed else device_type,
                document_version=None if relaxed else document_version,
                manual_type=None if relaxed else manual_type,
                record_type="manual",
                status="ready",
            )

        text_vector = query_vectors.get("text_vector")
        image_vector = query_vectors.get("image_vector") or text_vector

        async def run_route(route: str, relaxed: bool = False) -> List[Dict]:
            route_filter = filter_for_route(route, relaxed=relaxed)
            route_name = self._route_name(route, relaxed=relaxed)
            if route == "keyword":
                if not hasattr(vector_service, "keyword_search"):
                    return []
                docs = await asyncio.to_thread(
                    vector_service.keyword_search,
                    query,
                    top_k=recall_k,
                    include_metadata=True,
                    filter=route_filter,
                )
                return [self._mark_route(doc, route_name) for doc in docs]

            route_vector = text_vector
            if route == "image_vector":
                route_vector = image_vector
            elif route == "image_summary":
                route_vector = text_vector or image_vector
            if not route_vector:
                return []
            docs = await asyncio.to_thread(
                vector_service.search,
                route_vector,
                top_k=recall_k,
                include_metadata=True,
                filter=route_filter,
            )
            return [self._mark_route(doc, route_name) for doc in docs]

        try:
            vector_service = get_vector_service()
            route_results = await asyncio.gather(*(run_route(route) for route in plan.routes))
            candidates = [doc for docs in route_results for doc in docs]

            if not candidates and optional_filter_used and not document_id:
                logger.info("No evidence matched optional retrieval filters; retrying without inferred metadata filters")
                relaxed_results = await asyncio.gather(*(run_route(route, relaxed=True) for route in plan.routes))
                candidates.extend(doc for docs in relaxed_results for doc in docs)
        except Exception as e:
            raise ToolException(code="SEARCH_FAILED", message=f"retrieval search failed: {e}")

        merged = self._merge_candidates(candidates)
        ranked = rank_candidates(query, merged, plan)
        expensive_rerank_recommended = should_use_expensive_rerank(plan, ranked)
        selected = diversify_candidates(ranked, top_k=top_k, intent=confidence_type)
        confidence = summarize_confidence(selected, intent=confidence_type)
        expanded_selected = expand_retrieval_context(selected, vector_service, max_expanded=6)
        expanded_count = max(0, len(expanded_selected) - len(selected))

        results: List[VectorSearchResult] = []
        for doc in expanded_selected:
            metadata = dict(doc.get("metadata") or {})
            if metadata.get("chunk_type") == "image_summary":
                metadata["source_chunk_type"] = "image_summary"
                metadata["chunk_type"] = "image"
            routes = sorted(set(doc.get("routes") or []))
            metadata["retrieval_routes"] = routes
            metadata["matched_types"] = confidence["matched_types"]
            metadata["retrieval_confidence"] = confidence["confidence"]
            metadata["retrieval_plan_intent"] = plan.intent
            metadata["requires_strict_evidence"] = plan.requires_strict_evidence
            metadata["expensive_rerank_recommended"] = expensive_rerank_recommended
            metadata["retrieval_context_expanded_count"] = expanded_count
            metadata["confidence_reason"] = {
                "best_relevance_score": confidence["best_relevance_score"],
                "candidate_count": confidence["candidate_count"],
                "dual_image_hit": confidence["dual_image_hit"],
                "intent": plan.intent,
                "confidence_intent": confidence_type,
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
