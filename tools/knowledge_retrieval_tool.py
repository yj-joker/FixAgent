"""Knowledge retrieval tool for multimodal RAG evidence."""

from __future__ import annotations

import logging
import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

from embeddings.multimodal_embedding import get_multimodal_embedding
from embeddings.text_embedding import get_text_embedding
from schemas.models import VectorSearchResult
from services.retrieval_planner import build_retrieval_plan, confidence_intent
from services.retrieval_ranker import rank_candidates, should_use_expensive_rerank
from services.retrieval_context_expander import expand_retrieval_context
from services.retrieval_fusion import DEFAULT_RRF_CONSTANT, reciprocal_rank_fusion
from services.retrieval_quality import evaluate_retrieval_quality
from services.retrieval_policy import (
    diversify_candidates,
    summarize_confidence,
)
from services.vector_service import build_redis_filter, escape_redis_tag_value, get_vector_service
from tools.base_tool import BaseTool, ToolException

logger = logging.getLogger(__name__)

DEFAULT_RECALL_TOP_N = 50


async def _emit_retrieval_event(
    event_sink: Optional[Callable[[Dict[str, Any]], Any]],
    event: str,
    data: Dict[str, Any],
) -> None:
    if not event_sink:
        return
    result = event_sink({"event": event, "data": data})
    if inspect.isawaitable(result):
        await result


def _count_selected_types(items: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items or []:
        metadata = item.get("metadata") or {}
        chunk_type = metadata.get("chunk_type") or "text"
        if chunk_type == "image_summary":
            chunk_type = "image"
        chunk_type = str(chunk_type or "text")
        counts[chunk_type] = counts.get(chunk_type, 0) + 1
    return counts


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
        if route == "text":
            route_name = "text_vector"
        else:
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
        _event_sink: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> List[VectorSearchResult]:
        query_vectors = await self._embed_query_vectors(query, image_urls)
        plan = build_retrieval_plan(query, has_images=bool(image_urls), explicit_chunk_type=chunk_type)
        confidence_type = confidence_intent(plan)
        final_top_k = max(int(top_k or 0), 0)
        recall_k = max(final_top_k * 3, DEFAULT_RECALL_TOP_N) if final_top_k else 0
        optional_filter_used = any((category, tags, device_type, document_version, manual_type))
        await _emit_retrieval_event(
            _event_sink,
            "retrieval_start",
            {
                "query": query,
                "intent": plan.intent,
                "routes": list(plan.routes),
                "topK": final_top_k,
                "recallTopN": recall_k,
                "hasImages": bool(image_urls),
            },
        )

        def filter_for_route(route: str, relaxed: bool = False) -> Optional[str]:
            route_chunk_type = chunk_type
            if not route_chunk_type:
                if route == "table":
                    route_chunk_type = "table"
                elif route == "text":
                    route_chunk_type = "text"
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

        async def run_route(route: str, relaxed: bool = False, limit: int = None) -> List[Dict]:
            route_filter = filter_for_route(route, relaxed=relaxed)
            route_name = self._route_name(route, relaxed=relaxed)
            route_top_k = limit or recall_k
            if route == "keyword":
                if not hasattr(vector_service, "keyword_search"):
                    await _emit_retrieval_event(
                        _event_sink,
                        "retrieval_route",
                        {
                            "route": route_name,
                            "sourceRoute": route,
                            "candidateCount": 0,
                            "limit": route_top_k,
                            "relaxed": relaxed,
                            "skipped": True,
                        },
                    )
                    return []
                docs = await asyncio.to_thread(
                    vector_service.keyword_search,
                    query,
                    top_k=route_top_k,
                    include_metadata=True,
                    filter=route_filter,
                )
                marked = [self._mark_route(doc, route_name) for doc in docs]
                await _emit_retrieval_event(
                    _event_sink,
                    "retrieval_route",
                    {
                        "route": route_name,
                        "sourceRoute": route,
                        "candidateCount": len(marked),
                        "limit": route_top_k,
                        "relaxed": relaxed,
                    },
                )
                return marked

            route_vector = text_vector
            if route == "image_vector":
                route_vector = image_vector
            elif route == "image_summary":
                route_vector = text_vector or image_vector
            if not route_vector:
                await _emit_retrieval_event(
                    _event_sink,
                    "retrieval_route",
                    {
                        "route": route_name,
                        "sourceRoute": route,
                        "candidateCount": 0,
                        "limit": route_top_k,
                        "relaxed": relaxed,
                        "skipped": True,
                    },
                )
                return []
            docs = await asyncio.to_thread(
                vector_service.search,
                route_vector,
                top_k=route_top_k,
                include_metadata=True,
                filter=route_filter,
            )
            marked = [self._mark_route(doc, route_name) for doc in docs]
            await _emit_retrieval_event(
                _event_sink,
                "retrieval_route",
                {
                    "route": route_name,
                    "sourceRoute": route,
                    "candidateCount": len(marked),
                    "limit": route_top_k,
                    "relaxed": relaxed,
                },
            )
            return marked

        try:
            vector_service = get_vector_service()
            route_results = await asyncio.gather(*(run_route(route) for route in plan.routes))
            candidate_lists = [list(docs) for docs in route_results]

            if not any(candidate_lists) and optional_filter_used and not document_id:
                logger.info("No evidence matched optional retrieval filters; retrying without inferred metadata filters")
                relaxed_results = await asyncio.gather(*(run_route(route, relaxed=True) for route in plan.routes))
                candidate_lists = [list(docs) for docs in relaxed_results]
        except Exception as e:
            raise ToolException(code="SEARCH_FAILED", message=f"retrieval search failed: {e}")

        fused = reciprocal_rank_fusion(
            candidate_lists,
            key_fn=self._canonical_id,
            top_k=recall_k,
            rrf_constant=DEFAULT_RRF_CONSTANT,
        )
        merged = self._merge_candidates(fused)
        ranked = rank_candidates(query, merged, plan)
        selected = diversify_candidates(ranked, top_k=final_top_k, intent=confidence_type)
        first_quality = evaluate_retrieval_quality(plan, ranked, selected, top_k=final_top_k)
        candidate_count_before = len(merged)
        supplemental_search_used = False
        supplemental_routes: List[str] = []
        await _emit_retrieval_event(
            _event_sink,
            "retrieval_quality",
            {
                "stage": "first_pass",
                "grade": first_quality.grade,
                "score": first_quality.score,
                "candidateCount": first_quality.candidate_count,
                "bestScore": first_quality.best_score,
                "matchedTypes": first_quality.matched_types,
                "requiredTypes": first_quality.required_types,
                "reasons": first_quality.reasons,
                "shouldSupplement": first_quality.should_supplement,
                "supplementalRoutes": first_quality.supplemental_routes,
            },
        )

        if first_quality.should_supplement:
            supplemental_search_used = True
            supplemental_routes = first_quality.supplemental_routes
            supplemental_limit = max(recall_k * 2, top_k * 6, 6)
            await _emit_retrieval_event(
                _event_sink,
                "retrieval_supplement",
                {
                    "routes": supplemental_routes,
                    "limit": supplemental_limit,
                    "reasons": first_quality.reasons,
                },
            )
            try:
                supplemental_results = await asyncio.gather(
                    *(run_route(route, limit=supplemental_limit) for route in supplemental_routes)
                )
            except Exception as e:
                raise ToolException(code="SEARCH_FAILED", message=f"supplemental retrieval failed: {e}")
            candidate_lists.extend(list(docs) for docs in supplemental_results)
            fused = reciprocal_rank_fusion(
                candidate_lists,
                key_fn=self._canonical_id,
                top_k=max(recall_k, supplemental_limit),
                rrf_constant=DEFAULT_RRF_CONSTANT,
            )
            merged = self._merge_candidates(fused)
            ranked = rank_candidates(query, merged, plan)
            selected = diversify_candidates(ranked, top_k=final_top_k, intent=confidence_type)

        final_quality = evaluate_retrieval_quality(plan, ranked, selected, top_k=final_top_k)
        candidate_count_after = len(merged)
        await _emit_retrieval_event(
            _event_sink,
            "retrieval_quality",
            {
                "stage": "final",
                "grade": final_quality.grade,
                "score": final_quality.score,
                "candidateCount": final_quality.candidate_count,
                "bestScore": final_quality.best_score,
                "matchedTypes": final_quality.matched_types,
                "requiredTypes": final_quality.required_types,
                "reasons": final_quality.reasons,
                "shouldSupplement": False,
                "supplementalRoutes": supplemental_routes,
            },
        )
        expensive_rerank_recommended = should_use_expensive_rerank(plan, ranked)
        confidence = summarize_confidence(selected, intent=confidence_type)
        expanded_selected = expand_retrieval_context(selected, vector_service, max_expanded=6)
        expanded_count = max(0, len(expanded_selected) - len(selected))
        await _emit_retrieval_event(
            _event_sink,
            "retrieval_expand",
            {
                "expandedCount": expanded_count,
                "primaryCount": len(selected),
                "totalCount": len(expanded_selected),
                "strategy": "parent_section_context",
            },
        )

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
            metadata["adaptive_rag_enabled"] = True
            metadata["recall_top_n"] = recall_k
            metadata["final_top_k"] = final_top_k
            metadata.setdefault("rrf_enabled", False)
            metadata.setdefault("rrf_constant", DEFAULT_RRF_CONSTANT)
            metadata["first_pass_quality"] = first_quality.grade
            metadata["final_quality"] = final_quality.grade
            metadata["first_pass_quality_score"] = first_quality.score
            metadata["final_quality_score"] = final_quality.score
            metadata["first_pass_quality_reasons"] = first_quality.reasons
            metadata["final_quality_reasons"] = final_quality.reasons
            metadata["quality_reasons"] = final_quality.reasons
            metadata["required_evidence_types"] = final_quality.required_types
            metadata["supplemental_search_used"] = supplemental_search_used
            metadata["supplemental_routes"] = supplemental_routes
            metadata["candidate_count_before"] = candidate_count_before
            metadata["candidate_count_after"] = candidate_count_after
            metadata["confidence_reason"] = {
                "best_relevance_score": confidence["best_relevance_score"],
                "candidate_count": confidence["candidate_count"],
                "dual_image_hit": confidence["dual_image_hit"],
                "intent": plan.intent,
                "confidence_intent": confidence_type,
                "first_pass_quality": first_quality.grade,
                "final_quality": final_quality.grade,
            }
            if confidence["confidence"] == "low" or final_quality.grade == "low":
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
        await _emit_retrieval_event(
            _event_sink,
            "retrieval_done",
            {
                "selectedCount": len(results),
                "primaryCount": len(selected),
                "expandedCount": expanded_count,
                "candidateCountBefore": candidate_count_before,
                "candidateCountAfter": candidate_count_after,
                "countsByType": _count_selected_types(expanded_selected),
                "finalQuality": final_quality.grade,
                "finalQualityScore": final_quality.score,
                "supplementalSearchUsed": supplemental_search_used,
                "supplementalRoutes": supplemental_routes,
            },
        )
        return results


_retrieval_tool: Optional[KnowledgeRetrievalTool] = None


def get_knowledge_retrieval_tool() -> KnowledgeRetrievalTool:
    global _retrieval_tool
    if _retrieval_tool is None:
        _retrieval_tool = KnowledgeRetrievalTool()
    return _retrieval_tool
