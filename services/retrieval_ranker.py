"""Lightweight domain ranking for maintenance RAG candidates."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from services.retrieval_planner import RetrievalPlan


UNIT_RE = re.compile(r"(?:N[·路\.]?m|kW|KW|V|A|MPa|kPa|Pa|mm|cm|m/s|r/min|rpm|℃|°C|L|mL|kg|g|%)", re.IGNORECASE)


def _metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return candidate.get("metadata") or {}


def _chunk_type(candidate: Dict[str, Any]) -> str:
    value = _metadata(candidate).get("chunk_type", "text")
    return "image" if value == "image_summary" else value


def _chunk_label(candidate: Dict[str, Any]) -> str:
    return _metadata(candidate).get("chunk_label", "")


def _content(candidate: Dict[str, Any]) -> str:
    metadata = _metadata(candidate)
    values = [
        candidate.get("text", ""),
        metadata.get("section_title", ""),
        metadata.get("caption", ""),
        metadata.get("image_title", ""),
        metadata.get("image_summary", ""),
        metadata.get("parameter_names", ""),
        metadata.get("units", ""),
    ]
    return " ".join(str(value) for value in values if value)


def _query_terms(query: str) -> List[str]:
    text = query or ""
    terms = [part for part in re.split(r"[\s,，。；;、:：?？!！]+", text) if part]
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    terms.extend(chinese)
    seen = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return seen


def _lexical_bonus(query: str, candidate: Dict[str, Any]) -> float:
    content = _content(candidate)
    overlap = 0
    for term in _query_terms(query):
        if term and term in content:
            overlap += 1
    return min(0.12, overlap * 0.04)


def _route_bonus(candidate: Dict[str, Any], plan: RetrievalPlan) -> float:
    routes = set(candidate.get("routes") or [])
    if not routes and candidate.get("retrieval_route"):
        routes.add(candidate["retrieval_route"])
    return min(0.2, sum(plan.route_weights.get(route, 0.0) for route in routes))


def _domain_bonus(query: str, candidate: Dict[str, Any], plan: RetrievalPlan) -> float:
    metadata = _metadata(candidate)
    chunk_type = _chunk_type(candidate)
    chunk_label = _chunk_label(candidate)
    content = _content(candidate)
    bonus = 0.0

    if plan.intent == "parameter":
        if chunk_type == "table":
            bonus += 0.10
        if chunk_label == "table_row":
            bonus += 0.16
        elif chunk_label == "table_full":
            bonus += 0.10
        if metadata.get("units") or UNIT_RE.search(content) or UNIT_RE.search(query or ""):
            bonus += 0.08
        if metadata.get("parameter_names"):
            bonus += 0.04
    elif plan.intent == "procedure":
        if chunk_label == "step":
            bonus += 0.14
        if chunk_label == "safety":
            bonus += 0.04
    elif plan.intent == "diagnosis":
        if chunk_label == "troubleshooting":
            bonus += 0.15
        if any(word in content for word in ("故障", "原因", "处理", "排除")):
            bonus += 0.04
    elif plan.intent == "image_identification":
        if chunk_type == "image":
            bonus += 0.10
        if chunk_label in {"image", "image_summary"}:
            bonus += 0.08

    if any("relaxed" in route for route in candidate.get("routes") or []):
        bonus -= 0.08
    return bonus


def rank_candidates(query: str, candidates: Iterable[Dict[str, Any]], plan: RetrievalPlan) -> List[Dict[str, Any]]:
    """Rank candidates using route, lexical and maintenance-domain signals."""
    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        base = candidate.get("relevance_score")
        if base is None:
            base = candidate.get("score", 0.0)
        score = float(base or 0.0)
        score += _route_bonus(candidate, plan)
        score += _lexical_bonus(query, candidate)
        score += _domain_bonus(query, candidate, plan)

        item = dict(candidate)
        item["metadata"] = dict(candidate.get("metadata") or {})
        item["rerank_score"] = round(max(0.0, min(1.0, score)), 6)
        item["metadata"]["retrieval_plan_intent"] = plan.intent
        ranked.append(item)

    return sorted(ranked, key=lambda item: item.get("rerank_score", 0.0), reverse=True)


def should_use_expensive_rerank(plan: RetrievalPlan, ranked_candidates: Iterable[Dict[str, Any]]) -> bool:
    """Gate a future external reranker to high-value ambiguous cases."""
    if plan.intent not in {"parameter", "procedure", "diagnosis", "image_identification"}:
        return False
    items = sorted(
        list(ranked_candidates),
        key=lambda item: item.get("rerank_score", item.get("relevance_score", 0.0)),
        reverse=True,
    )
    if not items:
        return False
    best = float(items[0].get("rerank_score", items[0].get("relevance_score", 0.0)) or 0.0)
    if best < 0.65:
        return True
    if len(items) >= 2:
        second = float(items[1].get("rerank_score", items[1].get("relevance_score", 0.0)) or 0.0)
        return abs(best - second) <= 0.04
    return False
