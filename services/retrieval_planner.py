"""Query planning for knowledge retrieval routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PARAMETER_HINTS = (
    "参数",
    "规格",
    "型号",
    "扭矩",
    "力矩",
    "间隙",
    "标准",
    "数值",
    "多少",
    "单位",
    "N·m",
    "N路m",
    "mm",
    "MPa",
    "kPa",
    "电压",
    "电流",
    "torque",
    "spec",
    "specification",
    "parameter",
    "clearance",
)
PROCEDURE_HINTS = ("怎么", "如何", "步骤", "流程", "拆", "装", "更换", "维修", "检修", "安装", "调整", "操作")
DIAGNOSIS_HINTS = ("故障", "原因", "过热", "异响", "漏油", "启动不了", "报警", "异常", "怎么回事", "排除")
IMAGE_HINTS = ("图片", "图中", "图里", "图上", "识别", "这是什么", "照片", "示意图", "结构图", "位置图")


@dataclass(frozen=True)
class RetrievalPlan:
    intent: str
    routes: List[str]
    route_weights: Dict[str, float] = field(default_factory=dict)
    requires_strict_evidence: bool = False
    use_expensive_rerank: bool = False


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _plan_for_chunk_type(chunk_type: Optional[str]) -> Optional[RetrievalPlan]:
    if not chunk_type:
        return None
    if chunk_type == "table":
        return RetrievalPlan(
            intent="parameter",
            routes=["table", "keyword"],
            route_weights={"table": 0.18, "table_vector": 0.18, "keyword": 0.08},
            requires_strict_evidence=True,
        )
    if chunk_type == "image":
        return RetrievalPlan(
            intent="image_identification",
            routes=["image_vector"],
            route_weights={"image_vector": 0.18},
            requires_strict_evidence=True,
        )
    if chunk_type == "image_summary":
        return RetrievalPlan(
            intent="image_identification",
            routes=["image_summary"],
            route_weights={"image_summary": 0.16},
            requires_strict_evidence=True,
        )
    return RetrievalPlan(
        intent="general",
        routes=["semantic", "keyword"],
        route_weights={"semantic": 0.03, "keyword": 0.06},
    )


def build_retrieval_plan(
    query: str,
    has_images: bool = False,
    explicit_chunk_type: Optional[str] = None,
) -> RetrievalPlan:
    """Choose recall routes from the user query and optional uploaded images."""
    explicit_plan = _plan_for_chunk_type(explicit_chunk_type)
    if explicit_plan:
        return explicit_plan

    text = query or ""
    if has_images or _contains_any(text, IMAGE_HINTS):
        return RetrievalPlan(
            intent="image_identification",
            routes=["image_vector", "image_summary", "semantic"],
            route_weights={"image_vector": 0.18, "image_summary": 0.16, "semantic": 0.03},
            requires_strict_evidence=True,
        )

    if _contains_any(text, PARAMETER_HINTS):
        return RetrievalPlan(
            intent="parameter",
            routes=["table", "keyword", "semantic"],
            route_weights={"table": 0.18, "table_vector": 0.18, "keyword": 0.08, "semantic": 0.02},
            requires_strict_evidence=True,
        )

    if _contains_any(text, PROCEDURE_HINTS):
        return RetrievalPlan(
            intent="procedure",
            routes=["semantic", "keyword"],
            route_weights={"semantic": 0.04, "keyword": 0.07},
            requires_strict_evidence=True,
        )

    if _contains_any(text, DIAGNOSIS_HINTS):
        return RetrievalPlan(
            intent="diagnosis",
            routes=["semantic", "keyword"],
            route_weights={"semantic": 0.04, "keyword": 0.07},
            requires_strict_evidence=True,
        )

    return RetrievalPlan(
        intent="general",
        routes=["semantic", "keyword"],
        route_weights={"semantic": 0.03, "keyword": 0.06},
    )


def confidence_intent(plan: RetrievalPlan) -> str:
    """Map planner intents to the existing confidence helper's type vocabulary."""
    if plan.intent == "parameter":
        return "table"
    if plan.intent in {"procedure", "diagnosis"}:
        return "text"
    if plan.intent == "image_identification":
        return "image"
    return "mixed"
