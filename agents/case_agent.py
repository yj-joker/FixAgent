"""
检修案例沉淀 Agent

提供两个纯 JSON 能力，供 Java 编排调用：
1. draft_case  —— 把原始材料整理成结构化检修案例（含一轮 Basic Reflection 自检）
2. check_compliance —— 门控 LLM，判断文本是否可纳入设备检修知识库

设计原则：Java 负责编排，本模块只产出 JSON。
"""

import json
import logging

from config.settings import get_settings
from services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

_DRAFT_SYS = """你是设备检修案例整理助手。把给定材料整理成结构化检修案例，只输出 JSON：
{"title","summary","diagnosis","resolution","result","experience_summary","tags","downtime","cost"}
要求：忠于材料，不编造；experience_summary 提炼可复用经验；tags 用逗号分隔。"""

_REFLECT_SYS = """检查上一版案例 JSON 是否有：编造材料里没有的事实、遗漏关键步骤、字段错填。
若有问题输出修正后的完整 JSON；若没有问题，原样输出该 JSON。只输出 JSON。"""

_COMPLY_SYS = """你是内容合规审核员。判断文本是否可纳入设备检修知识库，只输出 JSON：
{"relevance":bool,"legality":bool,"reason":str}
relevance=是否属于设备检修/维修经验；legality=是否不含违法/有害/敏感/人身攻击。reason 说明拦截原因（中文）。"""

_TASK_VALIDATE_SYS = """判断输入是否是一个"有效的设备检修问题/故障现象"，而非乱码、空话或与检修无关的闲聊。
只输出 JSON：{"valid":bool,"reason":str}。
判定要宽松：只要像真实的设备故障/检修需求就算有效，简短描述（如"发动机异响"）也算有效；
仅当明显是乱码、测试字符、或与设备检修完全无关时才 valid=false，reason 用中文说明。"""

_GRAPH_VALIDATE_SYS = """判断以下"待录入知识图谱的检修实体清单"是否像真实、有意义的设备检修知识。
输入是 设备/部件/故障/解决方案 的名称清单。只输出 JSON：{"valid":bool,"reason":str}。
判定适度宽松：只要实体名看起来是真实的设备/部件/故障/检修方案就算有效；
仅当明显是乱码、测试字符、占位符（如"未知""test""aaa""无"）或与设备检修完全无关时才 valid=false，reason 用中文说明拦截原因。"""


async def draft_case(req) -> dict:
    """整理原始材料为结构化检修案例，含一轮 Basic Reflection 自检。"""
    llm = get_llm_service()
    material = req.task_context or req.raw_text or ""
    r1 = await llm.chat(
        messages=[
            {"role": "system", "content": _DRAFT_SYS},
            {"role": "user", "content": material},
        ],
        response_format={"type": "json_object"},
    )
    draft = _safe_json(r1["content"])
    r2 = await llm.chat(
        messages=[
            {"role": "system", "content": _REFLECT_SYS},
            {"role": "user", "content": json.dumps(draft or {}, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    refined = _safe_json(r2["content"]) or draft
    return _normalize_draft(refined or {})


def _normalize_draft(d: dict) -> dict:
    """把模型返回规整成 CaseDraftResponse 能接受的类型（LLM 常把 tags 输出成数组、downtime 输出成小数）。"""
    if not isinstance(d, dict):
        return {}
    # tags: 数组 → 逗号分隔字符串
    tags = d.get("tags")
    if isinstance(tags, list):
        d["tags"] = ",".join(str(t).strip() for t in tags if str(t).strip())
    elif tags is not None and not isinstance(tags, str):
        d["tags"] = str(tags)
    # downtime: 小数/字符串 → 整数（分钟）
    dt = d.get("downtime")
    if isinstance(dt, float):
        d["downtime"] = int(round(dt))
    elif isinstance(dt, str):
        try:
            d["downtime"] = int(round(float(dt)))
        except (ValueError, TypeError):
            d["downtime"] = None
    # cost: 字符串 → 浮点
    cost = d.get("cost")
    if isinstance(cost, str):
        try:
            d["cost"] = float(cost)
        except (ValueError, TypeError):
            d["cost"] = None
    # 其余结构化文本字段：若被输出成数组/对象，降级为字符串，避免响应校验失败
    for k in ("title", "summary", "diagnosis", "resolution", "result", "experience_summary"):
        v = d.get(k)
        if v is not None and not isinstance(v, str):
            d[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
    return d


async def check_compliance(text: str) -> dict:
    """门控 LLM：判断文本是否相关且合法。"""
    llm = get_llm_service()
    r = await llm.chat(
        messages=[
            {"role": "system", "content": _COMPLY_SYS},
            {"role": "user", "content": text[:4000]},
        ],
        response_format={"type": "json_object"},
    )
    d = _safe_json(r["content"]) or {}
    relevance = bool(d.get("relevance", False))
    legality = bool(d.get("legality", False))
    return {
        "compliant": relevance and legality,
        "relevance": relevance,
        "legality": legality,
        "reason": d.get("reason", ""),
    }


async def validate_task_text(text: str) -> dict:
    """轻量任务入口校验：宽松判定输入是否是有效的设备检修问题（用便宜快模型）。"""
    llm = get_llm_service()
    r = await llm.chat(
        messages=[
            {"role": "system", "content": _TASK_VALIDATE_SYS},
            {"role": "user", "content": text[:1000]},
        ],
        response_format={"type": "json_object"},
        model=get_settings().intent_router_model,
    )
    d = _safe_json(r["content"]) or {}
    # 缺省放行（宁松勿严）：模型没给 valid 时默认有效
    return {"valid": bool(d.get("valid", True)), "reason": d.get("reason", "")}


async def validate_graph_entities(text: str) -> dict:
    """图谱沉淀守门：判断待入图谱的抽取实体（设备/部件/故障/方案）是否像真实检修知识。"""
    llm = get_llm_service()
    r = await llm.chat(
        messages=[
            {"role": "system", "content": _GRAPH_VALIDATE_SYS},
            {"role": "user", "content": text[:2000]},
        ],
        response_format={"type": "json_object"},
        model=get_settings().intent_router_model,
    )
    d = _safe_json(r["content"]) or {}
    return {"valid": bool(d.get("valid", True)), "reason": d.get("reason", "")}


def _safe_json(s: str):
    """从模型文本中提取首个 JSON 对象，失败返回 None。"""
    try:
        import re
        m = re.search(r"\{.*\}", s or "", re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None
