"""
检修案例沉淀 Agent

提供两个纯 JSON 能力，供 Java 编排调用：
1. draft_case  —— 把原始材料整理成结构化检修案例（含一轮 Basic Reflection 自检）
2. check_compliance —— 门控 LLM，判断文本是否可纳入设备检修知识库

设计原则：Java 负责编排，本模块只产出 JSON。
"""

import json
import logging

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
    return refined or {}


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


def _safe_json(s: str):
    """从模型文本中提取首个 JSON 对象，失败返回 None。"""
    try:
        import re
        m = re.search(r"\{.*\}", s or "", re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None
