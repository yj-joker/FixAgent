"""手册 → 知识图谱候选实体抽取。

输入：import_document 解析得到的 sections。
输出：符合 /weixiu/graph/ingest 契约的 graphData(dict)。
只对故障/排障/零件类章节抽取，不全文三元组化。
"""
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 命中任一关键词的章节才进入抽取
_SCHEMA_SECTION_HINTS = (
    "故障", "排障", "排除", "诊断", "维修", "检修", "异常", "报警",
    "零件", "部件", "配件", "元件", "备件",
)


def _section_text(section: Dict[str, Any]) -> str:
    """把一个 section 的所有文本块 + 表格拼成纯文本。"""
    texts = [c.get("text", "") if isinstance(c, dict) else str(c)
             for c in section.get("text_chunks", [])]
    table_text = "\n".join(
        " | ".join(str(x) for x in row)
        for t in section.get("tables", []) for row in t.get("rows", []) if row
    )
    return ("\n".join(texts) + "\n" + table_text).strip()


def select_schema_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """挑出 schema 相关章节：标题命中关键词，或正文/表格内容命中关键词。

    上游解析器常把整本手册归到单一无意义标题（如「前言」），
    因此不能只看标题，必须同时按正文内容兜底匹配。
    """
    kept = []
    for s in sections:
        title = s.get("section_title", "") or ""
        has_content = bool(s.get("text_chunks")) or bool(s.get("tables"))
        if not has_content:
            continue
        if any(h in title for h in _SCHEMA_SECTION_HINTS) or \
                any(h in _section_text(s) for h in _SCHEMA_SECTION_HINTS):
            kept.append(s)
    return kept


_EXTRACT_PROMPT = """你是设备维修知识抽取器。从给定的手册章节文本中，抽取「部件、故障、解决方案」三类实体及其关系。

严格只输出 JSON，结构如下（不要任何解释文字）：
{
  "components": [{"name":"部件名","specification":"规格(可空)"}],
  "faults": [{"name":"故障名","description":"描述","severity":"轻微/一般/严重/致命","category":"机械/电气/软件/其他","relatedComponent":"所属部件名(可空)"}],
  "solutions": [{"title":"方案名","summary":"做法摘要","toolsRequired":"所需工具(可空)","difficulty":"简单/中等/复杂(可空)","estimatedTime":整数分钟(可空),"relatedFault":"对应故障名"}]
}

规则：
- 只抽手册中明确写到的，不要编造。
- 名称用规范短词（如「主轴轴承」而非整句）。
- relatedComponent / relatedFault 必须引用本次抽取里出现过的 name，否则留空。
- 没有可抽内容时输出空数组。"""


# 单次 LLM 抽取的字符窗口，以及单个 section 最多切多少窗口（控制成本）
_WINDOW = 6000
_MAX_WINDOWS_PER_SECTION = 12


async def _extract_content(llm_service, title: str, content: str) -> Dict[str, Any]:
    """对一段文本做一次 LLM 结构化抽取。"""
    content = (content or "").strip()
    if len(content) < 20:
        return {"components": [], "faults": [], "solutions": []}
    messages = [
        {"role": "system", "content": _EXTRACT_PROMPT},
        {"role": "user", "content": f"【章节】{title}\n{content}"},
    ]
    resp = await llm_service.chat(messages)
    # chat() 非流式返回 dict({"content":...})，流式/其他实现可能直接返回字符串，两者都兼容
    text = resp.get("content", "") if isinstance(resp, dict) else (resp or "")
    try:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])
    except Exception as e:
        logger.warning("抽取JSON解析失败 section=%s err=%s", title, e)
        return {"components": [], "faults": [], "solutions": []}


async def _extract_section(llm_service, section: Dict[str, Any]) -> Dict[str, Any]:
    """抽取一个 section：内容过长时按窗口切分多次抽取再合并。"""
    title = section.get("section_title", "")
    full = _section_text(section)
    merged = {"components": [], "faults": [], "solutions": []}
    windows = [full[i:i + _WINDOW] for i in range(0, len(full), _WINDOW)][:_MAX_WINDOWS_PER_SECTION]
    for win in windows:
        part = await _extract_content(llm_service, title, win)
        for k in merged:
            merged[k].extend(part.get(k, []))
    return merged


def normalize_graph_data(raw: Dict[str, Any], manual_id: int,
                         document_id: str, device_names: List[str]) -> Dict[str, Any]:
    """合并去重 + 分配 tempId + 把 name 引用改写成 tempId 引用。"""
    comp_temp_by_name: Dict[str, str] = {}
    components = []
    for c in raw.get("components", []):
        name = (c.get("name") or "").strip()
        if not name or name in comp_temp_by_name:
            continue
        temp = f"c{len(components) + 1}"
        comp_temp_by_name[name] = temp
        components.append({"tempId": temp, "name": name,
                           "specification": (c.get("specification") or "").strip() or None})

    fault_temp_by_name: Dict[str, str] = {}
    faults = []
    for f in raw.get("faults", []):
        name = (f.get("name") or "").strip()
        if not name or name in fault_temp_by_name:
            continue
        temp = f"f{len(faults) + 1}"
        fault_temp_by_name[name] = temp
        faults.append({
            "tempId": temp, "name": name,
            "description": (f.get("description") or "").strip() or None,
            "severity": (f.get("severity") or "一般"),
            "category": (f.get("category") or "其他"),
            "relatedComponentTempId": comp_temp_by_name.get((f.get("relatedComponent") or "").strip()),
        })

    solutions = []
    for s in raw.get("solutions", []):
        title = (s.get("title") or "").strip()
        if not title:
            continue
        solutions.append({
            "tempId": f"s{len(solutions) + 1}", "title": title,
            "summary": (s.get("summary") or "").strip() or None,
            "toolsRequired": (s.get("toolsRequired") or "").strip() or None,
            "difficulty": (s.get("difficulty") or "").strip() or None,
            "estimatedTime": s.get("estimatedTime") if isinstance(s.get("estimatedTime"), int) else None,
            "relatedFaultTempId": fault_temp_by_name.get((s.get("relatedFault") or "").strip()),
        })

    return {"manualId": manual_id, "documentId": document_id,
            "deviceNames": device_names or [],
            "components": components, "faults": faults, "solutions": solutions}


async def extract_graph_data(llm_service, sections: List[Dict[str, Any]],
                             manual_id: int, document_id: str,
                             device_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """主入口：筛章节 → 逐章 LLM 抽取 → 合并 → 规范化。"""
    merged = {"components": [], "faults": [], "solutions": []}
    for section in select_schema_sections(sections):
        part = await _extract_section(llm_service, section)
        for k in merged:
            merged[k].extend(part.get(k, []))
    return normalize_graph_data(merged, manual_id, document_id, device_names or [])
