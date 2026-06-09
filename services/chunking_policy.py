"""Structured chunking policy for maintenance manual evidence."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


GENERAL_CHUNK_TARGET = 520
GENERAL_CHUNK_OVERLAP = 90

STEP_HINT_RE = re.compile(r"(^|\n)\s*(?:第?\d+[\.、．)]|\(\d+\)|[一二三四五六七八九十]+[、.．])\s*")
SAFETY_HINTS = (
    "注意",
    "警告",
    "危险",
    "断电",
    "停机",
    "高温",
    "冷却后",
    "泄压",
    "防护",
    "护目镜",
    "不得",
    "禁止",
)
TROUBLESHOOTING_HINTS = ("故障", "原因", "处理", "解决", "排除", "异常", "报警")
UNIT_RE = re.compile(
    r"(?:N[·路\.]?m|kW|KW|V|A|MPa|kPa|Pa|mm|cm|m/s|r/min|rpm|℃|°C|L|mL|kg|g|%)",
    re.IGNORECASE,
)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _section_id(section_index: int) -> str:
    return f"sec:{section_index:04d}"


def _base_metadata(section: Dict[str, Any], section_index: int) -> Dict[str, Any]:
    return {
        "record_type": "manual",
        "status": "ready",
        "section_index": section_index,
        "section_title": _as_text(section.get("section_title")),
        "page_range": _as_text(section.get("page_range")),
        "parent_section_id": _section_id(section_index),
    }


def _emit_chunk(
    chunks: List[Dict[str, Any]],
    *,
    text: str,
    chunk_type: str,
    chunk_label: str,
    section: Dict[str, Any],
    section_index: int,
    page: Any = None,
    source_index: int | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    clean = _as_text(text)
    if not clean:
        return None
    local_id = f"{_section_id(section_index)}:{chunk_type}:{len(chunks):04d}"
    chunk_metadata = {
        **_base_metadata(section, section_index),
        "chunk_uid": local_id,
        "chunk_type": chunk_type,
        "chunk_label": chunk_label,
    }
    if source_index is not None:
        chunk_metadata["source_index"] = source_index
    if metadata:
        chunk_metadata.update({k: v for k, v in metadata.items() if v not in (None, "")})
    chunk = {
        "id": local_id,
        "text": clean,
        "page": page,
        "chunk_type": chunk_type,
        "chunk_label": chunk_label,
        "metadata": chunk_metadata,
    }
    chunks.append(chunk)
    return chunk


def _split_numbered_steps(text: str) -> List[str]:
    clean = _as_text(text)
    if not clean:
        return []
    matches = list(STEP_HINT_RE.finditer(clean))
    if len(matches) <= 1:
        return [clean]
    parts = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        part = clean[start:end].strip()
        if part:
            parts.append(part)
    return parts or [clean]


def _split_safety_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[。！？!?；;])\s*|\n+", _as_text(text))
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and any(hint in sentence for hint in SAFETY_HINTS)
    ]


def _looks_like_safety(text: str) -> bool:
    return any(hint in _as_text(text) for hint in SAFETY_HINTS)


def _looks_like_troubleshooting(text: str) -> bool:
    clean = _as_text(text)
    return sum(1 for hint in TROUBLESHOOTING_HINTS if hint in clean) >= 2


def _looks_like_step(text: str, label: str = "") -> bool:
    return label == "step" or bool(STEP_HINT_RE.search(_as_text(text)))


def _split_general_text(text: str) -> List[str]:
    clean = _as_text(text)
    if len(clean) <= GENERAL_CHUNK_TARGET:
        return [clean] if clean else []

    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + GENERAL_CHUNK_TARGET)
        boundary = max(
            clean.rfind("。", start, end),
            clean.rfind("\n", start, end),
            clean.rfind("；", start, end),
        )
        if boundary > start + 160:
            end = boundary + 1
        part = clean[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(clean):
            break
        start = max(end - GENERAL_CHUNK_OVERLAP, start + 1)
    return chunks


def _table_rows(table: Dict[str, Any]) -> List[List[str]]:
    rows = table.get("rows") or []
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append([_as_text(k) + "=" + _as_text(v) for k, v in row.items()])
        elif isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            normalized.append([_as_text(cell) for cell in row])
    return [row for row in normalized if any(row)]


def _table_to_text(table: Dict[str, Any]) -> str:
    rows = _table_rows(table)
    lines = []
    caption = _as_text(table.get("caption"))
    if caption:
        lines.append(f"表格：{caption}")
    lines.extend(" | ".join(cell for cell in row if cell) for row in rows)
    return "\n".join(line for line in lines if line)


def _table_headers(rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    headers = [_as_text(cell) for cell in rows[0]]
    if any(headers):
        return [header or f"col_{idx + 1}" for idx, header in enumerate(headers)]
    max_cols = max(len(row) for row in rows)
    return [f"col_{idx + 1}" for idx in range(max_cols)]


def _table_data_rows(rows: List[List[str]], headers: List[str]) -> List[List[str]]:
    if not rows:
        return []
    if headers and rows[0] == headers:
        return rows[1:]
    return rows[1:] if len(rows) > 1 and any(rows[0]) else rows


def _row_to_text(caption: str, headers: List[str], row: List[str]) -> str:
    pairs = []
    for idx, value in enumerate(row):
        clean = _as_text(value)
        if not clean:
            continue
        header = headers[idx] if idx < len(headers) else f"col_{idx + 1}"
        pairs.append(f"{header}={clean}")
    prefix = f"表格：{caption}\n" if caption else ""
    return prefix + "；".join(pairs)


def _extract_units(values: Iterable[Any]) -> List[str]:
    units = []
    for value in values:
        for match in UNIT_RE.findall(_as_text(value)):
            if match not in units:
                units.append(match)
    return units


def _link_neighbors(chunks: List[Dict[str, Any]]) -> None:
    for idx, chunk in enumerate(chunks):
        metadata = chunk.setdefault("metadata", {})
        if idx > 0:
            metadata["prev_chunk_id"] = chunks[idx - 1]["id"]
        if idx + 1 < len(chunks):
            metadata["next_chunk_id"] = chunks[idx + 1]["id"]


def build_section_index_chunks(section: Dict[str, Any], section_index: int = 0) -> List[Dict[str, Any]]:
    """Build retrieval-ready child chunks for one parsed manual section."""
    chunks: List[Dict[str, Any]] = []
    step_chunk_ids: List[str] = []
    text_chunk_ids: List[str] = []

    for source_index, raw in enumerate(section.get("text_chunks") or []):
        if isinstance(raw, dict):
            text = _as_text(raw.get("text"))
            page = raw.get("page")
            label = _as_text(raw.get("chunk_label")) or "general"
            context = {
                "context_before": raw.get("context_before", ""),
                "context_after": raw.get("context_after", ""),
            }
        else:
            text = _as_text(raw)
            page = None
            label = "general"
            context = {}

        if not text:
            continue

        emitted_primary = []
        if _looks_like_step(text, label):
            for part in _split_numbered_steps(text):
                chunk = _emit_chunk(
                    chunks,
                    text=part,
                    chunk_type="text",
                    chunk_label="step",
                    section=section,
                    section_index=section_index,
                    page=page,
                    source_index=source_index,
                    metadata=context,
                )
                if chunk:
                    emitted_primary.append(chunk)
                    step_chunk_ids.append(chunk["id"])
                    text_chunk_ids.append(chunk["id"])
        elif _looks_like_troubleshooting(text):
            chunk = _emit_chunk(
                chunks,
                text=text,
                chunk_type="text",
                chunk_label="troubleshooting",
                section=section,
                section_index=section_index,
                page=page,
                source_index=source_index,
                metadata=context,
            )
            if chunk:
                emitted_primary.append(chunk)
                text_chunk_ids.append(chunk["id"])
        else:
            label = "safety" if _looks_like_safety(text) else "general"
            for part in _split_general_text(text):
                chunk = _emit_chunk(
                    chunks,
                    text=part,
                    chunk_type="text",
                    chunk_label=label,
                    section=section,
                    section_index=section_index,
                    page=page,
                    source_index=source_index,
                    metadata=context,
                )
                if chunk:
                    emitted_primary.append(chunk)
                    text_chunk_ids.append(chunk["id"])

        if label != "safety":
            seen_safety_texts = {chunk["text"] for chunk in emitted_primary if chunk.get("chunk_label") == "safety"}
            for safety_text in _split_safety_sentences(text):
                if safety_text in seen_safety_texts:
                    continue
                chunk = _emit_chunk(
                    chunks,
                    text=safety_text,
                    chunk_type="text",
                    chunk_label="safety",
                    section=section,
                    section_index=section_index,
                    page=page,
                    source_index=source_index,
                    metadata=context,
                )
                if chunk:
                    text_chunk_ids.append(chunk["id"])

    for table_index, table in enumerate(section.get("tables") or []):
        table_text = _table_to_text(table)
        rows = _table_rows(table)
        headers = _table_headers(rows)
        data_rows = _table_data_rows(rows, headers)
        caption = _as_text(table.get("caption"))
        page = table.get("page")
        table_units = _extract_units(cell for row in rows for cell in row)
        table_meta = {
            "table_index": table_index,
            "caption": caption,
            "headers": headers,
            "table_rows": len(data_rows),
            "units": table_units,
        }
        table_full_chunk = None
        if table_text:
            table_full_chunk = _emit_chunk(
                chunks,
                text=table_text,
                chunk_type="table",
                chunk_label="table_full",
                section=section,
                section_index=section_index,
                page=page,
                source_index=table_index,
                metadata=table_meta,
            )

        for row_index, row in enumerate(data_rows):
            row_text = _row_to_text(caption, headers, row)
            units = _extract_units(row + headers)
            parameter_names = [row[0]] if row and row[0] else []
            _emit_chunk(
                chunks,
                text=row_text,
                chunk_type="table",
                chunk_label="table_row",
                section=section,
                section_index=section_index,
                page=page,
                source_index=table_index,
                metadata={
                    **table_meta,
                    "row_index": row_index,
                    "units": units or table_units,
                    "parameter_names": parameter_names,
                    "parent_table_chunk_id": table_full_chunk["id"] if table_full_chunk else "",
                },
            )

    for image_index, image in enumerate(section.get("images") or []):
        caption = _as_text(image.get("caption"))
        image_name = _as_text(image.get("image_name")) or f"img_{image_index}"
        page = image.get("page")
        text = caption or f"{_as_text(section.get('section_title'))} 第{page or '?'}页插图"
        _emit_chunk(
            chunks,
            text=text,
            chunk_type="image",
            chunk_label="image",
            section=section,
            section_index=section_index,
            page=page,
            source_index=image_index,
            metadata={
                "image_index": image_index,
                "image_name": image_name,
                "caption": caption,
                "related_step_chunk_ids": step_chunk_ids[:5],
                "related_text_chunk_ids": text_chunk_ids[:5],
            },
        )

    _link_neighbors(chunks)
    return chunks
