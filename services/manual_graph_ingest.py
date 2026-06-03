"""手册图谱抽取编排：manifest.kg_sections → 抽取 → POST Java ingest → 写 kg_status。"""
import logging
import httpx
from services.vector_service import get_vector_service
from services.llm_service import get_llm_service
from services.manual_graph_extractor import extract_graph_data
from config.settings import get_settings

logger = logging.getLogger(__name__)

MAX_RETRY = 3
_RETRYABLE = (TimeoutError, ConnectionError, httpx.HTTPError)


def classify_error(exc: Exception) -> str:
    return "failed_retryable" if isinstance(exc, _RETRYABLE) else "failed_permanent"


async def run_kg_extraction(manual_id, document_id: str, device_type=None) -> dict:
    settings = get_settings()
    vector_svc = get_vector_service()
    manifest = vector_svc.get_document_manifest(document_id) or {}
    sections = manifest.get("kg_sections") or []
    if not sections:
        vector_svc.put_document_manifest(document_id, {**manifest, "kg_status": "failed_permanent",
                                                       "kg_error": "no schema sections"})
        raise ValueError("no schema sections")

    vector_svc.put_document_manifest(document_id, {**manifest, "kg_status": "extracting"})
    device_names = [manifest.get("device_type")] if manifest.get("device_type") else (
        [device_type] if device_type else [])

    try:
        graph_data = await extract_graph_data(
            get_llm_service(), sections, manual_id, document_id, device_names)
        # Java 入库需为每个部件/故障逐个生成向量(调百炼),大批量手册耗时可达数分钟,
        # 超时给足，避免实际已入库却被误判失败、触发 sweeper 无谓重试
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{settings.java_service_url}/weixiu/graph/ingest",
                json=graph_data,
                headers={"X-Internal-Token": settings.internal_token},
            )
            resp.raise_for_status()
        vector_svc.put_document_manifest(document_id, {
            **manifest, "kg_status": "ready", "kg_error": "",
            "kg_counts": {"components": len(graph_data["components"]),
                          "faults": len(graph_data["faults"]),
                          "solutions": len(graph_data["solutions"])},
        })
        return graph_data
    except Exception as exc:
        retry = int(manifest.get("kg_retry_count", 0)) + 1
        status = classify_error(exc)
        if status == "failed_retryable" and retry >= MAX_RETRY:
            status = "failed_permanent"
        vector_svc.put_document_manifest(document_id, {
            **manifest, "kg_status": status, "kg_retry_count": retry, "kg_error": str(exc)})
        logger.error("[KG抽取] 失败 doc=%s status=%s retry=%d err=%s",
                     document_id, status, retry, exc)
        raise
