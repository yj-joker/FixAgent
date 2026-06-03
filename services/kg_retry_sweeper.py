import asyncio, logging
from services.vector_service import get_vector_service
from services.manual_graph_ingest import run_kg_extraction

logger = logging.getLogger(__name__)
SWEEP_INTERVAL_SECONDS = 1800  # 30分钟


async def sweep_once():
    vs = get_vector_service()
    for doc_id in vs.list_documents_by_kg_status("failed_retryable"):
        m = vs.get_document_manifest(doc_id) or {}
        manual_id = m.get("manual_id") or m.get("manualId")
        try:
            await run_kg_extraction(manual_id, doc_id, m.get("device_type"))
        except Exception as e:
            logger.warning("[KG重试] doc=%s 仍失败: %s", doc_id, e)


async def start_kg_retry_sweeper():
    while True:
        try:
            await sweep_once()
        except Exception as e:
            logger.error("[KG重试] sweep异常: %s", e)
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
