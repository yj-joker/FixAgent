from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import run_async, run_auto_cases, run_menu


def vec(v):
    return [float(v)] * 1024


def parse_result():
    return {
        "file_name": "manual.pdf",
        "total_pages": 1,
        "sections": [{
            "section_title": "发动机",
            "page_range": "1-1",
            "text_chunks": [{
                "text": "1. 排放机油\n拆下发动机左侧放油螺栓。",
                "page": 1,
                "chunk_label": "step",
                "context_before": "3.2 拆卸发动机",
                "context_after": "",
            }],
            "tables": [],
            "images": [{
                "page": 1,
                "image_name": "page_001_img_01.png",
                "local_path": "D:/manual_images/page_001_img_01.png",
                "caption": "",
                "context_before": "3.2 拆卸发动机\n1. 排放机油",
                "context_after": "拆下发动机左侧放油螺栓。",
            }],
        }],
        "extraction_summary": {"text_chunks_total": 1, "tables_total": 0, "images_total": 1},
    }


def build_service():
    with patch("services.knowledge_service.get_document_parser") as get_parser, \
         patch("services.knowledge_service.get_text_embedding") as get_text, \
         patch("services.knowledge_service.get_image_embedding") as get_image, \
         patch("services.knowledge_service.get_vector_service") as get_vector, \
         patch("services.knowledge_service.get_file_storage") as get_storage, \
         patch("services.knowledge_service.get_image_summary_service") as get_summary:
        from services.knowledge_service import KnowledgeService

        parser = MagicMock()
        parser._execute = AsyncMock(return_value=parse_result())
        text = MagicMock()
        text.embed_batch = AsyncMock(return_value=[vec(1)])
        text.embed = AsyncMock(return_value=vec(2))
        image = MagicMock()
        image.embed = AsyncMock(return_value=vec(3))
        vector = MagicMock()
        vector.add_vector_batch.return_value = 1
        vector.add_vector.return_value = True
        storage = MagicMock()
        storage.ensure_public_url.return_value = "/files/manual_images/page_001_img_01.png"
        summary = MagicMock()
        summary.summarize = AsyncMock(return_value={
            "image_title": "发动机放油螺栓位置图",
            "image_summary": "图片展示发动机左侧放油螺栓所在位置。",
        })

        get_parser.return_value = parser
        get_text.return_value = text
        get_image.return_value = image
        get_vector.return_value = vector
        get_storage.return_value = storage
        get_summary.return_value = summary
        service = KnowledgeService()
    return service, text, image, vector, storage, summary


def auto_test():
    async def import_case():
        service, text, image, vector, storage, summary = build_service()
        result = await service.import_document("manual.pdf", category="motor", tags=["engine"])
        stored_docs = vector.add_vector.call_args_list
        image_meta = stored_docs[0].kwargs["metadata"]
        summary_meta = stored_docs[1].kwargs["metadata"]
        return {
            "result": result,
            "image_calls": image.embed.await_count,
            "summary_embed_calls": text.embed.await_count,
            "storage_calls": storage.ensure_public_url.call_count,
            "summary_calls": summary.summarize.await_count,
            "image_meta": image_meta,
            "summary_meta": summary_meta,
        }

    run_auto_cases([
        {
            "name": "PDF image import stores image vector and semantic summary vector",
            "input": "local extracted image",
            "expected": "image URL + dual records",
            "run": lambda: run_async(import_case()),
            "check": lambda x: x["result"]["image_count"] == 1
            and x["image_calls"] == 1
            and x["summary_embed_calls"] == 1
            and x["storage_calls"] == 1
            and x["summary_calls"] == 1
            and x["image_meta"]["document_id"]
            and x["image_meta"]["image_url"].startswith("/files/")
            and x["summary_meta"]["retrieval_route"] == "image_summary",
        },
    ])


def manual_test():
    auto_test()


if __name__ == "__main__":
    run_menu("knowledge multimodal import", auto_test, manual_test)
