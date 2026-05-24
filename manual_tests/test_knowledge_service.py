from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def vec(v=0.1):
    return [float(v)] * 1024


def sample_parse_result(text_chunks=None, tables=None, images=None):
    return {
        "file_name": "manual.pdf",
        "total_pages": 2,
        "sections": [{
            "section_title": "第一章",
            "page_range": "1-2",
            "text_chunks": text_chunks if text_chunks is not None else ["这是一个足够长的文本块，用于入库测试。"],
            "tables": tables if tables is not None else [{"page": 1, "caption": "参数表", "rows": [["名称", "值"], ["电压", "380V"]]}],
            "images": images if images is not None else [{"page": 2, "image_name": "img1.png", "local_path": "D:/img1.png", "caption": "轴承结构图"}],
        }],
        "extraction_summary": {"text_chunks_total": 1, "tables_total": 1, "images_total": 1},
    }


def build_service(parse_result):
    with patch("services.knowledge_service.get_document_parser") as get_parser, \
         patch("services.knowledge_service.get_text_embedding") as get_emb, \
         patch("services.knowledge_service.get_image_embedding") as get_img_emb, \
         patch("services.knowledge_service.get_file_storage") as get_file_storage, \
         patch("services.knowledge_service.get_image_summary_service") as get_image_summary, \
         patch("services.knowledge_service.get_vector_service") as get_vec:
        from services.knowledge_service import KnowledgeService
        parser = MagicMock()
        parser._execute = AsyncMock(return_value=parse_result)
        emb = MagicMock()
        emb.embed_batch = AsyncMock(side_effect=lambda batch: [vec(i + 1) for i, _ in enumerate(batch)])
        emb.embed = AsyncMock(return_value=vec(9))
        img_emb = MagicMock()
        img_emb.embed = AsyncMock(return_value=vec(7))
        file_storage = MagicMock()
        file_storage.ensure_document_url.side_effect = lambda file_url: file_url
        file_storage.ensure_public_url.side_effect = lambda image: image.get("image_url", "")
        image_summary = MagicMock()
        image_summary.summarize = AsyncMock(return_value={
            "image_summary": "图片摘要",
            "image_title": "图片标题",
            "summary_source": "test",
        })
        vector = MagicMock()
        vector.add_vector_batch.side_effect = lambda docs: len(docs)
        vector.add_vector.return_value = True
        vector.put_document_manifest.return_value = True
        get_parser.return_value = parser
        get_emb.return_value = emb
        get_img_emb.return_value = img_emb
        get_file_storage.return_value = file_storage
        get_image_summary.return_value = image_summary
        get_vec.return_value = vector
        svc = KnowledgeService()
    return svc, parser, emb, img_emb, file_storage, vector


def auto_test():
    import services.knowledge_service as module

    async def full_pipeline():
        svc, parser, emb, img_emb, file_storage, vector = build_service(sample_parse_result())
        file_storage.ensure_public_url.side_effect = None
        file_storage.ensure_public_url.return_value = "http://localhost:9000/weixiu-public-tupian/img1.png"
        result = await svc.import_document("manual.pdf", "pdf", category="motor", tags=["bearing"])
        image_metadata = next(
            call.kwargs["metadata"]
            for call in vector.add_vector.call_args_list
            if call.kwargs["metadata"]["chunk_type"] == "image"
        )
        summary_metadata = next(
            call.kwargs["metadata"]
            for call in vector.add_vector.call_args_list
            if call.kwargs["metadata"]["chunk_type"] == "image_summary"
        )
        return {
            "result": result,
            "parse_calls": parser._execute.await_count,
            "batch_calls": emb.embed_batch.await_count,
            "single_embed_calls": emb.embed.await_count,
            "image_embed_calls": img_emb.embed.await_count,
            "image_embed_input": img_emb.embed.await_args.args[0],
            "image_metadata": image_metadata,
            "summary_metadata": summary_metadata,
            "batch_docs": vector.add_vector_batch.call_args.args[0],
            "add_vector_calls": vector.add_vector.call_count,
            "ready_manifest": vector.put_document_manifest.call_args_list[-1].args[1],
        }

    async def batching_case():
        chunks = [f"这是第{i}个足够长的文本块，用于测试分批行为。" for i in range(21)]
        svc, _, emb, _, _, vector = build_service(sample_parse_result(text_chunks=chunks, tables=[], images=[]))
        result = await svc.import_document("manual.pdf")
        return {"text_count": result["text_count"], "batch_calls": emb.embed_batch.await_count, "add_batch_calls": vector.add_vector_batch.call_count}

    async def empty_doc():
        svc, _, _, _, _, vector = build_service(sample_parse_result(text_chunks=[], tables=[], images=[]))
        result = await svc.import_document("empty.pdf")
        return {"text_count": result["text_count"], "image_count": result["image_count"], "table_count": result["table_count"], "writes": vector.add_vector.call_count}

    async def image_url_case():
        images = [{
            "page": 2,
            "image_name": "img1.png",
            "local_path": "",
            "image_url": "https://cdn.example.com/img1.png",
            "caption": "轴承结构图",
        }]
        svc, _, emb, img_emb, _, vector = build_service(sample_parse_result(text_chunks=[], tables=[], images=images))
        result = await svc.import_document("manual.pdf")
        image_call = vector.add_vector.call_args_list[0]
        summary_call = vector.add_vector.call_args_list[-1]
        return {
            "image_count": result["image_count"],
            "text_embed_calls": emb.embed.await_count,
            "image_embed_calls": img_emb.embed.await_count,
            "image_embed_url": img_emb.embed.await_args.args[0],
            "vector_first": image_call.kwargs["vector"][0],
            "metadata": image_call.kwargs["metadata"],
            "summary_metadata": summary_call.kwargs["metadata"],
        }

    async def failed_write_marks_manifest_failed():
        svc, _, _, _, _, vector = build_service(sample_parse_result(tables=[], images=[]))
        vector.add_vector_batch.side_effect = None
        vector.add_vector_batch.return_value = 0
        try:
            await svc.import_document("manual.pdf", document_id="manual-001")
        except RuntimeError:
            pass
        return [call.args[1]["status"] for call in vector.put_document_manifest.call_args_list]

    def table_to_text():
        return module.KnowledgeService._table_to_text({"caption": "参数表", "rows": [["名称", "值"], ["电压", "380V"]]})

    def singleton():
        with patch("services.knowledge_service.get_document_parser"), patch("services.knowledge_service.get_text_embedding"), patch("services.knowledge_service.get_image_embedding"), patch("services.knowledge_service.get_vector_service"):
            module._knowledge_service = None
            return module.get_knowledge_service() is module.get_knowledge_service()

    run_auto_cases([
        {
            "name": "完整管线：解析、向量化、文本/表格/图片入库并返回统计",
            "input": "1文本 + 1表格 + 1图片",
            "expected": "text_count=1,image_count=1,table_count=1",
            "run": lambda: run_async(full_pipeline()),
            "check": lambda x: x["result"]["text_count"] == 1 and x["result"]["table_count"] == 1 and x["result"]["image_count"] == 1 and x["result"]["image_summary_count"] == 1 and x["add_vector_calls"] == 3 and x["image_embed_calls"] == 1 and x["image_embed_input"] == "D:/img1.png" and x["image_metadata"]["embedding_source"] == "local_image" and x["image_metadata"]["image_url"] == "http://localhost:9000/weixiu-public-tupian/img1.png" and x["summary_metadata"]["retrieval_route"] == "image_summary" and x["ready_manifest"]["image_summary_count"] == 1,
        },
        {
            "name": "文本块按 BATCH_SIZE=20 分批 embed_batch 入库",
            "input": "21 个有效文本块",
            "expected": "embed_batch 调用 2 次",
            "run": lambda: run_async(batching_case()),
            "check": lambda x: x["text_count"] == 21 and x["batch_calls"] == 2 and x["add_batch_calls"] == 2,
        },
        {
            "name": "_table_to_text 将表格转 markdown 风格文本",
            "input": "caption + rows",
            "expected": "包含表格标题与 380V",
            "run": table_to_text,
            "check": lambda x: "表格：参数表" in x and "电压 | 380V" in x,
        },
        {
            "name": "图片含 image_url 时使用 ImageEmbedding 入库并记录图片向量来源",
            "input": "1 张带公网 URL 的 PDF 图片",
            "expected": {"embedding_source": "image", "image_embed_calls": 1},
            "run": lambda: run_async(image_url_case()),
            "check": lambda x: x["image_count"] == 1 and x["text_embed_calls"] == 1 and x["image_embed_calls"] == 1 and x["image_embed_url"] == "https://cdn.example.com/img1.png" and x["vector_first"] == 7.0 and x["metadata"]["embedding_source"] == "image_url" and x["metadata"]["image_url"] == "https://cdn.example.com/img1.png" and x["summary_metadata"]["retrieval_route"] == "image_summary",
        },
        {
            "name": "空文档无内容时统计为 0 且不写入",
            "input": "sections 内无 text/table/image",
            "expected": {"text_count": 0, "writes": 0},
            "run": lambda: run_async(empty_doc()),
            "check": lambda x: x == {"text_count": 0, "image_count": 0, "table_count": 0, "writes": 0},
        },
        {
            "name": "get_knowledge_service() 返回同一实例",
            "input": "连续调用",
            "expected": True,
            "run": singleton,
            "check": lambda x: x is True,
        },
        {
            "name": "failed vector writes persist failed document status",
            "input": "text vector batch write fails",
            "expected": "failed manifest status",
            "run": lambda: run_async(failed_write_marks_manifest_failed()),
            "check": lambda x: x[-1] == "failed" and "indexing" in x,
        },
    ])


def manual_test():
    from services.knowledge_service import get_knowledge_service

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    file_url = ask("请输入 PDF 路径或 URL", "manual.pdf")
    category = ask("category（可空）", "")
    tags_raw = ask("tags，英文逗号分隔（可空）", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or None
    result = run_async(get_knowledge_service().import_document(file_url=file_url, file_type="pdf", category=category or None, tags=tags))
    print_json(result)


if __name__ == "__main__":
    run_menu("services/knowledge_service.py", auto_test, manual_test)
