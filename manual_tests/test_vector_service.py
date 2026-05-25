from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_real_dependency, run_async, run_auto_cases, run_menu


def vec():
    return [0.1] * 1024


def auto_test():
    import services.vector_service as module

    def filter_case():
        return {
            "category": module.build_redis_filter(category="motor"),
            "tags": module.build_redis_filter(tags=["bearing", "overheat"]),
            "both": module.build_redis_filter(category="motor", tags=["bearing"]),
            "document_id": module.build_redis_filter(document_id="motorcycle-engine-manual-real-pdf"),
        }

    def add_vector_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            client.execute_command.return_value = ["num_docs", "0"]
            svc = module.VectorService()
            ok = svc.add_vector("doc1", "中文内容", vec(), {"来源": "手册"}, category="motor", tags=["bearing"])
            mapping = client.hset.call_args.kwargs["mapping"]
            return {"ok": ok, "key": client.hset.call_args.args[0], "metadata": mapping["metadata"], "tags": mapping["tags"]}

    def search_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            def execute_command(*args):
                if args[0] == "FT.INFO":
                    return ["num_docs", "1"]
                if args[0] == "FT.SEARCH":
                    return [1, b"doc:doc1", [b"id", b"doc1", b"text", "轴承过热".encode("utf-8"), b"score", b"0.12", b"metadata", '{"category":"motor"}'.encode("utf-8")]]
                return "OK"
            client.execute_command.side_effect = execute_command
            svc = module.VectorService()
            result = svc.search(vec(), top_k=1)
            return result

    async def search_by_text_case():
        with patch("services.vector_service.redis.Redis") as redis_cls, patch("embeddings.text_embedding.get_text_embedding") as get_emb:
            client = MagicMock()
            redis_cls.return_value = client
            def execute_command(*args):
                if args[0] == "FT.INFO":
                    return ["num_docs", "1"]
                if args[0] == "FT.SEARCH":
                    return [0]
                return "OK"
            client.execute_command.side_effect = execute_command
            emb = MagicMock()
            emb.embed = AsyncMock(return_value=vec())
            get_emb.return_value = emb
            svc = module.VectorService()
            result = await svc.search_by_text("电动机故障", top_k=1)
            return {"result": result, "embed_calls": emb.embed.await_count}

    def delete_count_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            def execute_command(*args):
                if args[0] == "FT.INFO":
                    return ["num_docs", "7"]
                return "OK"
            client.execute_command.side_effect = execute_command
            client.delete.return_value = 1
            svc = module.VectorService()
            return {"deleted": svc.delete("doc1"), "count": svc.count()}

    def bytes_count_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            client.execute_command.return_value = [b"num_docs", b"7"]
            svc = module.VectorService()
            return svc.count()

    def delete_by_document_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            searches = []

            def execute_command(*args):
                if args[0] == "FT.INFO":
                    return ["num_docs", "2"]
                if args[0] == "FT.SEARCH":
                    searches.append(args)
                    if len(searches) == 1:
                        return [2, b"doc:a", [b"id", b"a"], b"doc:b", [b"id", b"b"]]
                    return [0]
                return "OK"

            client.execute_command.side_effect = execute_command
            client.delete.return_value = 1
            svc = module.VectorService()
            deleted = svc.delete_by_document("motorcycle-engine-manual-real-pdf")
            return {
                "deleted": deleted,
                "search_count": len(searches),
                "first_search": searches[0],
                "deleted_keys": [call.args[0] for call in client.delete.call_args_list],
            }

    def index_prefix_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            client.execute_command.side_effect = [module.redis.exceptions.ResponseError("missing"), "OK"]
            module.VectorService()
            return client.execute_command.call_args_list[1].args

    def storage_stats_and_cache_cleanup_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client
            client.execute_command.return_value = ["num_docs", "3"]
            client.scan_iter.side_effect = lambda match, count=1000: iter({
                "doc:*": [b"doc:a", b"doc:b", b"doc:c"],
                "document:*": [b"document:manual"],
                "cache:emb:text:*": [b"cache:emb:text:a", b"cache:emb:text:b"],
                "cache:emb:image:*": [b"cache:emb:image:a"],
            }.get(match, []))
            client.delete.return_value = 1
            svc = module.VectorService()
            stats = svc.get_storage_stats()
            cleared = svc.clear_embedding_cache()
            return {
                "stats": stats,
                "cleared": cleared,
                "deleted": [call.args[0] for call in client.delete.call_args_list],
            }

    run_auto_cases([
        {
            "name": "build_redis_filter 生成 category/tags 过滤表达式",
            "input": "category=motor,tags=bearing|overheat",
            "expected": "包含 @category 和 @tags",
            "run": filter_case,
            "check": lambda x: "@category:{motor}" in x["category"]
            and "@tags:{bearing|overheat}" in x["tags"]
            and "@category:{motor}" in x["both"]
            and r"@document_id:{motorcycle\-engine\-manual\-real\-pdf}" in x["document_id"],
        },
        {
            "name": "add_vector 写入 doc:{doc_id}，中文 metadata 不转义",
            "input": "doc1 + 中文 metadata",
            "expected": "hset key=doc:doc1 且 metadata 含中文",
            "run": add_vector_case,
            "check": lambda x: x["ok"] is True and x["key"] == "doc:doc1" and "来源" in x["metadata"] and x["tags"] == "bearing",
        },
        {
            "name": "search 解析 Redis FT.SEARCH 返回结果",
            "input": "Mock FT.SEARCH 返回 1 条",
            "expected": "doc_id/text/score/metadata",
            "run": search_case,
            "check": lambda x: len(x) == 1 and x[0]["doc_id"] == "doc1" and x[0]["metadata"]["category"] == "motor",
        },
        {
            "name": "search_by_text 先向量化再 search",
            "input": "电动机故障",
            "expected": "embed 调用 1 次",
            "run": lambda: run_async(search_by_text_case()),
            "check": lambda x: x["embed_calls"] == 1,
        },
        {
            "name": "delete 和 count 返回正确结果",
            "input": "delete doc1; FT.INFO num_docs=7",
            "expected": {"deleted": True, "count": 7},
            "run": delete_count_case,
            "check": lambda x: x == {"deleted": True, "count": 7},
        },
        {
            "name": "count supports FT.INFO bytes fields",
            "input": "FT.INFO returns bytes num_docs",
            "expected": 7,
            "run": bytes_count_case,
            "check": lambda x: x == 7,
        },
        {
            "name": "delete_by_document uses escaped paged RediSearch",
            "input": "document_id with hyphens",
            "expected": "LIMIT <= 10000 and all keys deleted",
            "run": delete_by_document_case,
            "check": lambda x: x["deleted"] == 2
            and x["search_count"] == 2
            and x["deleted_keys"] == [b"doc:a", b"doc:b"]
            and "10000" in x["first_search"]
            and r"motorcycle\-engine\-manual\-real\-pdf" in x["first_search"][2],
        },
        {
            "name": "vector index is scoped to long-lived doc keys only",
            "input": "new FT.CREATE command",
            "expected": "PREFIX 1 doc:",
            "run": index_prefix_case,
            "check": lambda x: "PREFIX" in x and "doc:" in x,
        },
        {
            "name": "storage statistics distinguish vectors, manifests and disposable cache",
            "input": "three key categories",
            "expected": "cleanup deletes cache keys only",
            "run": storage_stats_and_cache_cleanup_case,
            "check": lambda x: x["stats"]["vector_records"] == 3
            and x["stats"]["document_manifests"] == 1
            and x["stats"]["cache"]["text"] == 2
            and x["stats"]["cache"]["image"] == 1
            and x["cleared"]["total_deleted"] == 3
            and x["deleted"] == [b"cache:emb:text:a", b"cache:emb:text:b", b"cache:emb:image:a"],
        },
    ])


def manual_test():
    from services.vector_service import get_vector_service

    require_real_dependency("redis", "pip install redis")
    action = ask("操作: add/search/search_text/delete/count", "count")
    svc = get_vector_service()
    svc.redis.ping()
    if action == "count":
        print_json({"count": svc.count()})
    elif action == "delete":
        doc_id = ask("doc_id", "doc1")
        print_json({"deleted": svc.delete(doc_id)})
    elif action == "search":
        top_k = int(ask("top_k", "5"))
        print_json(svc.search([0.1] * 1024, top_k=top_k))
    elif action == "search_text":
        query = ask("查询文本", "轴承怎么保养")
        top_k = int(ask("top_k", "5"))
        print_json(run_async(svc.search_by_text(query, top_k=top_k)))
    else:
        from embeddings.text_embedding import get_text_embedding
        doc_id = ask("doc_id", "manual_doc")
        text = ask("text", "手动写入测试")
        embedding = get_text_embedding()
        vector = run_async(embedding.embed(text))
        print_json({"added": svc.add_vector(doc_id, text, vector, {"source": "manual"})})


if __name__ == "__main__":
    run_menu("services/vector_service.py", auto_test, manual_test)
