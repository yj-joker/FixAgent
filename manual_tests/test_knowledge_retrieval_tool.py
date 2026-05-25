from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def vec(v):
    return [float(v)] * 1024


def auto_test():
    from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool

    def filter_case():
        tool = KnowledgeRetrievalTool()
        return {
            "category": tool._build_filter(category="motor"),
            "tags": tool._build_filter(tags=["bearing", "overheat"]),
            "hyphen_tags": tool._build_filter(tags=["maintenance-manual"]),
            "both": tool._build_filter(category="motor", tags=["bearing"]),
        }

    async def text_search():
        with patch("tools.knowledge_retrieval_tool.get_text_embedding") as get_emb, patch("tools.knowledge_retrieval_tool.get_vector_service") as get_vec:
            emb = MagicMock()
            emb.embed = AsyncMock(return_value=vec(1))
            get_emb.return_value = emb
            service = MagicMock()
            service.search.return_value = [{"doc_id": "doc1", "score": 0.9, "text": "轴承过热", "metadata": {"category": "motor"}}]
            get_vec.return_value = service
            result = await KnowledgeRetrievalTool().run(query="轴承", top_k=1)
            return result.model_dump()

    async def multimodal_search():
        with patch("tools.knowledge_retrieval_tool.get_multimodal_embedding") as get_mm, patch("tools.knowledge_retrieval_tool.get_vector_service") as get_vec:
            mm = MagicMock()
            mm.embed = AsyncMock(return_value={"text_vector": vec(1), "image_vectors": [vec(3)]})
            get_mm.return_value = mm
            service = MagicMock()
            service.search.return_value = [{"doc_id": "doc2", "score": 0.8, "text": "图文结果", "metadata": {"match_type": "multimodal"}}]
            get_vec.return_value = service
            result = await KnowledgeRetrievalTool().run(query="轴承", image_urls=["img1"])
            called_vector = service.search.call_args.args[0]
            return {"result": result.model_dump(), "fused_first": called_vector[0]}

    async def embedding_fail():
        with patch("tools.knowledge_retrieval_tool.get_text_embedding") as get_emb:
            emb = MagicMock()
            emb.embed = AsyncMock(side_effect=RuntimeError("API不可用"))
            get_emb.return_value = emb
            return (await KnowledgeRetrievalTool().run(query="轴承")).model_dump()

    run_auto_cases([
        {
            "name": "_build_filter 支持 category/tags/组合过滤",
            "input": "category=motor,tags=bearing|overheat",
            "expected": "生成 RediSearch filter",
            "run": filter_case,
            "check": lambda x: x["category"] == "@category:{motor}"
            and x["tags"] == "@tags:{bearing|overheat}"
            and x["hyphen_tags"] == r"@tags:{maintenance\-manual}"
            and x["both"] == "(@category:{motor} @tags:{bearing})",
        },
        {
            "name": "纯文本检索：embed → vector_service.search → VectorSearchResult",
            "input": "query='轴承'",
            "expected": "返回 doc1",
            "run": lambda: run_async(text_search()),
            "check": lambda x: x["success"] is True and x["data"][0]["id"] == "doc1",
        },
        {
            "name": "图文混合检索：文本向量和图片向量逐维平均",
            "input": "text_vector=1, image_vector=3",
            "expected": "融合向量首维=2.0",
            "run": lambda: run_async(multimodal_search()),
            "check": lambda x: x["fused_first"] == 2.0 and x["result"]["success"] is True,
        },
        {
            "name": "向量化失败返回 EMBEDDING_FAILED",
            "input": "embed 抛 RuntimeError",
            "expected": {"error.code": "EMBEDDING_FAILED"},
            "run": lambda: run_async(embedding_fail()),
            "check": lambda x: x["success"] is False and x["error"]["code"] == "EMBEDDING_FAILED",
        },
    ])


def manual_test():
    from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    query = ask("query", "轴承过热")
    top_k = int(ask("top_k", "5"))
    result = run_async(KnowledgeRetrievalTool().run(query=query, top_k=top_k))
    print_json(result.model_dump())


if __name__ == "__main__":
    run_menu("tools/knowledge_retrieval_tool.py", auto_test, manual_test)
