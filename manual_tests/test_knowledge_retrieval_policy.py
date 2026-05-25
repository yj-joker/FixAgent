from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import run_async, run_auto_cases, run_menu


def vector_doc(doc_id, score, chunk_type, route=None, text=None, **metadata):
    metadata = {"chunk_type": chunk_type, **metadata}
    if route:
        metadata["retrieval_route"] = route
    return {
        "doc_id": doc_id,
        "score": score,
        "raw_score": score,
        "raw_score_type": "cosine_distance",
        "relevance_score": round(1 - score, 6),
        "text": text or doc_id,
        "metadata": metadata,
    }


async def image_query_case():
    with patch("tools.knowledge_retrieval_tool.get_text_embedding") as get_emb, \
         patch("tools.knowledge_retrieval_tool.get_vector_service") as get_vec:
        from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool

        emb = MagicMock()
        emb.embed = AsyncMock(return_value=[0.1] * 1024)
        vector = MagicMock()
        vector.search.side_effect = [
            [vector_doc("img:1", 0.08, "image", route="image_vector", image_url="/files/img1.png")],
            [vector_doc("img:1", 0.12, "image", route="image_summary", image_url="/files/img1.png")],
        ]
        get_emb.return_value = emb
        get_vec.return_value = vector
        result = await KnowledgeRetrievalTool()._execute("发动机图片", top_k=3)
        return {
            "result": result,
            "filters": [call.kwargs["filter"] for call in vector.search.call_args_list],
        }


async def mixed_query_case():
    with patch("tools.knowledge_retrieval_tool.get_text_embedding") as get_emb, \
         patch("tools.knowledge_retrieval_tool.get_vector_service") as get_vec:
        from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool

        emb = MagicMock()
        emb.embed = AsyncMock(return_value=[0.2] * 1024)
        vector = MagicMock()
        vector.search.return_value = [
            vector_doc("txt:1", 0.04, "text"),
            vector_doc("txt:2", 0.05, "text"),
            vector_doc("tbl:1", 0.1, "table"),
            vector_doc("img:1", 0.15, "image", route="image_summary", image_url="/files/img1.png"),
        ]
        get_emb.return_value = emb
        get_vec.return_value = vector
        return await KnowledgeRetrievalTool()._execute("发动机", top_k=3)


def auto_test():
    run_auto_cases([
        {
            "name": "图片意图触发图片本体与 summary 双路召回并汇总置信度",
            "input": "发动机图片",
            "expected": "同一图片含两条 route 且 confidence=high",
            "run": lambda: run_async(image_query_case()),
            "check": lambda x: len(x["result"]) == 1
            and x["result"][0].metadata["retrieval_routes"] == ["image_summary", "image_vector"]
            and x["result"][0].metadata["retrieval_confidence"] == "high"
            and "@chunk_type:{image}" in x["filters"][0]
            and "@chunk_type:{image_summary}" in x["filters"][1],
        },
        {
            "name": "泛词查询保留文本表格图片类型多样性",
            "input": "发动机",
            "expected": "top3 覆盖 text/table/image",
            "run": lambda: run_async(mixed_query_case()),
            "check": lambda x: len(x) == 3 and {item.metadata["chunk_type"] for item in x} == {"text", "table", "image"},
        },
    ])


if __name__ == "__main__":
    run_menu("tools/knowledge_retrieval_tool.py policy", auto_test, lambda: None)
