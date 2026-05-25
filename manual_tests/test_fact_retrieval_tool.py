import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def auto_test():
    from tools.fact_retrieval_tool import FactRetrievalTool

    async def batch_search():
        with patch("services.vector_service.get_vector_service") as get_vec:
            service = MagicMock()
            service.search_by_text = AsyncMock(side_effect=[
                [{"doc_id": "fact1", "text": "电动机型号X200", "score": 0.91}],
                [{"doc_id": "fact2", "text": "故障码E-5013", "score": 0.86}],
            ])
            get_vec.return_value = service
            result = await FactRetrievalTool().run(queries=["电动机型号X200", "轴承故障码E-5013"], top_k=3)
            return result.model_dump()

    async def empty_queries():
        with patch("services.vector_service.get_vector_service") as get_vec:
            service = MagicMock()
            service.search_by_text = AsyncMock()
            get_vec.return_value = service
            result = await FactRetrievalTool().run(queries=[])
            return {"result": result.model_dump(), "calls": service.search_by_text.await_count}

    async def top_k_case():
        with patch("services.vector_service.get_vector_service") as get_vec:
            service = MagicMock()
            service.search_by_text = AsyncMock(return_value=[])
            get_vec.return_value = service
            await FactRetrievalTool().run(queries=["q"], top_k=5)
            return service.search_by_text.call_args.kwargs["top_k"]

    run_auto_cases([
        {
            "name": "正常批量检索返回每条 query 的相似事实",
            "input": "2 条 queries",
            "expected": "results 含两组 key",
            "run": lambda: run_async(batch_search()),
            "check": lambda x: x["success"] is True and len(x["data"]["results"]) == 2 and x["data"]["results"]["电动机型号X200"][0]["id"] == "fact1",
        },
        {
            "name": "空 queries 返回空 results 且不调用搜索",
            "input": "queries=[]",
            "expected": {"results": {}, "calls": 0},
            "run": lambda: run_async(empty_queries()),
            "check": lambda x: x["result"]["data"]["results"] == {} and x["calls"] == 0,
        },
        {
            "name": "top_k 参数传递给 vector_service.search_by_text",
            "input": "top_k=5",
            "expected": 5,
            "run": lambda: run_async(top_k_case()),
            "check": lambda x: x == 5,
        },
        {
            "name": "to_openai_schema() 注册 search_similar_facts",
            "input": "FactRetrievalTool().to_openai_schema()",
            "expected": "function.name=search_similar_facts",
            "run": lambda: FactRetrievalTool().to_openai_schema(),
            "check": lambda x: x["function"]["name"] == "search_similar_facts" and "queries" in x["function"]["parameters"]["properties"],
        },
    ])


def manual_test():
    from tools.fact_retrieval_tool import FactRetrievalTool

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    raw = ask("请输入 queries，多个用英文逗号分隔", "电动机型号X200,轴承故障码E-5013")
    queries = [q.strip() for q in raw.split(",") if q.strip()]
    top_k = int(ask("top_k", "3"))
    result = run_async(FactRetrievalTool().run(queries=queries, top_k=top_k))
    print_json(result.model_dump())


if __name__ == "__main__":
    run_menu("tools/fact_retrieval_tool.py", auto_test, manual_test)
