from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def auto_test():
    from tools.graph_query_tool import GraphQueryTool, GraphSearchDeviceTool

    async def full_flow():
        with patch("tools.graph_query_tool.get_text_embedding") as get_emb, patch("tools.graph_query_tool.get_graph_service") as get_graph:
            emb = MagicMock()
            emb.embed = AsyncMock(return_value=[0.1] * 1024)
            get_emb.return_value = emb
            graph = MagicMock()
            graph.search_components_by_embedding.return_value = [{"id": "comp_001", "score": 0.7}]
            graph.search_faults_by_embedding.return_value = [{"id": "fault_001", "score": 0.9}]
            graph.find_diagnosis_paths.return_value = {
                "records": [{
                    "device_id": "dev1",
                    "device_name": "电动机",
                    "component_id": "comp_001",
                    "component_name": "轴承",
                    "fault_id": "fault_001",
                    "fault_name": "过热",
                    "solution_title": "更换轴承",
                    "fault_score": 0.9,
                    "component_score": 0.7,
                    "path_text": "电动机 -> 轴承 -> 过热",
                }],
                "total": 1,
            }
            get_graph.return_value = graph
            result = await GraphQueryTool().run(keyword="电动机", component_description="轴承", fault_description="过热")
            return result.model_dump()

    async def no_match():
        with patch("tools.graph_query_tool.get_text_embedding") as get_emb, patch("tools.graph_query_tool.get_graph_service") as get_graph:
            emb = MagicMock()
            emb.embed = AsyncMock(return_value=[0.1] * 1024)
            get_emb.return_value = emb
            graph = MagicMock()
            graph.search_components_by_embedding.return_value = []
            graph.search_faults_by_embedding.return_value = []
            get_graph.return_value = graph
            return (await GraphQueryTool().run(component_description="未知", fault_description="未知")).model_dump()

    async def embedding_fail():
        with patch("tools.graph_query_tool.get_text_embedding") as get_emb, patch("tools.graph_query_tool.get_graph_service") as get_graph:
            emb = MagicMock()
            emb.embed = AsyncMock(side_effect=RuntimeError("API失败"))
            get_emb.return_value = emb
            get_graph.return_value = MagicMock()
            return (await GraphQueryTool().run(component_description="轴承")).model_dump()

    async def device_search():
        with patch("tools.graph_query_tool.get_graph_service") as get_graph:
            graph = MagicMock()
            graph.find_devices.return_value = [{"id": "dev1", "name": "1号泵", "code": "P001"}]
            get_graph.return_value = graph
            return (await GraphSearchDeviceTool().run(keyword="泵")).model_dump()

    run_auto_cases([
        {
            "name": "完整流程：部件/故障描述向量化后查询诊断路径",
            "input": "keyword+component_description+fault_description",
            "expected": "返回路径列表",
            "run": lambda: run_async(full_flow()),
            "check": lambda x: x["success"] is True and x["data"][0]["component_id"] == "comp_001",
        },
        {
            "name": "无向量匹配结果时返回 NO_MATCH",
            "input": "向量检索空列表",
            "expected": {"error.code": "NO_MATCH"},
            "run": lambda: run_async(no_match()),
            "check": lambda x: x["success"] is False and x["error"]["code"] == "NO_MATCH",
        },
        {
            "name": "embedding 失败返回 EMBEDDING_FAILED",
            "input": "embed 抛异常",
            "expected": {"error.code": "EMBEDDING_FAILED"},
            "run": lambda: run_async(embedding_fail()),
            "check": lambda x: x["success"] is False and x["error"]["code"] == "EMBEDDING_FAILED",
        },
        {
            "name": "graph_search_devices 返回设备列表",
            "input": "keyword='泵'",
            "expected": "返回 dev1",
            "run": lambda: run_async(device_search()),
            "check": lambda x: x["success"] is True and x["data"][0]["id"] == "dev1",
        },
    ])


def manual_test():
    from tools.graph_query_tool import GraphQueryTool

    require_real_dependency("neo4j", "pip install neo4j")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    keyword = ask("keyword（设备关键字，可空）", "电动机")
    component = ask("component_description（可空）", "轴承")
    fault = ask("fault_description（可空）", "过热")
    result = run_async(GraphQueryTool().run(
        keyword=keyword or None,
        component_description=component or None,
        fault_description=fault or None,
    ))
    print_json(result.model_dump())


if __name__ == "__main__":
    run_menu("tools/graph_query_tool.py", auto_test, manual_test)
