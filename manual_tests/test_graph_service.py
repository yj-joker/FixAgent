from unittest.mock import MagicMock, patch

from test_runner import ask, print_json, require_real_dependency, run_auto_cases, run_menu


def build_service():
    with patch("services.graph_service.GraphDatabase.driver"):
        from services.graph_service import GraphService
        svc = GraphService()
    return svc


def auto_test():
    def search_components_case():
        svc = build_service()
        svc._execute_query = MagicMock(return_value=[{"id": "comp_001", "name": "轴承", "score": 0.91}])
        result = svc.search_components_by_embedding([0.1] * 1024, limit=5, min_score=0.5)
        return {"result": result, "params": svc._execute_query.call_args.args[1]}

    def search_faults_case():
        svc = build_service()
        svc._execute_query = MagicMock(return_value=[{"id": "fault_001", "name": "过热", "score": 0.88}])
        result = svc.search_faults_by_embedding([0.1] * 1024, limit=5, min_score=0.8)
        return {"result": result, "params": svc._execute_query.call_args.args[1]}

    def diagnosis_path_case():
        svc = build_service()
        svc.find_devices = MagicMock(return_value=[{"id": "dev_001", "name": "电动机"}])
        svc._branch_device_component_fault = MagicMock(return_value=([
            {
                "device_id": "dev_001",
                "device_name": "电动机",
                "component_id": "comp_001",
                "component_name": "轴承",
                "fault_id": "fault_001",
                "fault_name": "过热",
                "solution_title": "更换轴承",
                "verified": True,
                "has_history": True,
            }
        ], 1))
        result = svc.find_diagnosis_paths(
            keyword="电动机",
            component_ids=["comp_001"],
            fault_ids=["fault_001"],
            component_score_map={"comp_001": 0.77},
            fault_score_map={"fault_001": 0.88},
        )
        return result

    def empty_case():
        svc = build_service()
        return svc.find_diagnosis_paths(keyword=None, component_ids=[], fault_ids=[])

    run_auto_cases([
        {
            "name": "search_components_by_embedding 调用 component_embedding_index 并传入 minScore",
            "input": "1024维向量, limit=5, min_score=0.5",
            "expected": "返回 comp_001",
            "run": search_components_case,
            "check": lambda x: x["result"][0]["id"] == "comp_001" and x["params"]["minScore"] == 0.5,
        },
        {
            "name": "search_faults_by_embedding 调用 fault_embedding_index 并传入 minScore",
            "input": "1024维向量, limit=5, min_score=0.8",
            "expected": "返回 fault_001",
            "run": search_faults_case,
            "check": lambda x: x["result"][0]["id"] == "fault_001" and x["params"]["minScore"] == 0.8,
        },
        {
            "name": "find_diagnosis_paths 分支1返回路径字段和 score/path_text",
            "input": "设备+部件+故障",
            "expected": "records[0] 含 fault_score/component_score/path_text",
            "run": diagnosis_path_case,
            "check": lambda x: x["total"] == 1 and x["records"][0]["fault_score"] == 0.88 and "电动机" in x["records"][0]["path_text"],
        },
        {
            "name": "find_diagnosis_paths 空部件且空故障时返回空结果",
            "input": "component_ids=[], fault_ids=[]",
            "expected": {"records": [], "total": 0},
            "run": empty_case,
            "check": lambda x: x["records"] == [] and x["total"] == 0,
        },
    ])


def manual_test():
    from services.graph_service import get_graph_service

    require_real_dependency("neo4j", "pip install neo4j")
    svc = get_graph_service()
    if hasattr(svc.driver, "verify_connectivity"):
        svc.driver.verify_connectivity()
    action = ask("操作: devices/components/faults/paths", "devices")
    if action == "devices":
        keyword = ask("keyword", "电动机")
        print_json(svc.find_devices(keyword=keyword, limit=10))
    elif action == "paths":
        keyword = ask("keyword", "电动机")
        comp = ask("component_id", "comp_001")
        fault = ask("fault_id", "fault_001")
        print_json(svc.find_diagnosis_paths(keyword=keyword, component_ids=[comp], fault_ids=[fault]))
    else:
        print("此手动项请根据真实 Neo4j 数据调用对应方法。")


if __name__ == "__main__":
    run_menu("services/graph_service.py", auto_test, manual_test)
