from pydantic import ValidationError

from test_runner import ask, print_json, run_auto_cases, run_menu


def auto_test():
    from schemas.models import AgentMode, BaseResponse, GraphNode, GraphRelation, PaginationMeta, VectorSearchResult

    run_auto_cases([
        {
            "name": "AgentMode 所有枚举值可正常创建",
            "input": ["chat", "retrieval", "diagnosis", "guidance", "full"],
            "expected": "全部可创建",
            "run": lambda: [AgentMode(v).value for v in ["chat", "retrieval", "diagnosis", "guidance", "full"]],
            "check": lambda x: x == ["chat", "retrieval", "diagnosis", "guidance", "full"],
        },
        {
            "name": "BaseResponse 默认 success=True, code=200 且可序列化",
            "input": "BaseResponse()",
            "expected": {"success": True, "code": 200},
            "run": lambda: BaseResponse().model_dump(),
            "check": lambda x: x["success"] is True and x["code"] == 200,
        },
        {
            "name": "PaginationMeta.create 正常分页与 total=0 边界",
            "input": "total=100,page=2,page_size=10; total=0",
            "expected": {"normal_total_pages": 10, "zero_total_pages": 0},
            "run": lambda: {
                "normal_total_pages": PaginationMeta.create(100, 2, 10).total_pages,
                "zero_total_pages": PaginationMeta.create(0, 1, 10).total_pages,
            },
            "check": lambda x: x == {"normal_total_pages": 10, "zero_total_pages": 0},
        },
        {
            "name": "VectorSearchResult metadata 默认空 dict",
            "input": "id/score/content",
            "expected": {"metadata": {}},
            "run": lambda: VectorSearchResult(id="doc1", score=0.9, content="轴承过热").model_dump(),
            "check": lambda x: x["metadata"] == {},
        },
        {
            "name": "GraphNode / GraphRelation properties 默认空 dict，必填字段缺失会校验失败",
            "input": "GraphNode + 缺失字段 GraphRelation",
            "expected": "默认空 dict 且缺失字段报错",
            "run": lambda: {
                "node_props": GraphNode(id="n1", label="Device").properties,
                "relation_error": isinstance(_validation_error(lambda: GraphRelation(source_id="n1", target_id="n2")), ValidationError),
            },
            "check": lambda x: x["node_props"] == {} and x["relation_error"] is True,
        },
    ])


def _validation_error(fn):
    try:
        fn()
        return None
    except Exception as exc:
        return exc


def score_fields_case():
    from schemas.models import VectorSearchResult

    return VectorSearchResult(
        id="doc2",
        score=0.2,
        raw_score=0.2,
        raw_score_type="cosine_distance",
        relevance_score=0.8,
        retrieval_route="semantic",
        content="engine text",
    ).model_dump()


def manual_test():
    from schemas.models import AgentMode, BaseResponse, PaginationMeta

    print("可测试项: 1.AgentMode 2.BaseResponse 3.PaginationMeta")
    choice = ask("请选择", "1")
    if choice == "1":
        value = ask("输入枚举值", "chat")
        print_json({"result": AgentMode(value).value})
    elif choice == "2":
        print_json(BaseResponse().model_dump())
    else:
        total = int(ask("total", "100"))
        page = int(ask("page", "2"))
        page_size = int(ask("page_size", "10"))
        print_json(PaginationMeta.create(total, page, page_size).model_dump())


if __name__ == "__main__":
    run_menu("schemas/models.py", auto_test, manual_test)
