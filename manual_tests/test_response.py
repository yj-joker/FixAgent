from datetime import datetime

from test_runner import print_json, run_auto_cases, run_menu


def auto_test():
    from schemas.models import VectorSearchResult
    from schemas.response import (
        ChatResponse,
        KnowledgeImportResponse,
        KnowledgeSearchResponse,
        MemoryConsolidateResponse,
        MemorySummary,
    )

    run_auto_cases([
        {
            "name": "ChatResponse verification=None 时可通过 exclude_none 不输出",
            "input": "verification=None",
            "expected": "verification 不在序列化结果中",
            "run": lambda: ChatResponse(session_id="s1", message="ok", verification=None).model_dump(exclude_none=True),
            "check": lambda x: "verification" not in x,
        },
        {
            "name": "KnowledgeSearchResponse data 为 VectorSearchResult 列表时正确序列化",
            "input": "1条向量检索结果",
            "expected": "data[0].id=doc1",
            "run": lambda: KnowledgeSearchResponse(
                data=[VectorSearchResult(id="doc1", score=0.9, content="轴承过热")],
                total=1,
                query_time_ms=12,
            ).model_dump(),
            "check": lambda x: x["data"][0]["id"] == "doc1" and x["total"] == 1,
        },
        {
            "name": "MemoryConsolidateResponse camelCase 别名正确映射",
            "input": "model_dump(by_alias=True)",
            "expected": "sessionId/originalCount/consolidatedAt/summary.newFacts 等存在",
            "run": lambda: MemoryConsolidateResponse(
                session_id="s1",
                summary=MemorySummary(brief_summary="摘要"),
                original_count=2,
                consolidated_at=datetime.now().isoformat(),
            ).model_dump(by_alias=True),
            "check": lambda x: "sessionId" in x and "originalCount" in x and "briefSummary" in x["summary"],
        },
        {
            "name": "KnowledgeImportResponse sections 和 extraction_summary 正确序列化",
            "input": "sections=[dict], extraction_summary=dict",
            "expected": "dict/list 保持结构",
            "run": lambda: KnowledgeImportResponse(
                file_name="manual.pdf",
                total_pages=1,
                text_count=1,
                image_count=0,
                table_count=1,
                sections=[{"section_title": "第一章"}],
                extraction_summary={"tables_total": 1},
                process_time_ms=10,
            ).model_dump(),
            "check": lambda x: x["sections"][0]["section_title"] == "第一章" and x["extraction_summary"]["tables_total"] == 1,
        },
    ])


def manual_test():
    from schemas.response import ChatResponse

    resp = ChatResponse(session_id="sess_manual", message="手动响应", tools_used=["knowledge_retrieval"])
    print_json(resp.model_dump(exclude_none=True))


if __name__ == "__main__":
    run_menu("schemas/response.py", auto_test, manual_test)
