import json
from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, run_async, run_auto_cases, run_menu


def test_chat_request_allows_image_only_and_rejects_empty_request():
    import pytest
    from pydantic import ValidationError
    from schemas.request import ChatRequest

    request = ChatRequest(session_id="s1", message="", images=["data:image/png;base64,abc"])

    assert request.message == ""
    assert request.images == ["data:image/png;base64,abc"]

    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="", images=[])


def test_chat_image_only_uses_default_prompt_and_enhanced_query():
    import api.main as main
    from agents.base_agent import AgentOutput
    from schemas.request import ChatRequest

    captured = {}

    image_summary_service = MagicMock()
    image_summary_service.understand_user_image = AsyncMock(
        return_value={
            "image_title": "火花塞结构图",
            "image_summary": "图片中是摩托车发动机火花塞，标注 a 表示电极间隙。",
            "keywords": ["火花塞", "电极间隙", "摩托车发动机"],
            "summary_source": "multimodal_llm",
        }
    )
    image_summary_service.summarize = AsyncMock(return_value={})

    fix_agent = MagicMock()

    async def run_with_react(input_data):
        captured["input_data"] = input_data
        return AgentOutput(
            agent_name="fix_agent",
            message="从图片看，这是火花塞。",
            tools_used=["knowledge_retrieval"],
            metadata={"status": "ok"},
            latency_ms=10,
        )

    fix_agent.run_with_react = run_with_react
    review_agent = MagicMock()
    review_agent.review = AsyncMock(
        return_value=AgentOutput(
            agent_name="review_agent",
            message="从图片看，这是火花塞。",
            tools_used=["knowledge_retrieval"],
            metadata={"verification_has_issues": False, "verification": {}},
            latency_ms=20,
        )
    )

    router = MagicMock()
    router.classify = AsyncMock(return_value=MagicMock(
        requires_image_understanding=True,
        intent="visual_identification",
        model_dump=lambda: {
            "intent": "visual_identification",
            "requires_image_understanding": True,
            "requires_manual_evidence": False,
            "requires_safety_notice": False,
            "allowed_tools": ["knowledge_retrieval", "graph_search_java"],
        },
    ))

    with patch.object(main, "get_intent_router", return_value=router), patch.object(
        main, "get_image_summary_service", return_value=image_summary_service
    ), patch.object(
        main, "get_fix_agent", return_value=fix_agent
    ), patch.object(main, "get_review_agent", return_value=review_agent):
        response = run_async(main.chat(ChatRequest(session_id="s1", message="", images=["img://spark-plug"])))

    input_data = captured["input_data"]
    assert "请识别图片中的设备或部件" in input_data.user_message
    assert input_data.images == ["img://spark-plug"]
    assert "火花塞" in input_data.context["enhanced_retrieval_query"]
    assert input_data.context["intent_decision"]["intent"] == "visual_identification"
    assert input_data.context["image_understanding"]["summaries"][0]["image_title"] == "火花塞结构图"
    assert response.message == "从图片看，这是火花塞。"


def test_search_facts_empty_query_returns_empty_result_without_embedding_call():
    import api.main as main

    response = run_async(main.search_facts(query="", top_k=5))

    assert response["facts"] == []
    assert response["query_time_ms"] == 0


def auto_test():
    import api.main as main
    from agents.base_agent import AgentOutput
    from schemas.request import (
        ChatRequest,
        KnowledgeImportRequest,
        KnowledgeSearchRequest,
        MemoryConsolidateRequest,
        MemoryMessage,
        TemporaryPlanGenerateRequest,
    )
    from tools.base_tool import ToolResult

    async def chat_success():
        fix_agent = MagicMock()
        fix_agent.run_with_react = AsyncMock(
            return_value=AgentOutput(
                agent_name="fix_agent",
                message="check bearing lubrication",
                tools_used=["knowledge_retrieval"],
                metadata={"status": "ok"},
                latency_ms=10,
            )
        )
        review_agent = MagicMock()
        review_agent.review = AsyncMock(
            return_value=AgentOutput(
                agent_name="review_agent",
                message="verified answer",
                tools_used=["knowledge_retrieval"],
                metadata={"verification_has_issues": False, "verification": {"grounding": {}}},
                latency_ms=20,
            )
        )
        with patch.object(main, "get_fix_agent", return_value=fix_agent), patch.object(
            main, "get_review_agent", return_value=review_agent
        ):
            response = await main.chat(ChatRequest(session_id="s1", message="bearing hot", stream=False))
            return response.model_dump()

    async def chat_fix_error_skips_review():
        fix_agent = MagicMock()
        fix_agent.run_with_react = AsyncMock(
            return_value=AgentOutput(
                agent_name="fix_agent",
                message="temporary unavailable",
                tools_used=[],
                metadata={"status": "error", "error_detail": "boom"},
                latency_ms=5,
            )
        )
        review_agent = MagicMock()
        review_agent.review = AsyncMock()
        with patch.object(main, "get_fix_agent", return_value=fix_agent), patch.object(
            main, "get_review_agent", return_value=review_agent
        ):
            response = await main.chat(ChatRequest(session_id="s1", message="hello", stream=False))
            if hasattr(response, "model_dump"):
                content = response.model_dump()
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = response.body
            if isinstance(content, bytes):
                content = json.loads(content.decode("utf-8"))
            return {
                "status_code": getattr(response, "status_code", 200),
                "response": content,
                "review_calls": review_agent.review.call_count,
            }

    async def chat_stream_success():
        async def events(_input):
            yield {"event": "status", "data": {"stage": "start"}}
            yield {"event": "tool", "data": {"tool": "knowledge"}}
            yield {"event": "token", "data": {"content": "O"}}
            yield {"event": "token", "data": {"content": "K"}}
            yield {"event": "done", "data": {"tools_used": ["knowledge"], "latency_ms": 12}}

        fix_agent = MagicMock()
        fix_agent.run_with_react_stream = events
        review_agent = MagicMock()
        review_agent.review = AsyncMock(
            return_value=AgentOutput(
                agent_name="review_agent",
                message="OK",
                tools_used=["knowledge"],
                metadata={"verification_has_issues": False, "verification": {}},
                latency_ms=18,
            )
        )
        review_agent.get_inline_markers.return_value = []
        with patch.object(main, "get_fix_agent", return_value=fix_agent), patch.object(
            main, "get_review_agent", return_value=review_agent
        ):
            response = await main.chat_stream(ChatRequest(session_id="s1", message="stream", stream=True))
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return "".join(chunks)

    async def knowledge_import_success():
        svc = MagicMock()
        svc.import_document = AsyncMock(
            return_value={
                "file_name": "manual.pdf",
                "total_pages": 2,
                "text_count": 3,
                "image_count": 1,
                "table_count": 1,
                "sections": [{"title": "A"}],
                "extraction_summary": {"ok": True},
                "process_time_ms": 99,
            }
        )
        with patch("services.knowledge_service.get_knowledge_service", return_value=svc):
            response = await main.knowledge_import(
                KnowledgeImportRequest(file_url="manual.pdf", file_type="pdf", category="motor", tags=["a"])
            )
            return response.model_dump()

    async def knowledge_search_success():
        tool = MagicMock()
        tool.run = AsyncMock(
            return_value=ToolResult(
                tool_name="knowledge_retrieval",
                success=True,
                data=[
                    {
                        "id": "doc1",
                        "score": 0.9,
                        "content": "bearing guide",
                        "metadata": {"category": "motor"},
                    }
                ],
            )
        )
        with patch.object(main, "get_knowledge_retrieval_tool", return_value=tool):
            response = await main.knowledge_search(KnowledgeSearchRequest(query="bearing", top_k=1))
            return response.model_dump()

    async def knowledge_storage_stats_success():
        svc = MagicMock()
        svc.get_storage_stats.return_value = {
            "vector_records": 3,
            "document_manifests": 1,
            "cache": {"text": 2, "image": 1, "total": 3},
        }
        with patch.object(main, "get_vector_service", return_value=svc):
            response = await main.knowledge_storage_stats()
            return response.model_dump()

    async def temporary_plan_generate_success():
        svc = MagicMock()
        from schemas.response import TemporaryPlanDraftResponse

        svc.generate = AsyncMock(
            return_value=TemporaryPlanDraftResponse(
                request_id="req-1",
                status="PENDING_REVIEW",
                device_type="发动机总成",
                title="待审核临时计划",
                warnings=["审核通过后方可执行"],
            )
        )
        with patch.object(main, "get_temporary_plan_service", return_value=svc):
            response = await main.temporary_plan_generate(
                TemporaryPlanGenerateRequest(
                    request_id="req-1",
                    device_type="发动机总成",
                    fault_description="异响",
                )
            )
            return response.model_dump()

    async def knowledge_clear_cache_success():
        svc = MagicMock()
        svc.clear_embedding_cache.return_value = {"text_deleted": 2, "image_deleted": 1, "total_deleted": 3}
        with patch.object(main, "get_vector_service", return_value=svc):
            response = await main.knowledge_clear_embedding_cache()
            return response.model_dump()

    async def memory_consolidate_error_response():
        agent = MagicMock()
        agent.run = AsyncMock(
            return_value=AgentOutput(
                agent_name="memory_agent",
                message="error",
                tools_used=[],
                metadata={"status": "error", "error_type": "JsonParseError", "error_detail": "bad json"},
                latency_ms=7,
            )
        )
        request = MemoryConsolidateRequest(
            sessionId="s1",
            memoryMessages=[MemoryMessage(role="user", content="hello")],
        )
        with patch.object(main, "get_memory_agent", return_value=agent):
            response = await main.memory_consolidate(request)
            content = getattr(response, "content", None)
            if content is None and hasattr(response, "body"):
                content = json.loads(response.body.decode("utf-8"))
            return {"status_code": response.status_code, "content": content}

    async def search_facts_filters_type():
        svc = MagicMock()
        svc.search_by_text = AsyncMock(
            return_value=[
                {
                    "doc_id": "fact:s1:1",
                    "text": "user device is X200",
                    "score": 0.12,
                    "metadata": {"type": "fact", "keywords": "X200", "session_id": "s1"},
                },
                {
                    "doc_id": "kb:1",
                    "text": "manual",
                    "score": 0.2,
                    "metadata": {"type": "knowledge"},
                },
            ]
        )
        with patch.object(main, "get_vector_service", return_value=svc):
            return await main.search_facts(query="X200", top_k=5)

    async def realtime_update_success():
        agent = MagicMock()
        agent.run = AsyncMock(
            return_value=AgentOutput(
                agent_name="realtime_memory_agent",
                message="updated",
                tools_used=[],
                metadata={
                    "result": {
                        "has_update": True,
                        "fact_corrections": [{"correct_content": "E5013"}],
                        "preference_changes": [],
                        "superseded_fact_ids": ["fact:s1:old"],
                    }
                },
                latency_ms=8,
            )
        )
        with patch("agents.realtime_memory_agent.get_realtime_memory_agent", return_value=agent):
            request = main.RealtimeUpdateRequest.model_construct(
                session_id="s1",
                user_message="correct code",
                ai_response="old",
                recent_facts='["old fact"]',
            )
            return await main.realtime_memory_update(request)

    run_auto_cases(
        [
            {
                "name": "chat() returns reviewed response on FixAgent success",
                "input": "ChatRequest",
                "expected": "message=verified answer and tools_used kept",
                "run": lambda: run_async(chat_success()),
                "check": lambda x: x["message"] == "verified answer"
                and x["tools_used"] == ["knowledge_retrieval"]
                and x["verification"] is None,
            },
            {
                "name": "chat() returns failure response and skips ReviewAgent when FixAgent errors",
                "input": "fix metadata.status=error",
                "expected": "HTTP 500, success=False, review call count is 0",
                "run": lambda: run_async(chat_fix_error_skips_review()),
                "check": lambda x: x["status_code"] == 500
                and x["response"]["success"] is False
                and x["response"]["code"] == 500
                and x["response"]["message"] == "temporary unavailable"
                and x["review_calls"] == 0,
            },
            {
                "name": "chat_stream() emits session/status/tool/token/verification/done SSE chunks",
                "input": "mock stream events",
                "expected": "SSE text contains key events",
                "run": lambda: run_async(chat_stream_success()),
                "check": lambda x: '"event": "session_id"' in x
                and '"event": "status"' in x
                and '"event": "tool"' in x
                and '"event": "verification"' in x
                and '"event": "done"' in x,
            },
            {
                "name": "knowledge_import() maps service result to response model",
                "input": "mock import_document result",
                "expected": "file_name/manual stats",
                "run": lambda: run_async(knowledge_import_success()),
                "check": lambda x: x["file_name"] == "manual.pdf"
                and x["total_pages"] == 2
                and x["text_count"] == 3,
            },
            {
                "name": "knowledge_search() returns total and data from tool success",
                "input": "ToolResult success",
                "expected": "total=1",
                "run": lambda: run_async(knowledge_search_success()),
                "check": lambda x: x["total"] == 1 and x["data"][0]["id"] == "doc1",
            },
            {
                "name": "knowledge_storage_stats() exposes separated vector and cache counts",
                "input": "mock storage stats",
                "expected": "cache and vectors separated",
                "run": lambda: run_async(knowledge_storage_stats_success()),
                "check": lambda x: x["vector_records"] == 3 and x["cache"]["total"] == 3,
            },
            {
                "name": "knowledge_clear_embedding_cache() only reports cache deletion",
                "input": "mock cache deletion",
                "expected": "three cache keys deleted",
                "run": lambda: run_async(knowledge_clear_cache_success()),
                "check": lambda x: x["total_deleted"] == 3,
            },
            {
                "name": "temporary_plan_generate() returns review-required draft",
                "input": "TemporaryPlanGenerateRequest",
                "expected": "status=PENDING_REVIEW and review_required=True",
                "run": lambda: run_async(temporary_plan_generate_success()),
                "check": lambda x: x["status"] == "PENDING_REVIEW"
                and x["review_required"] is True,
            },
            {
                "name": "memory_consolidate() returns JSONResponse when agent status=error",
                "input": "MemoryAgent error metadata",
                "expected": "status_code=200 and content.status=error",
                "run": lambda: run_async(memory_consolidate_error_response()),
                "check": lambda x: x["status_code"] == 200 and x["content"]["status"] == "error",
            },
            {
                "name": "search_facts() filters vector results to metadata.type=fact",
                "input": "fact and knowledge results",
                "expected": "only fact item returned",
                "run": lambda: run_async(search_facts_filters_type()),
                "check": lambda x: len(x["facts"]) == 1 and x["facts"][0]["doc_id"] == "fact:s1:1",
            },
            {
                "name": "realtime_memory_update() parses recent_facts string and returns update payload",
                "input": "RealtimeUpdateRequest",
                "expected": "has_update=True and superseded id returned",
                "run": lambda: run_async(realtime_update_success()),
                "check": lambda x: x["has_update"] is True
                and x["superseded_fact_ids"] == ["fact:s1:old"],
            },
        ]
    )


def manual_test():
    import api.main as main
    from schemas.request import ChatRequest

    message = ask("message", "bearing is overheating")
    response = run_async(main.chat(ChatRequest(session_id="manual", message=message, stream=False)))
    print_json(response.model_dump())


if __name__ == "__main__":
    run_menu("api/main.py", auto_test, manual_test)
