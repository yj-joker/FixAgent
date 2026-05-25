from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def make_agent(llm=None):
    from agents.realtime_memory_agent import RealtimeMemoryAgent

    return RealtimeMemoryAgent(llm or MagicMock())


def auto_test():
    import agents.realtime_memory_agent as module
    from agents.base_agent import AgentInput
    from agents.realtime_memory_agent import FactCorrection, get_realtime_memory_agent

    def parse_json_block():
        agent = make_agent()
        result = agent._parse_result(
            """```json
{"has_update": true, "fact_corrections": [], "preference_changes": [{"action": "upsert", "content": "reply short", "category": "style", "preferenceCategory": 0, "sourceType": "explicit"}]}
```"""
        )
        return result.model_dump()

    def trim_multiple_preferences():
        agent = make_agent()
        result = agent._parse_result(
            '{"has_update": true, "fact_corrections": [], "preference_changes": ['
            '{"action": "upsert", "content": "first"},'
            '{"action": "upsert", "content": "second"}'
            "]}"
        )
        return result.model_dump()

    def reject_non_dict_json():
        try:
            make_agent()._parse_result("[]")
            return False
        except ValueError:
            return True

    async def run_no_update():
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(
            return_value={
                "content": '{"has_update": false, "fact_corrections": [], "preference_changes": []}'
            }
        )
        out = await make_agent(llm).run(
            AgentInput(
                user_message="normal question",
                session_id="s1",
                context={"user_message": "normal question", "ai_response": "normal answer"},
            )
        )
        return out.model_dump()

    async def run_llm_error():
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(side_effect=RuntimeError("llm down"))
        out = await make_agent(llm).run(AgentInput(user_message="hello", session_id="s1"))
        return out.model_dump()

    async def apply_fact_correction():
        agent = make_agent()
        correction = FactCorrection(
            wrong_content="old fault code E4012",
            correct_content="correct fault code E5013",
            keywords="fault code",
        )
        with patch("services.vector_service.get_vector_service") as get_vec, patch(
            "embeddings.text_embedding.get_text_embedding"
        ) as get_emb:
            vector = MagicMock()
            vector.search.return_value = [
                {"doc_id": "fact:s1:old", "score": 0.2, "metadata": {"type": "fact"}}
            ]
            vector.delete.return_value = True
            vector.add_vector.return_value = True
            emb = MagicMock()
            emb.embed = AsyncMock(return_value=[0.1] * 1024)
            get_vec.return_value = vector
            get_emb.return_value = emb

            result = await agent._apply_fact_corrections([correction], "s1")
            return {
                "result": result,
                "delete_calls": vector.delete.call_count,
                "add_calls": vector.add_vector.call_count,
                "new_doc_id": vector.add_vector.call_args.kwargs["doc_id"],
            }

    def singleton():
        with patch("services.llm_service.get_llm_service") as get_llm:
            get_llm.return_value = MagicMock()
            module._realtime_agent = None
            return get_realtime_memory_agent() is get_realtime_memory_agent()

    run_auto_cases(
        [
            {
                "name": "_parse_result supports markdown json block",
                "input": "```json ... ```",
                "expected": "has_update=True and one preference",
                "run": parse_json_block,
                "check": lambda x: x["has_update"] is True and len(x["preference_changes"]) == 1,
            },
            {
                "name": "_parse_result keeps only first preference change",
                "input": "two preference changes",
                "expected": "one item, content=first",
                "run": trim_multiple_preferences,
                "check": lambda x: len(x["preference_changes"]) == 1
                and x["preference_changes"][0]["content"] == "first",
            },
            {
                "name": "_parse_result rejects non-dict JSON",
                "input": "[]",
                "expected": "ValueError",
                "run": reject_non_dict_json,
                "check": lambda x: x is True,
            },
            {
                "name": "run() returns ok metadata when there is no update",
                "input": "LLM returns has_update=false",
                "expected": "metadata.status=ok, has_update=False",
                "run": lambda: run_async(run_no_update()),
                "check": lambda x: x["metadata"]["status"] == "ok"
                and x["metadata"]["has_update"] is False,
            },
            {
                "name": "run() returns error metadata on LLM failure",
                "input": "chat_with_tools raises RuntimeError",
                "expected": "metadata.status=error",
                "run": lambda: run_async(run_llm_error()),
                "check": lambda x: x["metadata"]["status"] == "error"
                and x["metadata"]["has_update"] is False,
            },
            {
                "name": "_apply_fact_corrections deletes matched fact and writes new fact",
                "input": "one correction with old match score=0.2",
                "expected": "superseded old id and new fact:s1:rt_ id",
                "run": lambda: run_async(apply_fact_correction()),
                "check": lambda x: x["result"]["superseded_ids"] == ["fact:s1:old"]
                and x["delete_calls"] == 1
                and x["add_calls"] == 1
                and x["new_doc_id"].startswith("fact:s1:rt_"),
            },
            {
                "name": "get_realtime_memory_agent() returns singleton",
                "input": "two calls",
                "expected": True,
                "run": singleton,
                "check": lambda x: x is True,
            },
        ]
    )


def manual_test():
    from agents.base_agent import AgentInput
    from agents.realtime_memory_agent import get_realtime_memory_agent

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    user_message = ask("user message", "Actually the fault code is E5013, not E4012")
    ai_response = ask("ai response", "Previous answer mentioned E4012.")
    out = run_async(
        get_realtime_memory_agent().run(
            AgentInput(
                user_message=user_message,
                session_id="manual",
                context={"user_message": user_message, "ai_response": ai_response, "recent_facts": []},
            )
        )
    )
    print_json(out.model_dump())


if __name__ == "__main__":
    run_menu("agents/realtime_memory_agent.py", auto_test, manual_test)
