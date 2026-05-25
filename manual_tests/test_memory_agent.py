from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, run_async, run_auto_cases, run_menu


def make_agent(llm=None):
    from agents.memory_agent import MemoryAgent
    return MemoryAgent(llm or MagicMock())


def auto_test():
    import agents.memory_agent as module
    from agents.base_agent import AgentInput
    from agents.memory_agent import get_memory_agent

    def format_conversations():
        agent = make_agent()
        return agent._format_conversations([
            {"seq": 1, "role": "user", "content": "我的设备是X200"},
            {"seq": 2, "role": "assistant", "content": "建议检查轴承"},
        ])

    def build_messages():
        agent = make_agent()
        messages = agent._build_messages(AgentInput(
            user_message="整理",
            session_id="s1",
            context={
                "conversations": [{"seq": 1, "role": "user", "content": "以后用中文"}],
                "old_preferences": [{"content": "简洁", "category": "交互风格", "preferenceCategory": 0}],
                "old_unresolved": [{"id": 7, "content": "继续排查", "type": "进行中任务", "status": "active"}],
                "previous_summary": "旧摘要",
            },
        ))
        return messages

    def extract_json_cases():
        agent = make_agent()
        normal = agent._extract_json('```json\n{"brief_summary":"摘要"}\n```').brief_summary
        direct = agent._extract_json('{"brief_summary":"裸JSON"}').brief_summary
        try:
            agent._extract_json("123")
            bad = False
        except ValueError:
            bad = True
        return {"normal": normal, "direct": direct, "bad": bad}

    async def retry_success():
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(side_effect=[
            {"content": "123"},
            {"content": '{"new_facts":[{"content":"用户设备型号为X200"}],"brief_summary":"摘要"}'},
        ])
        agent = make_agent(llm)
        with patch.object(agent, "_store_facts_to_vector", AsyncMock(return_value=["fact:s1:1"])):
            out = await agent.run(AgentInput(user_message="整理", session_id="s1", context={"conversations": []}))
        return out.model_dump()

    async def retry_exhausted():
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(return_value={"content": "123"})
        out = await make_agent(llm).run(AgentInput(user_message="整理", session_id="s1", context={"conversations": []}))
        return out.model_dump()

    async def store_facts():
        agent = make_agent()
        with patch("services.vector_service.get_vector_service") as get_vec, patch("embeddings.text_embedding.get_text_embedding") as get_emb:
            vector = MagicMock()
            vector.search.return_value = []
            vector.add_vector.return_value = True
            emb = MagicMock()
            emb.embed = AsyncMock(return_value=[0.1] * 1024)
            get_vec.return_value = vector
            get_emb.return_value = emb
            ids = await agent._store_facts_to_vector([{"content": "用户设备型号为X200", "keywords": "X200"}], "s1")
            return {"ids": ids, "add_calls": vector.add_vector.call_count}

    def singleton():
        with patch("services.llm_service.get_llm_service") as get_llm:
            get_llm.return_value = MagicMock()
            module._memory_agent = None
            return get_memory_agent() is get_memory_agent()

    run_auto_cases([
        {
            "name": "_format_conversations 正确标注【用户】/【助手】和序号",
            "input": "2条对话",
            "expected": "包含第1轮、【用户】、【助手】",
            "run": format_conversations,
            "check": lambda x: "第1轮" in x and "【用户】" in x and "【助手】" in x,
        },
        {
            "name": "_build_messages 包含旧偏好、旧待办、previous_summary 和新对话",
            "input": "context 完整",
            "expected": "user 消息含旧摘要/id=7/偏好",
            "run": build_messages,
            "check": lambda x: x[0]["role"] == "system" and "旧摘要" in x[1]["content"] and "| 7 |" in x[1]["content"],
        },
        {
            "name": "_extract_json 支持 markdown 代码块和裸 JSON，拒绝非 dict",
            "input": "代码块/裸JSON/123",
            "expected": "前两者成功，非 dict ValueError",
            "run": extract_json_cases,
            "check": lambda x: x == {"normal": "摘要", "direct": "裸JSON", "bad": True},
        },
        {
            "name": "JSON 第一次解析失败后重试成功",
            "input": "第一次 content=123，第二次合法 JSON",
            "expected": "status 正常且 message=摘要",
            "run": lambda: run_async(retry_success()),
            "check": lambda x: x["message"] == "摘要" and x["metadata"]["summary"]["fact_ids"] == ["fact:s1:1"],
        },
        {
            "name": "重试用尽返回 JsonParseError",
            "input": "两次均返回非 dict JSON",
            "expected": "metadata.status=error",
            "run": lambda: run_async(retry_exhausted()),
            "check": lambda x: x["metadata"]["status"] == "error" and x["metadata"]["error_type"] == "JsonParseError",
        },
        {
            "name": "_store_facts_to_vector 写入 fact:{session_id}:... doc_id",
            "input": "1条新事实",
            "expected": "返回 fact:s1 前缀",
            "run": lambda: run_async(store_facts()),
            "check": lambda x: x["ids"][0].startswith("fact:s1:") and x["add_calls"] == 1,
        },
        {
            "name": "get_memory_agent() 返回同一实例",
            "input": "连续调用",
            "expected": True,
            "run": singleton,
            "check": lambda x: x is True,
        },
    ])


def manual_test():
    from agents.base_agent import AgentInput
    from agents.memory_agent import get_memory_agent

    text = ask("请输入用户对话内容", "我的设备型号是X200")
    out = run_async(get_memory_agent().run(AgentInput(
        user_message="整理",
        session_id="manual",
        context={"conversations": [{"seq": 1, "role": "user", "content": text}]},
    )))
    print_json(out.model_dump())


if __name__ == "__main__":
    run_menu("agents/memory_agent.py", auto_test, manual_test)
