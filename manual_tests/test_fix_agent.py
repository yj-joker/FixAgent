from unittest.mock import MagicMock, patch

from test_runner import print_json, run_async, run_auto_cases, run_menu


def auto_test():
    import agents.fix_agent as module
    from agents.base_agent import AgentInput
    from agents.fix_agent import FIX_AGENT_SYSTEM_PROMPT, FixAgent, get_fix_agent

    def prompt_case():
        return {
            "knowledge": "知识检索" in FIX_AGENT_SYSTEM_PROMPT,
            "diagnosis": "故障诊断" in FIX_AGENT_SYSTEM_PROMPT,
            "guidance": "维修指引" in FIX_AGENT_SYSTEM_PROMPT,
            "tools": all(name in FIX_AGENT_SYSTEM_PROMPT for name in ["knowledge_retrieval", "graph_query_diagnosis_path", "graph_search_devices"]),
        }

    def lazy_tools():
        agent = FixAgent(MagicMock())
        before = agent._tools is None
        with patch("tools.knowledge_retrieval_tool.get_knowledge_retrieval_tool") as k, \
             patch("tools.graph_query_tool.get_graph_query_tool") as g, \
             patch("tools.graph_query_tool.get_graph_search_device_tool") as d:
            k.return_value = MagicMock(name="knowledge_tool")
            g.return_value = MagicMock(name="graph_tool")
            d.return_value = MagicMock(name="device_tool")
            tools = agent.get_tools()
        return {"before": before, "count": len(tools), "after": agent._tools is not None}

    def tool_schema_names():
        agent = FixAgent(MagicMock())
        tools = []
        for name in ["knowledge_retrieval", "graph_query_diagnosis_path", "graph_search_devices"]:
            t = MagicMock()
            t.name = name
            t.to_openai_schema.return_value = {"type": "function", "function": {"name": name}}
            tools.append(t)
        agent._tools = tools
        return [t.to_openai_schema()["function"]["name"] == t.name for t in agent.get_tools()]

    async def react_stream():
        llm = MagicMock()
        llm.chat_with_tools = __import__("unittest.mock").mock.AsyncMock(return_value={"content": "你好", "trace": []})
        agent = FixAgent(llm)
        agent._tools = []
        events = []
        async for event in agent.run_with_react_stream(AgentInput(user_message="你好", session_id="s1")):
            events.append(event["event"])
        return events

    def singleton():
        with patch("services.llm_service.get_llm_service") as get_llm:
            get_llm.return_value = MagicMock()
            module._fix_agent = None
            return get_fix_agent() is get_fix_agent()

    run_auto_cases([
        {
            "name": "系统提示词包含三大职责、工具说明和回答规范",
            "input": "FIX_AGENT_SYSTEM_PROMPT",
            "expected": "核心关键词齐全",
            "run": prompt_case,
            "check": lambda x: all(x.values()),
        },
        {
            "name": "工具懒加载：首次 get_tools() 才创建工具实例",
            "input": "_tools is None",
            "expected": "返回 3 个工具",
            "run": lazy_tools,
            "check": lambda x: x == {"before": True, "count": 3, "after": True},
        },
        {
            "name": "工具注册名称与 OpenAI schema function.name 一致",
            "input": "3 个工具 mock",
            "expected": "[True, True, True]",
            "run": tool_schema_names,
            "check": lambda x: x == [True, True, True],
        },
        {
            "name": "run_with_react_stream() 可产生完整事件流",
            "input": "无工具，LLM直接回复",
            "expected": "status/token/done",
            "run": lambda: run_async(react_stream()),
            "check": lambda x: x[0] == "status" and "token" in x and x[-1] == "done",
        },
        {
            "name": "get_fix_agent() 返回同一实例",
            "input": "连续调用",
            "expected": True,
            "run": singleton,
            "check": lambda x: x is True,
        },
    ])


def manual_test():
    from agents.fix_agent import get_fix_agent

    agent = get_fix_agent()
    print_json({"name": agent.name, "description": agent.description, "tools": [t.name for t in agent.get_tools()]})


if __name__ == "__main__":
    run_menu("agents/fix_agent.py", auto_test, manual_test)
