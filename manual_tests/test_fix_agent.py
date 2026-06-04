from unittest.mock import MagicMock, patch
from unittest.mock import AsyncMock

from test_runner import print_json, run_async, run_auto_cases, run_menu


def test_fix_agent_injects_images_and_enhanced_query_into_retrieval_tools():
    from agents.fix_agent import FixAgent

    agent = FixAgent(MagicMock())
    agent._current_images = ["img://spark-plug"]
    agent._current_enhanced_query = "火花塞 电极间隙 摩托车发动机"

    retrieval_kwargs = agent._customize_tool_kwargs(
        "knowledge_retrieval",
        {"query": "这个是什么", "top_k": 5},
    )
    graph_kwargs = agent._customize_tool_kwargs(
        "graph_search_java",
        {"fault_description": "无法启动"},
    )

    assert retrieval_kwargs["image_urls"] == ["img://spark-plug"]
    assert "这个是什么" in retrieval_kwargs["query"]
    assert "火花塞 电极间隙" in retrieval_kwargs["query"]
    assert graph_kwargs["image_urls"] == ["img://spark-plug"]


def test_fix_agent_filters_tools_by_intent_decision():
    from agents.fix_agent import FixAgent

    agent = FixAgent(MagicMock())
    tools = []
    for name in ["knowledge_retrieval", "graph_search_java", "procedure_recommend"]:
        tool = MagicMock()
        tool.name = name
        tools.append(tool)
    agent._tools = tools
    agent._current_allowed_tools = ["knowledge_retrieval"]

    assert [tool.name for tool in agent.get_tools()] == ["knowledge_retrieval"]

    agent._current_allowed_tools = []
    assert agent.get_tools() == []


def test_fix_agent_reruns_once_when_tool_scope_is_insufficient():
    from agents.base_agent import AgentInput
    from agents.fix_agent import FixAgent

    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        {"content": "NEEDS_MORE_TOOLS: 需要图谱工具", "trace": []},
        {"content": "已补充图谱检索后回答。", "trace": []},
    ])
    agent = FixAgent(llm)
    knowledge_tool = MagicMock()
    knowledge_tool.name = "knowledge_retrieval"
    knowledge_tool.to_openai_schema.return_value = {"type": "function", "function": {"name": "knowledge_retrieval"}}
    graph_tool = MagicMock()
    graph_tool.name = "graph_search_java"
    graph_tool.to_openai_schema.return_value = {"type": "function", "function": {"name": "graph_search_java"}}
    agent._tools = [knowledge_tool, graph_tool]

    result = run_async(agent.run_with_react(AgentInput(
        user_message="帮我分析故障",
        session_id="s1",
        context={
            "intent_decision": {
                "intent": "knowledge_query",
                "answer_style": "plain_conversational",
                "allowed_tools": ["knowledge_retrieval"],
            }
        },
    )))

    assert result.message == "已补充图谱检索后回答。"
    assert result.metadata["intent_rerun_with_full_tools"] is True
    assert llm.chat_with_tools.call_count == 2


def test_fix_agent_reruns_once_for_structured_needs_more_tools_status():
    from agents.base_agent import AgentInput
    from agents.fix_agent import FixAgent

    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        {
            "content": (
                '{"status":"needs_more_tools",'
                '"needed_tools":["graph_search_java"],'
                '"reason":"需要图谱确认部件和故障路径"}'
            ),
            "trace": [],
        },
        {"content": "已使用扩展工具完成回答。", "trace": []},
    ])
    agent = FixAgent(llm)
    knowledge_tool = MagicMock()
    knowledge_tool.name = "knowledge_retrieval"
    knowledge_tool.to_openai_schema.return_value = {"type": "function", "function": {"name": "knowledge_retrieval"}}
    graph_tool = MagicMock()
    graph_tool.name = "graph_search_java"
    graph_tool.to_openai_schema.return_value = {"type": "function", "function": {"name": "graph_search_java"}}
    agent._tools = [knowledge_tool, graph_tool]

    result = run_async(agent.run_with_react(AgentInput(
        user_message="启动不了怎么处理",
        session_id="s1",
        context={
            "intent_decision": {
                "intent": "maintenance_guidance",
                "task_action": "repair_guidance",
                "policy": {
                    "tool_scope": ["knowledge_retrieval"],
                    "response_style": "step_guidance",
                    "evidence_level": "required",
                    "safety_level": "operation",
                },
            }
        },
    )))

    assert result.message == "已使用扩展工具完成回答。"
    assert result.metadata["intent_rerun_with_full_tools"] is True
    assert result.metadata["react_status_before_rerun"]["needed_tools"] == ["graph_search_java"]
    assert "图谱确认" in result.metadata["intent_rerun_reason"]


def test_fix_agent_formats_structured_user_clarification_status():
    from agents.base_agent import AgentInput
    from agents.fix_agent import FixAgent

    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(return_value={
        "content": (
            '{"status":"needs_user_clarification",'
            '"answer_mode":"general_then_ask",'
            '"general_answer":"可以先从外观、启动、燃油、点火和压缩这几项做通用排查。",'
            '"questions":["摩托车型号或发动机型号是什么？","主要故障现象是启动不了、异响、漏油还是过热？"],'
            '"reason":"缺少车型和故障现象，无法生成可靠检修步骤"}'
        ),
        "trace": [],
    })
    agent = FixAgent(llm)
    agent._tools = []

    result = run_async(agent.run_with_react(AgentInput(
        user_message="发动机坏了怎么修",
        session_id="s1",
        context={
            "intent_decision": {
                "intent": "maintenance_guidance",
                "task_action": "repair_guidance",
                "policy": {
                    "tool_scope": [],
                    "response_style": "step_guidance",
                    "evidence_level": "required",
                    "safety_level": "operation",
                },
            }
        },
    )))

    assert result.metadata["react_status"]["status"] == "needs_user_clarification"
    assert "可以先从外观、启动、燃油、点火和压缩" in result.message
    assert "摩托车型号或发动机型号是什么" in result.message
    assert "主要故障现象" in result.message
    assert '"status"' not in result.message


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
