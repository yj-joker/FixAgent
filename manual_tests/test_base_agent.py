from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError

from test_runner import print_json, run_async, run_auto_cases, run_menu


def make_agent(llm=None):
    from agents.base_agent import BaseAgent

    class DemoAgent(BaseAgent):
        @property
        def name(self):
            return "demo_agent"

        @property
        def description(self):
            return "演示 Agent"

        def get_system_prompt(self):
            return "系统提示"

    return DemoAgent(llm or MagicMock())


def auto_test():
    from agents.base_agent import AgentInput

    def input_validation():
        try:
            AgentInput(user_message="缺少 session_id")
            return False
        except ValidationError:
            return True

    def build_messages_context_images():
        agent = make_agent()
        msg = agent._build_messages(AgentInput(
            user_message="当前问题",
            session_id="s1",
            images=["http://img"],
            conversation_history=[{"role": "user", "content": "历史问"}, {"role": "assistant", "content": "历史答"}],
            context={
                "previous_summary": "旧摘要",
                "relevant_facts": [{"text": "历史事实"}],
                "user_preferences": [{"content": "中文"}],
                "session_preferences": [{"content": "简短"}],
                "unresolved_items": [{"type": "待办", "content": "继续排查"}],
            }
        ))
        return msg

    async def run_success():
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={"content": "回答"})
        output = await make_agent(llm).run(AgentInput(user_message="你好", session_id="s1"))
        return output.model_dump()

    async def run_error():
        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM失败"))
        output = await make_agent(llm).run(AgentInput(user_message="你好", session_id="s1"))
        return output.model_dump()

    async def react_success():
        tool = MagicMock()
        tool.name = "demo_tool"
        tool.to_openai_schema.return_value = {"type": "function", "function": {"name": "demo_tool"}}
        tool.run = AsyncMock(return_value=MagicMock(success=True, data={"ok": True}, error=None))
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(return_value={"content": "ReAct回答", "trace": [{"action": "finish"}]})
        agent = make_agent(llm)
        agent.get_tools = lambda: [tool]
        output = await agent.run_with_react(AgentInput(user_message="查", session_id="s1"))
        return output.model_dump()

    async def stream_events():
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(return_value={"content": "OK", "trace": [{"action": "finish"}]})
        agent = make_agent(llm)
        events = []
        async for event in agent.run_with_react_stream(AgentInput(user_message="查", session_id="s1")):
            events.append(event["event"])
        return events

    run_auto_cases([
        {
            "name": "AgentInput user_message 和 session_id 必填",
            "input": "缺少 session_id",
            "expected": "ValidationError",
            "run": input_validation,
            "check": lambda x: x is True,
        },
        {
            "name": "_build_messages 注入历史、上下文和图片",
            "input": "history/context/images",
            "expected": "system 含旧摘要，user 含图片 URL",
            "run": build_messages_context_images,
            "check": lambda x: x[0]["role"] == "system" and "旧摘要" in x[0]["content"] and x[-1]["role"] == "user" and "http://img" in x[-1]["content"],
        },
        {
            "name": "run() 正常返回 AgentOutput 并记录 latency_ms",
            "input": "LLM 返回 content",
            "expected": "message=回答",
            "run": lambda: run_async(run_success()),
            "check": lambda x: x["message"] == "回答" and x["agent_name"] == "demo_agent" and x["latency_ms"] >= 0,
        },
        {
            "name": "run() 异常返回友好错误和 metadata.status=error",
            "input": "LLM 抛 RuntimeError",
            "expected": "AI服务暂时不可用",
            "run": lambda: run_async(run_error()),
            "check": lambda x: x["metadata"]["status"] == "error" and "AI服务暂时不可用" in x["message"],
        },
        {
            "name": "run_with_react() 注册工具 schema 并返回 react metadata",
            "input": "1 个工具",
            "expected": "execution_mode=react",
            "run": lambda: run_async(react_success()),
            "check": lambda x: x["metadata"]["execution_mode"] == "react" and x["tools_used"] == ["demo_tool"],
        },
        {
            "name": "run_with_react_stream() 事件包含 status/token/done",
            "input": "content='OK'",
            "expected": "status, token, done",
            "run": lambda: run_async(stream_events()),
            "check": lambda x: x[0] == "status" and "token" in x and x[-1] == "done",
        },
    ])


def manual_test():
    agent = make_agent(MagicMock())
    msg = agent._build_messages(__import__("agents.base_agent", fromlist=["AgentInput"]).AgentInput(user_message="手动问题", session_id="manual"))
    print_json(msg)


if __name__ == "__main__":
    run_menu("agents/base_agent.py", auto_test, manual_test)
