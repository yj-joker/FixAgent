from unittest.mock import AsyncMock, MagicMock

from test_runner import print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def auto_test():
    from services.llm_service import LLMService

    async def chat_non_stream():
        svc = LLMService()
        fake_response = MagicMock()
        fake_response.json.return_value = {"id": "req1", "choices": [{"message": {"content": "你好"}}], "usage": {"total_tokens": 3}}
        fake_response.raise_for_status.return_value = None
        svc.client.post = AsyncMock(return_value=fake_response)
        return await svc.chat([{"role": "user", "content": "你好"}])

    async def chat_with_tools_no_tool():
        svc = LLMService()
        svc._sync_chat_with_tools = AsyncMock(return_value={"content": "直接回答", "tool_calls": [], "finish_reason": "stop"})
        result = await svc.chat_with_tools([{"role": "user", "content": "你好"}], [], {})
        return {"content": result["content"], "trace_action": result["trace"][0]["action"]}

    async def chat_with_single_tool():
        svc = LLMService()
        svc._sync_chat_with_tools = AsyncMock(side_effect=[
            {
                "content": "",
                "tool_calls": [{
                    "id": "call1",
                    "function": {"name": "demo_tool", "arguments": "{\"query\":\"轴承\"}"},
                }],
                "finish_reason": "tool_calls",
            },
            {"content": "工具后回答", "tool_calls": [], "finish_reason": "stop"},
        ])

        async def handler(query):
            return {"query": query, "result": "ok"}

        result = await svc.chat_with_tools([{"role": "user", "content": "查轴承"}], [{"type": "function", "function": {"name": "demo_tool"}}], {"demo_tool": handler})
        return {"content": result["content"], "trace_len": len(result["trace"]), "first_action": result["trace"][0]["action"]}

    async def max_iterations():
        svc = LLMService()
        svc._sync_chat_with_tools = AsyncMock(return_value={
            "content": "",
            "tool_calls": [{"id": "call1", "function": {"name": "missing", "arguments": "{}"}}],
        })
        try:
            await svc.chat_with_tools([{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "missing"}}], {}, max_iterations=1)
            return "未抛异常"
        except RuntimeError as exc:
            return str(exc)

    run_auto_cases([
        {
            "name": "chat 非流式返回 content/usage/request_id",
            "input": "单条 user 消息",
            "expected": {"content": "你好", "request_id": "req1"},
            "run": lambda: run_async(chat_non_stream()),
            "check": lambda x: x["content"] == "你好" and x["request_id"] == "req1",
        },
        {
            "name": "chat_with_tools 无工具调用时直接 finish",
            "input": "tools=[]",
            "expected": {"trace_action": "finish"},
            "run": lambda: run_async(chat_with_tools_no_tool()),
            "check": lambda x: x["content"] == "直接回答" and x["trace_action"] == "finish",
        },
        {
            "name": "chat_with_tools 单次工具调用后继续生成最终回答",
            "input": "demo_tool",
            "expected": "trace 包含 tool_call 和 finish",
            "run": lambda: run_async(chat_with_single_tool()),
            "check": lambda x: x["content"] == "工具后回答" and x["trace_len"] == 2 and x["first_action"] == "tool_call",
        },
        {
            "name": "chat_with_tools 超过 max_iterations 抛 RuntimeError",
            "input": "max_iterations=1 且一直 tool_call",
            "expected": "Tool calling 超出最大迭代次数",
            "run": lambda: run_async(max_iterations()),
            "check": lambda x: "超出最大迭代次数" in x,
        },
    ])


def manual_test():
    from services.llm_service import get_llm_service

    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    message = input("请输入要发送给 DashScope 的消息（回车默认: 你好）: ").strip() or "你好"
    result = run_async(get_llm_service().chat([{"role": "user", "content": message}], stream=False))
    print_json(result)


if __name__ == "__main__":
    run_menu("services/llm_service.py", auto_test, manual_test)
