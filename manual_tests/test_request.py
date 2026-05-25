from pydantic import ValidationError

from test_runner import ask, print_json, run_auto_cases, run_menu


def _ok(fn):
    try:
        return fn()
    except Exception as exc:
        return exc


def auto_test():
    from schemas.models import AgentMode
    from schemas.request import (
        ChatRequest,
        KnowledgeImportRequest,
        KnowledgeSearchRequest,
        MemoryConsolidateRequest,
        MemoryPreferenceVO,
        MemoryUnresolvedVO,
    )

    run_auto_cases([
        {
            "name": "ChatRequest 拦截空 message、超长 message、超过 10 张图片",
            "input": "message='' / 50001字符 / 11张图片",
            "expected": "全部 ValidationError",
            "run": lambda: [
                isinstance(_ok(lambda: ChatRequest(session_id="s", message="")), ValidationError),
                isinstance(_ok(lambda: ChatRequest(session_id="s", message="x" * 50001)), ValidationError),
                isinstance(_ok(lambda: ChatRequest(session_id="s", message="ok", images=["u"] * 11)), ValidationError),
            ],
            "check": lambda x: x == [True, True, True],
        },
        {
            "name": "ChatRequest 默认值正确",
            "input": {"session_id": "s1", "message": "你好"},
            "expected": "mode=chat, stream=True, conversation_history/context=None",
            "run": lambda: ChatRequest(session_id="s1", message="你好").model_dump(),
            "check": lambda x: x["mode"] == AgentMode.CHAT and x["stream"] is True and x["conversation_history"] is None and x["context"] is None,
        },
        {
            "name": "KnowledgeImportRequest file_type 默认 pdf，category/tags 可选",
            "input": {"file_url": "manual.pdf"},
            "expected": {"file_type": "pdf"},
            "run": lambda: KnowledgeImportRequest(file_url="manual.pdf").model_dump(),
            "check": lambda x: x["file_type"] == "pdf" and x["category"] is None and x["tags"] is None,
        },
        {
            "name": "KnowledgeSearchRequest top_k 范围为 1-50",
            "input": "top_k=0, 51, 10",
            "expected": "0/51失败，10成功",
            "run": lambda: [
                isinstance(_ok(lambda: KnowledgeSearchRequest(query="轴承", top_k=0)), ValidationError),
                isinstance(_ok(lambda: KnowledgeSearchRequest(query="轴承", top_k=51)), ValidationError),
                KnowledgeSearchRequest(query="轴承", top_k=10).top_k,
            ],
            "check": lambda x: x == [True, True, 10],
        },
        {
            "name": "MemoryConsolidateRequest sessionId 别名与 memoryMessages 校验",
            "input": {"sessionId": "s1", "memoryMessages": [{"role": "user", "content": "故障码E-5013"}]},
            "expected": "session_id=s1",
            "run": lambda: MemoryConsolidateRequest(
                sessionId="s1",
                memoryMessages=[{"role": "user", "content": "故障码E-5013"}],
            ).session_id,
            "check": lambda x: x == "s1",
        },
        {
            "name": "MemoryPreferenceVO / MemoryUnresolvedVO 基本字段合法",
            "input": "preferenceCategory=0/1",
            "expected": "两个对象可创建",
            "run": lambda: [
                MemoryPreferenceVO(content="用中文", category="交互风格", preferenceCategory=0).preferenceCategory,
                MemoryPreferenceVO(content="本轮简短", category="格式要求", preferenceCategory=1).preferenceCategory,
                MemoryUnresolvedVO(content="继续排查", type="进行中任务", status="active").status,
            ],
            "check": lambda x: x == [0, 1, "active"],
        },
    ])


def manual_test():
    from schemas.request import ChatRequest

    session_id = ask("session_id", "sess_manual")
    message = ask("message", "你好")
    stream = ask("stream true/false", "true").lower() == "true"
    req = ChatRequest(session_id=session_id, message=message, stream=stream)
    print_json(req.model_dump())


if __name__ == "__main__":
    run_menu("schemas/request.py", auto_test, manual_test)
