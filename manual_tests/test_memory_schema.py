import json

from pydantic import ValidationError

from test_runner import print_json, run_auto_cases, run_menu


def _error(fn):
    try:
        fn()
        return None
    except Exception as exc:
        return exc


def auto_test():
    from schemas.memory import FactItem, MemorySummary, PreferenceItem

    run_auto_cases([
        {
            "name": "FactItem content 必填，keywords/source_seq_range 默认空字符串",
            "input": "content='用户设备型号X200'",
            "expected": {"keywords": "", "source_seq_range": ""},
            "run": lambda: FactItem(content="用户设备型号X200").model_dump(),
            "check": lambda x: x["keywords"] == "" and x["source_seq_range"] == "",
        },
        {
            "name": "FactItem 缺少 content 时校验失败",
            "input": "{}",
            "expected": "ValidationError",
            "run": lambda: isinstance(_error(lambda: FactItem()), ValidationError),
            "check": lambda x: x is True,
        },
        {
            "name": "PreferenceItem 默认 preferenceCategory=0, sourceType=inferred",
            "input": "content/category",
            "expected": {"preferenceCategory": 0, "sourceType": "inferred"},
            "run": lambda: PreferenceItem(content="用中文", category="交互风格").model_dump(),
            "check": lambda x: x["preferenceCategory"] == 0 and x["sourceType"] == "inferred",
        },
        {
            "name": "MemorySummary 完整字段可解析",
            "input": "包含 facts/preferences/unresolved/resolved/summary",
            "expected": "全部字段存在",
            "run": lambda: MemorySummary(
                new_facts=[{"content": "型号X200"}],
                superseded_ids=["fact:old"],
                updated_preferences=[{"content": "简短回答"}],
                updated_unresolved=[{"content": "继续排查"}],
                resolved_item_ids=[1],
                brief_summary="用户在排查故障",
            ).model_dump(),
            "check": lambda x: len(x["new_facts"]) == 1 and x["resolved_item_ids"] == [1],
        },
        {
            "name": "典型 JSON 可反序列化为空列表默认不报错",
            "input": "{\"brief_summary\":\"摘要\"}",
            "expected": "new_facts 默认 []",
            "run": lambda: MemorySummary(**json.loads('{"brief_summary":"摘要"}')).model_dump(),
            "check": lambda x: x["new_facts"] == [] and x["brief_summary"] == "摘要",
        },
    ])


def manual_test():
    from schemas.memory import MemorySummary

    print("请输入 MemorySummary JSON，直接回车使用默认示例。")
    raw = input("> ").strip() or '{"new_facts":[{"content":"设备型号为X200"}],"brief_summary":"记录设备型号"}'
    print_json(MemorySummary(**json.loads(raw)).model_dump())


if __name__ == "__main__":
    run_menu("schemas/memory.py", auto_test, manual_test)
