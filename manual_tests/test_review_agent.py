from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import print_json, run_async, run_auto_cases, run_menu


def auto_test():
    from agents.base_agent import AgentOutput
    from agents.review_agent import _GraphCheck, _GroundingCheck, _SafetyCheck, ReviewAgent, get_review_agent

    async def grounding_no_evidence():
        return await _GroundingCheck.run("建议检查轴承温度是否超过100°C。", [])

    async def grounding_embedding_fail():
        trace = [{"action": "tool_call", "tool_calls": [{"name": "knowledge_retrieval", "arguments": {"query": "轴承"}, "result_summary": "证据"}]}]
        with patch("embeddings.text_embedding.get_text_embedding") as get_emb:
            emb = MagicMock()
            emb.embed_batch = AsyncMock(side_effect=RuntimeError("向量失败"))
            get_emb.return_value = emb
            return await _GroundingCheck.run("建议检查轴承温度是否超过100°C。", trace)

    async def review_pipeline():
        output = AgentOutput(
            agent_name="fix_agent",
            message="发动机过热需要检查冷却液。",
            tools_used=["knowledge_retrieval"],
            metadata={"react_trace": []},
            latency_ms=10,
        )
        return (await ReviewAgent().review(output)).model_dump()

    def inline_markers():
        answer = "轴承过热：更换轴承。"
        verification = {
            "grounding": {"unverified_claims": [{"sentence": "轴承过热：更换轴承", "max_similarity": 0.1}]},
            "graph": {"unverified_paths": [{"fault_name": "轴承过热", "reason": "故障名不在图谱中"}]},
        }
        return ReviewAgent().get_inline_markers(answer, verification)

    run_auto_cases([
        {
            "name": "_split_sentences 和 _is_factual_claim 识别事实句",
            "input": "含建议/100°C/你好",
            "expected": "事实句 True，闲聊 False",
            "run": lambda: {
                "sentences": _GroundingCheck._split_sentences("你好。\n建议检查轴承温度100°C。"),
                "fact": _GroundingCheck._is_factual_claim("建议检查轴承温度100°C"),
                "chat": _GroundingCheck._is_factual_claim("你好欢迎使用"),
            },
            "check": lambda x: x["fact"] is True and x["chat"] is False,
        },
        {
            "name": "无证据时事实句全部标记未验证",
            "input": "react_trace=[]",
            "expected": "unverified_count > 0",
            "run": lambda: run_async(grounding_no_evidence()),
            "check": lambda x: x["unverified_count"] > 0 and "无工具调用记录" in x["note"],
        },
        {
            "name": "向量化失败时 grounding 默认通过",
            "input": "embed_batch 抛异常",
            "expected": "unverified_count=0",
            "run": lambda: run_async(grounding_embedding_fail()),
            "check": lambda x: x["unverified_count"] == 0 and "向量化失败" in x["note"],
        },
        {
            "name": "_GraphCheck 提取故障-方案对和 trace 结果",
            "input": "1. 轴承过热：立即更换轴承并检查润滑",
            "expected": "提取到 pair",
            "run": lambda: {
                "pairs": _GraphCheck._extract_pairs("1. 轴承过热：立即更换轴承并检查润滑"),
                "trace": _GraphCheck._parse_trace_results([{"action": "tool_call", "tool_calls": [{"name": "graph_query_diagnosis_path", "result_summary": '[{\"fault_name\":\"轴承过热\",\"solution_title\":\"立即更换轴承并检查润滑\"}'}]}]),
            },
            "check": lambda x: x["pairs"][0]["fault_name"] == "轴承过热" and x["trace"][0]["solution_title"] == "立即更换轴承并检查润滑",
        },
        {
            "name": "_SafetyCheck 缺少安全提醒时追加警告，已有关键词不重复",
            "input": "发动机过热 / 高压断电验电",
            "expected": "高温追加，高压不缺失",
            "run": lambda: {
                "hot": _SafetyCheck.run("发动机过热需要检查。"),
                "high_voltage": _SafetyCheck.run("高压设备操作前需要断电并验电。"),
            },
            "check": lambda x: x["hot"]["missing_count"] >= 1 and x["high_voltage"]["missing_count"] == 0,
        },
        {
            "name": "review() 聚合三层 verification 并追加安全警告",
            "input": "发动机过热需要检查冷却液",
            "expected": "metadata.verification 存在",
            "run": lambda: run_async(review_pipeline()),
            "check": lambda x: "verification" in x["metadata"] and x["metadata"]["verification_has_issues"] is True,
        },
        {
            "name": "get_inline_markers() 返回按位置排序且去重的标记",
            "input": "grounding + graph 同一位置",
            "expected": "同一 char_pos 不重复",
            "run": inline_markers,
            "check": lambda x: len({m["char_pos"] for m in x}) == len(x) and x == sorted(x, key=lambda m: m["char_pos"]),
        },
    ])


def manual_test():
    from agents.review_agent import _SafetyCheck

    text = input("请输入要检查安全提醒的回答: ").strip() or "发动机过热需要检查。"
    print_json(_SafetyCheck.run(text))


if __name__ == "__main__":
    run_menu("agents/review_agent.py", auto_test, manual_test)
