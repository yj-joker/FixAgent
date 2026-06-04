from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import print_json, run_async, run_auto_cases, run_menu


def test_review_pipeline_components_are_importable_and_sanitize_json_leaks():
    from agents.review_agent import _OutputSanitizer, _ResponseComposer, _SafetyReviewer

    dirty = (
        "我先调用工具。\n"
        "```json\n"
        '{"query":"识别图片","image_urls":["img://x"],"top_k":5}'
        "\n```\n"
        "根据检索结果，这是火花塞。"
    )

    cleaned = _OutputSanitizer.sanitize(dirty)
    safety = _SafetyReviewer.review(
        "拆卸火花塞前需要拔下高压线。",
        user_message="怎么拆卸火花塞？",
        policy={"safety_level": "operation"},
    )
    final = _ResponseComposer.compose(
        base_message=cleaned,
        safety=safety,
        policy={"evidence_level": "optional", "safety_level": "operation"},
    )

    assert "```json" not in cleaned
    assert "image_urls" not in cleaned
    assert "火花塞" in cleaned
    assert "安全提醒" in final


def test_evidence_verifier_runs_grounding_and_graph_by_level():
    from agents.review_agent import _EvidenceVerifier

    grounding = {"total_claims": 1, "verified_count": 1, "unverified_count": 0}
    graph = {"total_paths": 0, "verified_count": 0, "unverified_count": 0}

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)) as grounding_run, patch(
        "agents.review_agent._GraphCheck.run", new=AsyncMock(return_value=graph)
    ) as graph_run:
        result = run_async(_EvidenceVerifier.verify("火花塞是点火部件。", [], review_level="full"))

    assert result["review_level"] == "full"
    assert result["grounding"] == grounding
    assert result["graph"] == graph
    assert grounding_run.call_count == 1
    assert graph_run.call_count == 1


def test_review_policy_wins_when_legacy_flags_conflict():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message="这是普通交流，可以直接自然回复。",
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "你好",
            "intent_decision": {
                "intent": "chat_social",
                "requires_manual_evidence": True,
                "requires_safety_notice": True,
                "policy": {
                    "evidence_level": "none",
                    "safety_level": "none",
                    "response_style": "plain_conversational",
                },
            },
        },
        latency_ms=10,
    )

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock()) as grounding_run:
        result = run_async(ReviewAgent().review(output))

    assert grounding_run.call_count == 0
    assert "安全提醒" not in result.message
    assert "待确认信息" not in result.message


def test_response_composer_plain_conversational_compacts_markdown():
    from agents.review_agent import _ResponseComposer

    final = _ResponseComposer.compose(
        base_message="### 设备识别结论\n- **部件名称**：火花塞\n\n- **所属系统**：点火系统",
        safety={"appended_text": ""},
        policy={"response_style": "plain_conversational", "safety_level": "none"},
    )

    assert "###" not in final
    assert "**" not in final
    assert final.count("\n") <= 2
    assert "火花塞" in final


def test_review_blocks_formal_guidance_when_all_claims_are_unsupported():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "根据维修手册知识库检索结果（含《Honda Service Manual》《Yamaha Technical Training》），"
            "安装摩托车发动机火花塞需要按15-20 N·m紧固，并检查0.7-0.8 mm电极间隙。"
        ),
        tools_used=["knowledge_retrieval"],
        metadata={"react_trace": []},
        latency_ms=10,
    )

    result = run_async(ReviewAgent().review(output))

    assert "当前资料不足以形成可确认的正式操作指引" in result.message
    assert "根据维修手册知识库检索结果" not in result.message.split("## 待确认信息")[0]
    assert "Honda Service Manual" not in result.message.split("## 待确认信息")[0]
    assert "15-20 N·m" not in result.message.split("## 待确认信息")[0]


def test_review_removes_unsupported_source_claim_even_when_partial_content_remains():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "根据维修手册知识库检索结果，火花塞安装需要保持清洁。"
            "如果没有具体车型，建议补充车型或发动机型号。"
        ),
        tools_used=["knowledge_retrieval"],
        metadata={
            "react_trace": [{
                "action": "tool_call",
                "tool_calls": [{
                    "name": "knowledge_retrieval",
                    "result_data": [{"content": "火花塞安装前应清洁安装孔周边。"}],
                }],
            }]
        },
        latency_ms=10,
    )

    async def fake_grounding(*_args, **_kwargs):
        return {
            "unverified_claims": [{
                "sentence": "根据维修手册知识库检索结果，火花塞安装需要保持清洁",
                "max_similarity": 0.2,
            }],
            "verified_claims": [{
                "sentence": "如果没有具体车型，建议补充车型或发动机型号",
                "max_similarity": 0.8,
            }],
            "total_claims": 2,
            "verified_count": 1,
            "unverified_count": 1,
        }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(side_effect=fake_grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    assert "根据维修手册知识库检索结果" not in result.message.split("## 待确认信息")[0]
    assert "补充车型或发动机型号" in result.message


def test_safety_check_skips_identification_questions_without_operation_intent():
    from agents.review_agent import _SafetyCheck

    result = _SafetyCheck.run(
        "这看起来是火花塞，安装在发动机上，图中 a 通常表示电极间隙。",
        user_query="这个是摩托车上的部件吗，你认识这是什么吗",
    )

    assert result["missing_count"] == 0
    assert result["appended_text"] == ""


def test_safety_check_keeps_operation_questions_and_filters_irrelevant_rules():
    from agents.review_agent import _SafetyCheck

    result = _SafetyCheck.run(
        "拆卸火花塞前需要接触发动机上方部件，并拔下高压线。",
        user_query="怎么拆卸火花塞？",
    )

    assert result["missing_count"] >= 1
    assert "高温防护" in result["triggered_rules"]
    assert "化学品防护" not in result["triggered_rules"]
    assert "压力容器/管路安全" not in result["triggered_rules"]


def test_review_removes_unsupported_page_and_manual_citations_without_blocking_visual_id():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "从图片看，这很可能是火花塞，图中 a 通常表示电极间隙。\n"
            "根据《摩托车发动机维修手册 (1).pdf》第7页「3.2 点火系统组件」内容，"
            "必须按照制造商推荐型号更换，并确保电极间隙符合标准。"
        ),
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "这个是摩托车上的部件吗，你认识这是什么吗",
        },
        latency_ms=10,
    )

    grounding = {
        "unverified_claims": [{
            "sentence": "根据《摩托车发动机维修手册 (1).pdf》第7页「3.2 点火系统组件」内容，必须按照制造商推荐型号更换，并确保电极间隙符合标准",
            "max_similarity": 0.0,
        }],
        "verified_claims": [],
        "total_claims": 1,
        "verified_count": 0,
        "unverified_count": 1,
    }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    formal = result.message.split("## 待确认信息")[0]
    assert "从图片看，这很可能是火花塞" in formal
    assert "根据《摩托车发动机维修手册" not in formal
    assert "第7页" not in formal
    assert "系统通用安全提醒" not in result.message


def test_review_does_not_block_image_comparison_part_identification_question():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "从图片看，这两个是同一类部件，都是火花塞。"
            "第一张是真实火花塞照片，第二张是火花塞结构示意图。"
            "它通常安装在汽油发动机气缸盖上，属于点火系统。"
        ),
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "这两个是一样的东西吗，这是摩托车发动机上的配件吗",
        },
        latency_ms=10,
    )

    grounding = {
        "unverified_claims": [{
            "sentence": "它通常安装在汽油发动机气缸盖上，属于点火系统",
            "max_similarity": 0.0,
        }],
        "verified_claims": [],
        "total_claims": 1,
        "verified_count": 0,
        "unverified_count": 1,
    }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    assert "当前知识库未检索到可支撑" not in result.message
    assert "都是火花塞" in result.message


def test_review_keeps_identification_but_removes_unsolicited_repair_guidance():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "这两张图展示的是同一类部件：火花塞。\n"
            "图1是火花塞结构示意图，图2是火花塞实物照片。\n\n"
            "### 🔧 维修建议（若更换火花塞）\n"
            "### Step 1: 断开电源并冷却发动机\n"
            "- 操作内容：关闭摩托车电源开关，等待发动机完全冷却。\n"
            "### Step 4: 安装新火花塞\n"
            "- 操作内容：按推荐扭矩（一般15–20Nm）拧紧。\n"
            "> ⚠️ 提示：建议每行驶1万公里更换火花塞。"
        ),
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "这两个是一样的东西吗，这是摩托车发动机上的配件吗",
        },
        latency_ms=10,
    )

    grounding = {
        "unverified_claims": [{
            "sentence": "按推荐扭矩（一般15–20Nm）拧紧",
            "critical_claims": ["15–20Nm"],
        }, {
            "sentence": "建议每行驶1万公里更换火花塞",
            "critical_claims": ["1万公里"],
        }],
        "verified_claims": [],
        "total_claims": 2,
        "verified_count": 0,
        "unverified_count": 2,
    }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    formal = result.message.split("## 待确认信息")[0]
    assert "当前知识库未检索到可支撑" not in result.message
    assert "同一类部件：火花塞" in formal
    assert "维修建议" not in formal
    assert "Step 1" not in formal
    assert "15–20Nm" not in formal
    assert "1万公里" not in formal


def test_review_keeps_casual_chat_plain_without_grounding_or_safety_sections():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "挺适合转的。你有钳工基础，做摩托车维修会很占优势，"
            "尤其是工具手感、测量意识和机械结构理解。刚开始可以先从保养、点火系统和供油系统入手。"
            "你现在更想先学日常保养，还是发动机拆装？"
        ),
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "你好，我是一个钳工，最近转行做摩托车修理",
        },
        latency_ms=10,
    )

    grounding = {
        "unverified_claims": [{
            "sentence": "你有钳工基础，做摩托车维修会很占优势",
            "max_similarity": 0.0,
        }],
        "verified_claims": [],
        "total_claims": 1,
        "verified_count": 0,
        "unverified_count": 1,
    }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    assert "当前知识库未检索到可支撑" not in result.message
    assert "待确认信息" not in result.message
    assert "系统通用安全提醒" not in result.message
    assert "钳工基础" in result.message


def test_review_uses_chat_social_intent_to_skip_grounding_and_safety():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message="你有钳工基础，转摩托车维修会比较顺手。可以先从保养和点火系统入门。",
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "你好，我是一个钳工，最近转行做摩托车修理",
            "intent_decision": {
                "intent": "chat_social",
                "requires_manual_evidence": False,
                "requires_safety_notice": False,
            },
        },
        latency_ms=10,
    )

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock()) as grounding:
        result = run_async(ReviewAgent().review(output))

    assert grounding.call_count == 0
    assert "当前知识库未检索到可支撑" not in result.message
    assert "待确认信息" not in result.message
    assert "安全提醒" not in result.message
    assert "钳工基础" in result.message


def test_review_visual_intent_does_not_block_unverified_identification_claims():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message="这两张图是同一类东西，都是火花塞，常见于摩托车汽油发动机点火系统。",
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "这两个是一样的东西吗，这是摩托车发动机上的配件吗",
            "intent_decision": {
                "intent": "visual_identification",
                "requires_manual_evidence": False,
                "requires_safety_notice": False,
                "allow_visual_answer_without_manual": True,
            },
        },
        latency_ms=10,
    )
    grounding = {
        "unverified_claims": [{
            "sentence": "常见于摩托车汽油发动机点火系统",
            "max_similarity": 0.0,
        }],
        "verified_claims": [],
        "total_claims": 1,
        "verified_count": 0,
        "unverified_count": 1,
    }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    assert "当前知识库未检索到可支撑" not in result.message
    assert "待确认信息" not in result.message
    assert "都是火花塞" in result.message


def test_review_limits_pending_section_for_formal_queries():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "火花塞安装步骤如下：\n"
            "- Step 1: 断电。\n"
            "- Step 2: 拆下旧火花塞。\n"
            "- Step 3: 按15-20Nm安装新火花塞。\n"
            "- Step 4: 每1万公里更换一次。"
        ),
        tools_used=[],
        metadata={"react_trace": [], "user_message": "怎么更换火花塞？"},
        latency_ms=10,
    )
    grounding = {
        "unverified_claims": [
            {"sentence": "- Step 1: 断电。", "critical_claims": ["断电"]},
            {"sentence": "- Step 2: 拆下旧火花塞。", "critical_claims": ["拆下旧火花塞"]},
            {"sentence": "- Step 3: 按15-20Nm安装新火花塞。", "critical_claims": ["15-20Nm"]},
            {"sentence": "- Step 4: 每1万公里更换一次。", "critical_claims": ["1万公里"]},
        ],
        "verified_claims": [],
        "total_claims": 4,
        "verified_count": 0,
        "unverified_count": 4,
    }

    with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
        "agents.review_agent._GraphCheck.run",
        new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
    ):
        result = run_async(ReviewAgent().review(output))

    assert "待确认信息" in result.message
    assert result.message.count("- ") <= 4
    assert "另有" in result.message


def test_review_removes_leaked_tool_argument_json_blocks():
    from agents.base_agent import AgentOutput
    from agents.review_agent import ReviewAgent

    output = AgentOutput(
        agent_name="fix_agent",
        message=(
            "我已收到您上传的图片，正在分析其中的设备或部件。请稍等。\n\n"
            "首先，我将使用图文混合检索来识别该部件，并结合知识库判断其所属系统。\n\n"
            "```json\n"
            "{\n"
            '  "query": "识别图片中的设备或部件",\n'
            '  "image_urls": ["https://example.com/img.jpeg"],\n'
            '  "top_k": 5\n'
            "}\n"
            "```\n\n"
            "```json\n"
            "{\n"
            '  "keyword": "",\n'
            '  "component_description": "带有螺纹和电极的金属部件",\n'
            '  "limit": 10\n'
            "}\n"
            "```\n\n"
            "根据检索结果，我确认这是一个**火花塞**。"
        ),
        tools_used=[],
        metadata={
            "react_trace": [],
            "user_message": "请识别图片中的设备或部件，并结合知识库判断它可能属于什么系统。",
        },
        latency_ms=10,
    )

    result = run_async(ReviewAgent().review(output, level="light"))

    assert "```json" not in result.message
    assert '"image_urls"' not in result.message
    assert '"component_description"' not in result.message
    assert "正在分析" not in result.message
    assert "图文混合检索" not in result.message
    assert "火花塞" in result.message


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
            return await _GroundingCheck.run("建议检查轴承是否存在异常磨损。", trace)

    async def grounding_critical_claims_require_literal_evidence():
        trace = [{
            "action": "tool_call",
            "tool_calls": [{
                "name": "knowledge_retrieval",
                "arguments": {"query": "火花塞检查与安装"},
                "result_summary": "火花塞间隙标准值为0.7～0.9 mm，安装扭矩为20 ± 2 N·m，使用16 mm工具。",
                "result_data": [{"content": "火花塞间隙标准值为0.7～0.9 mm，安装扭矩为20 ± 2 N·m，使用16 mm工具。"}],
            }],
        }]
        with patch("embeddings.text_embedding.get_text_embedding") as get_emb:
            emb = MagicMock()
            emb.embed_batch = AsyncMock(return_value=[[1.0, 0.0]] * 8)
            get_emb.return_value = emb
            return await _GroundingCheck.run(
                "- 火花塞间隙：0.7～0.9 mm。\n"
                "- 安装扭矩：20 ± 2 N·m。\n"
                "- 每10,000 km或6个月检查一次。\n"
                "- 推荐型号：NGK CR7E。\n"
                "- 安装前确认型号匹配（如NGK CR7E vs. DENSO U24ESR-U）。\n"
                "- 发动机温度保持80～90℃后再操作。",
                trace,
            )

    async def review_moves_unsupported_critical_lines_out_of_guidance():
        output = AgentOutput(
            agent_name="fix_agent",
            message=(
                "## 操作步骤\n"
                "- 火花塞间隙：0.7～0.9 mm。\n"
                "- 安装扭矩：20 ± 2 N·m（18～22 N·m）。\n"
                "- 每10,000 km或6个月检查一次。请记录检查日期。\n"
                "- 发动机停止后进行拆卸。"
            ),
            tools_used=["knowledge_retrieval"],
            metadata={"react_trace": []},
            latency_ms=10,
        )
        grounding = {
            "unverified_claims": [{
                "sentence": "- 每10,000 km或6个月检查一次。",
                "critical_claims": ["10,000 km", "6个月"],
                "reason": "关键内容未找到明确依据",
            }, {
                "sentence": "- 安装扭矩：20 ± 2 N·m（18～22 N·m）。",
                "critical_claims": ["20 ± 2 N·m", "18～22 N·m"],
                "matched_claims": ["20 ± 2 N·m"],
                "unmatched_claims": ["18～22 N·m"],
                "reason": "关键内容未找到明确依据",
            }],
            "verified_claims": [{"sentence": "- 火花塞间隙：0.7～0.9 mm。"}],
            "total_claims": 2,
            "verified_count": 1,
            "unverified_count": 1,
        }
        with patch("agents.review_agent._GroundingCheck.run", new=AsyncMock(return_value=grounding)), patch(
            "agents.review_agent._GraphCheck.run",
            new=AsyncMock(return_value={"unverified_paths": [], "verified_paths": [], "total_paths": 0, "verified_count": 0, "unverified_count": 0}),
        ):
            return (await ReviewAgent().review(output)).model_dump()

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
            "name": "向量化失败时关键维修声明不得默认通过",
            "input": "embed_batch 抛异常",
            "expected": "unverified_count>0",
            "run": lambda: run_async(grounding_embedding_fail()),
            "check": lambda x: x["unverified_count"] > 0 and "无法确认" in x["note"],
        },
        {
            "name": "关键参数、周期和型号必须分别有原文依据",
            "input": "手册有间隙和扭矩，没有周期和型号",
            "expected": "周期、型号示例和温度均未验证",
            "run": lambda: run_async(grounding_critical_claims_require_literal_evidence()),
            "check": lambda x: x["verified_count"] == 2
            and x["unverified_count"] == 4
            and any("10,000" in c["sentence"] for c in x["unverified_claims"])
            and sum("CR7E" in c["sentence"] for c in x["unverified_claims"]) == 2,
        },
        {
            "name": "_GraphCheck 提取故障-方案对和 trace 结果",
            "input": "1. 轴承过热：立即更换轴承并检查润滑",
            "expected": "提取到 pair",
            "run": lambda: {
                "pairs": _GraphCheck._extract_pairs("1. 轴承过热：立即更换轴承并检查润滑"),
                "trace": _GraphCheck._parse_trace_results([{"action": "tool_call", "tool_calls": [{"name": "graph_search_java", "result_summary": '[{\"fault_name\":\"轴承过热\",\"solution_title\":\"立即更换轴承并检查润滑\"}]'}]}]),
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
            "name": "review() 将无依据关键内容移出正式指引并标明安全提示来源",
            "input": "周期无依据且回答触发发动机安全规则",
            "expected": "正式段无周期，待确认段含周期，安全提醒有来源声明",
            "run": lambda: run_async(review_moves_unsupported_critical_lines_out_of_guidance()),
            "check": lambda x: "10,000" not in x["message"].split("## 待确认信息")[0]
            and "20 ± 2 N·m" in x["message"].split("## 待确认信息")[0]
            and "## 待确认信息" in x["message"]
            and "10,000" in x["message"].split("## 待确认信息")[1]
            and "18～22 N·m" in x["message"].split("## 待确认信息")[1]
            and "## 系统通用安全提醒" in x["message"]
            and "不代表当前手册原文" in x["message"],
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
