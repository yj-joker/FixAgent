"""
统一诊断 Agent（FixAgent）

持有全部工具的 ReAct Agent，在单次循环中自主决策工具调用。
替代原有的 Orchestrator + RetrievalAgent + DiagnosisAgent + GuidanceAgent 四层架构。

【核心能力】
- 知识检索：从向量知识库检索维修手册相关内容
- 故障诊断：通过图谱查询分析设备→部件→故障→解决方案链路
- 维修指引：综合检索和诊断结果生成标准化维修步骤

【执行模式】
- run_with_react()：非流式，返回 AgentOutput
- run_with_react_stream()：流式，yield SSE 事件

【调用链】
api/main.py → FixAgent.run_with_react() → chat_with_tools() → 工具调用循环 → 最终回答
              → ReviewAgent.run() → 审核 → 返回

【关联】
- 继承 BaseAgent，使用 run_with_react() 进入 ReAct 循环
- 工具来源：tools/knowledge_retrieval_tool.py, tools/graph_java_tool.py
- 下游：ReviewAgent 对输出做最终校验
"""

import json
import logging
import time
from typing import List, Any, Optional, Dict

from agents.base_agent import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


FIX_AGENT_SYSTEM_PROMPT = """你是一名专业的设备检修AI助手，具备知识检索、故障诊断和维修指引三大核心能力。

## 你的职责
1. **知识检索**：根据用户问题从维修手册知识库中检索相关内容，支持图文混合检索
2. **故障诊断**：分析设备故障现象，推理可能原因，给出诊断结论
3. **维修指引**：生成详细、步骤化的维修操作指引，每步必须包含安全注意事项

## 可用工具

### knowledge_retrieval
从向量知识库中检索与查询语义最相似的文档。支持纯文本查询和图文混合查询。
- 适用：用户询问设备知识、故障原因、维修方法等需要查阅资料的情况
- 参数：
  - query（查询文本，必填）
  - top_k（返回数量，默认5）
  - category（分类过滤，可选）
  - tags（标签过滤，可选）
  - document_id / device_type / manual_type / document_version / chunk_type（检索范围过滤，可选）
  - image_urls（图片URL列表，用户上传图片时传入，启用图文混合检索）
- 使用策略：优先使用，获取维修手册中的相关知识作为诊断和指引的依据。用户有图片时必须传入 image_urls
- 结果中的 retrieval_confidence、matched_types、retrieval_routes、relevance_score、rerank_score 是检索侧证据强度信号；
  retrieval_confidence=low 时不得把结果写成确定性技术结论，应追问、限定结论或明确说明依据不足

### java_graph_diagnosis_path
从设备检修知识图谱中查询诊断路径：设备→部件→故障→解决方案。
通过文本向量 + 图片向量 + 设备关键字三维度 OR 召回，按匹配度排序。
- 适用：需要分析设备故障的因果关系、查找已知解决方案
- 参数：
  - keyword（设备名称关键字，模糊匹配，可选）
  - fault_description（故障现象描述，语义匹配故障节点，可选）
  - component_description（部件描述，语义匹配部件节点，可选）
  - image_urls（故障图片URL列表，图片向量检索，可选）
  - limit（返回数量上限，默认10）
- 使用策略：
  - 从用户描述中拆分出故障现象和部件信息，分别传入 fault_description 和 component_description
  - 用户明确说了设备名称时传 keyword
  - 用户上传了图片时必须传入 image_urls
  - 四个参数至少传一个

### java_graph_device_search
从知识图谱中按关键字搜索设备节点。
- 适用：不确定设备全名时搜索设备列表，为诊断路径查询缩小范围
- 参数：keyword（搜索关键字）、limit（数量上限，默认10）
- 使用策略：当用户提到的设备名称模糊或不确定时，先搜索确认设备

### procedure_recommend
根据设备类型和故障信息推荐标准作业流程。
- 适用：用户明确提到设备类型，或描述故障并需要检修操作指引时
- 参数：
  - device_type（设备类型，必填）
  - maintenance_level（检修等级，可选）
  - fault_description（故障描述，可选，用于说明推荐上下文）
- 使用策略：
  - 用户说了设备类型时直接传入
  - 用户只描述故障时，先从 java_graph_diagnosis_path 结果中提取设备类型
  - 推荐结果可引导用户在检修任务模块中启动对应流程

### recall_conversation_detail
召回历史对话的原始细节。当你发现上下文中的事实摘要不够详细，无法回答用户追问的具体细节时使用。
- 适用：用户追问之前讨论过的具体代码片段、配置值、字段名、操作步骤、设备参数等细节
- 参数：keywords（检索关键词，从用户问题中提取核心术语）
- 使用策略：
  - 当「相关历史事实」中有相关摘要但缺少细节时，用事实中的关键词调用此工具
  - 不要每次都调用，只在用户明确追问细节且当前上下文不足时才用
  - 关键词要精准，如设备名+部件名、故障码、配置项名

## 工具调用策略

**简单知识查询**（如"什么是曲轴"）：
→ knowledge_retrieval 检索 → 直接回答

**故障诊断**（如"发动机过热怎么回事"）：
→ knowledge_retrieval 检索相关知识
→ java_graph_diagnosis_path 查询诊断路径（拆分 fault_description 和 component_description）
→ 综合分析后给出诊断结论

**图片故障诊断**（用户上传了故障图片）：
→ java_graph_diagnosis_path 查询（传入 image_urls + fault_description）
→ knowledge_retrieval 检索（传入 image_urls + query）
→ 综合图谱证据链和知识库内容给出诊断

**维修指引**（如"怎么更换气缸垫"）：
→ knowledge_retrieval 检索维修步骤
→ java_graph_diagnosis_path 确认故障-方案对应关系
→ procedure_recommend 推荐匹配的标准作业流程
→ 综合证据与推荐流程生成标准化维修步骤

**不确定设备**（如"那个什么泵坏了"）：
→ java_graph_device_search 搜索匹配设备
→ 确认后再做诊断检索

**细节追问**（如"之前说的那个间隙值是多少来着"、"上次提到的维修步骤具体怎么做"）：
→ 先检查「相关历史事实」中是否有相关摘要
→ 如果摘要存在但缺少细节 → recall_conversation_detail 召回原始对话
→ 结合召回的原始对话内容给出详细回答

**闲聊/无关问题**：
→ 不调用工具，直接用自身知识友好回复，并引导用户描述设备问题

## 回答规范

1. **有据可依**：回答必须基于工具检索到的知识，不要凭空编造技术细节
2. **步骤化输出**：维修指引必须使用普通文本格式，每一步都要写全：
   ```
   诊断结论：
   故障原因分析

   操作步骤：

   步骤一：操作名称
   操作内容：具体做什么
   所需工具：需要什么工具
   安全注意：这一步的安全风险及防护措施

   步骤二：操作名称
   ...
   ```
3. **安全优先**：涉及高压、高温、化学品、旋转部件、重物吊装等操作时，安全注意必须写具体（如佩戴绝缘手套、切断电源并挂牌、降温至常温等）
4. **设备类型处理**：用户如果提到了设备类型，直接使用；如果没提到，先从知识库检索确认设备类型再回答
5. **追问引导**：信息不足时主动追问（设备型号、故障现象、发生时间等）
6. **证据展示**：回答引用图片时保留图片证据及 image_url；引用表格和文本时说明来源页码或章节（若工具结果提供）
7. **中文回复**：始终使用中文回答
8. **未验证证据处理**：图谱中标记「⚠未验证(手册推断)」的解决方案来自手册自动抽取，尚未经真实检修验证。引用时必须说明「以下方案依据手册推断，建议现场确认」，不得表述为已验证的确定结论
9. **结构化诊断输出**：当回答包含多个故障排查项、优先级列表或原本适合表格展示的诊断结果时，最终回答必须输出一个 JSON 对象，格式如下：
   ```
   {
     "message": "简短说明",
     "diagnosisItems": [
       {
         "priority": "一级",
         "faultPart": "故障部位",
         "rootCause": "根本原因说明",
         "knowledgeBasis": "知识库依据"
       }
     ]
   }
   ```
   diagnosisItems 中每个对象必须包含 priority、faultPart、rootCause、knowledgeBasis 四个字段。不要输出 Markdown 表格。
10. **纯文本兼容**：禁止使用 emoji。禁止使用 Markdown 表格。禁止使用 #、*、- 作为标题、加粗或列表符号。禁止使用 | 作为表格分隔符。非 JSON 回答只能使用普通中文、中文序号、冒号和正常换行。
11. **段落与换行**：不允许把多个信息点挤在同一整段中。普通解释使用自然段；当内容包含编号、清单、选项、步骤或文件列表时，每一项必须单独换行。编号格式使用“1. 内容”“2. 内容”，不要把多个编号写在同一行。每个文件、步骤或选项之间用空行分隔，便于阅读。

## 多模态处理

如果用户上传了图片，图片URL会附在用户消息中。
- 调用 java_graph_diagnosis_path 时必须将图片URL通过 image_urls 参数传入，启用图片向量检索
- 调用 knowledge_retrieval 时也必须将图片URL通过 image_urls 参数传入，启用图文混合检索
- 同时结合图片内容和文本描述进行综合分析
- 工具调用必须通过系统提供的 function calling 完成，禁止在最终回答中展示工具参数 JSON、image_urls、top_k、component_description 等内部调用参数。
- 当用户只是在问“这是什么 / 是否同一类 / 是否是某设备配件”时，只回答识别、对比和所属系统；不要主动生成拆装步骤、维修建议、扭矩、间隙标准或更换周期，除非用户明确追问。
- 当用户是寒暄、自我介绍、学习交流或职业转型聊天时，用自然短段落回答，最多给一个追问；不要输出表格、大标题、长项目符号清单或系统安全提醒。只有用户明确要求检修步骤、参数表或正式方案时，才使用结构化标题和列表。
"""

FIX_AGENT_SYSTEM_PROMPT += """

## 知识库文件清单规则
当用户询问知识库中有哪些文件、PDF、文档或手册时，必须使用 knowledge_inventory 工具。
回答只能基于该工具返回的 MySQL 结构化清单。
每个条目只展示“手册名称”，不要额外展示“文件名”字段。
文件清单必须使用编号分段格式：编号和手册名称单独一行，统计信息单独一行；不同文件之间用空行分隔。
禁止根据 Redis 向量 chunk、检索片段、解析中间文件名或文档内容反推文件名、文件数量或已导入文档。
如果 knowledge_inventory 没有结果或不可用，必须说明暂时无法确认知识库文件列表，不得编造文件。
"""


class FixAgent(BaseAgent):
    """
    统一诊断 Agent

    持有全部工具（知识检索 + 图谱诊断 + 设备搜索），
    在 ReAct 循环中自主决策调用哪些工具、以什么顺序调用。

    替代原有的 Orchestrator 意图路由 + 3 个子 Agent 的架构，
    减少一轮 LLM 意图识别调用的延迟。
    """

    def __init__(self, llm_service):
        super().__init__(llm_service)
        self._tools = None
        self._current_intent_decision: Optional[Dict[str, Any]] = None
        self._current_allowed_tools: Optional[List[str]] = None

    @property
    def name(self) -> str:
        return "fix_agent"

    @property
    def description(self) -> str:
        return "设备检修AI助手：知识检索、故障诊断、维修指引"

    def get_system_prompt(self) -> str:
        prompt = FIX_AGENT_SYSTEM_PROMPT
        decision = self._current_intent_decision or {}
        policy = decision.get("policy") or {}
        if decision:
            prompt += (
                "\n\n## 当前意图路由\n"
                f"- intent: {decision.get('intent')}\n"
                f"- task_action: {decision.get('task_action')}\n"
                f"- response_style: {policy.get('response_style') or decision.get('answer_style')}\n"
                f"- evidence_level: {policy.get('evidence_level')}\n"
                f"- safety_level: {policy.get('safety_level')}\n"
                f"- allow_visual_answer_without_manual: "
                f"{policy.get('allow_visual_answer_without_manual', decision.get('allow_visual_answer_without_manual'))}\n"
                "\n请按当前意图调整回答风格。若当前工具不足以完成用户问题，"
                "输出 JSON："
                "`{\"status\":\"needs_more_tools\",\"needed_tools\":[\"工具名\"],\"reason\":\"原因\"}`。"
                "不要编造工具结果。"
            )
        return prompt

    def get_tools(self) -> List[Any]:
        if self._tools is None:
            from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
            from tools.knowledge_inventory_tool import get_knowledge_inventory_tool
            from tools.graph_java_tool import (
                get_java_graph_device_search_tool,
                get_java_graph_diagnosis_path_tool,
            )
            from tools.conversation_detail_tool import get_conversation_detail_tool
            from tools.procedure_recommend_tool import get_procedure_recommend_tool

            self._tools = [
                get_knowledge_retrieval_tool(),
                get_knowledge_inventory_tool(),
                get_java_graph_diagnosis_path_tool(),
                get_java_graph_device_search_tool(),
                get_conversation_detail_tool(),
                get_procedure_recommend_tool(),
            ]
        allowed = self._current_allowed_tools
        if allowed is None:
            return self._tools
        allowed_set = set(allowed)
        return [tool for tool in self._tools if tool.name in allowed_set]

    def _customize_tool_kwargs(self, tool_name: str, kwargs: dict) -> dict:
        """为 recall_conversation_detail 注入 user_id"""
        if tool_name == "recall_conversation_detail" and hasattr(self, "_current_user_id"):
            kwargs["user_id"] = self._current_user_id or ""
        if tool_name in ("knowledge_retrieval", "java_graph_diagnosis_path"):
            images = getattr(self, "_current_images", None)
            if images and not kwargs.get("image_urls"):
                kwargs["image_urls"] = images
        if tool_name == "knowledge_retrieval":
            enhanced_query = getattr(self, "_current_enhanced_query", None)
            if enhanced_query:
                query = str(kwargs.get("query") or "").strip()
                kwargs["query"] = enhanced_query if not query else f"{query} {enhanced_query}"
        return kwargs

    async def run_with_react(self, input_data: AgentInput, max_iterations: int = 10) -> AgentOutput:
        """
        重写 ReAct 入口，提取 user_id 供 recall_conversation_detail 工具使用。
        """
        self._current_user_id = None
        self._current_images = input_data.images or []
        self._current_enhanced_query = None
        self._current_intent_decision = None
        self._current_allowed_tools = None
        if input_data.context and input_data.context.get("user_id"):
            self._current_user_id = str(input_data.context["user_id"])
        if input_data.context and input_data.context.get("enhanced_retrieval_query"):
            self._current_enhanced_query = str(input_data.context["enhanced_retrieval_query"])
        if input_data.context and input_data.context.get("intent_decision"):
            self._current_intent_decision = dict(input_data.context["intent_decision"])
            policy = self._current_intent_decision.get("policy") or {}
            allowed_tools = policy.get("tool_scope") or self._current_intent_decision.get("allowed_tools")
            if isinstance(allowed_tools, list):
                self._current_allowed_tools = [str(name) for name in allowed_tools]

        if self._is_knowledge_inventory_intent():
            return await self._run_knowledge_inventory_direct()

        output = await super().run_with_react(input_data, max_iterations)
        if self._current_intent_decision:
            output.metadata["intent_decision"] = self._current_intent_decision

        react_status = self._parse_react_status(output.message)
        if react_status:
            output.metadata["react_status"] = react_status
            if react_status.get("status") == "needs_user_clarification":
                output.message = self._format_user_clarification_message(react_status)

        if self._needs_more_tools(output) and self._current_allowed_tools is not None:
            logger.info("[fix_agent] intent tool scope insufficient, rerunning once with full tools")
            self._current_allowed_tools = None
            rerun = await super().run_with_react(input_data, max_iterations)
            rerun.metadata["intent_decision"] = self._current_intent_decision or {}
            rerun.metadata["intent_rerun_reason"] = react_status.get("reason") if react_status else output.message
            if react_status:
                rerun.metadata["react_status_before_rerun"] = react_status
            rerun.metadata["intent_rerun_with_full_tools"] = True
            return rerun

        return output

    def _is_knowledge_inventory_intent(self) -> bool:
        decision = self._current_intent_decision or {}
        return decision.get("intent") == "knowledge_inventory"

    async def _run_knowledge_inventory_direct(self) -> AgentOutput:
        start_time = time.time()
        tools = self.get_tools()
        inventory_tool = next((tool for tool in tools if tool.name == "knowledge_inventory"), None)
        if inventory_tool is None:
            return AgentOutput(
                agent_name=self.name,
                message="暂时无法确认知识库文件列表：缺少 knowledge_inventory 工具。",
                tools_used=[],
                metadata={
                    "execution_mode": "knowledge_inventory_direct",
                    "intent_decision": self._current_intent_decision or {},
                    "status": "tool_missing",
                },
                latency_ms=int((time.time() - start_time) * 1000),
            )

        result = await inventory_tool.run()
        if not result.success:
            error_message = result.error.message if result.error else "unknown error"
            return AgentOutput(
                agent_name=self.name,
                message=f"暂时无法确认知识库文件列表：{error_message}",
                tools_used=["knowledge_inventory"],
                metadata={
                    "execution_mode": "knowledge_inventory_direct",
                    "intent_decision": self._current_intent_decision or {},
                    "status": "tool_error",
                    "error_detail": error_message,
                },
                latency_ms=int((time.time() - start_time) * 1000),
            )

        data = result.data or {}
        documents = data.get("documents") or []
        message = self._format_knowledge_inventory_message(documents)
        return AgentOutput(
            agent_name=self.name,
            message=message,
            tools_used=["knowledge_inventory"],
            metadata={
                "execution_mode": "knowledge_inventory_direct",
                "intent_decision": self._current_intent_decision or {},
                "knowledge_inventory_total": len(documents),
                "knowledge_inventory_source": data.get("source"),
            },
            latency_ms=int((time.time() - start_time) * 1000),
        )

    @staticmethod
    def _format_knowledge_inventory_message(documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "知识库中目前没有已导入的知识文件。"

        lines = [f"知识库中目前共有{len(documents)}个已导入的知识文件，具体如下："]
        for index, doc in enumerate(documents, start=1):
            name = str(doc.get("manual_name") or "").strip() or f"未命名手册 {index}"
            status = str(doc.get("status") or "-").strip()
            text_count = int(doc.get("text_count") or 0)
            image_count = int(doc.get("image_count") or 0)
            table_count = int(doc.get("table_count") or 0)
            created_at = str(doc.get("created_at") or "").strip()
            detail = f"含{text_count}段文本、{image_count}张图片、{table_count}个表格，状态为 {status}"
            if created_at:
                detail += f"，入库时间：{created_at}"
            detail += "。"

            lines.append("")
            lines.append(f"{index}. 《{name}》")
            lines.append(detail)

        lines.append("")
        lines.append("请告诉我你最关注的信息：")
        lines.append("")
        lines.append("1. 具体设备、部件或故障现象")
        lines.append("2. 维修步骤或安全注意事项")
        lines.append("3. 参数标准、图片内容或表格信息")

        return "\n".join(lines).strip()

    async def run_with_react_stream(self, input_data: AgentInput, max_iterations: int = 10):
        """重写流式 ReAct 入口，同样提取 user_id"""
        self._current_user_id = None
        self._current_images = input_data.images or []
        self._current_enhanced_query = None
        self._current_intent_decision = None
        self._current_allowed_tools = None
        if input_data.context and input_data.context.get("user_id"):
            self._current_user_id = str(input_data.context["user_id"])
        if input_data.context and input_data.context.get("enhanced_retrieval_query"):
            self._current_enhanced_query = str(input_data.context["enhanced_retrieval_query"])
        if input_data.context and input_data.context.get("intent_decision"):
            self._current_intent_decision = dict(input_data.context["intent_decision"])
            policy = self._current_intent_decision.get("policy") or {}
            allowed_tools = policy.get("tool_scope") or self._current_intent_decision.get("allowed_tools")
            if isinstance(allowed_tools, list):
                self._current_allowed_tools = [str(name) for name in allowed_tools]

        async for event in super().run_with_react_stream(input_data, max_iterations):
            yield event

    @staticmethod
    def _needs_more_tools(output: AgentOutput) -> bool:
        status = output.metadata.get("react_status") or FixAgent._parse_react_status(output.message)
        if status and status.get("status") == "needs_more_tools":
            return True
        message = (output.message or "").strip()
        return message.startswith("NEEDS_MORE_TOOLS:")

    @staticmethod
    def _parse_react_status(message: str) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if status not in {"needs_more_tools", "needs_user_clarification", "final_answer"}:
            return None
        needed_tools = data.get("needed_tools")
        if needed_tools is not None and not isinstance(needed_tools, list):
            data["needed_tools"] = [str(needed_tools)]
        return data

    @staticmethod
    def _format_user_clarification_message(status: Dict[str, Any]) -> str:
        parts: List[str] = []
        general_answer = str(status.get("general_answer") or "").strip()
        if general_answer:
            parts.append(general_answer)

        questions = status.get("questions") or []
        if questions:
            question_lines = []
            for question in questions[:3]:
                text = str(question or "").strip()
                if text:
                    question_lines.append(f"- {text}")
            if question_lines:
                parts.append("为了进一步查询知识库并给出更准确的判断，请补充：\n" + "\n".join(question_lines))

        if not parts:
            reason = str(status.get("reason") or "").strip()
            if reason:
                parts.append(f"还需要补充信息后才能继续判断：{reason}")
            else:
                parts.append("还需要补充车型、部件型号或故障现象后，我才能继续判断。")
        return "\n\n".join(parts)


# 单例
_fix_agent = None


def get_fix_agent() -> FixAgent:
    global _fix_agent
    if _fix_agent is None:
        from services.llm_service import get_llm_service
        _fix_agent = FixAgent(get_llm_service())
    return _fix_agent
