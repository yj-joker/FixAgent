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
- 工具来源：tools/knowledge_retrieval_tool.py, tools/graph_query_tool.py
- 下游：ReviewAgent 对输出做最终校验
"""

import logging
from typing import List, Any

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
  - image_urls（图片URL列表，用户上传图片时传入，启用图文混合检索）
- 使用策略：优先使用，获取维修手册中的相关知识作为诊断和指引的依据。用户有图片时必须传入 image_urls

### graph_search_java
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

### graph_search_devices
从知识图谱中按关键字搜索设备节点。
- 适用：不确定设备全名时搜索设备列表，为诊断路径查询缩小范围
- 参数：keyword（搜索关键字）、limit（数量上限，默认10）
- 使用策略：当用户提到的设备名称模糊或不确定时，先搜索确认设备

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
→ graph_search_java 查询诊断路径（拆分 fault_description 和 component_description）
→ 综合分析后给出诊断结论

**图片故障诊断**（用户上传了故障图片）：
→ graph_search_java 查询（传入 image_urls + fault_description）
→ knowledge_retrieval 检索（传入 image_urls + query）
→ 综合图谱证据链和知识库内容给出诊断

**维修指引**（如"怎么更换气缸垫"）：
→ knowledge_retrieval 检索维修步骤
→ graph_search_java 确认故障-方案对应关系
→ 生成标准化维修步骤

**不确定设备**（如"那个什么泵坏了"）：
→ graph_search_devices 搜索匹配设备
→ 确认后再做诊断检索

**细节追问**（如"之前说的那个间隙值是多少来着"、"上次提到的维修步骤具体怎么做"）：
→ 先检查「相关历史事实」中是否有相关摘要
→ 如果摘要存在但缺少细节 → recall_conversation_detail 召回原始对话
→ 结合召回的原始对话内容给出详细回答

**闲聊/无关问题**：
→ 不调用工具，直接用自身知识友好回复，并引导用户描述设备问题

## 回答规范

1. **有据可依**：回答必须基于工具检索到的知识，不要凭空编造技术细节
2. **步骤化输出**：维修指引必须使用以下格式，每一步都要写全：
   ```
   ## 诊断结论
   （故障原因分析）

   ## 操作步骤
   ### Step 1: [操作名称]
   - **操作内容**：（具体做什么）
   - **所需工具**：（需要什么工具）
   - **安全注意**：（这一步的安全风险及防护措施）

   ### Step 2: [操作名称]
   ...
   ```
3. **安全优先**：涉及高压、高温、化学品、旋转部件、重物吊装等操作时，安全注意必须写具体（如佩戴绝缘手套、切断电源并挂牌、降温至常温等）
4. **设备类型处理**：用户如果提到了设备类型，直接使用；如果没提到，先从知识库检索确认设备类型再回答
5. **追问引导**：信息不足时主动追问（设备型号、故障现象、发生时间等）
6. **中文回复**：始终使用中文回答

## 多模态处理

如果用户上传了图片，图片URL会附在用户消息中。
- 调用 graph_search_java 时必须将图片URL通过 image_urls 参数传入，启用图片向量检索
- 调用 knowledge_retrieval 时也必须将图片URL通过 image_urls 参数传入，启用图文混合检索
- 同时结合图片内容和文本描述进行综合分析
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

    @property
    def name(self) -> str:
        return "fix_agent"

    @property
    def description(self) -> str:
        return "设备检修AI助手：知识检索、故障诊断、维修指引"

    def get_system_prompt(self) -> str:
        return FIX_AGENT_SYSTEM_PROMPT

    def get_tools(self) -> List[Any]:
        if self._tools is None:
            from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
            from tools.graph_java_tool import get_graph_java_tool
            from tools.graph_query_tool import get_graph_search_device_tool
            from tools.conversation_detail_tool import get_conversation_detail_tool

            self._tools = [
                get_knowledge_retrieval_tool(),
                get_graph_java_tool(),
                get_graph_search_device_tool(),
                get_conversation_detail_tool(),
            ]
        return self._tools

    def _customize_tool_kwargs(self, tool_name: str, kwargs: dict) -> dict:
        """为 recall_conversation_detail 注入 user_id"""
        if tool_name == "recall_conversation_detail" and hasattr(self, "_current_user_id"):
            kwargs["user_id"] = self._current_user_id or ""
        return kwargs

    async def run_with_react(self, input_data: AgentInput, max_iterations: int = 10) -> AgentOutput:
        """
        重写 ReAct 入口，提取 user_id 供 recall_conversation_detail 工具使用。
        """
        self._current_user_id = None
        if input_data.context and input_data.context.get("user_id"):
            self._current_user_id = str(input_data.context["user_id"])

        return await super().run_with_react(input_data, max_iterations)

    async def run_with_react_stream(self, input_data: AgentInput, max_iterations: int = 10):
        """重写流式 ReAct 入口，同样提取 user_id"""
        self._current_user_id = None
        if input_data.context and input_data.context.get("user_id"):
            self._current_user_id = str(input_data.context["user_id"])

        async for event in super().run_with_react_stream(input_data, max_iterations):
            yield event


# 单例
_fix_agent = None


def get_fix_agent() -> FixAgent:
    global _fix_agent
    if _fix_agent is None:
        from services.llm_service import get_llm_service
        _fix_agent = FixAgent(get_llm_service())
    return _fix_agent
