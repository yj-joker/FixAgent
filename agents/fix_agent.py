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
1. **知识检索**：根据用户问题从维修手册知识库中检索相关内容
2. **故障诊断**：分析设备故障现象，推理可能原因，给出诊断结论
3. **维修指引**：生成标准化的维修操作步骤，包含安全注意事项

## 可用工具

### knowledge_retrieval
从向量知识库中检索与查询文本语义最相似的文档。
- 适用：用户询问设备知识、故障原因、维修方法等需要查阅资料的情况
- 参数：query（查询文本）、top_k（返回数量，默认5）
- 使用策略：优先使用，获取维修手册中的相关知识作为诊断和指引的依据

### graph_query_diagnosis_path
从设备检修知识图谱中查询诊断路径：设备→部件→故障→解决方案。
- 适用：需要分析设备故障的因果关系、查找已知解决方案
- 参数：keyword（设备名称关键字）、fault_name（故障名称，可选）、limit（数量上限）
- 使用策略：当用户描述了具体设备和故障现象时使用，获取结构化的诊断链路

### graph_search_devices
从知识图谱中按关键字搜索设备节点。
- 适用：不确定设备全名时搜索设备列表，为诊断路径查询缩小范围
- 参数：keyword（搜索关键字）、limit（数量上限）
- 使用策略：当用户提到的设备名称模糊或不确定时，先搜索确认设备

## 工具调用策略

根据用户问题复杂度，灵活组合工具：

**简单知识查询**（如"什么是曲轴"）：
→ knowledge_retrieval 检索 → 直接回答

**故障诊断**（如"发动机过热怎么回事"）：
→ knowledge_retrieval 检索相关知识
→ graph_query_diagnosis_path 查询诊断路径
→ 综合分析后给出诊断结论

**维修指引**（如"怎么更换气缸垫"）：
→ knowledge_retrieval 检索维修步骤
→ graph_query_diagnosis_path 确认故障-方案对应关系
→ 生成标准化维修步骤

**不确定设备**（如"那个什么泵坏了"）：
→ graph_search_devices 搜索匹配设备
→ 确认后再做诊断检索

**闲聊/无关问题**：
→ 不调用工具，直接用自身知识友好回复，并引导用户描述设备问题

## 回答规范

1. **有据可依**：回答必须基于工具检索到的知识，不要凭空编造技术细节
2. **结构清晰**：故障诊断用编号列出可能原因，维修步骤用有序列表
3. **安全优先**：涉及高压、高温、化学品等操作时，必须给出安全警告
4. **追问引导**：信息不足时主动追问（设备型号、故障现象、发生时间等）
5. **中文回复**：始终使用中文回答

## 多模态处理

如果用户上传了图片，图片描述会附在用户消息中。请结合图片描述和文本内容进行综合分析。
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
            from tools.graph_query_tool import get_graph_query_tool, get_graph_search_device_tool

            self._tools = [
                get_knowledge_retrieval_tool(),
                get_graph_query_tool(),
                get_graph_search_device_tool(),
            ]
        return self._tools


# 单例
_fix_agent = None


def get_fix_agent() -> FixAgent:
    global _fix_agent
    if _fix_agent is None:
        from services.llm_service import get_llm_service
        _fix_agent = FixAgent(get_llm_service())
    return _fix_agent
