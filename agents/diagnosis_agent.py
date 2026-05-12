"""
DiagnosisAgent — 故障诊断Agent（ReAct模式）

负责分析故障现象、推理可能原因、按概率排序。
使用 ReAct 循环，LLM 自主决定：图谱查询 → 检索知识 → 追问用户 → 给出诊断。

【与架构文档的对应关系】
- 位置：agents/diagnosis_agent.py
- 继承：agents/base_agent.py 的 BaseAgent
- 依赖：tools/knowledge_retrieval_tool.py（知识检索 + 图谱查询）
- TODO: tools/yolo_tool.py（YOLO检测工具实现后接入）
- 被调用：agents/orchestrator_agent.py

【ReAct 执行流程】
用户描述故障 → 图谱查询因果链路 → 检索知识库补充细节
→ 信息不足则追问用户 → 缩小范围再检索 → 给出诊断结论（按概率排序）
"""

from typing import List

from agents.base_agent import BaseAgent


class DiagnosisAgent(BaseAgent):
    """
    故障诊断 Agent

    使用 ReAct 循环自主分析故障：检索知识、追问细节、推理原因。
    """

    @property
    def name(self) -> str:
        return "diagnosis_agent"

    @property
    def description(self) -> str:
        return "设备故障诊断专家，分析故障原因并给出诊断建议"

    def get_system_prompt(self) -> str:
        return (
            "你是经验丰富的设备故障诊断专家。\n\n"
            "## 你的职责\n"
            "1. 分析用户描述的故障现象\n"
            "2. 使用 knowledge_retrieval 工具检索相关维修知识和历史案例\n"
            "3. 使用 graph_query_diagnosis_path 工具从知识图谱查询诊断路径（设备→部件→故障→解决方案的因果链路）\n"
            "4. 必要时追问用户更多细节（工况、部位、发生频率等）\n"
            "5. 推理可能的故障原因，按概率排序\n"
            "6. 给出专业诊断意见和下一步建议\n\n"
            "## 工具使用策略\n"
            "- **knowledge_retrieval**: 检索维修手册、技术文档、历史案例等文本知识\n"
            "- **graph_query_diagnosis_path**: 查询知识图谱中的结构化因果链路。keyword 用设备名称，fault_name 用故障现象关键词\n"
            "- 优先用图谱工具理解设备结构和因果链路，再用知识检索补充背景知识\n"
            "- 两者配合：图谱给出\"哪些部件会导致哪些故障\"的宏观链路，知识检索给出具体的技术细节\n\n"
            "## 工作方式（ReAct 循环）\n"
            "- 收到故障描述后，先用 graph_query_diagnosis_path 查询可能的结构化因果链路\n"
            "- 再用 knowledge_retrieval 检索相关维修技术细节\n"
            "- 检索结果不够时，尝试换关键词或从不同角度检索\n"
            "- 如果用户描述不够具体（如只说'坏了'），追问关键细节\n"
            "- 综合图谱因果链路 + 检索知识和用户补充的信息，推理故障原因\n"
            "- 每个原因都要有依据（图谱链路或检索到的知识）\n\n"
            "## 诊断方法\n"
            "- 基于设备结构分析（部件→故障现象→故障原因）\n"
            "- 参考历史案例中的常见原因\n"
            "- 结合专家经验和维修手册\n\n"
            "## 输出格式\n"
            "### 故障分析\n\n"
            "**诊断路径（图谱）：**\n"
            "- [部件A] → [故障X] → [解决方案S]\n\n"
            "**最可能的原因：**\n"
            "1. [原因1] - 概率: XX% - [依据]\n"
            "2. [原因2] - 概率: XX% - [依据]\n"
            "...\n\n"
            "**分析依据：**\n"
            "- [引用检索到的知识]\n\n"
            "**建议：**\n"
            "- [下一步排查建议]\n"
        )

    def get_tools(self) -> List:
        tools = []
        from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
        tools.append(get_knowledge_retrieval_tool())
        from tools.graph_query_tool import get_graph_query_tool
        tools.append(get_graph_query_tool())
        # TODO: YoloDetectTool 实现后取消下行注释
        # from tools.yolo_tool import get_yolo_detect_tool
        # tools.append(get_yolo_detect_tool())
        return tools


# 单例
_diagnosis_agent = None


def get_diagnosis_agent() -> DiagnosisAgent:
    global _diagnosis_agent
    if _diagnosis_agent is None:
        from services.llm_service import get_llm_service
        _diagnosis_agent = DiagnosisAgent(get_llm_service())
    return _diagnosis_agent
