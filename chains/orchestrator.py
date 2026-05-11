"""
Orchestrator 路由链

负责意图→模式的映射和按模式分发执行，被 OrchestratorAgent 调用。

【与架构文档的对应关系】
- 位置：chains/orchestrator.py
- 职责：意图到执行模式的映射、根据模式路由到对应的 handler
- 被调用：agents/orchestrator_agent.py

【路由逻辑】
用户消息 → IntentionRecognizer 识别意图 → map_intention_to_mode() 映射为 AgentMode
  → OrchestratorAgent 根据 mode 调用对应的 handler / 子 Agent

【扩展点】
- 新增意图类型时，在 map_intention_to_mode() 中添加映射
- 子 Agent 实现后，在 OrchestratorAgent 对应 handler 中取消 TODO 注释
"""

from schemas.models import IntentionType, AgentMode


def map_intention_to_mode(intention: IntentionType) -> AgentMode:
    """
    将用户意图映射为 Agent 执行模式

    【映射规则】
    | 用户意图          | 执行模式    | 说明                     |
    |------------------|------------|-------------------------|
    | query_knowledge  | retrieval  | 查知识 → 向量检索         |
    | troubleshoot     | diagnosis  | 故障排查 → 诊断分析       |
    | seek_guidance    | guidance   | 寻求指导 → 生成维修步骤   |
    | submit_case      | retrieval  | 提交案例 → 检索相似案例   |
    | general_chat     | chat       | 闲聊 → 直接 LLM 对话     |

    Args:
        intention: IntentionRecognizer 识别出的用户意图

    Returns:
        对应的 AgentMode 枚举值
    """
    mapping = {
        IntentionType.QUERY_KNOWLEDGE: AgentMode.RETRIEVAL,
        IntentionType.TROUBLESHOOT: AgentMode.DIAGNOSIS,
        IntentionType.SEEK_GUIDANCE: AgentMode.GUIDANCE,
        IntentionType.SUBMIT_CASE: AgentMode.RETRIEVAL,
        IntentionType.GENERAL_CHAT: AgentMode.CHAT,
    }
    return mapping.get(intention, AgentMode.CHAT)
