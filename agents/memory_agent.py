"""
工作记忆整理 Agent

将多条原始对话记录压缩为结构化记忆摘要。
Java 端在对话达到阈值（如30条）时触发整理，调用本 Agent 提取关键信息。

采用单次 function calling 架构：
LLM 读取对话 → 提取候选事实 → 调用 search_similar_facts 检索 → 判断冲突 → 输出结果

【与其他模块的关系】
- 继承 BaseAgent，覆盖 run() 实现 function calling 流程
- 由 api/main.py 的 /ai/memory/consolidate 端点调用
- 调用 services/llm_service.py 的 chat_with_tools()
- 工具注册：tools/fact_retrieval_tool.py 提供向量检索能力
"""

import json
import logging
import re
import time

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from pydantic import ValidationError
from schemas.memory import MemorySummary

logger = logging.getLogger(__name__)


MEMORY_SYSTEM_PROMPT = """你是工作记忆整理助手。从对话记录中提取并整理记忆，输出结构化结果。

## 可用工具
- **search_similar_facts**: 在已有事实库中批量搜索与候选事实语义相似的历史事实。
  用法：先读完所有对话，识别出全部候选事实，提取每条事实的核心关键词（设备型号、错误码、故障部位等），然后一次性批量调用。

## 分类标准

### 事实（客观、已确认、不会改变）
属于事实：设备型号/参数/配置、已完成的诊断过程和结果、已验证的技术结论
不是事实（归为偏好）：主观评价、工作习惯、未完成的任务

### 偏好（主观、可改变）
属于偏好：交互风格要求、格式/视觉偏好、工作习惯/经验、关注领域

### 未完成事项（悬而未决）
属于未完成：问了但没得到答案的问题、进行中的任务、用户提出的待办
注意：一旦事项在新对话中得到解决，应转为事实，不再作为未完成事项

## 冲突判断规则（仅针对事实）
根据工具返回的相似事实判断：
- 无相似结果（score < 0.7）或结果为空 → 正常新增
- 有相似且内容相同 → 不重复添加
- 有相似、同话题但结论不同 → 以新对话中的结论为准，在 superseded_ids 中标记旧事实的 id
- 有相似且互相印证 → 合并为一条更完整的表述

偏好和未完成事项不调用工具，按以下规则处理：
- 同类别、同级别偏好有矛盾 → 以最新表述为准
- 未完成事项已解决 → 移入 resolved_items
- 已有偏好和未完成事项可能附带 preferenceCategory 和 status 字段：
  - preferenceCategory: 0=用户级（所有对话公用）, 1=会话级（仅本次会话有效）
  - status: active=进行中, superseded=已放弃。已放弃的事项无需处理

## 输出格式
严格按以下 JSON 输出，不要输出其他内容：
```json
{
  "new_facts": [
    {"content": "事实描述", "keywords": "检索用关键词", "source_seq_range": "3-5"}
  ],
  "superseded_ids": ["要标记为无效的旧事实ID"],
  "updated_preferences": [
    {"content": "偏好描述", "category": "交互风格|格式要求|工作习惯|关注领域|其他", "preferenceCategory": 0}
  ],
  "updated_unresolved": [
    {"content": "待解决描述", "type": "未答复问题|进行中任务|用户待办", "status": "active"}
  ],
  "resolved_items": ["已解决的事项描述"],
  "brief_summary": "200字以内的整体摘要"
}
```

- preferenceCategory: 0=用户级, 1=会话级。新发现的偏好根据内容判断：涉及个人习惯的为用户级，仅针对本次任务的为会话级
- status: active=进行中, superseded=已放弃。新发现的未完成事项默认为 active
"""


class MemoryAgent(BaseAgent):
    """
    工作记忆整理 Agent

    单次 function calling 架构：
    1. 构建消息（含已有偏好/未完成 + 新对话）
    2. 注册 search_similar_facts 工具
    3. 调用 LLM（自动处理工具调用循环）
    4. 解析 JSON 返回结构化数据
    """

    @property
    def name(self) -> str:
        return "memory_agent"

    @property
    def description(self) -> str:
        return "工作记忆整理Agent：将多条原始对话压缩为结构化摘要"

    def get_system_prompt(self) -> str:
        return MEMORY_SYSTEM_PROMPT

    def _format_conversations(self, conversations: list) -> str:
        """将对话列表格式化为 LLM 可读的文本块"""
        lines = ["## 新对话记录\n"]
        for item in conversations:
            role_label = "用户" if item.get("role") == "user" else "助手"
            seq = item.get("seq", "?")
            content = item.get("content", "")
            lines.append(f"[{seq}] {role_label}：{content}")
        return "\n".join(lines)

    def _build_messages(self, input_data: AgentInput) -> list:
        """构建消息列表，包含已有记忆上下文和待整理的新对话"""
        ctx = input_data.context or {}
        conversations = ctx.get("conversations", [])
        old_preferences = ctx.get("old_preferences", [])
        old_unresolved = ctx.get("old_unresolved", [])

        parts = []

        if old_preferences:
            parts.append("## 已有偏好（需与对话中的新偏好合并）\n")
            parts.append("| 偏好内容 | 分类 | 级别 |")
            parts.append("|----------|------|------|")
            for p in old_preferences:
                level = "用户级" if p.get('preferenceCategory') == 0 else "会话级"
                parts.append(f"| {p.get('content', '')} | {p.get('category', '其他')} | {level} |")
            parts.append("")

        if old_unresolved:
            parts.append("## 已有未完成事项（需根据新对话判断是否已解决）\n")
            parts.append("| 事项内容 | 类型 | 状态 |")
            parts.append("|----------|------|------|")
            for u in old_unresolved:
                status_label = "进行中" if u.get('status') == 'active' else "已放弃"
                parts.append(f"| {u.get('content', '')} | {u.get('type', '待办')} | {status_label} |")
            parts.append("")

        parts.append(self._format_conversations(conversations))

        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_content}
        ]

    def _extract_json(self, text: str) -> MemorySummary:
        """从 LLM 返回内容中提取 JSON（兼容 markdown 代码块包裹），Pydantic 校验"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        return MemorySummary(**data)

    async def _store_facts_to_vector(self, facts: list[dict], session_id: str):
        """
        将提取的事实写入 Redis 向量库

        每条事实生成向量后存入 Redis，供后续 MemoryAgent 整理时做冲突检索。
        存入的向量与知识库共用同一索引，通过 metadata.type="fact" 区分。

        Args:
            facts: LLM 输出的 new_facts 列表 [{"content": "", "keywords": "", "source_seq_range": ""}]
            session_id: 会话ID，用于 doc_id 前缀
        """
        if not facts:
            return

        from services.vector_service import get_vector_service
        from embeddings.text_embedding import get_text_embedding

        vector_service = get_vector_service()
        embedding_service = get_text_embedding()
        batch_ts = str(int(time.time() * 1000))

        for i, fact in enumerate(facts):
            content = fact.get("content", "")
            keywords = fact.get("keywords", "")
            search_text = f"{keywords} {content}" if keywords else content

            try:
                vector = await embedding_service.embed(search_text)
            except Exception:
                continue

            doc_id = f"fact:{session_id}:{batch_ts}_{i}"

            vector_service.add_vector(
                doc_id=doc_id,
                text=content,
                vector=vector,
                metadata={
                    "type": "fact",
                    "session_id": session_id,
                    "keywords": keywords,
                    "source_seq_range": fact.get("source_seq_range", "")
                }
            )

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        执行记忆整理（function calling 模式）

        流程：构建消息 → 注册工具 → chat_with_tools → 解析 JSON
        """
        start_time = time.time()

        messages = self._build_messages(input_data)

        from tools.fact_retrieval_tool import get_fact_retrieval_tool
        fact_tool = get_fact_retrieval_tool()
        tools = [fact_tool.to_openai_schema()]
        async def fact_handler(**kwargs):
            result = await fact_tool.run(**kwargs)
            if result.success:
                return result.data if result.data is not None else {"result": "success"}
            return {"error": result.error.message if result.error else "unknown error"}

        tool_handlers = {"search_similar_facts": fact_handler}

        try:
            response = await self.llm_service.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_handlers=tool_handlers,
                response_format={"type": "json_object"}
            )
            content = response.get("content") or ""
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return AgentOutput(
                agent_name=self.name,
                message="记忆整理失败，请稍后重试",
                intention=None,
                tools_used=[],
                metadata={
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_detail": str(e),
                    "latency_ms": latency_ms
                },
                latency_ms=latency_ms
            )

        latency_ms = int((time.time() - start_time) * 1000)

        try:
            summary = self._extract_json(content)
            # 将提取的事实写入 Redis 向量库（失败不影响主流程）
            if summary.new_facts:
                try:
                    await self._store_facts_to_vector(
                        [f.model_dump() for f in summary.new_facts],
                        input_data.session_id
                    )
                except Exception:
                    logger.exception("Failed to store facts to Redis vector DB")
        except (json.JSONDecodeError, ValueError, ValidationError, AttributeError, TypeError) as e:
            logger.warning(f"JSON parse failed, using fallback: {e}, raw content: {content[:100]}")
            summary = MemorySummary(brief_summary=content[:200] if content else "")

        return AgentOutput(
            agent_name=self.name,
            message=summary.brief_summary,
            intention=None,
            tools_used=["search_similar_facts"] if summary.new_facts else [],
            metadata={
                "summary": summary.model_dump(),
                "latency_ms": latency_ms
            },
            latency_ms=latency_ms
        )


# 单例
_memory_agent = None


def get_memory_agent() -> MemoryAgent:
    global _memory_agent
    if _memory_agent is None:
        from services.llm_service import get_llm_service
        _memory_agent = MemoryAgent(get_llm_service())
    return _memory_agent
