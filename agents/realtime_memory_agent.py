"""
实时记忆更新 Agent

在每轮对话时轻量级检测用户是否纠正了事实或改变了偏好，
如果检测到就立即更新，不等待定时整合。

【为什么需要实时更新？】
定时整合（每4轮一次）有延迟：
- 用户在第1轮说了错误事实 → 第4轮才能修正
- 用户改变偏好 → 要等到下次整合才生效
- 中间这段时间，AI会用错误的事实/过时的偏好回答用户

【设计思路：轻量 + 快速】
不做完整整合（那太慢了40s+），而是用一个简短的 prompt
让 LLM 只判断"当前消息是否包含纠正/偏好变更"。
如果是 → 返回具体操作；如果不是 → 返回空操作。
整个过程 < 3秒，不阻塞主对话流。

【与完整整合的关系】
- 实时更新：处理"纠正旧事实"和"偏好变更"这两种紧急情况
- 定时整合：处理"提取新事实"、"新待办"、"摘要更新"等非紧急任务
- 两者互不冲突，实时更新是定时整合的补充，不是替代

【调用链路】
Java端每轮对话完成后 → 异步调用 POST /ai/memory/realtime_update →
本Agent快速判断 → 返回操作列表 → Java端立即执行更新
"""

import json
import logging
import time
from typing import Optional

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ========== 实时更新的 prompt（极简，只做判断） ==========
REALTIME_DETECT_PROMPT = """你是记忆实时更新检测器。你的唯一任务是判断用户最新消息是否包含以下两类操作：

## 1. 事实纠正
用户明确修正了之前说过的或AI认为正确的信息。
示例：
- "不对，故障码应该是E-5013不是E-4012"
- "我上次说错了，那个设备型号是X200不是X100"
- "更正一下，维修日期是3月15号"

## 2. 偏好变更
用户明确表达了新的偏好、改变了旧偏好、或者撤销了旧偏好。

### 2a. 新增/修改偏好（action="upsert"）
用户表达了新的持久性偏好：
- "以后回复用英文" → upsert
- "不要再写那么长的回复了，简短一点" → upsert
- "从现在开始每次回复都带上参考来源" → upsert

### 2b. 撤销/删除偏好（action="delete"）
用户明确表示不再需要某个之前的偏好：
- "不需要用英文了" → delete（撤销"用英文回复"）
- "算了，注释还是要的" → delete（撤销"不要写注释"）
- "回复长度你自己看着办吧，不用刻意简短" → delete（撤销"简短回复"）
- "不用每次都带参考来源了" → delete（撤销"带参考来源"）

## 不是纠正/偏好变更的情况（忽略）：
- 普通提问或讨论
- 补充新信息（不是修正旧的）
- 临时性请求（"这次帮我..."不是偏好变更）
- 仅针对当前这一次的要求（"这个用英文写" ≠ 永久偏好）

## 输出格式（JSON）：
```json
{
  "has_update": true/false,
  "fact_corrections": [
    {
      "wrong_content": "之前错误的事实描述（用于向量匹配定位旧事实）",
      "correct_content": "纠正后的正确事实",
      "keywords": "检索关键词"
    }
  ],
  "preference_changes": [
    {
      "action": "upsert 或 delete",
      "content": "偏好描述（upsert时为新偏好内容，delete时为要删除的偏好描述）",
      "category": "交互风格|格式要求|工作习惯|关注领域|其他",
      "preferenceCategory": 0,
      "sourceType": "explicit"
    }
  ]
}
```

如果没有任何纠正或偏好变更，返回：
```json
{"has_update": false, "fact_corrections": [], "preference_changes": []}
```

注意：
- 只检测用户的最新消息，不要分析历史消息
- 偏好变更的 sourceType 永远是 "explicit"（因为是用户明确说的）
- fact_corrections 中的 wrong_content 要尽量准确，用于后续向量匹配
- delete 操作的 content 要描述被撤销的那条偏好是什么（用于在数据库中模糊定位）
"""


# ========== Pydantic 输出模型 ==========
class FactCorrection(BaseModel):
    """事实纠正项"""
    wrong_content: str = Field(description="之前错误的事实描述，用于向量匹配定位")
    correct_content: str = Field(description="纠正后的正确事实")
    keywords: str = Field(default="", description="检索关键词")


class PreferenceChange(BaseModel):
    """
    偏好变更项

    action 说明：
    - "upsert": 新增或修改偏好（如果同类已存在则覆盖内容，不存在则新建）
    - "delete": 撤销/删除偏好（用户明确表示不再需要某个偏好，从数据库物理删除）
    """
    action: str = Field(default="upsert", description="操作类型: upsert=新增或修改, delete=删除")
    content: str = Field(description="偏好描述（upsert时为新内容，delete时为要删除的偏好描述）")
    category: str = Field(default="其他", description="分类")
    preferenceCategory: int = Field(default=0, description="0=用户级, 1=会话级")
    sourceType: str = Field(default="explicit", description="来源类型，实时检测永远是explicit")


class RealtimeUpdateResult(BaseModel):
    """实时更新检测结果"""
    has_update: bool = Field(default=False)
    fact_corrections: list[FactCorrection] = Field(default_factory=list)
    preference_changes: list[PreferenceChange] = Field(default_factory=list)


# ========== Agent 实现 ==========
class RealtimeMemoryAgent(BaseAgent):
    """
    实时记忆更新 Agent

    特点：
    - 不用 function calling（太慢），直接 JSON 输出
    - 不带工具，纯判断
    - 只看当前这一轮的用户消息 + AI回复
    - 2-3秒内完成
    """

    @property
    def name(self) -> str:
        return "realtime_memory_agent"

    @property
    def description(self) -> str:
        return "实时检测事实纠正和偏好变更"

    def get_system_prompt(self) -> str:
        return REALTIME_DETECT_PROMPT

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        执行实时检测

        input_data.context 应包含：
        - user_message: 用户本轮消息
        - ai_response: AI本轮回复（可选，用于判断AI是否用了错误信息）
        - recent_facts: 本轮注入的相关事实列表（可选，帮助LLM判断纠正了哪条）
        """
        start_time = time.time()

        context = input_data.context or {}
        user_msg = context.get("user_message", input_data.user_message)
        ai_response = context.get("ai_response", "")
        recent_facts = context.get("recent_facts", [])

        # 构建用户消息（给LLM看的上下文）
        user_content_parts = []
        if recent_facts:
            user_content_parts.append("【当前注入的历史事实】")
            for f in recent_facts:
                user_content_parts.append(f"- {f}")
            user_content_parts.append("")

        if ai_response:
            user_content_parts.append(f"【AI回复】{ai_response[:300]}")
            user_content_parts.append("")

        user_content_parts.append(f"【用户最新消息】{user_msg}")

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": "\n".join(user_content_parts)}
        ]

        try:
            # 直接调用 LLM chat 接口（无需 function calling），返回 JSON 格式
            response = await self.llm_service.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.get("content") or ""

            # 解析结果
            result = self._parse_result(content)

        except Exception as e:
            logger.warning(f"[realtime_memory] 检测失败: {e}")
            latency_ms = int((time.time() - start_time) * 1000)
            return AgentOutput(
                agent_name=self.name,
                message="检测失败",
                intention=None,
                tools_used=[],
                metadata={"status": "error", "has_update": False},
                latency_ms=latency_ms
            )

        latency_ms = int((time.time() - start_time) * 1000)

        # 如果有更新，执行向量库操作并收集被替代的旧事实ID
        superseded_fact_ids = []
        if result.has_update and result.fact_corrections:
            superseded_fact_ids = await self._apply_fact_corrections(result.fact_corrections, input_data.session_id)

        # 将被替代的旧事实 fact_id（向量库doc_id）返回给Java端
        # Java端用这些ID去MySQL中标记旧事实为 superseded
        result_dict = result.model_dump()
        result_dict["superseded_fact_ids"] = superseded_fact_ids

        return AgentOutput(
            agent_name=self.name,
            message="检测完成" if not result.has_update else "发现记忆更新",
            intention=None,
            tools_used=[],
            metadata={
                "status": "ok",
                "has_update": result.has_update,
                "result": result_dict,
                "latency_ms": latency_ms
            },
            latency_ms=latency_ms
        )

    def _parse_result(self, content: str) -> RealtimeUpdateResult:
        """解析LLM返回的JSON"""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            import re
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError(f"期望dict，实际为 {type(data).__name__}")

        return RealtimeUpdateResult(**data)

    async def _apply_fact_corrections(self, corrections: list[FactCorrection], session_id: str) -> list[str]:
        """
        立即在向量库中修正事实

        流程：
        1. 用 wrong_content 向量检索找到旧事实
        2. 从向量库删除旧事实
        3. 将 correct_content 作为新事实写入向量库
        4. 返回被删除的旧事实 doc_id 列表，供Java端在MySQL中标记 superseded

        Returns:
            被替代的旧事实 fact_id 列表（即向量库中的 doc_id）
            Java端用这些 fact_id 在 memory_fact 表中 WHERE fact_id IN (...) 标记为 superseded
        """
        from services.vector_service import get_vector_service
        from embeddings.text_embedding import get_text_embedding

        vector_service = get_vector_service()
        embedding_service = get_text_embedding()
        superseded_ids = []  # 收集被替代的旧事实ID

        for correction in corrections:
            try:
                # 1. 用错误内容的语义去检索旧事实
                old_vector = await embedding_service.embed(correction.wrong_content)
                old_results = vector_service.search(old_vector, top_k=3)

                # 找到匹配的旧事实（score越低越相似，COSINE距离）
                for old in old_results:
                    metadata = old.get("metadata", {})
                    if metadata.get("type") == "fact" and old.get("score", 1) < 0.3:
                        # 删除旧事实向量
                        doc_id = old.get("doc_id", "")
                        if doc_id:
                            vector_service.delete(doc_id)
                            superseded_ids.append(doc_id)
                            logger.info(f"[realtime] 删除旧事实向量: {doc_id}")
                        break

                # 2. 写入新的正确事实
                search_text = f"{correction.keywords} {correction.correct_content}" if correction.keywords else correction.correct_content
                new_vector = await embedding_service.embed(search_text)
                batch_ts = str(int(time.time() * 1000))
                new_doc_id = f"fact:{session_id}:rt_{batch_ts}"

                vector_service.add_vector(
                    doc_id=new_doc_id,
                    text=correction.correct_content,
                    vector=new_vector,
                    metadata={
                        "type": "fact",
                        "session_id": session_id,
                        "keywords": correction.keywords,
                        "source": "realtime_correction"  # 标记来源是实时纠正
                    }
                )
                logger.info(f"[realtime] 写入纠正事实: {correction.correct_content[:50]}")

            except Exception as e:
                logger.error(f"[realtime] 事实纠正失败: {e}")

        return superseded_ids


# ========== 单例 ==========
_realtime_agent: Optional[RealtimeMemoryAgent] = None


def get_realtime_memory_agent() -> RealtimeMemoryAgent:
    """获取实时记忆更新Agent单例"""
    global _realtime_agent
    if _realtime_agent is None:
        from services.llm_service import get_llm_service
        _realtime_agent = RealtimeMemoryAgent(llm_service=get_llm_service())
    return _realtime_agent
