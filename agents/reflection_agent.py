"""
用户画像反思 Agent

定期从用户的所有 active 事实中归纳高层画像，包括：
- device_expertise: 擅长/常修哪些设备
- fault_pattern: 常遇到的故障模式
- work_style: 偏好简短 vs 详细、偏安全 vs 快速
- safety_awareness: 安全意识水平
- overall: 综合画像摘要（200字以内）
"""

import json
import logging
import re
import time
from typing import Optional

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


REFLECTION_PROMPT = """你是用户画像分析师。根据提供的用户历史事实记录，归纳该用户的画像。

## 画像维度

### 1. device_expertise（设备专长）
用户常维修/讨论哪些设备类型？擅长哪些？用简洁列表描述。
示例："常修液压泵和电动机，熟悉HYD-3000系列参数，对变速箱相关问题较少涉及"

### 2. fault_pattern（常见故障模式）
用户经常遇到哪类故障？有哪些反复出现的问题模式？
示例："频繁遇到轴承过热问题，电机异响是常见主诉，液压系统泄漏出现过3次"

### 3. work_style（工作风格）
从用户提问和交互方式推断其偏好：
- 简洁 vs 详细：用户喜欢简短回复还是详细解释？
- 效率 vs 安全：用户更关心快速解决还是安全规程？
- 理论 vs 实操：用户更关注原理还是操作步骤？
示例："偏好简洁直接的回答，注重实操步骤，会主动确认安全注意事项"

### 4. safety_awareness（安全意识）
用户在维修过程中对安全的关注程度：
示例："安全意识较高，经常询问防护措施，会主动确认断电流程"

### 5. overall（综合画像）
200字以内的综合描述，概括用户特点。

## 输出格式（JSON）
```json
{
  "reflections": [
    {"type": "device_expertise", "content": "画像描述", "confidence": 0.85},
    {"type": "fault_pattern", "content": "画像描述", "confidence": 0.80},
    {"type": "work_style", "content": "画像描述", "confidence": 0.75},
    {"type": "safety_awareness", "content": "画像描述", "confidence": 0.70},
    {"type": "overall", "content": "综合画像200字以内", "confidence": 0.80}
  ]
}
```

## 注意
- 只从提供的事实中归纳，不要编造
- 事实数量少时，降低 confidence
- 如果某个维度事实不足（< 3条相关），该维度 confidence 设为 < 0.5
- 保持客观中性，不做价值判断
"""


class ReflectionItem(BaseModel):
    type: str = Field(description="画像类型")
    content: str = Field(description="画像内容")
    confidence: float = Field(default=0.70, description="置信度")


class ReflectionResult(BaseModel):
    reflections: list[ReflectionItem] = Field(default_factory=list)


class MemoryReflectionAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "memory_reflection_agent"

    @property
    def description(self) -> str:
        return "从用户历史事实归纳高层画像"

    def get_system_prompt(self) -> str:
        return REFLECTION_PROMPT

    async def run(self, input_data: AgentInput) -> AgentOutput:
        start_time = time.time()

        context = input_data.context or {}
        facts = context.get("facts", [])
        user_id = context.get("user_id", "unknown")

        if not facts:
            return AgentOutput(
                agent_name=self.name,
                message="无事实可用于反思",
                intention=None,
                tools_used=[],
                metadata={"status": "skip", "reason": "no_facts"},
                latency_ms=0
            )

        # 构建 LLM 输入
        fact_lines = []
        for i, f in enumerate(facts, 1):
            content = f.get("content", "") if isinstance(f, dict) else str(f)
            device = f.get("device_type", "") if isinstance(f, dict) else ""
            suffix = f" [设备:{device}]" if device else ""
            fact_lines.append(f"{i}. {content}{suffix}")

        user_content = f"## 用户 {user_id} 的历史事实（共{len(facts)}条）\n\n" + "\n".join(fact_lines)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_content}
        ]

        try:
            response = await self.llm_service.chat_with_tools(
                messages=messages,
                tools=[],
                tool_handlers={},
                response_format={"type": "json_object"}
            )
            content = response.get("content") or ""
            result = self._parse_result(content)
        except Exception as e:
            logger.error(f"[reflection] 反思失败: {e}")
            latency_ms = int((time.time() - start_time) * 1000)
            return AgentOutput(
                agent_name=self.name, message="反思失败",
                intention=None, tools_used=[],
                metadata={"status": "error", "error": str(e)},
                latency_ms=latency_ms
            )

        latency_ms = int((time.time() - start_time) * 1000)

        return AgentOutput(
            agent_name=self.name,
            message=f"完成用户 {user_id} 画像反思，{len(result.reflections)} 个维度",
            intention=None,
            tools_used=[],
            metadata={
                "status": "ok",
                "reflections": [r.model_dump() for r in result.reflections],
                "fact_count": len(facts),
                "latency_ms": latency_ms
            },
            latency_ms=latency_ms
        )

    def _parse_result(self, content: str) -> ReflectionResult:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        return ReflectionResult(**data)


_reflection_agent: Optional[MemoryReflectionAgent] = None

def get_reflection_agent() -> MemoryReflectionAgent:
    global _reflection_agent
    if _reflection_agent is None:
        from services.llm_service import get_llm_service
        _reflection_agent = MemoryReflectionAgent(llm_service=get_llm_service())
    return _reflection_agent
