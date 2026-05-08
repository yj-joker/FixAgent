

# Agents 模块

## 模块概述

Agent是系统的**核心智能组件**，负责理解用户意图、分解任务、调用工具并返回结果。本模块基于LangChain Agent实现，提供：

- **意图识别**: 判断用户查询的类型（检索/诊断/作业/对话）
- **任务分解**: 将复杂任务拆分为可执行的子任务
- **工具编排**: 协调多个工具完成复杂流程
- **结果生成**: 汇总工具结果，生成最终响应

架构设计参考 `架构` 文件中的系统架构图，采用**Orchestrator模式**，由调度Agent统一协调专业Agent。

## 架构设计

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Orchestrator Agent (调度中枢)                    │
│                                                                  │
│  ┌───────────────┐                                             │
│  │  意图识别     │ ──► 识别用户意图类型                          │
│  └───────────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌───────────────┐                                             │
│  │  任务分解     │ ──► 分解为子任务                             │
│  └───────────────┘                                             │
│          │                                                      │
│          └──────────────────────────────────────────────────┐   │
│                                                             │   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │   │
│  │ RetrievalAgent  │  │ DiagnosisAgent  │  │ GuidanceAgent│ │   │
│  │                 │  │                 │  │             │ │   │
│  │ • 知识检索      │  │ • 原因分析      │  │ • 步骤生成  │ │   │
│  │ • 案例匹配      │  │ • 概率排序      │  │ • 合规校验  │ │   │
│  │ • 多模态融合    │  │ • 图谱推理      │  │ • 模板填充  │ │   │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │   │
│                           │                                   │   │
│                           └───────────────────────────────────┘   │
│                                       │                          │
│                                       ▼                          │
│                           ┌───────────────────┐                  │
│                           │   结果汇总输出    │                  │
│                           │ • 自然语言回复   │                  │
│                           │ • 结构化JSON    │                  │
│                           └───────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Agent列表

| Agent | 文件 | 职责 | 核心能力 |
|-------|------|------|---------|
| `orchestrator_agent` | orchestrator_agent.py | 调度中枢 | 意图识别、任务分解、结果汇总 |
| `retrieval_agent` | retrieval_agent.py | 知识检索 | 文本检索、图片检索、混合检索 |
| `diagnosis_agent` | diagnosis_agent.py | 故障诊断 | 原因分析、概率排序、图谱推理 |
| `guidance_agent` | guidance_agent.py | 作业指引 | 步骤生成、合规校验、模板填充 |
| `intention_recognizer` | intention/recognizer.py | 意图识别 | LLM轻量级识别、关键词兜底 |

## 意图识别模块

### 模块概述

意图识别是 Orchestrator Agent 的核心能力，负责判断用户查询类型并选择合适的处理路径。

采用 **LLM 轻量级识别 + 关键词兜底** 的混合方案：

```
用户消息
    │
    ├─── 方案A：LLM识别 ──► recognizer.py
    │         │  优先使用 LLM 判断意图
    │         │  成功 → 返回结果
    │         │  失败（超时/异常）→ 触发方案B
    │         ▼
    │    判断结果 ✓
    │
    └─── 方案B：关键词兜底 ──► fallback.py
              │  当方案A失败时启用
              │  快速、免费、简单
              ▼
         判断结果 ✓
```

### 意图类型

| 意图 | 值 | 触发场景举例 | 后续分发 |
|-----|---|-------------|---------|
| `QUERY_KNOWLEDGE` | query_knowledge | "什么是轴承？"、"查询知识库" | → RetrievalAgent |
| `TROUBLESHOOT` | troubleshoot | "坏了"、"不转"、"过热"、"原因" | → DiagnosisAgent |
| `SEEK_GUIDANCE` | seek_guidance | "怎么修"、"操作步骤"、"指引" | → GuidanceAgent |
| `SUBMIT_CASE` | submit_case | "提交案例"、"上传案例" | → RetrievalAgent |
| `GENERAL_CHAT` | general_chat | 其他闲聊内容 | → 直接LLM对话 |

### 文件结构

```
agents/intention/
├── __init__.py            # 模块说明
├── recognizer.py          # 意图识别器（LLM识别）
├── prompts.py             # 识别提示词模板
└── fallback.py            # 关键词兜底策略
```

### recognizer.py - 核心识别器

```python
# recognizer.py
"""
意图识别核心模块

采用 LLM 轻量级识别 + 关键词兜底的混合方案。
"""

class IntentionResult:
    """意图识别结果"""
    intention: IntentionType    # 意图类型
    confidence: float          # 置信度 0.0~1.0
    reasoning: str             # 识别理由

    def __init__(
        self,
        intention: IntentionType,
        confidence: float,
        reasoning: str
    ):
        self.intention = intention
        self.confidence = confidence
        self.reasoning = reasoning


async def recognize(message: str) -> IntentionResult:
    # 1. 尝试 LLM 识别
    try:
        result = await _llm_recognize(message)
        return result
    except Exception as e:
        # 2. LLM 失败时降级到关键词兜底
        return fallback_recognize(message)


# 返回值示例
{
    "intention": IntentionType.TROUBLESHOOT,  # 意图类型
    "confidence": 0.95,                         # 置信度
    "reasoning": "用户询问故障原因，符合 troubleshoot 模式"  # 识别理由
}
```

### prompts.py - 提示词模板

```python
# prompts.py
"""
意图识别提示词模板
"""

SYSTEM_PROMPT = """你是一个意图识别专家。
用户会输入一条消息，你需要判断他的意图。

可选意图：
- query_knowledge: 想了解某个概念或知识
- troubleshoot: 设备坏了，想知道原因
- seek_guidance: 想了解如何维修或操作
- submit_case: 想提交或上传案例
- general_chat: 闲聊

请只返回意图名称，不要解释。
"""

USER_TEMPLATE = "用户输入：{message}"
```

### fallback.py - 关键词兜底

```python
# fallback.py
"""
关键词兜底策略

当 LLM 不可用时，使用关键词匹配作为降级方案。
"""

KEYWORD_MAP = {
    IntentionType.TROUBLESHOOT: ["坏", "不转", "过热", "故障", "原因"],
    IntentionType.SEEK_GUIDANCE: ["怎么修", "步骤", "指引", "操作"],
    IntentionType.QUERY_KNOWLEDGE: ["什么是", "查询", "知识"],
    IntentionType.SUBMIT_CASE: ["提交案例", "上传案例"],
}

def fallback_recognize(message: str) -> IntentionResult:
    """扫描关键词，返回匹配的意图"""
    for intention, keywords in KEYWORD_MAP.items():
        if any(kw in message for kw in keywords):
            return IntentionResult(
                intention=intention,
                confidence=0.6,  # 兜底方案置信度较低
                reasoning=f"关键词匹配: {keywords}"
            )
    return IntentionResult(
        intention=IntentionType.GENERAL_CHAT,
        confidence=0.5,
        reasoning="无关键词匹配，默认闲聊"
    )
```

### 技术选型

| 方案 | 延迟 | 成本 | 准确率 |
|-----|------|------|--------|
| LLM识别 | ~300ms | 低 | 高 |
| 关键词匹配 | <10ms | 无 | 中（兜底用） |

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| Agent框架 | LangChain Agent | 生态完善、工具集成方便 |
| 提示词管理 | LangChain PromptTemplate | 结构化、易维护 |
| 输出解析 | Pydantic + LangChain OutputParser | 类型安全 |

## 项目中的实现

### base_agent.py - Agent基类

```python
# agents/base_agent.py
"""
Agent基类模块

定义所有Agent的基类和通用接口。
采用模板方法模式，统一Agent执行流程。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel, Field
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from tools import get_langchain_tools


class AgentInput(BaseModel):
    """Agent输入模型"""
    user_message: str = Field(description="用户消息")
    session_id: str = Field(description="会话ID")
    images: Optional[List[str]] = Field(default=None, description="图片列表")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文信息")


class AgentOutput(BaseModel):
    """Agent输出模型"""
    agent_name: str = Field(description="Agent名称")
    message: str = Field(description="回复消息")
    intention: Optional[str] = Field(default=None, description="识别的意图")
    tools_used: List[str] = Field(default_factory=list, description="使用的工具")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    latency_ms: int = Field(description="执行时间")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="原始响应")


class BaseAgent(ABC):
    """
    Agent基类

    所有专业Agent继承此类，实现：
    - get_system_prompt(): 返回角色定义提示词
    - get_tools(): 返回可用工具列表
    - customize_output(): 自定义输出处理
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent描述"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示词

        应包含：
        - Agent角色定义
        - 能力范围
        - 输出格式要求
        """
        pass

    def get_tools(self) -> List:
        """获取可用工具列表，默认返回所有工具"""
        return get_langchain_tools()

    def customize_output(self, raw_output: Dict[str, Any]) -> AgentOutput:
        """自定义输出处理，子类可覆盖"""
        return raw_output

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent执行入口（模板方法）

        执行流程：
        1. 组装提示词
        2. 调用LLM
        3. 处理输出
        4. 返回结果
        """
        import time
        start_time = time.time()

        # 1. 组装消息
        messages = self._build_messages(input_data)

        # 2. 调用LLM
        response = await self.llm.agenerate([messages])
        raw_content = response.generations[0][0].text

        # 3. 构建输出
        output = AgentOutput(
            agent_name=self.name,
            message=raw_content,
            latency_ms=int((time.time() - start_time) * 1000)
        )

        # 4. 自定义处理
        return self.customize_output(output)

    async def run_stream(self, input_data: AgentInput) -> AsyncIterator[str]:
        """
        Agent流式执行入口

        Yields:
            每个token
        """
        messages = self._build_messages(input_data)

        async for token in self.llm.agenerate_stream([messages]):
            yield token

    def _build_messages(self, input_data: AgentInput) -> List:
        """构建LLM消息列表"""
        messages = [
            SystemMessage(content=self.get_system_prompt())
        ]

        # 添加上下文（如有）
        if input_data.context:
            context_str = "\n\n## 上下文信息\n"
            for k, v in input_data.context.items():
                context_str += f"- {k}: {v}\n"
            messages.append(SystemMessage(content=context_str))

        # 添加图片信息（如有）
        if input_data.images:
            images_str = "\n\n## 用户上传的图片\n"
            for i, img in enumerate(input_data.images):
                images_str += f"- 图片{i+1}: {img}\n"
            messages.append(SystemMessage(content=images_str))

        # 添加用户消息
        messages.append(HumanMessage(content=input_data.user_message))

        return messages
```

### orchestrator_agent.py - 调度Agent

```python
# agents/orchestrator_agent.py
"""
Orchestrator Agent - 调度中枢

负责：
1. 意图识别 - 判断用户查询类型
2. 任务分解 - 将复杂任务拆分为子任务
3. 结果汇总 - 收集各Agent结果并返回
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import Field

from .base_agent import BaseAgent, AgentInput, AgentOutput
from models import AgentMode, IntentionType
from services.llm_service import get_llm_service


class OrchestratorAgent:
    """
    调度Agent

    是系统的入口Agent，负责：
    - 意图识别
    - 任务分发
    - 结果汇总
    """

    # 意图识别关键词
    INTENTION_KEYWORDS = {
        IntentionType.TROUBLESHOOT: [
            "故障", "坏了", "不转", "过热", "异响", "振动",
            "原因", "为什么", "怎么回事", "维修", "检修"
        ],
        IntentionType.QUERY_KNOWLEDGE: [
            "什么是", "查询", "知识", "了解", "学习",
            "标准", "规范", "手册", "说明书"
        ],
        IntentionType.SEEK_GUIDANCE: [
            "怎么修", "如何处理", "操作步骤", "指引",
            "指导", "教程", "流程", "顺序"
        ],
        IntentionType.SUBMIT_CASE: [
            "提交案例", "上传案例", "分享经验", "记录故障"
        ]
    }

    def __init__(self):
        self.llm_service = get_llm_service()
        # 初始化子Agent
        from .retrieval_agent import RetrievalAgent
        from .diagnosis_agent import DiagnosisAgent
        from .guidance_agent import GuidanceAgent

        self.retrieval_agent = RetrievalAgent(self.llm_service)
        self.diagnosis_agent = DiagnosisAgent(self.llm_service)
        self.guidance_agent = GuidanceAgent(self.llm_service)

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        执行调度

        流程：
        1. 意图识别
        2. 模式选择
        3. 子Agent执行
        4. 结果汇总
        """
        import time
        start_time = time.time()

        # 1. 意图识别
        intention = self._recognize_intention(input_data.user_message)
        mode = self._intention_to_mode(intention)

        # 2. 根据模式执行
        if mode == AgentMode.FULL:
            # 完整流程：检索 -> 诊断 -> 指引
            result = await self._run_full_pipeline(input_data)
        else:
            # 单Agent执行
            result = await self._run_single_agent(input_data, mode)

        # 3. 汇总结果
        result.intention = intention.value
        result.latency_ms = int((time.time() - start_time) * 1000)

        return result

    def _recognize_intention(self, message: str) -> IntentionType:
        """
        意图识别

        基于关键词匹配判断用户意图。
        后续可升级为LLM识别。
        """
        message_lower = message.lower()
        scores = {}

        for intention, keywords in self.INTENTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            scores[intention] = score

        # 最高分意图
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return IntentionType.GENERAL_CHAT

    def _intention_to_mode(self, intention: IntentionType) -> AgentMode:
        """意图转换为执行模式"""
        mapping = {
            IntentionType.QUERY_KNOWLEDGE: AgentMode.RETRIEVAL,
            IntentionType.TROUBLESHOOT: AgentMode.DIAGNOSIS,
            IntentionType.SEEK_GUIDANCE: AgentMode.GUIDANCE,
            IntentionType.SUBMIT_CASE: AgentMode.RETRIEVAL,
            IntentionType.GENERAL_CHAT: AgentMode.CHAT
        }
        return mapping.get(intention, AgentMode.CHAT)

    async def _run_single_agent(
        self,
        input_data: AgentInput,
        mode: AgentMode
    ) -> AgentOutput:
        """运行单个Agent"""
        if mode == AgentMode.RETRIEVAL:
            return await self.retrieval_agent.run(input_data)
        elif mode == AgentMode.DIAGNOSIS:
            return await self.diagnosis_agent.run(input_data)
        elif mode == AgentMode.GUIDANCE:
            return await self.guidance_agent.run(input_data)
        else:
            # CHAT模式：简单对话
            return await self._run_chat(input_data)

    async def _run_full_pipeline(self, input_data: AgentInput) -> AgentOutput:
        """运行完整流程：检索 -> 诊断 -> 指引"""
        tools_used = []
        all_context = {}

        # 1. 检索
        retrieval_result = await self.retrieval_agent.run(input_data)
        tools_used.append("retrieval")
        all_context["retrieval"] = retrieval_result.message

        # 2. 诊断
        diagnosis_input = AgentInput(
            user_message=input_data.user_message,
            session_id=input_data.session_id,
            images=input_data.images,
            context={
                **input_data.context or {},
                "检索结果": retrieval_result.message
            }
        )
        diagnosis_result = await self.diagnosis_agent.run(diagnosis_input)
        tools_used.append("diagnosis")
        all_context["diagnosis"] = diagnosis_result.message

        # 3. 指引
        guidance_input = AgentInput(
            user_message=input_data.user_message,
            session_id=input_data.session_id,
            images=input_data.images,
            context={
                **all_context,
                "诊断结果": diagnosis_result.message
            }
        )
        guidance_result = await self.guidance_agent.run(guidance_input)
        tools_used.append("guidance")

        # 4. 汇总
        summary = self._summarize_results(all_context)

        return AgentOutput(
            agent_name="orchestrator",
            message=summary,
            tools_used=tools_used,
            metadata={
                "mode": "full_pipeline",
                "context": all_context
            }
        )

    async def _run_chat(self, input_data: AgentInput) -> AgentOutput:
        """简单对话"""
        messages = [
            {"role": "system", "content": "你是一个专业的设备检修助手。"},
            {"role": "user", "content": input_data.user_message}
        ]

        result = await self.llm_service.chat(messages)

        return AgentOutput(
            agent_name="orchestrator",
            message=result["content"],
            tools_used=[]
        )

    def _summarize_results(self, context: Dict[str, Any]) -> str:
        """汇总各Agent结果"""
        summary = "## 综合分析结果\n\n"

        if "retrieval" in context:
            summary += "### 相关信息\n" + context["retrieval"] + "\n\n"
        if "diagnosis" in context:
            summary += "### 故障分析\n" + context["diagnosis"] + "\n\n"
        if "guidance" in context:
            summary += "### 维修指引\n" + context["guidance"] + "\n"

        return summary
```

### retrieval_agent.py - 检索Agent

```python
# agents/retrieval_agent.py
"""
Retrieval Agent - 知识检索Agent

负责：
1. 知识检索 - 从向量库检索相关内容
2. 案例匹配 - 查找相似历史案例
3. 多模态融合 - 文本+图片联合检索
"""

from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI

from .base_agent import BaseAgent, AgentInput, AgentOutput
from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool


class RetrievalAgent(BaseAgent):
    """
    检索Agent

    专门负责从知识库中检索相关信息。
    """

    @property
    def name(self) -> str:
        return "retrieval_agent"

    @property
    def description(self) -> str:
        return "专业的设备检修知识检索助手，帮助用户查找相关的技术资料和历史案例。"

    def get_system_prompt(self) -> str:
        return """
你是设备检修知识检索专家。

你的职责：
1. 理解用户的技术问题
2. 从知识库中检索最相关的内容
3. 提供准确、有用的信息

回答要求：
- 简洁明了，直接回答问题
- 优先引用权威资料（检修手册、技术规格）
- 列出信息来源（知识库ID或来源）
- 如果信息不足，明确说明

输出格式：
- 自然语言回答
- 关键点用列表呈现
- 标注信息来源
"""

    def get_tools(self) -> List:
        """返回检索相关的工具"""
        return [
            KnowledgeRetrievalTool(),
        ]

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """执行检索"""
        import time
        start_time = time.time()

        retrieval_tool = KnowledgeRetrievalTool()

        # 调用检索工具
        result = await retrieval_tool.execute(
            query=input_data.user_message,
            images=input_data.images,
            top_k=5
        )

        if result.status == "success":
            data = result.data
            message = self._format_retrieval_results(data)
        else:
            message = f"检索失败: {result.error}"

        return AgentOutput(
            agent_name=self.name,
            message=message,
            tools_used=["knowledge_retrieval"],
            latency_ms=int((time.time() - start_time) * 1000)
        )

    def _format_retrieval_results(self, data: Dict[str, Any]) -> str:
        """格式化检索结果"""
        results = data.get("results", [])
        total = data.get("total", 0)

        if not results:
            return "未找到相关知识，建议您详细描述问题或提供故障图片。"

        output = f"找到 {total} 条相关信息：\n\n"

        for i, item in enumerate(results, 1):
            content = item.get("content", "")[:200]  # 限制长度
            score = item.get("score", 0)
            output += f"### {i}. 相关知识 (相似度: {score:.2f})\n"
            output += f"{content}...\n\n"

        output += "\n---\n*如需了解更多信息，请提供更具体的问题描述。*"

        return output
```

### diagnosis_agent.py - 诊断Agent

```python
# agents/diagnosis_agent.py
"""
Diagnosis Agent - 故障诊断Agent

负责：
1. 故障分析 - 根据症状分析可能原因
2. 原因推理 - 利用图谱推理因果关系
3. 概率排序 - 按可能性排序原因
"""

from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI

from .base_agent import BaseAgent, AgentInput, AgentOutput
from tools.graph_query_tool import GraphQueryTool, GraphPathTool
from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool


class DiagnosisAgent(BaseAgent):
    """
    诊断Agent

    专门负责故障分析和原因推理。
    """

    @property
    def name(self) -> str:
        return "diagnosis_agent"

    @property
    def description(self) -> str:
        return "经验丰富的设备故障诊断专家，能够分析故障原因并给出概率排序。"

    def get_system_prompt(self) -> str:
        return """
你是经验丰富的设备故障诊断专家。

你的职责：
1. 分析用户描述的故障现象
2. 结合图谱知识推理可能原因
3. 按可能性排序各原因
4. 给出专业诊断意见

分析方法：
- 基于设备结构（部件-故障现象-故障原因）
- 参考历史案例
- 结合专家经验

输出要求：
1. 列出可能的故障原因（按概率排序）
2. 说明分析依据
3. 给出简要建议

格式：
### 故障分析

**最可能的原因：**
1. [原因1] - 概率: XX% - [依据]
2. [原因2] - 概率: XX% - [依据]
...

**分析依据：**
- [依据1]
- [依据2]

**建议：**
- [下一步建议]
"""

    def get_tools(self) -> List:
        """返回诊断相关工具"""
        return [
            KnowledgeRetrievalTool(),
            GraphQueryTool(),
            GraphPathTool(),
        ]

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """执行诊断"""
        import time
        start_time = time.time()
        tools_used = []

        # 1. 先检索相关知识
        retrieval_tool = KnowledgeRetrievalTool()
        retrieval_result = await retrieval_tool.execute(
            query=input_data.user_message,
            images=input_data.images,
            top_k=3
        )
        tools_used.append("knowledge_retrieval")

        # 2. 图谱扩展
        graph_tool = GraphQueryTool()
        graph_result = await graph_tool.execute(
            entity_name=self._extract_symptom(input_data.user_message),
            depth=2
        )
        tools_used.append("graph_query")

        # 3. 综合分析
        context_summary = self._build_context_summary(
            retrieval_result.data if retrieval_result.status == "success" else {},
            graph_result.data if graph_result.status == "success" else {}
        )

        # 4. LLM生成诊断
        diagnosis_message = await self._generate_diagnosis(
            input_data.user_message,
            context_summary
        )

        return AgentOutput(
            agent_name=self.name,
            message=diagnosis_message,
            tools_used=tools_used,
            metadata={
                "retrieval_result": retrieval_result.data if retrieval_result.status == "success" else None,
                "graph_result": graph_result.data if graph_result.status == "success" else None
            },
            latency_ms=int((time.time() - start_time) * 1000)
        )

    def _extract_symptom(self, message: str) -> str:
        """从消息中提取故障现象关键词"""
        # 简单关键词提取
        symptoms = ["过热", "异响", "振动", "不转", "漏油", "磨损", "松动"]
        for symptom in symptoms:
            if symptom in message:
                return symptom
        return message[:10]  # 默认取前10字

    def _build_context_summary(
        self,
        retrieval_data: Dict,
        graph_data: Dict
    ) -> str:
        """构建上下文摘要"""
        context = "## 相关知识\n"

        if retrieval_data.get("results"):
            context += "检索到的相关知识：\n"
            for item in retrieval_data["results"][:2]:
                context += f"- {item.get('content', '')[:100]}...\n"

        if graph_data.get("nodes"):
            context += "\n图谱关联实体：\n"
            for node in graph_data["nodes"][:5]:
                context += f"- {node.get('label')}: {node.get('properties', {}).get('name', '')}\n"

        return context

    async def _generate_diagnosis(
        self,
        user_message: str,
        context: str
    ) -> str:
        """生成诊断结果"""
        from services.llm_service import get_llm_service
        llm = get_llm_service()

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"## 用户问题\n{user_message}\n\n{context}"}
        ]

        result = await llm.chat(messages)
        return result["content"]
```

### guidance_agent.py - 作业Agent

```python
# agents/guidance_agent.py
"""
Guidance Agent - 作业指引Agent

负责：
1. 步骤生成 - 生成标准化的维修步骤
2. 合规校验 - 检查是否符合安全规范
3. 模板填充 - 填充标准作业模板
"""

from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI

from .base_agent import BaseAgent, AgentInput, AgentOutput
from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool


class GuidanceAgent(BaseAgent):
    """
    作业指引Agent

    专门负责生成标准化的维修作业步骤。
    """

    @property
    def name(self) -> str:
        return "guidance_agent"

    @property
    def description(self) -> str:
        return "标准作业流程合规审核员，帮助生成规范化的维修步骤和操作指引。"

    def get_system_prompt(self) -> str:
        return """
你是标准作业流程合规审核员。

你的职责：
1. 根据诊断结果生成标准化的维修步骤
2. 确保步骤符合安全规范
3. 提供清晰的作业指引

步骤要求：
1. 按正确顺序排列
2. 每步有明确动作和检查点
3. 包含安全注意事项
4. 标注所需工具和材料

格式要求：
使用编号列表，每步格式：
[N]. [动作描述]
   - 检查点: [具体检查内容]
   - 安全注意: [安全事项]
   - 工具: [所需工具]
"""

    def get_tools(self) -> List:
        """返回作业相关工具"""
        return [
            KnowledgeRetrievalTool(),
        ]

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """生成作业指引"""
        import time
        start_time = time.time()
        tools_used = []

        # 1. 检索相关知识
        symptom = self._extract_symptom(input_data.user_message)
        retrieval_tool = KnowledgeRetrievalTool()
        retrieval_result = await retrieval_tool.execute(
            query=f"{symptom} 维修步骤",
            top_k=2
        )
        tools_used.append("knowledge_retrieval")

        # 2. 生成指引
        context = self._build_context(
            retrieval_result.data if retrieval_result.status == "success" else {},
            input_data.context or {}
        )

        guidance_message = await self._generate_guidance(
            input_data.user_message,
            context
        )

        return AgentOutput(
            agent_name=self.name,
            message=guidance_message,
            tools_used=tools_used,
            metadata={
                "symptom": symptom
            },
            latency_ms=int((time.time() - start_time) * 1000)
        )

    def _extract_symptom(self, message: str) -> str:
        """提取故障现象"""
        symptoms = ["过热", "异响", "振动", "不转", "漏油", "磨损", "松动"]
        for symptom in symptoms:
            if symptom in message:
                return symptom
        return "general"

    def _build_context(
        self,
        retrieval_data: Dict,
        agent_context: Dict
    ) -> str:
        """构建上下文"""
        context = ""

        # 诊断结果
        if "diagnosis" in agent_context:
            context += f"## 诊断结果\n{agent_context['diagnosis']}\n\n"

        # 检索知识
        if retrieval_data.get("results"):
            context += "## 相关维修知识\n"
            for item in retrieval_data["results"][:2]:
                context += f"- {item.get('content', '')[:150]}...\n"

        return context

    async def _generate_guidance(
        self,
        user_message: str,
        context: str
    ) -> str:
        """生成作业指引"""
        from services.llm_service import get_llm_service
        llm = get_llm_service()

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"## 维修需求\n{user_message}\n\n{context}"}
        ]

        result = await llm.chat(messages)
        return result["content"]
```

## 使用示例

### 1. Orchestrator统一入口

```python
# api/main.py
from agents.orchestrator_agent import OrchestratorAgent
from schemas import AgentInput

orchestrator = OrchestratorAgent()

# 处理用户请求
request = ChatRequest(
    session_id="sess_123",
    message="电动机轴承过热怎么办？",
    mode=AgentMode.AUTO  # 自动识别意图
)

result = await orchestrator.run(AgentInput(
    user_message=request.message,
    session_id=request.session_id,
    images=request.images
))

print(result.message)
print(f"意图: {result.intention}")
print(f"耗时: {result.latency_ms}ms")
```

### 2. 直接调用专业Agent

```python
# 直接使用检索Agent
from agents.retrieval_agent import RetrievalAgent
from services.llm_service import get_llm_service

retrieval = RetrievalAgent(get_llm_service())

result = await retrieval.run(AgentInput(
    user_message="查找轴承过热的知识",
    session_id="sess_123"
))
```

### 3. 流式输出

```python
# 支持流式输出
async for token in orchestrator.run_stream(input_data):
    # 实时推送token
    yield f"data: {token}\n\n"
```

## 与Java/LangChain4j的对应关系

| Python Agent | Java Agent (LangChain4j) | 说明 |
|-------------|--------------------------|------|
| `OrchestratorAgent` | `@Agent` + 意图识别 | 调度中枢 |
| `RetrievalAgent` | `@Agent` + RAG | 知识检索 |
| `DiagnosisAgent` | `@Agent` + Tool | 故障诊断 |
| `GuidanceAgent` | `@Agent` + Prompt | 作业指引 |

**关键区别**：
- Python版本使用LangChain（Python）
- Java版本使用LangChain4j，直接在Java中实现Agent逻辑
- Python的工具（YOLO/SAM）通过HTTP暴露给Java调用

## 文件结构

```
agents/
├── __init__.py
├── README.md                    # 本文件
├── base_agent.py               # Agent基类
├── orchestrator_agent.py       # 调度Agent
├── retrieval_agent.py          # 检索Agent
├── diagnosis_agent.py          # 诊断Agent
└── guidance_agent.py           # 作业Agent
```

## 注意事项

1. **意图识别的局限性**: 当前基于关键词匹配，建议升级为LLM识别
2. **工具调用开销**: 每次Agent执行可能调用多个工具，需优化并行调用
3. **流式输出**: 复杂流程的流式输出需要特殊处理（多Agent协调）
4. **上下文长度**: 多轮对话注意上下文长度控制
5. **错误恢复**: 某个工具失败时应有fallback策略
