"""
Agent基类模块

定义所有Agent的基类和通用接口。
采用模板方法模式，统一Agent执行流程。

【与架构文档的对应关系】
- 位置：agents/base_agent.py
- 职责：AI核心组件的父类，定义统一执行流程
- 被继承：FixAgent、ReviewAgent、MemoryAgent

【设计模式】
- 模板方法模式：run() 定义统一执行流程，子类实现具体逻辑
- 单例模式：各子Agent由调用方管理生命周期，BaseAgent不负责实例化
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime

from pydantic import BaseModel, Field

from services.llm_service import LLMService
from config.settings import get_settings

logger = logging.getLogger(__name__)


class AgentInput(BaseModel):
    """Agent输入模型"""
    user_message: str = Field(description="当前轮用户消息（纯文本）")
    session_id: str = Field(description="会话ID")
    images: Optional[List[str]] = Field(default=None, description="图片列表")
    context: Optional[Dict[str, Any]] = Field(default=None, description="结构化上下文（摘要、事实、偏好、待办）")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="多轮对话历史[{'role':'user','content':'...'}]")


class AgentOutput(BaseModel):
    """Agent输出模型"""
    agent_name: str = Field(description="Agent名称")
    message: str = Field(description="回复消息")
    intention: Optional[str] = Field(default=None, description="识别的意图")
    tools_used: List[str] = Field(default_factory=list, description="使用的工具")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    latency_ms: int = Field(default=0, description="执行时间")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="原始响应")


class BaseAgent(ABC):
    """
    Agent基类

    所有专业Agent继承此类，实现：
    - name: Agent名称（抽象属性）
    - description: Agent描述（抽象属性）
    - get_system_prompt(): 返回角色定义提示词（抽象方法）
    - _execute(): 执行具体逻辑（抽象方法，可选覆盖）

    【执行流程（模板方法）】
    1. 构建消息列表（_build_messages）
    2. 调用LLM（_call_llm）
    3. 处理输出（_process_response）
    4. 返回结果（run）

    【使用示例】
    ```python
    class MyAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "my_agent"

        @property
        def description(self) -> str:
            return "我的Agent"

        def get_system_prompt(self) -> str:
            return "你是一个专业的..."

        async def _execute(self, input_data: AgentInput) -> Dict[str, Any]:
            # 具体执行逻辑
            return {"message": "结果"}

    agent = MyAgent(llm_service)
    result = await agent.run(input_data)
    ```
    """

    def __init__(self, llm_service: LLMService):
        """
        初始化BaseAgent

        Args:
            llm_service: LLM服务实例，用于调用大模型
        """
        self.llm_service = llm_service

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

        Returns:
            系统提示词字符串
        """
        pass

    def get_tools(self) -> List[Any]:
        """
        获取可用工具列表

        默认返回空列表，子类可覆盖以提供具体工具。

        Returns:
            工具列表
        """
        return []

    def _customize_tool_kwargs(self, tool_name: str, kwargs: dict) -> dict:
        """
        为特定工具注入额外参数的钩子方法

        在 ReAct 循环中，LLM 生成的工具调用参数会经过此方法处理，
        子类可覆盖以注入上下文信息（如 user_id）到特定工具中。

        Args:
            tool_name: 被调用的工具名
            kwargs: LLM 生成的原始参数

        Returns:
            处理后的参数字典
        """
        return kwargs

    def _build_messages(self, input_data: AgentInput) -> List[Dict[str, str]]:
        """
        构建LLM消息列表（支持多轮对话历史和结构化上下文）

        消息结构：
        1. system: 角色定义 + 上下文信息（摘要/事实/偏好/待办）
        2. 历史对话: 按user/assistant交替排列（多轮记忆）
        3. 当前user消息: 本轮用户输入

        Args:
            input_data: Agent输入数据

        Returns:
            消息列表，格式：[{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]
        """
        # ===== 1. 构建system prompt（角色 + 上下文） =====
        system_content = self.get_system_prompt()

        # 将结构化上下文注入system prompt，让LLM知道背景信息
        if input_data.context:
            context_parts = []

            # 之前的对话摘要
            if input_data.context.get("previous_summary"):
                context_parts.append(f"## 之前的对话摘要\n{input_data.context['previous_summary']}")

            # 相关历史事实（向量检索得到）
            if input_data.context.get("relevant_facts"):
                facts = input_data.context["relevant_facts"]
                facts_str = "\n".join(f"- {f.get('text', f) if isinstance(f, dict) else f}" for f in facts)
                context_parts.append(f"## 相关历史事实\n{facts_str}")

            # 用户偏好
            if input_data.context.get("user_preferences"):
                prefs = input_data.context["user_preferences"]
                prefs_str = "\n".join(f"- {p.get('content', p) if isinstance(p, dict) else p}" for p in prefs)
                context_parts.append(f"## 用户偏好\n{prefs_str}")

            # 会话偏好
            if input_data.context.get("session_preferences"):
                prefs = input_data.context["session_preferences"]
                prefs_str = "\n".join(f"- {p.get('content', p) if isinstance(p, dict) else p}" for p in prefs)
                context_parts.append(f"## 当前会话偏好\n{prefs_str}")

            # 未解决事项
            if input_data.context.get("unresolved_items"):
                items = input_data.context["unresolved_items"]
                items_str = "\n".join(f"- [{i.get('type', '未知')}] {i.get('content', i) if isinstance(i, dict) else i}" for i in items)
                context_parts.append(f"## 待解决事项\n{items_str}")

            if context_parts:
                system_content += "\n\n---\n以下是当前对话的背景信息，请据此回答用户问题：\n\n" + "\n\n".join(context_parts)

        messages = [{"role": "system", "content": system_content}]

        # ===== 2. 添加多轮对话历史（保持user/assistant角色） =====
        if input_data.conversation_history:
            for turn in input_data.conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # ===== 3. 添加当前轮用户消息 =====
        user_content = input_data.user_message

        # 添加图片信息（如有）—— 使用多模态消息格式
        if input_data.images:
            # 构建多模态 content：[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:..."}}]
            content_parts = [{"type": "text", "text": user_content}]
            for img in input_data.images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_content})

        return messages

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        model: Optional[str] = None
    ) -> Dict[str, Any] | AsyncIterator[str]:
        """
        调用LLM服务

        Args:
            messages: 消息列表
            stream: 是否流式输出
            model: 模型覆盖（有图片时传 VLM 模型）

        Returns:
            非流式：完整响应字典
            流式：异步生成器yield每个token
        """
        return await self.llm_service.chat(messages, stream=stream, model=model)

    def _process_response(
        self,
        raw_response: Dict[str, Any],
        tools_used: List[str],
        metadata: Dict[str, Any],
        intention: Optional[str] = None
    ) -> AgentOutput:
        """
        处理LLM原始响应，转换为AgentOutput

        Args:
            raw_response: LLM返回的原始响应
            tools_used: 使用的工具列表
            metadata: 附加元数据
            intention: 识别的用户意图

        Returns:
            AgentOutput对象
        """
        return AgentOutput(
            agent_name=self.name,
            message=raw_response.get("content", ""),
            intention=intention,
            tools_used=tools_used,
            metadata=metadata,
            raw_response=raw_response
        )

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent执行入口（模板方法）

        执行流程：
        1. 构建消息
        2. 调用LLM
        3. 处理输出
        4. 返回结果

        异常处理：任意环节失败返回友好提示，
                  具体错误信息记录在 metadata 中供排查。

        Args:
            input_data: Agent输入数据

        Returns:
            AgentOutput对象
        """
        start_time = time.time()

        try:
            # 1. 构建消息
            messages = self._build_messages(input_data)

            # 2. 有图片时切换为视觉模型
            model_override = None
            if input_data.images:
                model_override = get_settings().vlm_model
                logger.info(f"[{self.name}] 检测到图片，切换模型: {model_override}")

            # 3. 调用LLM
            response = await self._call_llm(messages, stream=False, model=model_override)

            # 3. 处理输出
            intention = input_data.context.get("intention") if input_data.context else None
            output = self._process_response(
                raw_response=response,
                tools_used=self.get_tools_used(input_data),
                metadata={"latency_ms": 0},
                intention=intention
            )
            output.latency_ms = int((time.time() - start_time) * 1000)
            return output

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return AgentOutput(
                agent_name=self.name,
                message="AI服务暂时不可用，请稍后重试",
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

    async def run_with_react(
        self,
        input_data: AgentInput,
        max_iterations: int = 10
    ) -> AgentOutput:
        """
        ReAct 模式执行入口

        使用 LLM function calling 实现 Thought → Action → Observation 循环。
        LLM 自主决定每步调用哪个工具、何时结束、何时追问用户。

        流程：
        1. 构建消息（系统提示词 + 用户输入）
        2. 收集子类提供的工具列表
        3. 调用 chat_with_tools() 进入 ReAct 循环
        4. LLM 返回最终文本响应后退出循环
        5. 包装为 AgentOutput 返回

        Args:
            input_data: Agent 输入数据
            max_iterations: 最大工具调用轮数（防止无限循环）

        Returns:
            AgentOutput 对象
        """
        start_time = time.time()

        try:
            # 1. 构建消息
            messages = self._build_messages(input_data)

            # 2. 获取工具列表，转为 OpenAI schema + handler 映射
            tools = self.get_tools()
            tool_schemas = [t.to_openai_schema() for t in tools]
            tool_handlers = {}
            for tool in tools:
                def _make_handler(t):
                    async def handler(**kwargs):
                        # 允许子类为特定工具注入额外参数
                        kwargs = self._customize_tool_kwargs(t.name, kwargs)
                        result = await t.run(**kwargs)
                        if result.success:
                            return result.data if result.data is not None else {"result": "success"}
                        else:
                            return {"error": result.error.message if result.error else "unknown error"}
                    return handler
                tool_handlers[tool.name] = _make_handler(tool)

            # 3. 有图片时自动切换为视觉模型
            model_override = None
            if input_data.images:
                model_override = get_settings().vlm_model
                logger.info(f"[{self.name}] 检测到图片，切换模型: {model_override}")

            # 4. ReAct 循环（chat_with_tools 内部自动处理）
            response = await self.llm_service.chat_with_tools(
                messages=messages,
                tools=tool_schemas,
                tool_handlers=tool_handlers,
                max_iterations=max_iterations,
                model=model_override
            )

            # 4. 记录使用的工具
            tools_used = [t.name for t in tools]

            # 5. 处理响应
            intention = input_data.context.get("intention") if input_data.context else None
            react_trace = response.get("trace", [])
            output = self._process_response(
                raw_response=response,
                tools_used=tools_used,
                metadata={
                    "execution_mode": "react",
                    "react_trace": react_trace,
                    "react_iterations": len(react_trace)
                },
                intention=intention
            )
            output.latency_ms = int((time.time() - start_time) * 1000)
            return output

        except RuntimeError as e:
            # 工具调用超出最大迭代次数
            latency_ms = int((time.time() - start_time) * 1000)
            return AgentOutput(
                agent_name=self.name,
                message="AI推理步骤超出限制，请尝试简化问题后重新提问。",
                intention=None,
                tools_used=[],
                metadata={
                    "status": "max_iterations_exceeded",
                    "error_detail": str(e),
                    "latency_ms": latency_ms
                },
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return AgentOutput(
                agent_name=self.name,
                message="AI服务暂时不可用，请稍后重试",
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

    async def run_with_react_stream(
        self,
        input_data: AgentInput,
        max_iterations: int = 10
    ) -> AsyncIterator[dict]:
        """
        ReAct 模式流式执行入口

        先执行 ReAct 循环（工具调用阶段），完成后将最终回答和工具调用
        追踪以结构化事件流的形式逐 token 输出。

        与 run_with_react 的区别：
        - run_with_react: 返回 AgentOutput，适合非流式 API
        - run_with_react_stream: yield 事件 dict，适合 SSE 流式 API

        事件格式：
        - {"event": "status", "data": {"stage": "...", "mode": "..."}}
        - {"event": "tool", "data": {"tool": "knowledge_retrieval"}}
        - {"event": "token", "data": {"content": "..."}}
        - {"event": "done", "data": {}}
        - {"event": "error", "data": {"message": "..."}}
        """
        start_time = time.time()

        yield {
            "event": "status",
            "data": {"stage": f"{self.description}，正在分析...", "mode": self.name}
        }

        try:
            output = await self.run_with_react(input_data, max_iterations)

            if output.metadata.get("status") == "error":
                yield {"event": "error", "data": {"message": output.message}}
                yield {"event": "done", "data": {}}
                return

            # 输出工具调用事件
            react_trace = output.metadata.get("react_trace", [])
            for step in react_trace:
                if step.get("action") == "tool_call":
                    for tc in step.get("tool_calls", []):
                        yield {
                            "event": "tool",
                            "data": {"tool": tc.get("name", "unknown")}
                        }

            # 逐字流式输出最终回答
            message = output.message
            for i in range(0, len(message)):
                yield {"event": "token", "data": {"content": message[i]}}
                if i % 15 == 0:
                    await asyncio.sleep(0)

            # 附加耗时和 react_trace（供下游验证管线使用）
            latency = output.latency_ms or int((time.time() - start_time) * 1000)
            yield {
                "event": "done",
                "data": {
                    "latency_ms": latency,
                    "react_trace": react_trace,
                    "tools_used": output.tools_used
                }
            }

        except Exception as e:
            yield {"event": "error", "data": {"message": str(e)}}
            yield {"event": "done", "data": {}}

    async def run_stream(self, input_data: AgentInput) -> AsyncIterator[str]:
        """
        Agent流式执行入口

        Args:
            input_data: Agent输入数据

        Yields:
            每个token
        """
        messages = self._build_messages(input_data)
        model_override = get_settings().vlm_model if input_data.images else None
        stream_iter = await self._call_llm(messages, stream=True, model=model_override)

        async for token in stream_iter:
            yield token

    def get_tools_used(self, input_data: AgentInput) -> List[str]:
        """
        获取本次执行使用的工具列表

        默认返回空列表，子类可覆盖以记录实际使用的工具。

        Args:
            input_data: Agent输入数据

        Returns:
            工具名称列表
        """
        return []

    async def run_with_context(
        self,
        user_message: str,
        session_id: str,
        images: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        便捷执行方法

        创建一个AgentInput并执行。

        Args:
            user_message: 用户消息
            session_id: 会话ID
            images: 图片列表（可选）
            context: 上下文信息（可选）

        Returns:
            AgentOutput对象
        """
        input_data = AgentInput(
            user_message=user_message,
            session_id=session_id,
            images=images,
            context=context
        )
        return await self.run(input_data)
