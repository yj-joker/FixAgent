"""
OrchestratorAgent — 调度中枢

负责：意图识别 → 任务分发 → 结果汇总。
是整个 Agent 体系的入口，所有用户请求经由此处路由到对应子 Agent。

【与架构文档的对应关系】
- 位置：agents/orchestrator_agent.py
- 依赖：agents/intention/（意图识别）、chains/orchestrator.py（路由映射）
- 下游：RetrievalAgent / DiagnosisAgent / GuidanceAgent（子Agent）

【执行流程】
1. 解析 mode（用户显式指定 > AI自动识别）
2. 若 mode=CHAT，调用 IntentionRecognizer 自动识别意图
3. 根据 IntentionType 映射为 AgentMode
4. 按 mode 路由到对应 handler / 子 Agent
5. 汇总结果返回 AgentOutput
"""

import time
from typing import Optional, AsyncIterator

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from agents.intention.recognizer import get_intention_recognizer, IntentionRecognizer
from chains.orchestrator import map_intention_to_mode
from schemas.models import AgentMode, IntentionType, IntentionResult
from services.llm_service import LLMService


class OrchestratorAgent(BaseAgent):
    """
    调度中枢 Agent

    继承 BaseAgent，覆盖 run() 实现意图识别 → 路由分发流程。
    子 Agent 未就绪时，对应模式返回"开发中"状态，不降级实现。

    【子Agent注入】
    retrieval_agent / diagnosis_agent / guidance_agent 初始为 None，
    各 Agent 实现后通过 set_xxx_agent() 方法注入。
    """

    def __init__(self, llm_service: LLMService):
        super().__init__(llm_service)
        self.recognizer = get_intention_recognizer()

        # 子 Agent 引用 —— 实现后注入
        self.retrieval_agent = None     # type: Optional[BaseAgent]
        self.diagnosis_agent = None     # type: Optional[BaseAgent]
        self.guidance_agent = None      # type: Optional[BaseAgent]

    # ==================== Agent 元信息 ====================

    @property
    def name(self) -> str:
        return "orchestrator"

    @property
    def description(self) -> str:
        return "调度中枢：意图识别 → 任务分发 → 结果汇总"

    def get_system_prompt(self) -> str:
        return (
            "你是 FixAgent 智能助手，一个专业的设备维修与故障诊断 AI 系统。"
            "你能够理解用户描述的设备问题，查询相关知识库，分析故障原因，"
            "并生成标准化的维修作业指引。"
            "请用简洁、专业、易懂的中文回复用户。"
        )

    # ==================== 子Agent注入 ====================

    def set_retrieval_agent(self, agent):
        """注入检索Agent（RetrievalAgent实现后调用）"""
        self.retrieval_agent = agent

    def set_diagnosis_agent(self, agent):
        """注入诊断Agent（DiagnosisAgent实现后调用）"""
        self.diagnosis_agent = agent

    def set_guidance_agent(self, agent):
        """注入指引Agent（GuidanceAgent实现后调用）"""
        self.guidance_agent = agent

    # ==================== 核心执行入口 ====================

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        调度执行入口，覆盖父类模板方法

        流程：
        1. 解析 mode 来源（用户显式指定优先）
        2. 自动识别意图（mode=CHAT时）
        3. 映射 IntentionType → AgentMode
        4. 按 mode 路由到对应 handler
        5. 汇总返回
        """
        start_time = time.time()

        # 1. 解析用户指定的 mode
        user_mode = self._resolve_mode(input_data)

        # 2. 意图识别
        intention_result = None
        if user_mode == AgentMode.CHAT:
            # 用户未显式指定 → AI自动识别意图
            intention_result = await self.recognizer.recognize(
                input_data.user_message
            )
            effective_mode = map_intention_to_mode(intention_result.intention)
        else:
            # 用户显式指定了非CHAT模式 → 直接使用
            effective_mode = user_mode

        # 3. 按模式路由
        output = await self._dispatch(effective_mode, input_data, intention_result)

        # 4. 设置耗时和外层元数据
        latency_ms = int((time.time() - start_time) * 1000)
        output.latency_ms = latency_ms
        output.metadata["effective_mode"] = effective_mode.value
        output.metadata["user_mode"] = user_mode.value
        if intention_result:
            output.metadata["confidence"] = intention_result.confidence
            output.metadata["reasoning"] = intention_result.reasoning

        return output

    async def run_stream(self, input_data: AgentInput) -> AsyncIterator[str]:
        """
        流式执行入口，覆盖父类方法

        CHAT模式：流式输出LLM生成的token
        其他模式：TODO 子Agent实现流式接口后接入
        """
        user_mode = self._resolve_mode(input_data)

        # 意图识别
        intention_result = None
        if user_mode == AgentMode.CHAT:
            intention_result = await self.recognizer.recognize(
                input_data.user_message
            )
            effective_mode = map_intention_to_mode(intention_result.intention)
        else:
            effective_mode = user_mode

        # 流式路由
        if effective_mode == AgentMode.CHAT:
            messages = self._build_messages(input_data)
            stream_iter = await self._call_llm(messages, stream=True)
            async for token in stream_iter:
                yield token
        else:
            # TODO: 子Agent实现 run_stream() 后，改为调用对应子Agent的流式方法
            # 例：if effective_mode == AgentMode.RETRIEVAL:
            #         async for token in self.retrieval_agent.run_stream(input_data):
            #             yield token
            yield f"[{effective_mode.value}] 模式正在开发中，当前仅支持对话模式。请使用 mode=chat 或直接对话。"

    # ==================== 内部方法 ====================

    def _resolve_mode(self, input_data: AgentInput) -> AgentMode:
        """
        从 context 中提取用户指定的 mode

        ChatRequest.mode 通过 context["mode"] 传入。
        默认值为 CHAT，表示"未显式指定，需自动识别"。

        Returns:
            用户指定的 AgentMode，默认 CHAT
        """
        if input_data.context and "mode" in input_data.context:
            mode_str = input_data.context["mode"]
            try:
                return AgentMode(mode_str)
            except ValueError:
                return AgentMode.CHAT
        return AgentMode.CHAT

    async def _dispatch(
        self,
        mode: AgentMode,
        input_data: AgentInput,
        intention_result: Optional[IntentionResult]
    ) -> AgentOutput:
        """
        按模式分发到对应的 handler

        Args:
            mode: 执行模式
            input_data: Agent 输入
            intention_result: 意图识别结果（mode=CHAT时有值）

        Returns:
            AgentOutput
        """
        if mode == AgentMode.CHAT:
            return await self._execute_chat(input_data, intention_result)
        elif mode == AgentMode.RETRIEVAL:
            return await self._execute_retrieval(input_data, intention_result)
        elif mode == AgentMode.DIAGNOSIS:
            return await self._execute_diagnosis(input_data, intention_result)
        elif mode == AgentMode.GUIDANCE:
            return await self._execute_guidance(input_data, intention_result)
        elif mode == AgentMode.FULL:
            return await self._execute_full_pipeline(input_data, intention_result)
        else:
            return await self._execute_chat(input_data, intention_result)

    # ==================== 各模式 handler ====================

    async def _execute_chat(
        self,
        input_data: AgentInput,
        intention_result: Optional[IntentionResult]
    ) -> AgentOutput:
        """
        CHAT 模式 —— 直接 LLM 对话

        使用父类的 _build_messages → _call_llm → _process_response 流程。
        """
        messages = self._build_messages(input_data)
        response = await self._call_llm(messages, stream=False)

        intention_str = intention_result.intention.value if intention_result else None
        return self._process_response(
            raw_response=response,
            tools_used=[],
            metadata={"mode": AgentMode.CHAT.value},
            intention=intention_str
        )

    async def _execute_retrieval(
        self,
        input_data: AgentInput,
        intention_result: Optional[IntentionResult]
    ) -> AgentOutput:
        """
        RETRIEVAL 模式 —— 从知识库检索相关文档

        TODO: RetrievalAgent 实现后，取消下方注释，删除占位返回。
        """
        # TODO: RetrievalAgent 实现后启用
        # return await self.retrieval_agent.run(input_data)

        intention_str = intention_result.intention.value if intention_result else None
        return AgentOutput(
            agent_name=self.name,
            message="知识检索功能正在开发中，当前仅支持对话模式。请使用 mode=chat 或将问题直接发送给我。",
            intention=intention_str,
            tools_used=[],
            metadata={"mode": AgentMode.RETRIEVAL.value, "status": "not_implemented"}
        )

    async def _execute_diagnosis(
        self,
        input_data: AgentInput,
        intention_result: Optional[IntentionResult]
    ) -> AgentOutput:
        """
        DIAGNOSIS 模式 —— 故障诊断分析

        TODO: DiagnosisAgent 实现后，取消下方注释，删除占位返回。
        """
        # TODO: DiagnosisAgent 实现后启用
        # return await self.diagnosis_agent.run(input_data)

        intention_str = intention_result.intention.value if intention_result else None
        return AgentOutput(
            agent_name=self.name,
            message="故障诊断功能正在开发中，当前仅支持对话模式。请使用 mode=chat 或将问题直接发送给我。",
            intention=intention_str,
            tools_used=[],
            metadata={"mode": AgentMode.DIAGNOSIS.value, "status": "not_implemented"}
        )

    async def _execute_guidance(
        self,
        input_data: AgentInput,
        intention_result: Optional[IntentionResult]
    ) -> AgentOutput:
        """
        GUIDANCE 模式 —— 生成维修作业指引

        TODO: GuidanceAgent 实现后，取消下方注释，删除占位返回。
        """
        # TODO: GuidanceAgent 实现后启用
        # return await self.guidance_agent.run(input_data)

        intention_str = intention_result.intention.value if intention_result else None
        return AgentOutput(
            agent_name=self.name,
            message="维修指引功能正在开发中，当前仅支持对话模式。请使用 mode=chat 或将问题直接发送给我。",
            intention=intention_str,
            tools_used=[],
            metadata={"mode": AgentMode.GUIDANCE.value, "status": "not_implemented"}
        )

    async def _execute_full_pipeline(
        self,
        input_data: AgentInput,
        intention_result: Optional[IntentionResult]
    ) -> AgentOutput:
        """
        FULL 模式 —— 完整流程：检索 → 诊断 → 指引

        TODO: chains/pipeline.py 实现后，取消下方注释，删除占位返回。
        预期流程：
        1. RetrievalAgent.run()  → 从知识库检索相关文档
        2. DiagnosisAgent.run()  → 结合检索结果分析故障原因
        3. GuidanceAgent.run()   → 生成维修步骤
        4. 汇总三步结果返回
        """
        # TODO: chains/pipeline.py 实现后启用
        # from chains.pipeline import run_pipeline
        # return await run_pipeline(
        #     retrieval_agent=self.retrieval_agent,
        #     diagnosis_agent=self.diagnosis_agent,
        #     guidance_agent=self.guidance_agent,
        #     input_data=input_data
        # )

        intention_str = intention_result.intention.value if intention_result else None
        return AgentOutput(
            agent_name=self.name,
            message="完整诊断流程正在开发中，当前仅支持对话模式。请使用 mode=chat 或将问题直接发送给我。",
            intention=intention_str,
            tools_used=[],
            metadata={"mode": AgentMode.FULL.value, "status": "not_implemented"}
        )


# 单例
_orchestrator_agent: Optional[OrchestratorAgent] = None


def get_orchestrator_agent() -> OrchestratorAgent:
    """获取 OrchestratorAgent 单例"""
    global _orchestrator_agent
    if _orchestrator_agent is None:
        from services.llm_service import get_llm_service
        _orchestrator_agent = OrchestratorAgent(get_llm_service())
    return _orchestrator_agent
