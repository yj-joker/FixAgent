from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import json
from schemas.request import ChatRequest, KnowledgeSearchRequest, MemoryConsolidateRequest
from schemas.response import ChatResponse, KnowledgeSearchResponse, BaseResponse, MemoryConsolidateResponse
from schemas.models import AgentMode
from agents.orchestrator_agent import get_orchestrator_agent
from agents.memory_agent import get_memory_agent
from agents.base_agent import AgentInput

# Agent 惰性初始化（首次请求时创建，避免启动时加载模型）
_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = get_orchestrator_agent()
    return _orchestrator


app = FastAPI(
    title="FixAgent AI Module",
    version="1.0.0",
    description="AI推理引擎：故障诊断、知识检索、作业指引"
)

# CORS 中间件，允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""对话，非流式响应"""
@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    核心对话接口 —— 经 OrchestratorAgent 智能路由

    流程：
    1. 如果 mode != CHAT（用户显式指定），直接用该模式
    2. 如果 mode == CHAT（默认），AI 自动识别意图 → 映射为对应模式
    3. 按模式路由到子Agent处理（开发中的模式返回提示信息）
    """
    try:
        result = await _get_orchestrator().run_with_context(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images,
            context={"mode": request.mode.value}
        )
        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention=result.intention,
            tools_used=result.tools_used if result.tools_used else None,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""对话，流式响应"""
@app.post("/ai/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        orchestrator = _get_orchestrator()

        input_data = AgentInput(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images,
            context={"mode": request.mode.value}
        )

        yield f"data: {json.dumps({'event': 'session_id', 'data': {'session_id': request.session_id}})}\n\n"

        async for token in orchestrator.run_stream(input_data):
            yield f"data: {json.dumps({'event': 'token', 'data': {'content': token}})}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


"""检索"""
@app.post("/ai/retrieval", response_model=ChatResponse)
async def retrieval(request: ChatRequest) -> ChatResponse:
    #直接调用 RetrievalAgent，从向量库检索相关知识。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""诊断"""
@app.post("/ai/diagnosis", response_model=ChatResponse)
async def diagnosis(request: ChatRequest) -> ChatResponse:
    #直接调用 DiagnosisAgent，进行故障分析和原因推理。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""指引"""
@app.post("/ai/guidance", response_model=ChatResponse)
async def guidance(request: ChatRequest) -> ChatResponse:
    #直接调用 GuidanceAgent，生成标准化的维修作业步骤。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""完整流程"""
@app.post("/ai/pipeline", response_model=ChatResponse)
async def pipeline(request: ChatRequest) -> ChatResponse:
    #依次执行：检索 -> 诊断 -> 指引，返回综合分析结果。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""知识检索"""
@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    #直接调用向量检索服务，返回 TopK 相关片段。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""记忆整理"""
@app.post("/ai/memory/consolidate", response_model=MemoryConsolidateResponse, response_model_by_alias=True)
async def memory_consolidate(request: MemoryConsolidateRequest) -> MemoryConsolidateResponse:
    """
    将多条原始对话压缩为结构化记忆摘要（滑动窗口 + 分类记忆）。

    Java 端在会话对话数达到阈值（如30条）时调用此接口：
    1. 从数据库取出已有的偏好摘要和未完成事项
    2. 取出最近30条未压缩的原始对话
    3. 打包为 MemoryConsolidateRequest
    4. 调用本接口 —— LLM 提取事实并自动检索向量库做冲突检测
    5. Java 端存储：new_facts → 事实库（向量表），偏好/未完成 → 摘要表，
       superseded_ids → 标记旧事实为无效
    """
    from datetime import datetime

    try:
        conv_dicts = [{"seq": i + 1, "role": m.role, "content": m.content} for i, m in enumerate(request.memoryMessages)]
        agent_input = AgentInput(
            user_message="请整理以下对话记录",
            session_id=request.session_id,
            context={
                "conversations": conv_dicts,
                "old_preferences": [p.model_dump() for p in request.memoryPreferenceVOList],
                "old_unresolved": [u.model_dump() for u in request.memoryUnresolvedVOList]
            }
        )

        result = await get_memory_agent().run(agent_input)

        if result.metadata.get("status") == "error":
            error_type = result.metadata.get("error_type", "UnknownError")
            error_detail = result.metadata.get("error_detail", "记忆整理失败")
            raise HTTPException(status_code=500, detail=f"[{error_type}] {error_detail}")

        return MemoryConsolidateResponse(
            success=True,
            message="整理完成",
            code=200,
            session_id=request.session_id,
            summary=result.metadata.get("summary", {}),
            original_count=len(request.memoryMessages),
            consolidated_at=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""全局异常处理"""
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return BaseResponse(
        success=False,
        message=str(exc),
        code=500
    )