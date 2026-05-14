import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from api.asr_api import asr_router
from schemas.request import ChatRequest, KnowledgeImportRequest, KnowledgeSearchRequest, MemoryConsolidateRequest
from schemas.response import ChatResponse, KnowledgeImportResponse, KnowledgeSearchResponse, BaseResponse, MemoryConsolidateResponse
from agents.orchestrator_agent import get_orchestrator_agent
from agents.retrieval_agent import get_retrieval_agent
from agents.diagnosis_agent import get_diagnosis_agent
from agents.guidance_agent import get_guidance_agent
from agents.memory_agent import get_memory_agent
from agents.base_agent import AgentInput
from services.vector_service import get_vector_service

logger = logging.getLogger(__name__)

# 全局日志配置（项目级别唯一一处）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


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


app.include_router(asr_router)

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
        logger.info(f"[chat] session={request.session_id} mode={request.mode.value} msg_len={len(request.message)}")
        result = await _get_orchestrator().run_with_context(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images,
            context={"mode": request.mode.value}
        )
        logger.info(f"[chat] session={request.session_id} done latency={result.latency_ms}ms")
        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention=result.intention,
            tools_used=result.tools_used if result.tools_used else None,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        logger.exception(f"[chat] session={request.session_id} error")
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

        try:
            async for event in orchestrator.run_stream(input_data):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception(f"[chat_stream] session={request.session_id} error")
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


"""检索"""
@app.post("/ai/retrieval", response_model=ChatResponse)
async def retrieval(request: ChatRequest) -> ChatResponse:
    """直接调用 RetrievalAgent，从向量库检索相关知识。"""
    try:
        logger.info(f"[retrieval] session={request.session_id} msg_len={len(request.message)}")
        result = await get_retrieval_agent().run_with_react(AgentInput(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images
        ))
        logger.info(f"[retrieval] session={request.session_id} done latency={result.latency_ms}ms")
        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            tools_used=result.tools_used if result.tools_used else None,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        logger.exception(f"[retrieval] session={request.session_id} error")
        raise HTTPException(status_code=500, detail=str(e))


"""诊断"""
@app.post("/ai/diagnosis", response_model=ChatResponse)
async def diagnosis(request: ChatRequest) -> ChatResponse:
    """直接调用 DiagnosisAgent，进行故障分析和原因推理。"""
    try:
        logger.info(f"[diagnosis] session={request.session_id} msg_len={len(request.message)}")
        result = await get_diagnosis_agent().run_with_react(AgentInput(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images
        ))
        logger.info(f"[diagnosis] session={request.session_id} done latency={result.latency_ms}ms")
        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention=result.intention,
            tools_used=result.tools_used if result.tools_used else None,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        logger.exception(f"[diagnosis] session={request.session_id} error")
        raise HTTPException(status_code=500, detail=str(e))


"""指引"""
@app.post("/ai/guidance", response_model=ChatResponse)
async def guidance(request: ChatRequest) -> ChatResponse:
    """直接调用 GuidanceAgent，生成标准化的维修作业步骤。"""
    try:
        logger.info(f"[guidance] session={request.session_id} msg_len={len(request.message)}")
        result = await get_guidance_agent().run_with_react(AgentInput(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images
        ))
        logger.info(f"[guidance] session={request.session_id} done latency={result.latency_ms}ms")
        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            tools_used=result.tools_used if result.tools_used else None,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        logger.exception(f"[guidance] session={request.session_id} error")
        raise HTTPException(status_code=500, detail=str(e))


"""完整流程"""
@app.post("/ai/pipeline", response_model=ChatResponse)
async def pipeline(request: ChatRequest) -> ChatResponse:
    """依次执行：检索 -> 诊断 -> 指引，返回综合分析结果。"""
    try:
        logger.info(f"[pipeline] session={request.session_id} msg_len={len(request.message)}")
        result = await _get_orchestrator().run_with_context(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images,
            context={"mode": "full"}
        )
        logger.info(f"[pipeline] session={request.session_id} done latency={result.latency_ms}ms")
        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            tools_used=result.tools_used if result.tools_used else None,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        logger.exception(f"[pipeline] session={request.session_id} error")
        raise HTTPException(status_code=500, detail=str(e))


"""知识导入"""
@app.post("/ai/knowledge/import", response_model=KnowledgeImportResponse)
async def knowledge_import(request: KnowledgeImportRequest) -> KnowledgeImportResponse:
    """
    文档导入并入库：解析 PDF → 向量化 → 存入 Redis 向量库

    使用场景：
    1. 系统部署初始化时，Java 后端上传赛题提供的维修手册 PDF
    2. 后续运营中导入新的技术文档

    处理流程：
    DocumentParserTool 解析 → TextEmbedding 向量化 → VectorService 入库
    文本块直接向量化，表格转为 markdown 文本向量化，图片用图注文本向量化。
    """
    from services.knowledge_service import get_knowledge_service

    try:
        svc = get_knowledge_service()
        result = await svc.import_document(
            file_url=request.file_url,
            file_type=request.file_type,
            category=request.category,
            tags=request.tags
        )
        logger.info(f"[knowledge_import] file={result['file_name']} "
                    f"pages={result['total_pages']} "
                    f"text={result['text_count']} img={result['image_count']} tbl={result['table_count']} "
                    f"latency={result['process_time_ms']}ms")
        return KnowledgeImportResponse(
            success=True,
            message=f"导入完成：{result['file_name']}，共 {result['total_pages']} 页",
            code=200,
            file_name=result["file_name"],
            total_pages=result["total_pages"],
            text_count=result["text_count"],
            image_count=result["image_count"],
            table_count=result["table_count"],
            sections=result["sections"],
            extraction_summary=result["extraction_summary"],
            process_time_ms=result["process_time_ms"]
        )
    except Exception as e:
        logger.exception(f"[knowledge_import] error")
        raise HTTPException(status_code=500, detail=str(e))


"""知识检索"""
@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """直接调用向量检索服务，返回 TopK 相关片段。"""
    import time

    try:
        logger.info(f"[knowledge_search] q={request.query[:50]} top_k={request.top_k}")
        svc = get_vector_service()

        # 构建 RediSearch 过滤表达式
        filter_parts = []
        if request.category:
            filter_parts.append(f"@category:{{{request.category}}}")
        if request.tags:
            tag_str = "|".join(request.tags)
            filter_parts.append(f"@tags:{{{tag_str}}}")
        filter_str = " ".join(f"({p})" for p in filter_parts) if filter_parts else None

        t0 = time.time()
        results = await svc.search_by_text(
            text=request.query,
            top_k=request.top_k,
            filter=filter_str
        )
        query_time_ms = int((time.time() - t0) * 1000)

        from schemas.models import VectorSearchResult
        data = [
            VectorSearchResult(
                id=r["doc_id"],
                score=r["score"],
                content=r.get("text", ""),
                metadata=r.get("metadata", {})
            )
            for r in results
        ]

        logger.info(f"[knowledge_search] found={len(data)} latency={query_time_ms}ms")
        return KnowledgeSearchResponse(
            success=True,
            message=f"检索完成，找到 {len(data)} 条结果",
            code=200,
            data=data,
            total=len(data),
            query_time_ms=query_time_ms
        )
    except Exception as e:
        logger.exception(f"[knowledge_search] error")
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

        logger.info(f"[memory_consolidate] session={request.session_id} msg_count={len(request.memoryMessages)}")
        result = await get_memory_agent().run(agent_input)
        logger.info(f"[memory_consolidate] session={request.session_id} done latency={result.latency_ms}ms")

        if result.metadata.get("status") == "error":
            error_type = result.metadata.get("error_type", "UnknownError")
            error_detail = result.metadata.get("error_detail", "记忆整理失败")
            logger.error(f"[memory_consolidate] session={request.session_id} agent_error=[{error_type}] {error_detail}")
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
    return JSONResponse(
        status_code=500,
        content=BaseResponse(
            success=False,
            message=str(exc),
            code=500
        ).model_dump()
    )