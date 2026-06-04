import json
import logging
import os
import time
from functools import partial
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 全局替换：所有 json.dumps 默认保留中文原文，避免 \uXXXX 乱码
# 使用方法：文件内所有 json.dumps 调用都用 json_dumps 替代
json_dumps = partial(json.dumps, ensure_ascii=False)
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from schemas.request import (
    ChatRequest,
    KnowledgeImportRequest,
    KnowledgeSearchRequest,
    MemoryConsolidateRequest,
    TemporaryPlanGenerateRequest,
)
from schemas.response import (
    BaseResponse,
    ChatResponse,
    KnowledgeCacheClearResponse,
    KnowledgeImportResponse,
    KnowledgeSearchResponse,
    KnowledgeStorageStatsResponse,
    MemoryConsolidateResponse,
    TemporaryPlanDraftResponse,
)
from agents.fix_agent import get_fix_agent
from agents.review_agent import get_review_agent
from agents.memory_agent import get_memory_agent
from agents.base_agent import AgentInput, AgentOutput
from services.vector_service import get_vector_service
from services.llm_service import get_llm_service
from services.image_summary_service import get_image_summary_service
from services.intent_router import get_intent_router
from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
from services.temporary_plan_service import get_temporary_plan_service
from config.settings import get_settings
from schemas.models import AgentMode

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


from contextlib import asynccontextmanager


def _normalize_diagnosis_item(item: dict) -> dict:
    return {
        "priority": item.get("priority", ""),
        "fault_part": item.get("faultPart", item.get("fault_part", "")),
        "root_cause": item.get("rootCause", item.get("root_cause", "")),
        "knowledge_basis": item.get("knowledgeBasis", item.get("knowledge_basis", "")),
    }


def _serialize_diagnosis_items(items: list[dict]) -> list[dict]:
    return [
        {
            "priority": item.get("priority", ""),
            "faultPart": item.get("fault_part", item.get("faultPart", "")),
            "rootCause": item.get("root_cause", item.get("rootCause", "")),
            "knowledgeBasis": item.get("knowledge_basis", item.get("knowledgeBasis", "")),
        }
        for item in items
    ]


def _extract_structured_chat_payload(message: str) -> tuple[str, list[dict] | None]:
    """
    从模型最终文本中提取结构化诊断结果。

    兼容两种形式：
    1. 纯 JSON：{"message":"...","diagnosisItems":[...]}
    2. 普通文本：原样返回，不填 diagnosisItems
    """
    text = (message or "").strip()
    if not text:
        return message, None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return message, None

    if not isinstance(payload, dict):
        return message, None

    raw_items = payload.get("diagnosisItems") or payload.get("diagnosis_items")
    if not isinstance(raw_items, list):
        return payload.get("message", message), None

    diagnosis_items = [
        _normalize_diagnosis_item(item)
        for item in raw_items
        if isinstance(item, dict)
    ]

    return payload.get("message", message), diagnosis_items or None

@asynccontextmanager
async def lifespan(application: FastAPI):
    # 启动：开启 MQ 消费者
    close_connection = None
    try:
        from mq.consumer import start_consumers
        from mq.connection import close_connection
        await start_consumers()
        logger.info("[启动] RabbitMQ 消费者已启动")
        import asyncio
        from services.kg_retry_sweeper import start_kg_retry_sweeper
        asyncio.create_task(start_kg_retry_sweeper())
        logger.info("[启动] KG重试sweeper已启动")
    except Exception as e:
        logger.warning("[启动] RabbitMQ 消费者启动失败（MQ不可用时降级为HTTP模式）: %s", e)
    yield
    # 关闭：断开 MQ 连接
    if close_connection is not None:
        await close_connection()

app = FastAPI(
    title="FixAgent AI Module",
    version="2.0.0",
    description="AI推理引擎：FixAgent 统一诊断 + 3层确定性校验",
    lifespan=lifespan,
)

_settings = get_settings()
os.makedirs(_settings.local_file_storage_dir, exist_ok=True)
app.mount(_settings.file_public_base_url, StaticFiles(directory=_settings.local_file_storage_dir), name="rag_files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _should_use_rag_fast_path(request: ChatRequest) -> bool:
    """保守触发简单 RAG 快速路径，避免普通诊断问题误绕过 ReAct。"""
    if request.images:
        return False
    context = request.context or {}
    if context.get("disable_fast_path") or context.get("force_react"):
        return False
    if request.mode == AgentMode.RETRIEVAL:
        return True
    message = request.message or ""
    return any(
        keyword in message
        for keyword in ("根据知识库", "查知识库", "知识库回答", "只查资料", "根据资料", "根据手册")
    )


IMAGE_ONLY_DEFAULT_MESSAGE = "请识别图片中的设备或部件，并结合知识库判断它可能属于什么系统。"


def _compact_text(parts: list[str]) -> str:
    seen = set()
    compacted = []
    for part in parts:
        text = " ".join(str(part or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        compacted.append(text)
    return " ".join(compacted)


async def _build_image_understanding(images: list[str], user_message: str) -> dict:
    summaries = []
    for image_url in images:
        try:
            summary = await get_image_summary_service().understand_user_image(image_url, user_message=user_message)
            if not summary:
                summary = await get_image_summary_service().summarize(
                    image_url=image_url,
                    caption=user_message,
                    context_before="",
                    context_after="",
                    section_title="用户上传图片",
                )
        except Exception as exc:
            logger.warning("[chat][image_understanding] image summary failed: %s", exc)
            summary = {
                "image_title": "用户上传图片",
                "image_summary": "用户上传了一张待识别的设备或部件图片。",
                "summary_source": "fallback_error",
            }
        summaries.append({"image_url": image_url, **summary})

    enhanced_query = _compact_text(
        [user_message]
        + [
            " ".join(
                str(item.get(key, ""))
                for key in ("image_title", "image_summary")
                if item.get(key)
            )
            + " "
            + " ".join(str(keyword) for keyword in item.get("keywords", []) if keyword)
            for item in summaries
        ]
    )
    return {
        "summaries": summaries,
        "enhanced_query": enhanced_query or IMAGE_ONLY_DEFAULT_MESSAGE,
    }


async def _prepare_chat_agent_input(request: ChatRequest) -> AgentInput:
    raw_message = request.message or ""
    effective_message = raw_message.strip() or IMAGE_ONLY_DEFAULT_MESSAGE
    context = dict(request.context or {})

    intent_decision = await get_intent_router().classify(
        raw_message,
        images=request.images,
        context=context,
    )
    context["intent_decision"] = intent_decision.model_dump()
    context["intention"] = intent_decision.intent

    if request.images and intent_decision.requires_image_understanding:
        image_understanding = await _build_image_understanding(request.images, effective_message)
        context["image_understanding"] = image_understanding
        context["enhanced_retrieval_query"] = image_understanding["enhanced_query"]
        context["original_user_message"] = raw_message

    return AgentInput(
        user_message=effective_message,
        session_id=request.session_id,
        images=request.images,
        conversation_history=request.conversation_history,
        context=context,
    )


def _evidence_item_to_text(item, index: int) -> str:
    data = item.model_dump() if hasattr(item, "model_dump") else item
    metadata = data.get("metadata") or {}
    source = data.get("id") or metadata.get("document_id") or f"evidence-{index}"
    score = data.get("score", "")
    content = data.get("content") or data.get("text") or ""
    page = metadata.get("page_number") or metadata.get("page")
    page_text = f", page={page}" if page else ""
    return f"[证据{index}] source={source}, score={score}{page_text}\n{content}"


async def _run_rag_fast_path(request: ChatRequest) -> AgentOutput | None:
    """执行 RAG -> 单次 LLM 生成的轻量链路；失败时返回 None 交给 ReAct 回退。"""
    total_t0 = time.time()
    retrieval_t0 = time.time()
    retrieval = await get_knowledge_retrieval_tool().run(
        query=request.message,
        top_k=5,
    )
    retrieval_ms = int((time.time() - retrieval_t0) * 1000)
    if not retrieval.success or not retrieval.data:
        logger.warning(
            "[chat][fast_path] session=%s retrieval failed_or_empty duration_ms=%s error=%s",
            request.session_id,
            retrieval_ms,
            retrieval.error,
        )
        return None

    evidence_items = retrieval.data
    evidence_text = "\n\n".join(
        _evidence_item_to_text(item, idx)
        for idx, item in enumerate(evidence_items, start=1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是设备检修知识库问答助手。必须基于给定知识库证据回答；"
                "证据不足时明确说明不足，不要编造参数、型号或操作步骤。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{request.message}\n\n"
                f"知识库证据：\n{evidence_text}\n\n"
                "请用中文回答，必要时列出依据和不确定点。"
            ),
        },
    ]

    llm_t0 = time.time()
    response = await get_llm_service().chat(messages=messages, temperature=0.1)
    llm_ms = int((time.time() - llm_t0) * 1000)
    total_ms = int((time.time() - total_t0) * 1000)

    trace = [{
        "iteration": 1,
        "action": "tool_call",
        "duration_ms": retrieval_ms,
        "tool_calls": [{
            "name": "knowledge_retrieval",
            "arguments": {"query": request.message, "top_k": 5},
            "result_summary": str(evidence_items)[:200],
            "result_data": [item.model_dump() if hasattr(item, "model_dump") else item for item in evidence_items],
        }],
    }]
    logger.info(
        "[chat][fast_path] session=%s retrieval_ms=%s llm_ms=%s total_ms=%s evidence_count=%s",
        request.session_id,
        retrieval_ms,
        llm_ms,
        total_ms,
        len(evidence_items),
    )

    return AgentOutput(
        agent_name="fix_agent",
        message=response.get("content", ""),
        tools_used=["knowledge_retrieval"],
        metadata={
            "execution_mode": "rag_fast_path",
            "react_trace": trace,
            "react_iterations": 1,
            "phase_timings_ms": {
                "retrieval": retrieval_ms,
                "llm_generation": llm_ms,
                "fast_path_total": total_ms,
            },
        },
        latency_ms=total_ms,
        raw_response=response,
    )


@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    核心对话接口 —— FixAgent ReAct 推理 + 3层确定性校验

    流程：
    1. FixAgent 通过 ReAct 循环自主决策工具调用
    2. 3层校验：检索依据校验 → 图谱路径校验 → 安全规则引擎
    3. 返回最终结果（含校验标注和安全补充）
    """
    try:
        chat_t0 = time.time()
        logger.info(f"[chat] 会话={request.session_id} 消息长度={len(request.message)}")

        input_data = await _prepare_chat_agent_input(request)

        fix_t0 = time.time()
        fix_result = None
        review_level = "full"
        if _should_use_rag_fast_path(request):
            fix_result = await _run_rag_fast_path(request)
            if fix_result is not None:
                review_level = "light"

        if fix_result is None:
            fix_result = await get_fix_agent().run_with_react(input_data)
        fix_result.metadata["user_message"] = input_data.user_message
        fix_result.metadata["original_user_message"] = request.message
        if input_data.context and input_data.context.get("intent_decision"):
            fix_result.metadata["intent_decision"] = input_data.context["intent_decision"]
        fix_phase_ms = int((time.time() - fix_t0) * 1000)
        logger.info(
            "[chat][phase] session=%s execution_mode=%s fix_phase_ms=%s tools=%s",
            request.session_id,
            fix_result.metadata.get("execution_mode"),
            fix_phase_ms,
            fix_result.tools_used,
        )

        if fix_result.metadata.get("status") == "error":
            logger.warning(f"[chat] 会话={request.session_id} 诊断Agent错误: {fix_result.metadata.get('error_detail')}")
            return JSONResponse(
                status_code=500,
                content=ChatResponse(
                    success=False,
                    code=500,
                    session_id=request.session_id,
                    message=fix_result.message,
                    tools_used=None,
                    latency_ms=fix_result.latency_ms
                ).model_dump()
            )

        review_t0 = time.time()
        final_result = await get_review_agent().review(fix_result, level=review_level)
        review_phase_ms = int((time.time() - review_t0) * 1000)

        verification = final_result.metadata.get("verification", {})
        has_issues = final_result.metadata.get("verification_has_issues", False)
        total_phase_ms = int((time.time() - chat_t0) * 1000)

        logger.info(
            f"[chat] 会话={request.session_id} 完成 "
            f"有问题={'是' if has_issues else '否'} "
            f"review_level={review_level} "
            f"fix_phase={fix_phase_ms}ms review_phase={review_phase_ms}ms total={total_phase_ms}ms "
            f"返回耗时={final_result.latency_ms}ms"
        )

        response_message, diagnosis_items = _extract_structured_chat_payload(final_result.message)

        return ChatResponse(
            session_id=request.session_id,
            message=response_message,
            tools_used=final_result.tools_used if final_result.tools_used else None,
            latency_ms=final_result.latency_ms,
            verification=verification if has_issues else None,
            diagnosis_items=diagnosis_items,
        )
    except Exception as e:
        logger.exception(f"[chat] session={request.session_id} error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式对话接口（内联验证标记）

    采用「先缓冲再验证」策略：
    - ReAct 阶段实时推送 status / tool 事件（展示进度）
    - token 先缓冲不发送
    - ReAct 完成后运行 3 层验证（~300ms）
    - 逐字流式输出最终回答，在未验证内容前插入 marker 事件

    事件流：
    1. session_id 事件
    2. FixAgent ReAct 阶段：status / tool 事件（实时）
    3. 回答流式阶段：marker / token 事件（验证后输出）
    4. verification 事件（校验摘要）
    5. done 事件
    """
    async def event_generator():
        yield f"data: {json_dumps({'event': 'session_id', 'data': {'session_id': request.session_id}})}\n\n"

        input_data = await _prepare_chat_agent_input(request)

        try:
            fix_agent = get_fix_agent()

            # 执行 FixAgent ReAct，转发进度事件（status/tool），缓冲 token
            # 等 ReAct 完成 + 验证管线跑完后再流式输出带内联标记的回答
            import asyncio as _asyncio
            token_buffer: list = []
            done_data: dict = {}
            tools_in_stream: list = []
            error_occurred = False

            async for event in fix_agent.run_with_react_stream(input_data):
                ev_type = event.get("event")
                if ev_type == "status":
                    yield f"data: {json_dumps(event)}\n\n"
                elif ev_type == "tool":
                    tools_in_stream.append(event.get("data", {}).get("tool", ""))
                    yield f"data: {json_dumps(event)}\n\n"
                elif ev_type == "token":
                    token_buffer.append(event.get("data", {}).get("content", ""))
                elif ev_type == "done":
                    done_data = event.get("data", {})
                elif ev_type == "error":
                    error_occurred = True
                    yield f"data: {json_dumps(event)}\n\n"

            if error_occurred or not token_buffer:
                yield f"data: {json_dumps({'event': 'done', 'data': {}})}\n\n"
                return

            full_message = "".join(token_buffer)
            stream_react_trace = done_data.get("react_trace", [])
            stream_tools_used = done_data.get("tools_used", [])
            fix_latency = done_data.get("latency_ms", 0)

            # 构建 AgentOutput 供验证管线校验
            fix_output = AgentOutput(
                agent_name="fix_agent",
                message=full_message,
                intention=None,
                tools_used=tools_in_stream if tools_in_stream else stream_tools_used,
                metadata={
                    "react_trace": stream_react_trace,
                    "user_message": input_data.user_message,
                    "original_user_message": request.message,
                    "intent_decision": (input_data.context or {}).get("intent_decision"),
                },
                latency_ms=fix_latency
            )

            # 运行3层确定性校验（~300ms），获取内联标记位置
            verified_output = await get_review_agent().review(fix_output)
            verification = verified_output.metadata.get("verification", {})
            has_issues = verified_output.metadata.get("verification_has_issues", False)

            # 流式输出最终回答（逐字），在未验证语句前插入 marker 事件
            final_message, diagnosis_items = _extract_structured_chat_payload(verified_output.message)
            markers = get_review_agent().get_inline_markers(final_message, verification)
            marker_idx = 0
            for i, char in enumerate(final_message):
                while marker_idx < len(markers) and markers[marker_idx]["char_pos"] <= i:
                    m = markers[marker_idx]
                    yield f"data: {json_dumps({'event': 'marker', 'data': {'text': m['text'], 'type': m['type']}})}\n\n"
                    marker_idx += 1

                yield f"data: {json_dumps({'event': 'token', 'data': {'content': char}})}\n\n"
                if i % 15 == 0:
                    await _asyncio.sleep(0)

            # 末尾剩余标记（安全追加文本中可能出现的新段落）
            while marker_idx < len(markers):
                m = markers[marker_idx]
                yield f"data: {json_dumps({'event': 'marker', 'data': {'text': m['text'], 'type': m['type']}})}\n\n"
                marker_idx += 1

            # 验证摘要事件
            verification_event = {
                "event": "verification",
                "data": {
                    "has_issues": has_issues,
                    "summary": {
                        "grounding_unverified": verification.get("grounding", {}).get("unverified_count", 0),
                        "graph_unverified": verification.get("graph", {}).get("unverified_count", 0),
                        "safety_missing": verification.get("safety", {}).get("missing_count", 0)
                    }
                }
            }
            yield f"data: {json_dumps(verification_event)}\n\n"

            # 完成事件
            final_done = {
                "event": "done",
                "data": {
                    "tools_used": verified_output.tools_used,
                    "latency_ms": verified_output.latency_ms
                }
            }
            if diagnosis_items:
                final_done["data"]["diagnosisItems"] = _serialize_diagnosis_items(diagnosis_items)
            yield f"data: {json_dumps(final_done)}\n\n"

        except Exception as e:
            logger.exception(f"[chat_stream] session={request.session_id} error")
            yield f"data: {json_dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"
            yield f"data: {json_dumps({'event': 'done', 'data': {}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/ai/knowledge/import", response_model=KnowledgeImportResponse)
async def knowledge_import(request: KnowledgeImportRequest) -> KnowledgeImportResponse:
    """
    文档导入并入库：解析 PDF → 向量化 → 存入 Redis 向量库
    """
    from services.knowledge_service import get_knowledge_service

    try:
        svc = get_knowledge_service()
        result = await svc.import_document(
            file_url=request.file_url,
            file_type=request.file_type,
            category=request.category,
            tags=request.tags,
            document_id=request.document_id,
            device_type=request.device_type,
            manual_type=request.manual_type,
            document_version=request.document_version,
            replace_existing=request.replace_existing
        )
        logger.info(f"[knowledge_import] 文件={result['file_name']} "
                    f"页数={result['total_pages']} "
                    f"文本={result['text_count']} 图片={result['image_count']} 表格={result['table_count']} "
                    f"耗时={result['process_time_ms']}ms")
        return KnowledgeImportResponse(
            success=True,
            message=f"导入完成：{result['file_name']}，共 {result['total_pages']} 页",
            code=200,
            file_name=result["file_name"],
            total_pages=result["total_pages"],
            text_count=result["text_count"],
            image_count=result["image_count"],
            image_summary_count=result.get("image_summary_count", 0),
            table_count=result["table_count"],
            sections=result["sections"],
            extraction_summary=result["extraction_summary"],
            process_time_ms=result["process_time_ms"],
            document_id=result.get("document_id"),
            document_version=result.get("document_version"),
            source_file_url=result.get("source_file_url")
        )
    except Exception as e:
        logger.exception(f"[knowledge_import] error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/knowledge/storage/stats", response_model=KnowledgeStorageStatsResponse)
async def knowledge_storage_stats() -> KnowledgeStorageStatsResponse:
    stats = get_vector_service().get_storage_stats()
    return KnowledgeStorageStatsResponse(
        success=True,
        message="knowledge storage statistics",
        code=200,
        **stats,
    )


@app.delete("/ai/knowledge/cache/embedding", response_model=KnowledgeCacheClearResponse)
async def knowledge_clear_embedding_cache() -> KnowledgeCacheClearResponse:
    deleted = get_vector_service().clear_embedding_cache()
    return KnowledgeCacheClearResponse(
        success=True,
        message="embedding cache cleared",
        code=200,
        **deleted,
    )


@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """通过 KnowledgeRetrievalTool 进行向量检索，返回 TopK 相关片段。"""
    import time

    try:
        logger.info(f"[knowledge_search] 查询={request.query[:50]} 数量={request.top_k}")
        tool = get_knowledge_retrieval_tool()

        t0 = time.time()
        result = await tool.run(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            tags=request.tags,
            image_urls=request.images,
            document_id=request.document_id,
            chunk_type=request.chunk_type,
            device_type=request.device_type,
            document_version=request.document_version,
            manual_type=request.manual_type
        )
        query_time_ms = int((time.time() - t0) * 1000)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error.get("message", "检索失败") if result.error else "检索失败"
            )

        data = result.data
        if data:
            first_item = data[0]
            first_meta = first_item.metadata if hasattr(first_item, "metadata") else first_item.get("metadata", {})
        else:
            first_meta = {}

        logger.info(f"[knowledge_search] 找到={len(data)}条 耗时={query_time_ms}ms")
        return KnowledgeSearchResponse(
            success=True,
            message=f"检索完成，找到 {len(data)} 条结果",
            code=200,
            data=data,
            total=len(data),
            query_time_ms=query_time_ms,
            retrieval_confidence=first_meta.get("retrieval_confidence", "low"),
            matched_types=first_meta.get("matched_types", []),
            confidence_reason=first_meta.get("confidence_reason", {"candidate_count": 0})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[knowledge_search] error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/temporary-plan/generate", response_model=TemporaryPlanDraftResponse)
async def temporary_plan_generate(request: TemporaryPlanGenerateRequest) -> TemporaryPlanDraftResponse:
    """基于知识证据生成仅供审核的临时检修计划草稿。"""
    try:
        return await get_temporary_plan_service().generate(request)
    except Exception as e:
        logger.exception("[temporary_plan_generate] error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/memory/consolidate", response_model=MemoryConsolidateResponse, response_model_by_alias=True)
async def memory_consolidate(request: MemoryConsolidateRequest) -> MemoryConsolidateResponse:
    """
    将多条原始对话压缩为结构化记忆摘要（滑动窗口 + 分类记忆）。
    """
    from datetime import datetime

    try:
        # 将消息列表转为带序号的字典格式，方便LLM阅读
        conv_dicts = [{"seq": i + 1, "role": m.role, "content": m.content} for i, m in enumerate(request.memoryMessages)]
        agent_input = AgentInput(
            user_message="请整理以下对话记录",
            session_id=request.session_id,
            context={
                "conversations": conv_dicts,
                "old_preferences": [p.model_dump() for p in request.memoryPreferenceVOList],
                # unresolved 现在带 id 字段，让LLM能通过ID精确标记已解决的事项
                "old_unresolved": [u.model_dump() for u in request.memoryUnresolvedVOList],
                # 上一轮摘要：让LLM生成渐进式摘要，避免信息丢失
                "previous_summary": request.previousSummary,
            }
        )

        logger.info(f"[memory_consolidate] 会话={request.session_id} 消息数={len(request.memoryMessages)}")
        result = await get_memory_agent().run(agent_input)
        logger.info(f"[memory_consolidate] 会话={request.session_id} 完成 耗时={result.latency_ms}ms")

        if result.metadata.get("status") == "error":
            error_type = result.metadata.get("error_type", "UnknownError")
            error_detail = result.metadata.get("error_detail", "记忆整理失败")
            logger.error(f"[memory_consolidate] 会话={request.session_id} 记忆Agent错误=[{error_type}] {error_detail}")
            # 返回200但带error状态，让Java端重试逻辑能解析
            return JSONResponse(content={
                "status": "error",
                "error_type": error_type,
                "error_detail": error_detail,
                "session_id": request.session_id
            })

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


@app.post("/ai/memory/search_facts")
async def search_facts(query: str, top_k: int = 5, session_ids: str = "",
                       device_type: str = "", equipment_id: str = "",
                       site_id: str = "", task_id: str = ""):
    """
    事实记忆向量检索接口 — 带多因子重排序 + 业务维度感知

    Java 端在组装对话上下文时调用此接口，
    用当前用户消息作为 query 去向量库中检索最相关的历史事实。
    检索结果会被注入到 AI 对话上下文中，让 AI 能"记住"之前的事实。

    【两阶段排序】
    1. 粗筛：Redis KNN 取 top_k * 3 候选
    2. 精排：FactReranker 多因子综合排序（语义 + 新近性 + 重要度 + 频率 + 置信度 + 业务匹配）

    Args:
        query: 用户当前发送的消息文本，用于语义匹配
        top_k: 最多返回几条最相关的事实，默认5条
        session_ids: 当前用户的会话ID列表，逗号分隔。用于过滤非本用户的事实
        device_type: 当前设备类型（可选，用于业务维度加权）
        equipment_id: 当前设备ID（可选）
        site_id: 当前场地ID（可选）
        task_id: 当前检修任务ID（可选）

    Returns:
        {"facts": [{"doc_id": "fact:xxx", "content": "...", "score": 0.85, "final_score": 0.72, ...}, ...]}
    """
    import time as t
    from services.fact_reranker import rerank

    if not (query or "").strip():
        logger.info("[search_facts] empty query, skip fact vector search")
        return {"facts": [], "query_time_ms": 0}

    # 解析会话ID白名单
    allowed_sessions = set()
    if session_ids:
        allowed_sessions = {sid.strip() for sid in session_ids.split(",") if sid.strip()}

    try:
        t0 = t.time()
        svc = get_vector_service()
        # 粗筛：多取候选，留给 reranker 精排
        results = await svc.search_by_text(query, top_k=top_k * 3)

        # 过滤：只保留 type=fact, status=active, 属于当前用户
        candidates = []
        for r in results:
            metadata = r.get("metadata", {})
            if metadata.get("type") != "fact":
                continue
            # 过滤已废弃的事实（双重保障：即使旧向量未被删除，也不会返回）
            if metadata.get("status") and metadata.get("status") != "active":
                continue
            # 按会话ID过滤：只保留属于当前用户的事实
            fact_session = metadata.get("session_id", "")
            if allowed_sessions and fact_session not in allowed_sessions:
                continue
            candidates.append(r)

        # 构建业务上下文
        business_context = {}
        if device_type:
            business_context["device_type"] = device_type
        if equipment_id:
            business_context["equipment_id"] = equipment_id
        if site_id:
            business_context["site_id"] = site_id
        if task_id:
            business_context["task_id"] = task_id

        # 精排：多因子重排序（含业务维度）
        ranked = rerank(candidates, top_k=top_k,
                        business_context=business_context or None)

        # 格式化输出
        facts = []
        for r in ranked:
            metadata = r.get("metadata", {})
            facts.append({
                "content": r.get("text", ""),
                "score": round(r.get("score", 0), 4),
                "final_score": r.get("final_score", 0),
                "score_breakdown": r.get("score_breakdown", {}),
                "doc_id": r.get("doc_id", ""),
                "keywords": metadata.get("keywords", ""),
                "session_id": metadata.get("session_id", ""),
            })

        query_time_ms = int((t.time() - t0) * 1000)
        logger.info(f"[search_facts] 查询={query[:50]} 候选={len(candidates)} 精排后={len(facts)} 耗时={query_time_ms}ms")
        return {"facts": facts, "query_time_ms": query_time_ms}
    except Exception as e:
        logger.exception(f"[search_facts] error")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteFactsRequest(BaseModel):
    fact_ids: list[str]


@app.post("/ai/memory/delete_facts")
async def delete_facts(request: DeleteFactsRequest):
    """
    删除 Redis 向量库中的旧事实。
    Java 端整合产生 supersededIds 后调用此接口同步清理向量库。
    """
    if not request.fact_ids:
        return {"deleted": 0}

    svc = get_vector_service()
    deleted = svc.delete_batch(request.fact_ids)
    logger.info(f"[delete_facts] 删除旧事实向量 {deleted}/{len(request.fact_ids)} 条")
    return {"deleted": deleted}


class RealtimeUpdateRequest(BaseModel):
    """实时记忆更新请求体"""
    session_id: str
    user_message: str
    ai_response: str = ""
    recent_facts: list = []


@app.post("/ai/memory/realtime_update")
async def realtime_memory_update(request: RealtimeUpdateRequest):
    """
    实时记忆更新检测接口

    每轮对话完成后，Java端异步调用此接口。
    轻量级LLM判断用户是否纠正了事实或改变了偏好。
    如果检测到变更，立即更新向量库和返回偏好变更给Java端保存。

    【与定时整合的区别】
    - 本接口：只处理"纠正"和"偏好变更"，2-3秒完成
    - /consolidate：做完整整合（新事实、待办、摘要），40-60秒

    【调用时机】
    Java端在 doOnComplete 保存AI回复后立即异步调用。
    不阻塞主对话流，用户感知不到。

    Args:
        session_id: 会话ID
        user_message: 用户本轮消息
        ai_response: AI本轮回复（可选）
        recent_facts: JSON格式的本轮注入事实列表（可选）

    Returns:
        {
            "has_update": true/false,
            "fact_corrections": [...],  // 已在向量库中更新的事实
            "preference_changes": [...] // 需要Java端保存的偏好变更
        }
    """
    import time as t

    try:
        t0 = t.time()

        session_id = request.session_id
        user_message = request.user_message
        ai_response = request.ai_response
        recent_facts = request.recent_facts

        # 解析 recent_facts
        if isinstance(recent_facts, str):
            try:
                facts_list = json.loads(recent_facts) if recent_facts else []
            except (json.JSONDecodeError, TypeError):
                facts_list = []
        else:
            facts_list = recent_facts if recent_facts else []

        from agents.realtime_memory_agent import get_realtime_memory_agent
        agent = get_realtime_memory_agent()

        input_data = AgentInput(
            user_message=user_message,
            session_id=session_id,
            context={
                "user_message": user_message,
                "ai_response": ai_response,
                "recent_facts": facts_list
            }
        )

        result = await agent.run(input_data)
        latency_ms = int((t.time() - t0) * 1000)

        result_data = result.metadata.get("result", {})
        logger.info(
            f"[realtime_update] 会话={session_id} "
            f"有更新={result_data.get('has_update', False)} "
            f"耗时={latency_ms}ms"
        )

        return {
            "has_update": result_data.get("has_update", False),
            "fact_corrections": result_data.get("fact_corrections", []),
            "preference_changes": result_data.get("preference_changes", []),
            # 被替代的旧事实的向量库doc_id列表
            # Java端用这些ID在MySQL memory_fact表中标记旧事实为superseded
            "superseded_fact_ids": result_data.get("superseded_fact_ids", []),
            "latency_ms": latency_ms
        }

    except Exception as e:
        logger.exception(f"[realtime_update] error")
        return {
            "has_update": False,
            "fact_corrections": [],
            "preference_changes": [],
            "error": str(e)
        }

# ==================== 多模态向量化（文本或图片，不融合）====================

class MultimodalEmbeddingRequest(BaseModel):
    """多模态向量化请求 — 传 text 或 image_base64s 之一，不做融合"""
    text: str = ""
    image_base64s: list = []   # Java 端下载图片后转的 base64 data URI

@app.post("/ai/embedding/multimodal")
async def multimodal_embedding(req: MultimodalEmbeddingRequest):
    """
    使用多模态模型（qwen2.5-vl-embedding，1024维）向量化。
    传 text 或 image_base64s 之一：
    - 仅 text：返回文本在多模态空间的向量
    - 仅 image_base64s：返回图片向量（多张取均值）
    - 不做融合，调用方应分别调用

    image_base64s 格式: ["data:image/jpeg;base64,/9j/4AAQ..."]
    """
    import numpy as np
    from embeddings.image_embedding import get_image_embedding

    has_text = bool(req.text and req.text.strip())
    has_images = bool(req.image_base64s)

    if not has_text and not has_images:
        raise HTTPException(status_code=400, detail="text 和 image_base64s 不能同时为空")

    try:
        img_emb = get_image_embedding()

        if has_images:
            # 图片向量（多张取均值后归一化）
            img_vecs = await img_emb.embed_batch(req.image_base64s)
            vec = np.mean(img_vecs, axis=0)
        else:
            # 纯文本 → 通过多模态模型映射到 1024 维空间
            vec = np.array(await img_emb.embed_text_as_multimodal(req.text.strip()))

        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return {
            "vector": vec.tolist(),
            "dimension": len(vec),
            "has_text": has_text,
            "has_image": has_images
        }

    except Exception as e:
        logger.exception("[multimodal_embedding] error")
        raise HTTPException(status_code=500, detail=str(e))


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
