import json
import logging
from functools import partial
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse

# 全局替换：所有 json.dumps 默认保留中文原文，避免 \uXXXX 乱码
# 使用方法：文件内所有 json.dumps 调用都用 json_dumps 替代
json_dumps = partial(json.dumps, ensure_ascii=False)
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from api.asr_api import asr_router
from schemas.request import ChatRequest, KnowledgeImportRequest, KnowledgeSearchRequest, MemoryConsolidateRequest
from schemas.response import ChatResponse, KnowledgeImportResponse, KnowledgeSearchResponse, BaseResponse, MemoryConsolidateResponse
from agents.fix_agent import get_fix_agent
from agents.review_agent import get_review_agent
from agents.memory_agent import get_memory_agent
from agents.base_agent import AgentInput, AgentOutput
from services.vector_service import get_vector_service

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


app = FastAPI(
    title="FixAgent AI Module",
    version="2.0.0",
    description="AI推理引擎：FixAgent 统一诊断 + ReviewAgent 输出审核"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(asr_router)


@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    核心对话接口 —— FixAgent ReAct 推理 + ReviewAgent 审核

    流程：
    1. FixAgent 通过 ReAct 循环自主决策工具调用
    2. ReviewAgent 对 FixAgent 输出做质量审核
    3. 返回最终结果（原样或修正后）
    """
    try:
        logger.info(f"[chat] session={request.session_id} msg_len={len(request.message)}")

        input_data = AgentInput(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images,
            conversation_history=request.conversation_history,
            context=request.context
        )

        fix_result = await get_fix_agent().run_with_react(input_data)

        if fix_result.metadata.get("status") == "error":
            logger.warning(f"[chat] session={request.session_id} fix_agent error: {fix_result.metadata.get('error_detail')}")
            return ChatResponse(
                session_id=request.session_id,
                message=fix_result.message,
                tools_used=None,
                latency_ms=fix_result.latency_ms
            )

        final_result = await get_review_agent().review(fix_result)

        logger.info(
            f"[chat] session={request.session_id} done "
            f"review={final_result.metadata.get('review_status')} "
            f"latency={final_result.latency_ms}ms"
        )

        return ChatResponse(
            session_id=request.session_id,
            message=final_result.message,
            tools_used=final_result.tools_used if final_result.tools_used else None,
            latency_ms=final_result.latency_ms
        )
    except Exception as e:
        logger.exception(f"[chat] session={request.session_id} error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式对话接口

    事件流：
    1. session_id 事件
    2. FixAgent ReAct 阶段：status / tool / token 事件
    3. ReviewAgent 审核：review 事件
    4. done 事件
    """
    async def event_generator():
        yield f"data: {json_dumps({'event': 'session_id', 'data': {'session_id': request.session_id}})}\n\n"

        input_data = AgentInput(
            user_message=request.message,
            session_id=request.session_id,
            images=request.images,
            conversation_history=request.conversation_history,
            context=request.context
        )

        try:
            fix_agent = get_fix_agent()

            # run_with_react_stream 内部调用 run_with_react 一次，
            # 然后将工具调用事件和最终回答逐字符流式输出。
            # 我们拦截 done 事件，在其之前插入 ReviewAgent 审核。
            collected_events = []
            async for event in fix_agent.run_with_react_stream(input_data):
                collected_events.append(event)
                if event.get("event") != "done":
                    yield f"data: {json_dumps(event)}\n\n"

            # 从流式事件中重建 fix_output 用于审核
            # run_with_react_stream 的 done 事件包含 latency_ms
            done_event = next((e for e in collected_events if e.get("event") == "done"), {})
            fix_latency = done_event.get("data", {}).get("latency_ms", 0)

            # 收集流式输出的完整回答文本
            full_message = "".join(
                e.get("data", {}).get("content", "")
                for e in collected_events
                if e.get("event") == "token"
            )

            # 收集工具调用信息
            tools_used_in_stream = [
                e.get("data", {}).get("tool", "")
                for e in collected_events
                if e.get("event") == "tool"
            ]

            error_events = [e for e in collected_events if e.get("event") == "error"]
            if error_events:
                yield f"data: {json_dumps({'event': 'done', 'data': {}})}\n\n"
                return

            # 构建 AgentOutput 供 ReviewAgent 审核
            fix_output = AgentOutput(
                agent_name="fix_agent",
                message=full_message,
                intention=None,
                tools_used=tools_used_in_stream,
                metadata={"react_trace": []},
                latency_ms=fix_latency
            )

            review_result = await get_review_agent().review(fix_output)
            review_status = review_result.metadata.get("review_status", "approved")

            yield f"data: {json_dumps({'event': 'review', 'data': {'status': review_status}})}\n\n"

            if review_status == "revised":
                yield f"data: {json_dumps({'event': 'revised_content', 'data': {'content': review_result.message}})}\n\n"

            yield f"data: {json_dumps({'event': 'done', 'data': {'tools_used': review_result.tools_used, 'latency_ms': review_result.latency_ms}})}\n\n"

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


@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """直接调用向量检索服务，返回 TopK 相关片段。"""
    import time

    try:
        logger.info(f"[knowledge_search] q={request.query[:50]} top_k={request.top_k}")
        svc = get_vector_service()

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

        logger.info(f"[memory_consolidate] session={request.session_id} msg_count={len(request.memoryMessages)}")
        result = await get_memory_agent().run(agent_input)
        logger.info(f"[memory_consolidate] session={request.session_id} done latency={result.latency_ms}ms")

        if result.metadata.get("status") == "error":
            error_type = result.metadata.get("error_type", "UnknownError")
            error_detail = result.metadata.get("error_detail", "记忆整理失败")
            logger.error(f"[memory_consolidate] session={request.session_id} agent_error=[{error_type}] {error_detail}")
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
async def search_facts(query: str, top_k: int = 5):
    """
    事实记忆向量检索接口

    Java 端在组装对话上下文时调用此接口，
    用当前用户消息作为 query 去向量库中检索最相关的历史事实。
    检索结果会被注入到 AI 对话上下文中，让 AI 能"记住"之前的事实。

    【调用链路】
    用户发消息 → Java 端收到 → 调用本接口检索相关事实 →
    将事实注入上下文 → 发给 AI 生成回复

    【过滤逻辑】
    只返回 metadata.type == "fact" 的向量记录，
    排除知识库文档等其他类型的向量。

    Args:
        query: 用户当前发送的消息文本，用于语义匹配
        top_k: 最多返回几条最相关的事实，默认5条

    Returns:
        {"facts": [{"doc_id": "fact:xxx", "content": "...", "score": 0.85, ...}, ...]}
    """
    import time

    try:
        t0 = time.time()
        svc = get_vector_service()
        # 用用户消息做向量检索，在所有向量中搜索最相似的
        results = await svc.search_by_text(query, top_k=top_k * 2)
        # 只保留 type=fact 的结果（向量库中还有知识库文档等其他类型）
        facts = []
        for r in results:
            metadata = r.get("metadata", {})
            if metadata.get("type") == "fact":
                facts.append({
                    "content": r.get("text", ""),
                    "score": round(r.get("score", 0), 4),
                    "doc_id": r.get("doc_id", ""),
                    "keywords": metadata.get("keywords", ""),
                    "session_id": metadata.get("session_id", ""),
                })
        # 按相关度排序，只取 top_k 条
        facts = facts[:top_k]
        query_time_ms = int((time.time() - t0) * 1000)
        logger.info(f"[search_facts] query={query[:50]} found={len(facts)} latency={query_time_ms}ms")
        return {"facts": facts, "query_time_ms": query_time_ms}
    except Exception as e:
        logger.exception(f"[search_facts] error")
        raise HTTPException(status_code=500, detail=str(e))


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
            f"[realtime_update] session={session_id} "
            f"has_update={result_data.get('has_update', False)} "
            f"latency={latency_ms}ms"
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
