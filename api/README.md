# API 模块

## 模块职责

FastAPI Web 服务入口，HTTP 接口定义、请求路由、参数校验。所有 AI 推理逻辑由 `agents/` 完成，业务数据由 Java 后端管理。

## 接口列表

| 接口 | 方法 | 描述 | 状态 |
|------|------|------|------|
| `/ai/chat` | POST | 对话接口（意图识别 + 路由分发） | **已实现** |
| `/ai/chat/stream` | POST | SSE 流式响应（全模式可用） | **已实现** |
| `/ai/retrieval` | POST | 直接调用 RetrievalAgent | **已实现** |
| `/ai/diagnosis` | POST | 直接调用 DiagnosisAgent | **已实现** |
| `/ai/guidance` | POST | 直接调用 GuidanceAgent | **已实现** |
| `/ai/pipeline` | POST | 完整流程（检索→诊断→指引） | **已实现** |
| `/ai/knowledge/search` | POST | 直接调用 VectorService 检索 | **已实现** |
| `/ai/knowledge/import` | POST | 文档解析→向量化→入库管道 | **已实现** |
| `/ai/memory/consolidate` | POST | 记忆整理（function calling + 向量存储） | **已实现** |

## 请求模型

`schemas/request.py` 中定义：

- `ChatRequest` — session_id / message(max_length=50000) / mode / images / stream
- `KnowledgeImportRequest` — file_url / file_type / category / tags
- `KnowledgeSearchRequest` — query / top_k / category / tags
- `MemoryConsolidateRequest` — session_id / memoryMessages / memoryPreferenceVOList / memoryUnresolvedVOList

## 响应模型

`schemas/response.py` 中定义：

- `ChatResponse` — session_id / message / intention / tools_used / latency_ms
- `KnowledgeImportResponse` — file_name / total_pages / text_count / image_count / table_count / sections / extraction_summary
- `KnowledgeSearchResponse` — data(VectorSearchResult列表) / total / query_time_ms
- `MemoryConsolidateResponse` — session_id / summary(MemorySummary) / original_count / consolidated_at

## 日志输出点

关键位置输出 INFO 级别日志（控制台实时可见）：

| 接口 | 位置 | 日志内容 |
|------|------|---------|
| `/ai/chat` | api/main.py chat() | 请求入参（session/mode/消息长度）、完成耗时 |
| `/ai/knowledge/import` | api/main.py knowledge_import() | 文件名/页数/文本/图片/表格数量/耗时 |
| `/ai/knowledge/search` | api/main.py knowledge_search() | 查询词/top_k/命中数/耗时 |
| `/ai/memory/consolidate` | api/main.py memory_consolidate() | 对话条数、Agent 错误详情、完成耗时 |

日志格式：`2026-05-14 10:30:15 | INFO     | api.main | [chat] session=abc123 mode=DIAGNOSIS msg_len=24`

## 调用关系

```
api/main.py
    ├── schemas/request.py      — 请求模型
    ├── schemas/response.py     — 响应模型
    ├── agents/orchestrator_agent.py — 调度中枢（单例，惰性创建）
    ├── agents/retrieval_agent.py    — 知识检索
    ├── agents/diagnosis_agent.py    — 故障诊断
    ├── agents/guidance_agent.py     — 作业指引
    ├── agents/memory_agent.py       — 记忆整理
    └── services/vector_service.py  — 向量检索（knowledge/search）
        └── services/knowledge_service.py — 文档导入（knowledge/import）
```

## 与 Java 后端的交互

```
Java Backend                    FixAgent (Python)
  POST /ai/chat                     → OrchestratorAgent → 子Agent → ChatResponse
  POST /ai/chat/stream (SSE)       → OrchestratorAgent.run_stream() → SSE token 流
  POST /ai/retrieval               → RetrievalAgent → ChatResponse
  POST /ai/diagnosis               → DiagnosisAgent → ChatResponse
  POST /ai/guidance                → GuidanceAgent → ChatResponse
  POST /ai/pipeline                 → 三阶段串行 → ChatResponse
  POST /ai/knowledge/import        → KnowledgeService → KnowledgeImportResponse
  POST /ai/knowledge/search         → VectorService → KnowledgeSearchResponse
  POST /ai/memory/consolidate      → MemoryAgent → MemoryConsolidateResponse
```

## 错误处理

- Agent 执行失败（`metadata.status="error"`）→ API 层检测后 raise HTTPException(500)
- LLM 返回 content=null（tool_call 场景）→ `content or ""` 兜底，JSON 解析失败 → fallback + warning 日志
- 请求参数校验失败 → FastAPI 自动返回 422
- 全局异常捕获 → JSONResponse(status_code=500) 返回给 Java

## 启动方式

```bash
# 开发环境（热重载）
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 生产环境（多进程）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 文件结构

```
api/
├── __init__.py
└── main.py                     # FastAPI 入口，含 /ai/* 所有端点
```

## 注意事项

1. **日志级别**：生产环境将 `logging.basicConfig(level=logging.INFO)` 改为 `WARNING`
2. **Agent 惰性初始化**：应用启动时不加载 LLM，首次请求时才创建实例
3. **会话追踪**：`session_id` 由 Java 生成并传递，用于日志分片和链路追踪
4. **超时设置**：建议 HTTP 超时 > 60s（AI 推理耗时较长）
5. **SSE 协议**：流式接口推送 session_id / status / tool / token / done 事件
