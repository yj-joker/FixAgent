# API 模块

## 模块职责

FastAPI Web 服务入口，负责 HTTP 接口、请求路由、参数校验。

> **重要**：本模块仅负责接收请求和返回响应，所有 AI 推理逻辑由 `agents/` 和 `chains/` 完成。业务数据、会话管理由 Java 后端统一处理。

## 接口列表

| 接口 | 方法 | 描述 | Agent调用 |
|------|------|------|-----------|
| `/ai/chat` | POST | 对话接口（自动意图识别） | OrchestratorAgent |
| `/ai/retrieval` | POST | 纯检索接口 | RetrievalAgent |
| `/ai/diagnosis` | POST | 纯诊断接口 | DiagnosisAgent |
| `/ai/guidance` | POST | 纯指引接口 | GuidanceAgent |
| `/ai/pipeline` | POST | 完整流程接口 | All Agents |
| `/ai/memory/consolidate` | POST | 记忆整理接口 | MemoryAgent |

## 请求模型

参见 `schemas/request.py`：

- `ChatRequest` - 对话请求（session_id, message, mode, images, stream）
- `KnowledgeSearchRequest` - 知识检索请求
- `GraphQueryRequest` - 图谱查询请求
- `YoloDetectRequest` - YOLO检测请求
- `SamSegmentRequest` - SAM分割请求
- `ClipEmbedRequest` - CLIP向量化请求
- `DocumentParseRequest` - 文档解析请求
- `MemoryConsolidateRequest` - 记忆整理请求

## 响应模型

参见 `schemas/response.py`：

- `ChatResponse` - 对话响应（继承 BaseResponse：success, message, code）
- `KnowledgeSearchResponse` - 知识检索响应
- `GraphQueryResponse` - 图谱查询响应
- `YoloDetectResponse` - YOLO检测响应
- `SamSegmentResponse` - SAM分割响应
- `ClipEmbedResponse` - CLIP向量化响应
- `DocumentParseResponse` - 文档解析响应
- `MemoryConsolidateResponse` - 记忆整理响应

## 依赖关系

```
api/main.py
    │
    ├── schemas/request.py      # 请求模型
    ├── schemas/response.py     # 响应模型
    ├── schemas/models.py       # 枚举和常量（AgentMode, IntentionType等）
    │
    ├── agents/orchestrator_agent.py    # 调度Agent（意图识别+任务分解）
    ├── agents/retrieval_agent.py       # 检索Agent
    ├── agents/diagnosis_agent.py       # 诊断Agent
    ├── agents/guidance_agent.py       # 作业Agent
    ├── agents/memory_agent.py         # 记忆整理Agent
    │
    └── services/llm_service.py        # 阿里云百炼
    └── services/vector_service.py      # Redis向量库
    └── services/graph_service.py       # Neo4j图数据库
```

## 项目中的实现

### main.py - API入口

```python
# api/main.py
"""
FastAPI Web 服务入口

职责：
- HTTP 接口定义
- 请求参数校验（依赖 schemas/）
- 调用 agents/ 执行 AI 推理
- 返回结构化响应

边界：
- 仅负责 AI 推理，不碰业务数据
- 会话历史由 Java 后端管理（Redis ChatMemory）
- 错误码返回给 Java，由 Java 展示友好提示
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List

from schemas.request import ChatRequest, KnowledgeSearchRequest
from schemas.response import ChatResponse, KnowledgeSearchResponse, BaseResponse
from schemas.models import AgentMode

# Agent 初始化（应用启动时创建，避免每次请求创建）
orchestrator_agent = OrchestratorAgent()
retrieval_agent = RetrievalAgent()
diagnosis_agent = DiagnosisAgent()
guidance_agent = GuidanceAgent()

app = FastAPI(
    title="FixAgent AI Module",
    version="1.0.0",
    description="AI推理引擎：故障诊断、知识检索、作业指引"
)


# ==================== 对话相关接口 ====================

@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    对话接口（自动意图识别）

    流程：
    1. Java 后端组装 AIContext（包含会话历史）
    2. 调用本接口，传入用户消息和上下文
    3. OrchestratorAgent 识别意图并调度
    4. 返回 AI 推理结果

    参数：
        session_id: Java后端传递，用于日志追踪
        message: 用户消息
        mode: 运行模式（CHAT/RETRIEVAL/DIAGNOSIS/GUIDANCE/FULL）
        images: 图片URL列表
        stream: 是否流式输出
    """
    try:
        result = await orchestrator_agent.run(
            session_id=request.session_id,
            message=request.message,
            mode=request.mode,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention=result.intention,
            tools_used=result.tools_used,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/retrieval", response_model=ChatResponse)
async def retrieval(request: ChatRequest) -> ChatResponse:
    """
    纯检索接口

    直接调用 RetrievalAgent，从向量库检索相关知识。
    """
    try:
        result = await retrieval_agent.run(
            session_id=request.session_id,
            message=request.message,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="query_knowledge",
            tools_used=["knowledge_retrieval"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/diagnosis", response_model=ChatResponse)
async def diagnosis(request: ChatRequest) -> ChatResponse:
    """
    纯诊断接口

    直接调用 DiagnosisAgent，进行故障分析和原因推理。
    """
    try:
        result = await diagnosis_agent.run(
            session_id=request.session_id,
            message=request.message,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="troubleshoot",
            tools_used=["knowledge_retrieval", "graph_query"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/guidance", response_model=ChatResponse)
async def guidance(request: ChatRequest) -> ChatResponse:
    """
    纯指引接口

    直接调用 GuidanceAgent，生成标准化的维修作业步骤。
    """
    try:
        result = await guidance_agent.run(
            session_id=request.session_id,
            message=request.message,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="seek_guidance",
            tools_used=["knowledge_retrieval"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/pipeline", response_model=ChatResponse)
async def pipeline(request: ChatRequest) -> ChatResponse:
    """
    完整流程接口

    依次执行：检索 -> 诊断 -> 指引，返回综合分析结果。
    """
    try:
        result = await orchestrator_agent.run_full_pipeline(
            session_id=request.session_id,
            message=request.message,
            mode=AgentMode.FULL,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="full_pipeline",
            tools_used=["knowledge_retrieval", "graph_query"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 工具类接口 ====================

@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """
    知识检索接口

    直接调用向量检索服务，返回 TopK 相关片段。
    """
    try:
        result = await vector_service.search(
            query=request.query,
            images=request.images or [],
            top_k=request.top_k,
            category=request.category
        )

        return KnowledgeSearchResponse(
            data=result.results,
            total=result.total,
            query_time_ms=result.query_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 错误处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return BaseResponse(
        success=False,
        message=str(exc),
        code=500
    )
```

## 使用示例

### 1. 对话接口调用

```python
# Java 后端 - Feign 客户端
@FeignClient(name = "ai-service", url = "${ai.service.url}")
public interface AIAiClient {

    @PostMapping("/ai/chat")
    ChatResponse chat(@RequestBody ChatRequest request);

    @PostMapping("/ai/retrieval")
    ChatResponse retrieval(@RequestBody ChatRequest request);

    @PostMapping("/ai/diagnosis")
    ChatResponse diagnosis(@RequestBody ChatRequest request);

    @PostMapping("/ai/guidance")
    ChatResponse guidance(@RequestBody ChatRequest request);

    @PostMapping("/ai/pipeline")
    ChatResponse pipeline(@RequestBody ChatRequest request);
}

// Java 后端 - 调用对话接口
ChatRequest request = ChatRequest.builder()
    .sessionId("sess_abc123")                              // 会话ID（Java生成）
    .message("电动机轴承过热是什么原因？")                    // 用户消息
    .mode(AgentMode.DIAGNOSIS)                            // 运行模式
    .images(List.of("https://cdn.example.com/bearing.jpg")) // 图片（可选）
    .stream(false)                                         // 非流式
    .build();

ChatResponse response = aiClient.chat(request);

// 处理响应
if (response.isSuccess()) {
    System.out.println("AI回复: " + response.getMessage());
    System.out.println("识别意图: " + response.getIntention());
    System.out.println("使用工具: " + response.getToolsUsed());
    System.out.println("响应耗时: " + response.getLatencyMs() + "ms");
} else {
    System.out.println("AI服务错误: " + response.getMessage());
}
```

### 2. 知识检索接口调用

```python
# Java 后端 - Feign 客户端
@PostMapping("/ai/knowledge/search")
KnowledgeSearchResponse knowledgeSearch(@RequestBody KnowledgeSearchRequest request);

// Java 后端 - 调用检索接口
KnowledgeSearchRequest request = KnowledgeSearchRequest.builder()
    .query("轴承过热")                    // 查询文本
    .topK(10)                            // 返回数量
    .category("motor")                   // 分类过滤（可选）
    .build();

KnowledgeSearchResponse response = aiClient.knowledgeSearch(request);
```

### 3. Python 内部调用（Agent 间调用）

```python
# agents/orchestrator_agent.py
from api.main import orchestrator_agent, retrieval_agent

# Orchestrator 内部调度 retrieval_agent
async def run_full_pipeline(self, session_id, message, images):
    # 1. 检索阶段
    retrieval_result = await retrieval_agent.run(
        session_id=session_id,
        message=message,
        images=images
    )

    # 2. 诊断阶段（传入检索结果作为上下文）
    diagnosis_result = await diagnosis_agent.run(
        session_id=session_id,
        message=message,
        context={"retrieval": retrieval_result.message},
        images=images
    )

    # 3. 汇总结果
    return self._summarize(retrieval_result, diagnosis_result)
```

## 与 Java 后端的对应关系

| Python API | Java Controller | 说明 |
|------------|-----------------|------|
| `POST /ai/chat` | `AIController.chat()` | 对话接口（自动意图识别） |
| `POST /ai/retrieval` | `AIController.retrieval()` | 纯检索接口 |
| `POST /ai/diagnosis` | `AIController.diagnosis()` | 纯诊断接口 |
| `POST /ai/guidance` | `AIController.guidance()` | 纯指引接口 |
| `POST /ai/pipeline` | `AIController.pipeline()` | 完整流程接口 |
| `POST /ai/memory/consolidate` | `AIController.memoryConsolidate()` | 记忆整理接口 |

**调用流程**：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Java Backend (8080)                         │
│                                                                  │
│  1. 接收用户请求                                                 │
│  2. 从 Redis 获取 ChatMemory（会话历史）                          │
│  3. 组装 AIContext                                              │
│  4. 调用 Python AI 服务                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP POST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Python AI Module (8001)                        │
│                                                                  │
│  api/main.py → agents/orchestrator_agent.py                     │
│       │                    │                                     │
│       │                    ├── retrieval_agent.py                │
│       │                    │         │                           │
│       │                    │         └── tools/knowledge_...     │
│       │                    │         └── services/vector_...     │
│       │                    │                                     │
│       │                    ├── diagnosis_agent.py                │
│       │                    │         │                           │
│       │                    │         ├── tools/graph_query_...    │
│       │                    │         └── services/graph_...      │
│       │                    │                                     │
│       │                    └── guidance_agent.py                  │
│       │                                                      │
│  返回 ChatResponse (message + intention + tools_used)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Response
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Java Backend (8080)                         │
│                                                                  │
│  1. 存储对话历史到 Redis                                         │
│  2. 返回响应给客户端                                             │
└─────────────────────────────────────────────────────────────────┘
```

**关键区别**：

| 方面 | Python AI 模块 | Java 后端 |
|------|---------------|----------|
| 职责 | AI 推理（意图识别、诊断、检索） | 业务逻辑、数据管理、会话管理 |
| 会话 | 不管理，仅接收 session_id | 管理 ChatMemory（Redis） |
| 数据 | 不碰业务数据（案例、设备等） | 管理所有业务数据 |
| 错误处理 | 返回结构化错误码 | 转换为用户友好提示 |

## 环境变量

| 变量名 | 说明 | 来源 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key | `.env` |
| `REDIS_HOST` | Redis 地址 | `.env` |
| `REDIS_PORT` | Redis 端口 | `.env` |
| `NEO4J_URI` | Neo4j 连接URI | `.env` |
| `NEO4J_USERNAME` | Neo4j 用户名 | `.env` |
| `NEO4J_PASSWORD` | Neo4j 密码 | `.env` |

## 启动方式

```bash
# 开发环境
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# 生产环境
uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 4
```

## 文件结构

```
api/
├── __init__.py
├── README.md                    # 本文件
└── main.py                     # API 入口
```

## 意图识别

API 层不直接处理意图识别，而是委托给 OrchestratorAgent 的 `intention_recognizer` 模块。

**意图识别流程**：
```
请求进入 → OrchestratorAgent.run() → IntentionRecognizer.recognize() → 路由到子Agent
```

**识别类型**：
| 类型 | 说明 | 路由目标 |
|-----|------|---------|
| `troubleshoot` | 故障诊断 | DiagnosisAgent |
| `query_knowledge` | 知识检索 | RetrievalAgent |
| `seek_guidance` | 作业指引 | GuidanceAgent |
| `full_pipeline` | 完整流程 | All Agents |
| `general_chat` | 一般对话 | LLM直接回复 |

**技术方案**：采用 LLM 轻量级识别（qwen-turbo）+ 关键词兜底，详见 `agents/intention/` 模块。

## 注意事项

1. **边界清晰**：Python 仅负责 AI 推理，不碰业务数据
2. **会话追踪**：`session_id` 由 Java 后端生成并传递，用于日志追踪
3. **错误处理**：返回 `BaseResponse`（success, message, code），Java 负责用户友好提示
4. **流式输出**：`stream=True` 时使用 Server-Sent Events，需要特殊处理
5. **Agent 初始化**：Agent 实例应在应用启动时创建，避免重复创建开销
6. **超时控制**：AI 推理可能耗时较长，建议 HTTP 超时设置 > 60s
