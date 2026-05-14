# Services 模块

## 模块概述

Services 是系统的**核心服务层**，封装与外部服务的交互：

- **LLMService** — 阿里云百炼 DashScope API，对话 + function calling
- **VectorService** — Redis Stack 向量库，KNN 检索 + metadata 过滤
- **GraphService** — Neo4j 图数据库，设备-部件-故障因果链查询

设计原则：单一职责、接口统一、错误处理、异步优先、单例复用。

## 服务列表

| 服务 | 文件 | 职责 | 外部依赖 |
|------|------|------|---------|
| `llm_service` | llm_service.py | 大模型对话 + function calling + ReAct 循环 | 阿里云百炼 DashScope |
| `vector_service` | vector_service.py | 向量存储 / 检索 / 删除 / 计数 | Redis Stack (FT.SEARCH) |
| `graph_service` | graph_service.py | Neo4j CRUD + Cypher 查询 + 路径查询 | Neo4j |
| `knowledge_service` | knowledge_service.py | 文档导入编排：解析→向量化→入库 | DocumentParserTool + TextEmbedding + VectorService |

## LLMService — 核心接口

```python
# chat() — 普通对话
response = await llm_service.chat(messages)  # [{"role": "user", "content": "..."}]
response["content"]  # 文本回复

# chat(stream=True) — 流式
async for token in llm_service.chat(messages, stream=True):
    print(token, end="", flush=True)

# chat_with_tools() — ReAct 循环（含 trace）
response = await llm_service.chat_with_tools(messages, tools, handlers)
response["content"]      # 最终文本回复
response["trace"]       # 每轮工具调用记录
response["usage"]       # token 用量统计

# response_format — JSON 约束（MemoryAgent 专用）
response = await llm_service.chat_with_tools(..., response_format={"type": "json_object"})
```

## VectorService — 核心接口

```python
# 批量添加向量
vector_service.add_vector(doc_id, text, vector, metadata, category, tags)

# 按向量检索
results = vector_service.search(vector, top_k=5, filter=filter_expr)
# 返回: [{"doc_id": "...", "text": "...", "score": 0.92, "metadata": {...}}]

# 按文本检索（自动向量化）
results = await vector_service.search_by_text("电动机轴承过热", top_k=5)
```

索引 schema： `id(TEXT) text(TEXT) vector(VECTOR,HNSW,6,FLOAT32,1024,COSINE) metadata(TEXT) category(TAG) tags(TAG) created_at(NUMERIC)`

## GraphService — 核心接口

```python
# 查询诊断路径（Device → Component → Fault → Solution）
paths = graph_service.query_diagnosis_path(keyword="轴承过热", limit=10)

# 设备搜索
devices = graph_service.find_devices(keyword="电动机", limit=10)

# 设备概览
overview = graph_service.get_device_overview(device_id)

# 按部件查故障
faults = graph_service.find_faults_by_component(component_id, limit=10)

# 按故障查解决方案
solutions = graph_service.find_solutions_by_fault(fault_id, verified_only=True)
```

## KnowledgeService — 核心接口

```python
from services.knowledge_service import get_knowledge_service

svc = get_knowledge_service()

# 导入文档：解析 PDF → 向量化 → 存入 Redis 向量库
result = await svc.import_document(
    file_url="/path/to/manual.pdf",
    file_type="pdf",
    category="维修手册",
    tags=["电动机", "轴承"]
)
# result: {
#   "file_name": "...",
#   "total_pages": N,
#   "text_count": 125,      # 入库文本块数
#   "image_count": 30,      # 入库图片数
#   "table_count": 12,     # 入库表格数
#   "sections": [...],
#   "process_time_ms": 3200
# }
```

编排流程：DocumentParserTool 解析 → TextEmbedding 批量向量化 → VectorService 入库

## 文件结构

```
services/
├── __init__.py
├── llm_service.py              # LLM 调用（chat / chat_with_tools / ReAct trace）
├── vector_service.py           # Redis 向量库（search / add_vector / add_vector_batch）
├── graph_service.py            # Neo4j 图数据库（query_diagnosis_path / find_* / CRUD）
└── knowledge_service.py        # 文档导入编排（import_document: 解析→向量化→入库）
```

## 日志输出点

关键位置输出 DEBUG/INFO 级别日志：

| 位置 | 级别 | 内容 |
|------|------|------|
| `llm_service.py` chat() | DEBUG | model / stream / msg_count |
| `llm_service.py` chat_with_tools() finish | INFO | 迭代次数、总耗时 |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 大模型 | 阿里云百炼 DashScope (qwen-plus) | 国产、Qwen系列强、API稳定 |
| 向量库 | Redis Stack (FT.SEARCH KNN) | 国产环境友好、高性能 |
| 图数据库 | Neo4j | 国产化支持好、Java生态完善、Cypher简洁 |

## 注意事项

1. **连接池**：`httpx.AsyncClient(timeout=60s)` 复用连接（max_keepalive_connections=20）
2. **单例**：每个服务通过 `get_xxx_service()` 全局单例，避免重复创建连接
3. **向量维度**：统一 1024 维（text-embedding-v4），Redis schema 在首次访问时自动创建
4. **异步**：所有 I/O 操作使用 async/await，不阻塞事件循环
5. **Redis 索引迁移**：`add_vector` 时自动 FT.ALTER 添加 category/tags 字段（静默忽略已存在错误）