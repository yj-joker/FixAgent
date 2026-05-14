# Tools 模块

## 模块概述

Tools 是 Agent 的**能力扩展层**，将外部能力封装为统一接口，供 Agent 在 ReAct 循环中调用。

核心类 `BaseTool` 基于**模板方法模式**，统一异常处理 + OpenAI function calling schema 生成。

## 工具列表

| 工具 | 文件 | 描述 | 状态 |
|------|------|------|------|
| `knowledge_retrieval` | knowledge_retrieval_tool.py | 知识库向量检索 | **已实现** |
| `fact_retrieval` | fact_retrieval_tool.py | 事实向量检索（MemoryAgent专用） | **已实现** |
| `graph_query_diagnosis_path` | graph_query_tool.py | Neo4j图谱诊断路径查询 | **已实现** |
| `graph_search_devices` | graph_query_tool.py | Neo4j设备搜索 | **已实现** |
| `yolo_detect` | yolo_tool.py | YOLO目标检测 | TODO 占位 |
| `sam_segment` | sam_tool.py | SAM图像分割 | TODO 占位 |
| `document_parser` | document_tool.py | 文档解析 | **已实现**（PDF/Word/TXT，文本/表格/图片提取） |

## BaseTool 基类

所有工具继承 `BaseTool`，只需实现 `_execute()` 写业务逻辑：

```python
from tools.base_tool import BaseTool, ToolResult, ToolException

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "工具描述，供 LLM 理解何时调用"

    async def _execute(self, **kwargs) -> Any:
        # 正常业务逻辑
        # 失败时抛 ToolException(code, message)
        if failed:
            raise ToolException(code="MY_ERROR", message="业务错误描述")
        return result  # 任意可序列化类型
```

- `run(**kwargs) -> ToolResult` — 模板方法，统一 try/execute/catch，返回结构化结果
- `to_openai_schema() -> dict` — 生成 OpenAI function calling 格式定义，供 `chat_with_tools()` 使用
- `get_parameters_schema() -> dict` — 默认空参数 schema，子类可覆盖

## Agent 中使用工具

```python
# 子Agent返回工具列表（由 run_with_react() 收集）
class RetrievalAgent(BaseAgent):
    def get_tools(self) -> List[BaseTool]:
        return [get_knowledge_retrieval_tool()]

# Orchestrator 通过 set_xxx_agent() 注入子Agent
orchestrator = get_orchestrator_agent()
orchestrator.set_retrieval_agent(get_retrieval_agent())
```

ReAct 循环内部自动处理：LLM 判断调用时机 → `chat_with_tools()` 分发 handler → 结果写入 message history。

## 文件结构

```
tools/
├── __init__.py
├── base_tool.py                  # 基类: BaseTool / ToolResult / ToolException / ToolError
├── knowledge_retrieval_tool.py   # 知识库向量检索
├── fact_retrieval_tool.py        # 事实向量检索（MemoryAgent 专用）
├── graph_query_tool.py           # Neo4j 图谱查询（诊断路径 + 设备搜索）
├── yolo_tool.py                  # TODO: YOLO 目标检测
├── sam_tool.py                   # TODO: SAM 图像分割
└── document_tool.py              # 文档解析（PDF/Word/TXT，文本/表格/图片提取）
```

## 与其他模块的关系

```
tools/
    ├── services/vector_service.py — 向量检索（knowledge_retrieval / fact_retrieval）
    ├── services/graph_service.py — 图谱查询（graph_query_tool）
    ├── embeddings/text_embedding.py — 向量生成（fact_retrieval）
    └── agents/ — 通过 get_tools() 注入 ReAct 循环
```

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 向量库 | Redis Search (FT.SEARCH KNN) | 国产环境友好、高性能 |
| 图数据库 | Neo4j (Cypher) | 国产化支持好、Java生态完善 |
| 目标检测 | YOLOv8 | 速度快、预训练模型 |
| 图像分割 | SAM | Meta开源、高精度 |

## 注意事项

1. **异常规范**：业务错误抛 `ToolException(code, message)`，未知异常自动捕获为 `code="TOOL_ERROR"`
2. **参数命名**：OpenAI schema 中使用 snake_case，LLM 输出时自动转换
3. **工具数量**：ReAct 循环中工具过多会影响 LLM 决策准确率，建议每个 Agent 不超过 3 个工具
4. **TODO 工具**：yolo/sam/document 为预留接口，当前 ReAct 流程中未注册