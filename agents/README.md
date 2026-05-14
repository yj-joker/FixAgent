# Agents 模块

## 模块概述

Agent 是系统的**核心智能组件**，采用**混合架构**：

- **Orchestrator（编排器）**：意图识别 + 模式路由，简单对话直接走 LLM
- **ReAct 循环**：复杂任务（检索/诊断/指引）由子 Agent 内部自主决策工具调用
- **Function Calling**：MemoryAgent 用单次 function calling 做记忆整理

设计原则：简单场景快速响应，复杂场景深度推理。

## 架构图

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────┐
│               OrchestratorAgent (调度中枢)             │
│                                                        │
│  _resolve_mode() → 解析用户指定模式                   │
│  IntentionRecognizer.recognize() → CHAT模式自动识别意图 │
│  _map_intention_to_mode() → 映射为 AgentMode           │
│  _dispatch() → 路由到对应 handler                      │
└──────────────────────────────────────────────────────┘
           │
           ├── CHAT ──► 直接 LLM 对话
           │
           ├── RETRIEVAL ──► RetrievalAgent.run_with_react()
           │     ReAct 循环: Think → 检索 → 评估 → 追问/返回
           │
           ├── DIAGNOSIS ──► DiagnosisAgent.run_with_react()
           │     ReAct 循环: Think → 检索 → 图谱查询 → 推理 → 诊断
           │
           ├── GUIDANCE ──► GuidanceAgent.run_with_react()
           │     ReAct 循环: Think → 检索流程 → 生成步骤 → 安全校验
           │
           └── FULL ──► _execute_full_pipeline() 串行调用三个子Agent
```

## Agent 列表

| Agent | 文件 | 职责 | 执行模式 |
|-------|------|------|---------|
| `orchestrator_agent` | orchestrator_agent.py | 调度中枢 | 意图识别、模式分发、结果汇总 |
| `retrieval_agent` | retrieval_agent.py | 知识检索 | `run_with_react()` — ReAct |
| `diagnosis_agent` | diagnosis_agent.py | 故障诊断 | `run_with_react()` — ReAct |
| `guidance_agent` | guidance_agent.py | 作业指引 | `run_with_react()` — ReAct |
| `memory_agent` | memory_agent.py | 记忆整理 | `run()` — function calling + Pydantic 校验 |
| `intention_recognizer` | intention/recognizer.py | 意图识别 | LLM 轻量级识别 + 关键词兜底 |

## Agent 基类

`BaseAgent` 定义统一执行流程和异常处理模板：

- `run()` — 标准模板方法：`_build_messages` → `_call_llm` → `_process_response`
- `run_with_react()` — ReAct 入口，收集工具列表 → `chat_with_tools()` → 记录 `react_trace` 到 metadata
- `run_stream()` — 流式输出，yield 每个 token
- `run_with_context()` — 便捷方法，构造 `AgentInput` 后调用 `run()`

所有子 Agent 继承 `BaseAgent`，覆盖：

- `name` / `description` 属性
- `get_system_prompt()` — 返回角色定义提示词
- `get_tools()` — 返回可用工具列表（ReAct Agent 必须实现）
- `_build_messages()` — 自定义消息构建（MemoryAgent 覆盖）

异常处理：任意环节失败返回 `AgentOutput` 的友好提示 + `metadata.status="error"`，不抛出。

## 意图识别模块

`agents/intention/` — LLM 轻量级识别优先 + 关键词兜底：

```
recognize(message)
    ├── _llm_recognize() → qwen-turbo + temperature=0.1
    └── fallback_recognize() → 关键词扫描（置信度 0.6）
```

意图类型：`QUERY_KNOWLEDGE` / `TROUBLESHOOT` / `SEEK_GUIDANCE` / `SUBMIT_CASE` / `GENERAL_CHAT`

## 文件结构

```
agents/
├── __init__.py
├── base_agent.py                  # Agent基类，含 run()/run_with_react()/run_stream()
├── orchestrator_agent.py          # 调度中枢，全模式可用
├── retrieval_agent.py             # 知识检索 ReAct Agent
├── diagnosis_agent.py             # 故障诊断 ReAct Agent
├── guidance_agent.py              # 作业指引 ReAct Agent
├── memory_agent.py                # 记忆整理 function calling Agent
├── intention/                     # 意图识别子包
│   ├── recognizer.py              # LLM识别器（qwen-turbo）
│   ├── prompts.py                 # 识别提示词
│   └── fallback.py                # 关键词兜底
```

## 与其他模块的关系

```
agents/ (Agent层)
    ├── services/llm_service.py — chat()/chat_with_tools()
    ├── tools/ — 各 Agent 的可用工具（通过 get_tools() 注入）
    ├── embeddings/text_embedding.py — MemoryAgent 向量化存储
    └── services/vector_service.py — MemoryAgent 事实检索
```

## ReAct Trace 可观测性

`run_with_react()` 执行后，结果写入 `AgentOutput.metadata`：

```json
{
  "execution_mode": "react",
  "react_trace": [
    {
      "iteration": 1,
      "action": "tool_call",
      "tool_calls": [
        {
          "name": "knowledge_retrieval",
          "arguments": {"query": "电动机轴承过热", "top_k": 5},
          "result_summary": "找到5条相关知识..."
        }
      ],
      "duration_ms": 1840
    },
    {
      "iteration": 2,
      "action": "finish",
      "content_preview": "电动机轴承过热通常由以下原因引起...",
      "duration_ms": 1200
    }
  ],
  "react_iterations": 2
}
```

## 日志输出点

关键位置输出 INFO 级别日志：

| 位置 | 日志内容 |
|------|---------|
| `api/main.py` chat() | 请求入参（session/mode/消息长度）、完成耗时 |
| `api/main.py` memory_consolidate() | 对话条数、完成耗时、Agent 错误详情 |
| `services/llm_service.py` chat() | model/stream/msg_count |
| `services/llm_service.py` chat_with_tools() | 总迭代次数、最终耗时 |
| `agents/memory_agent.py` JSON解析失败 | raw content 前100字符 + 异常类型 |

## 注意事项

1. **日志级别**：生产环境将 `logging.basicConfig(level=logging.INFO)` 改为 `WARNING`
2. **ReAct 迭代上限**：默认 max_iterations=10，超出抛 RuntimeError
3. **MemoryAgent 独立性**：不通过 Orchestrator 调度，直接由 `/ai/memory/consolidate` 调用
4. **流式输出**：仅 CHAT 模式支持流式，其他模式返回占位提示