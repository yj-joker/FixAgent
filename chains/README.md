# Chains 模块

## 架构变更

原设计为 LangChain LCEL 管道式流程编排（检索链/诊断链/指引链/完整流程链），切换为**混合 ReAct 架构**后，执行策略从"固定管道"变为"LLM 自主编排"。

**已删除的链文件：**

| 文件 | 原因 |
|------|------|
| `retrieval_chain.py` | ReAct 替代 — RetrievalAgent 内部 LLM 自主编排检索策略 |
| `diagnosis_chain.py` | ReAct 替代 — DiagnosisAgent 内部 LLM 自主编排诊断策略 |
| `guidance_chain.py` | ReAct 替代 — GuidanceAgent 内部 LLM 自主编排指引策略 |
| `pipeline.py` | Orchestrator 替代 — `_execute_full_pipeline()` 直接串行调用子 Agent |

## 保留文件

| 文件 | 职责 | 原因 |
|------|------|------|
| `orchestrator.py` | `IntentionType → AgentMode` 路由映射 | 路由逻辑是分发层的，与 ReAct/管道无关，任何架构都需要 |
| `__init__.py` | 包初始化 | 保留 |

## 文件结构

```
chains/
├── __init__.py
├── README.md
└── orchestrator.py    # 意图到模式的映射函数
```
