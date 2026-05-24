# 检修任务闭环与证据浏览 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏既有知识搜索和 AI 对话兼容行为的前提下，完成以 Java 为业务权威、Python 为 AI/RAG 能力端的检修任务闭环 MVP 与可追溯证据浏览能力。

**Architecture:** Java 维护手册版本、索引任务、检修任务、计划审批、执行验收、案例审核与对外业务 API；Python 维护文档解析、向量索引、证据读取和结构化计划草案生成。耗时索引通过 RabbitMQ 异步交付，计划生成与 Java 图谱校验使用受保护的内部 HTTP 契约。

**Tech Stack:** Java 21, Spring Boot 3.5, MyBatis-Plus, Spring Data Neo4j, WebClient, RabbitMQ, MinIO, MySQL, Python 3, FastAPI, Pydantic 2, Redis Stack, `httpx`, `aio-pika`, `pdfplumber`, `PyMuPDF`.

---

## 执行门禁

本文件是实施规划，不是编码授权。任何代理或新会话开始修改业务代码前，必须先读取：

- `D:\FixAgent\优化方向.md`，它是业务与架构决策的唯一依据。
- `D:\FixAgent\docs\superpowers\plans\2026-05-24-maintenance-task-closed-loop-implementation.md`，它是任务顺序与验证依据。
- 用户针对“开始实现/帮我写代码”的明确批准。

本计划文档在当前文档分支中生成；它本身不包含业务代码修改。进入实现时，应在获得批准后为实际编码工作创建独立分支或工作树。

两仓库根目录：

| 仓库 | 根目录 | 职责 |
|---|---|---|
| Python AI/RAG | `D:\FixAgent` | 解析、索引、证据读取、计划草案、Java 图谱客户端 |
| Java Backend | `C:\Users\27202\Desktop\weixiu\weixiu` | 业务主数据、状态机、审批、图谱权威、对外 API |

提交策略：

- Python 与 Java 是两个 Git 仓库，每个任务在各自仓库独立提交。
- 跨端任务先提交提供契约的一端，再提交消费契约的一端；联调通过后再进入后续任务。
- 不将真实访问密钥、预签名 URL 或现场图片内容提交到仓库。
- 每个提交命令执行前先运行 `git status --short`；若列出的目录中存在非本任务改动，须将文档中的目录级 `git add` 展开成仅包含本任务文件的逐文件暂存命令。

## 当前基线与适配决策

| 已有实现 | 实施决策 |
|---|---|
| Python `KnowledgeService` 已保存 `document_id`、`document_version`、导入状态，并提供 `/ai/knowledge/import`、`/ai/knowledge/search` | 扩展为版本隔离的分块浏览与索引任务处理，不重新设计搜索返回 |
| Python `VectorService` 已支持 `document_id`、`chunk_type`、`document_version` TAG 过滤 | 在此基础上新增清单枚举、顺序分页和单块上下文读取 |
| Python 搜索已把 `image_summary` 映射为 `image` 并保留来源标记 | 加回归测试保护，不改已有兼容语义 |
| Java `AiChatRequest` 已包含 `images: List<String>`，`AiServiceImpl` 已将图片转换为 Base64 下发 | 不再做字段迁移，仅补契约与流式链路测试，并在任务计划生成复用受控图片输入 |
| Java `MioIOUpLoadServiceImpl.getObjectName()` 已设置上传 `contentType` | Java 侧保留实现并补测试；Python 自管文件上传路径仍需补 MIME 与稳定定位 |
| Java 已有 `MaintenanceManual`、详情预签名 URL、`KnowledgeImportProducer`/`KnowledgeResultListener` | 新增版本表、索引任务表和持久状态切换，替换仅存 Redis 临时任务状态的闭环不足 |
| Java 已有 `GraphQueryService`，但只有 `faultExists()` 和 `solutionExists()` | 新增真实故障到方案路径校验内部接口，作为 Python 唯一图谱验证权威 |
| Java `CaseRecordServiceImpl.save/update()` 当前立即生成 embedding 并写 Neo4j | 改为从审核通过的草稿发布正式案例，未审核内容不得入图谱 |
| Java 对外路由现用 `/weixiu/...` | 将设计中的 `/api/manuals`、`/api/maintenance-tasks` 映射到 `/weixiu/maintenance-manual` 与 `/weixiu/maintenance-task`，保持项目路由风格 |

## 文件结构规划

### Python 仓库

| 文件 | 动作 | 单一职责 |
|---|---|---|
| `requirements.txt` | Modify | 增加 RabbitMQ 异步客户端依赖 |
| `config/settings.py` | Modify | 增加 RabbitMQ、Java 内部接口与服务令牌配置 |
| `schemas/request.py` | Modify | 增加索引任务、分块查询、计划草案与图谱校验输入模型 |
| `schemas/response.py` | Modify | 增加文档清单、分块详情、计划草案与异步结果模型 |
| `services/vector_service.py` | Modify | 提供 manifest 列表、顺序分块读取、单块上下文与版本安全删除 |
| `services/file_storage.py` | Modify | 读取稳定对象定位，正确上传 MIME，不长期存储临时 PDF URL |
| `services/knowledge_service.py` | Modify | 保存 `table_rows`、稳定定位、索引版本与幂等导入状态 |
| `services/knowledge_queue_consumer.py` | Create | 消费 Java 索引任务并发布结果消息 |
| `services/java_graph_client.py` | Create | 携带服务令牌调用 Java 权威图谱校验接口 |
| `services/plan_draft_service.py` | Create | 限定手册范围后组装结构化计划草案 |
| `services/case_index_service.py` | Create | 仅索引审核通过的正式案例 |
| `tools/graph_query_tool.py` | Modify | 将图谱诊断工具切换至 Java 内部客户端 |
| `agents/review_agent.py` | Modify | 使用 Java 路径验证结果，不再以本地图节点存在作为通过条件 |
| `api/main.py` | Modify | 注册知识浏览与内部计划端点、服务认证、队列生命周期 |
| `manual_tests/test_runner.py` | Modify | 允许自动用例失败以非零状态结束 |
| `manual_tests/run_automated.py` | Create | 无交互运行指定自动测试模块 |
| `manual_tests/test_knowledge_service.py` | Modify | 覆盖表格结构与版本化导入 |
| `manual_tests/test_vector_service.py` | Modify | 覆盖 manifest 与分块读取 |
| `manual_tests/test_api_main.py` | Modify | 覆盖知识浏览和计划端点 |
| `manual_tests/test_java_graph_client.py` | Create | 覆盖内部认证与图谱降级 |
| `manual_tests/test_plan_draft_service.py` | Create | 覆盖有效证据与降级警告 |
| `manual_tests/test_knowledge_queue_consumer.py` | Create | 覆盖队列幂等与结果回传 |
| `manual_tests/test_case_index_service.py` | Create | 覆盖正式案例索引发布门禁 |
| `api/README.md`, `schemas/README.md`, `services/README.md` | Modify | 同步实际内部契约及兼容边界 |

### Java 仓库

| 文件 | 动作 | 单一职责 |
|---|---|---|
| `src/main/resources/application.yml` | Modify | 改为环境变量占位并声明 Python/内部令牌配置 |
| `src/main/resources/application-dev.yml` | Modify | 去除已跟踪真实凭据，使用开发环境变量占位 |
| `src/main/resources/fix.sql` | Modify | 新增手册版本、索引任务、检修闭环与案例草稿表 |
| `src/main/java/ai/weixiu/config/AiIntegrationProperties.java` | Create | 绑定 Python 地址与内部令牌 |
| `src/main/java/ai/weixiu/config/WebClientConfig.java` | Modify | 基于配置构造 Python 客户端并发送内部令牌 |
| `src/main/java/ai/weixiu/config/InternalApiInterceptor.java` | Create | 保护 Java 内部图谱端点 |
| `src/main/java/ai/weixiu/config/WebMvcConfig.java` | Modify | 注册 `/internal/**` 鉴权拦截 |
| `src/main/java/ai/weixiu/entity/MaintenanceManual.java` | Modify | 增加适用范围、服务版本和候选索引状态 |
| `src/main/java/ai/weixiu/entity/ManualVersion.java` | Create | 持久保存每个源文件版本及稳定对象定位 |
| `src/main/java/ai/weixiu/entity/ManualIndexJob.java` | Create | 持久保存异步索引生命周期 |
| `src/main/java/ai/weixiu/mapper/ManualVersionMapper.java` | Create | 手册版本 CRUD |
| `src/main/java/ai/weixiu/mapper/ManualIndexJobMapper.java` | Create | 索引任务 CRUD |
| `src/main/java/ai/weixiu/pojo/dto/ManualVersionCreateDTO.java` | Create | 上传版本输入 |
| `src/main/java/ai/weixiu/pojo/vo/ManualIndexStatusVO.java` | Create | 返回服务版本和索引结果 |
| `src/main/java/ai/weixiu/pojo/vo/ManualChunkVO.java` | Create | Java 证据分块响应 |
| `src/main/java/ai/weixiu/pojo/vo/ManualSourceVO.java` | Create | 原 PDF 临时访问响应 |
| `src/main/java/ai/weixiu/service/MaintenanceManualService.java` | Modify | 声明版本/索引/证据操作 |
| `src/main/java/ai/weixiu/service/impl/MaintenanceManualServiceImpl.java` | Modify | 编排上传、状态切换、原文访问 |
| `src/main/java/ai/weixiu/mq/KnowledgeImportProducer.java` | Modify | 发送稳定定位与版本化任务契约 |
| `src/main/java/ai/weixiu/mq/KnowledgeResultListener.java` | Modify | 幂等消费结果并切换服务版本 |
| `src/main/java/ai/weixiu/pojo/dto/KnowledgeIndexResultMessage.java` | Create | 映射 Python 索引结果契约 |
| `src/main/java/ai/weixiu/controller/MaintenanceManualController.java` | Modify | 对外暴露版本、索引与证据浏览路由 |
| `src/main/java/ai/weixiu/pojo/dto/GraphValidationRequest.java` | Create | 内部图谱验证输入 |
| `src/main/java/ai/weixiu/pojo/vo/GraphValidationVO.java` | Create | 真实路径校验结果 |
| `src/main/java/ai/weixiu/service/GraphQueryService.java` | Modify | 声明路径验证方法 |
| `src/main/java/ai/weixiu/service/impl/GraphQueryServiceImpl.java` | Modify | 查询真实 `Fault -> Solution` 路径 |
| `src/main/java/ai/weixiu/controller/InternalGraphController.java` | Create | 仅供 Python 调用的权威验证端点 |
| `src/main/java/ai/weixiu/entity/MaintenanceTask.java` | Create | 检修任务聚合根 |
| `src/main/java/ai/weixiu/entity/MaintenancePlan.java` | Create | 计划版本与审批状态 |
| `src/main/java/ai/weixiu/entity/MaintenanceStep.java` | Create | 步骤、风险与执行状态 |
| `src/main/java/ai/weixiu/entity/PlanEvidence.java` | Create | 固定手册/图谱证据 |
| `src/main/java/ai/weixiu/entity/ExecutionEvidence.java` | Create | 现场执行证据 |
| `src/main/java/ai/weixiu/entity/TaskAcceptance.java` | Create | 复测验收记录 |
| `src/main/java/ai/weixiu/entity/CaseRecordDraft.java` | Create | 待审核案例草稿 |
| `src/main/java/ai/weixiu/mapper/MaintenanceTaskMapper.java`、`MaintenancePlanMapper.java`、`MaintenanceStepMapper.java`、`PlanEvidenceMapper.java`、`ExecutionEvidenceMapper.java`、`TaskAcceptanceMapper.java`、`CaseRecordDraftMapper.java` | Create | 新增 MySQL 实体的 MyBatis-Plus 映射 |
| `src/main/java/ai/weixiu/service/MaintenanceTaskService.java` | Create | 任务状态机业务接口 |
| `src/main/java/ai/weixiu/service/impl/MaintenanceTaskServiceImpl.java` | Create | 生成、审批、执行、验收与草稿生成 |
| `src/main/java/ai/weixiu/service/PythonPlanClient.java` | Create | 调 Python 计划草案端点 |
| `src/main/java/ai/weixiu/service/impl/PythonPlanClientImpl.java` | Create | WebClient 调用与超时/警告转换 |
| `src/main/java/ai/weixiu/service/PythonCaseIndexClient.java` | Create | 通知 Python 索引审核通过案例 |
| `src/main/java/ai/weixiu/service/impl/PythonCaseIndexClientImpl.java` | Create | 调受保护案例索引端点 |
| `src/main/java/ai/weixiu/controller/MaintenanceTaskController.java` | Create | 任务闭环对外路由 |
| `src/main/java/ai/weixiu/service/CaseRecordService.java` | Modify | 提供审核通过后正式发布能力 |
| `src/main/java/ai/weixiu/service/impl/CaseRecordServiceImpl.java` | Modify | 阻止未审核数据直接 embedding/入图 |
| `src/main/java/ai/weixiu/controller/CaseRecordController.java` | Modify | 增加草稿审核入口 |
| 各任务 `Files` 小节中列出的 `src/test/java/ai/weixiu/**/**Test.java` | Create | 新增隔离单元/控制器测试类 |

## 契约约定

下列名称在所有任务中保持不变：

| 概念 | Java 表达 | Python 表达 |
|---|---|---|
| 手册业务 ID | `Long manualId` | `str manual_id` |
| 索引版本 | `String indexVersion`，格式 `v1`, `v2` | `str index_version` |
| Python 文档 ID | Java 构造 `manual:{manualId}:{indexVersion}` | `document_id` 原样使用 |
| 索引任务 ID | `String jobId`，Java 生成 | `job_id` 原样回传 |
| 计划请求去重 ID | `String requestId`，Java 生成 | `request_id` 原样回传 |
| 图谱校验 ID | Java 生成或回传 `validationId` | `validation_id` |
| 风险枚举 | `LOW`, `MEDIUM`, `HIGH` | 同名字符串 |
| 图谱降级警告 | `GRAPH_VALIDATION_UNAVAILABLE` | 同名警告码 |

## Task 0: 建立安全配置与可执行测试门禁

**Files:**
- Modify: `D:\FixAgent\manual_tests\test_runner.py`
- Create: `D:\FixAgent\manual_tests\run_automated.py`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\resources\application.yml`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\resources\application-dev.yml`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\config\ConfigurationSafetyTest.java`

- [ ] **Step 1: 为 Python 自动测试失败状态编写验证用例**

在 `manual_tests/run_automated.py` 中先加载一个故意返回失败的临时 case，断言 `run_auto_cases()` 能抛出 `AssertionError`；随后该 runner 才可作为后续任务的门禁入口。

```python
from test_runner import run_auto_cases

def verify_failure_propagates():
    try:
        run_auto_cases([{
            "name": "gate-probe",
            "run": lambda: False,
            "check": bool,
        }])
    except AssertionError:
        return True
    return False
```

- [ ] **Step 2: 运行失败验证，确认现状尚未提供非零门禁**

Run from `D:\FixAgent`:

```powershell
python -c "import sys; sys.path.insert(0, 'manual_tests'); from run_automated import verify_failure_propagates; raise SystemExit(0 if verify_failure_propagates() else 1)"
```

Expected: FAIL/exit `1`，因为当前 `run_auto_cases()` 仅打印失败，不抛异常。

- [ ] **Step 3: 最小化修改 Python runner 并提供批量运行入口**

在 `manual_tests/test_runner.py` 的 `run_auto_cases()` 结束处加入失败抛出逻辑：

```python
    if failed:
        raise AssertionError(f"自动测试失败: {failed}/{total}")
```

在 `manual_tests/run_automated.py` 中实现模块列表运行器，默认运行本计划涉及的自动模块，并允许命令行指定子集：

```python
DEFAULT_MODULES = [
    "test_api_main",
    "test_knowledge_service",
    "test_vector_service",
    "test_review_agent",
]

def main(modules=None):
    for name in modules or DEFAULT_MODULES:
        importlib.import_module(name).auto_test()

if __name__ == "__main__":
    main(sys.argv[1:] or None)
```

- [ ] **Step 4: 将 Java 配置中的已跟踪凭据改为环境变量引用并建立防回归测试**

`application*.yml` 中的敏感配置只保留环境变量引用，例如：

```yaml
spring:
  datasource:
    password: ${DB_PASSWORD:}
ai:
  python-service-url: ${AI_PYTHON_SERVICE_URL:http://127.0.0.1:8000}
  internal-token: ${AI_INTERNAL_TOKEN:}
minio:
  access-key: ${MINIO_ACCESS_KEY:}
  secret-key: ${MINIO_SECRET_KEY:}
```

在 `ConfigurationSafetyTest.java` 中读取两份 tracked YAML 文本并拒绝出现非环境引用的敏感键值：

```java
assertThat(yaml).contains("${DB_PASSWORD:}");
assertThat(yaml).contains("${AI_INTERNAL_TOKEN:}");
assertThat(yaml).doesNotContain("password: root");
```

实际凭据轮换属于部署操作，必须在代码替换占位后由密钥持有人完成，新的值不得写回 Git。

- [ ] **Step 5: 验证测试门禁与 Java 配置测试**

Run:

```powershell
python manual_tests/run_automated.py
```

Expected: Python 既有自动用例全部 `PASS` 且 exit `0`。

Run from Java root:

```powershell
mvn -Dtest=ConfigurationSafetyTest test
```

Expected: `BUILD SUCCESS`。

- [ ] **Step 6: 分仓提交**

Python:

```powershell
git add manual_tests/test_runner.py manual_tests/run_automated.py
git commit -m "test: make python automatic checks fail fast"
```

Java:

```powershell
git add src/main/resources/application.yml src/main/resources/application-dev.yml src/test/java/ai/weixiu/config/ConfigurationSafetyTest.java
git commit -m "security: remove tracked service credentials"
```

## Task 1: 扩展 Python 文档清单、结构化表格与证据读取原语

**Files:**
- Modify: `D:\FixAgent\schemas\request.py`
- Modify: `D:\FixAgent\schemas\response.py`
- Modify: `D:\FixAgent\services\knowledge_service.py`
- Modify: `D:\FixAgent\services\vector_service.py`
- Modify: `D:\FixAgent\services\file_storage.py`
- Modify: `D:\FixAgent\api\main.py`
- Modify: `D:\FixAgent\manual_tests\test_knowledge_service.py`
- Modify: `D:\FixAgent\manual_tests\test_vector_service.py`
- Modify: `D:\FixAgent\manual_tests\test_api_main.py`

- [ ] **Step 1: 写入表格元数据和版本清单失败用例**

在 `test_knowledge_service.py` 增加用例：导入 `document_id="manual:42:v2"` 和 `document_version="v2"` 时，表格调用必须携带原始行列，manifest 必须持有稳定版本字段：

```python
assert table_metadata["table_rows"] == [["名称", "值"], ["电压", "380V"]]
assert ready_manifest["document_id"] == "manual:42:v2"
assert ready_manifest["document_version"] == "v2"
assert ready_manifest["status"] == "ready"
```

- [ ] **Step 2: 写入 manifest 列表与顺序分块读取失败用例**

在 `test_vector_service.py` 中使用假的 Redis 返回两条 `document:*` 清单和同一版本的多个 `doc:*` 分块，要求：

```python
documents = svc.list_document_manifests()
page = svc.list_document_chunks("manual:42:v2", "table", page=1, page_size=20)
detail = svc.get_document_chunk("manual:42:v2", "chunk-2")

assert documents[0]["status"] == "ready"
assert page["items"][0]["metadata"]["page"] <= page["items"][1]["metadata"]["page"]
assert "vector" not in page["items"][0]
assert detail["context_before"] == ["前一块"]
```

- [ ] **Step 3: 运行新增测试以固定失败位置**

Run:

```powershell
python manual_tests/run_automated.py test_knowledge_service test_vector_service
```

Expected: FAIL，缺少 `table_rows`、`list_document_manifests()`、`list_document_chunks()` 与 `get_document_chunk()`。

- [ ] **Step 4: 实现最小存储能力**

`KnowledgeService` 保存表格行列：

```python
metadata={
    **common_metadata,
    "chunk_type": "table",
    "page": table.get("page"),
    "caption": table.get("caption", ""),
    "table_rows": table.get("rows", []),
}
```

`VectorService` 增加三项只读能力，所有分块查询强制 `document_id` 和允许类型集合。实现时将 Redis 响应解码集中到 `_decode_chunks()`，避免路由层解析存储结构：

```python
ALLOWED_KNOWLEDGE_CHUNK_TYPES = {"text", "table", "image", "image_summary"}

def _decode_hash_chunk(self, raw: dict) -> dict | None:
    if not raw:
        return None
    def value(name: str, default: str = "") -> str:
        item = raw.get(name.encode()) if name.encode() in raw else raw.get(name)
        return item.decode() if isinstance(item, bytes) else (item or default)
    return {
        "id": value("id"),
        "content": value("text"),
        "metadata": json.loads(value("metadata", "{}")),
    }

def _decode_chunks(self, search_result: list) -> list[dict]:
    items = []
    for index in range(1, len(search_result), 2):
        fields = search_result[index + 1]
        raw = {fields[offset]: fields[offset + 1] for offset in range(0, len(fields), 2)}
        item = self._decode_hash_chunk(raw)
        if item:
            items.append(item)
    return items

def list_document_manifests(self) -> list[dict]:
    items = []
    for key in self.redis.scan_iter(match=f"{self.DOCUMENT_KEY_PREFIX}*", count=1000):
        raw = self.redis.hget(key, "manifest")
        if raw:
            text = raw.decode() if isinstance(raw, bytes) else raw
            items.append(json.loads(text))
    return sorted(items, key=lambda item: item.get("updated_at", 0), reverse=True)

def _knowledge_chunks(self, document_id: str, chunk_type: str) -> list[dict]:
    if not document_id or chunk_type not in ALLOWED_KNOWLEDGE_CHUNK_TYPES:
        raise ValueError("unsupported knowledge chunk query")
    query = build_redis_filter(document_id=document_id, chunk_type=chunk_type)
    raw = self.redis.execute_command(
        "FT.SEARCH", self.INDEX_NAME, query,
        "RETURN", "3", "id", "text", "metadata",
        "LIMIT", "0", "10000", "DIALECT", "2",
    )
    return sorted(
        self._decode_chunks(raw),
        key=lambda item: (item["metadata"].get("page") or 0, item["id"]),
    )

def list_document_chunks(self, document_id: str, chunk_type: str, page: int, page_size: int) -> dict:
    size = max(1, min(page_size, 100))
    number = max(1, page)
    items = self._knowledge_chunks(document_id, chunk_type)
    start = (number - 1) * size
    return {"items": items[start:start + size], "page": number, "page_size": size, "total": len(items)}

def get_document_chunk(self, document_id: str, chunk_id: str) -> dict:
    raw = self.redis.hgetall(f"{self.VECTOR_KEY_PREFIX}{chunk_id}")
    item = self._decode_hash_chunk(raw)
    if not item or item["metadata"].get("document_id") != document_id:
        raise KeyError("knowledge chunk not found")
    ordered = self._knowledge_chunks(document_id, item["metadata"]["chunk_type"])
    index = next(i for i, value in enumerate(ordered) if value["id"] == chunk_id)
    return {"item": item, "context_before": ordered[max(0, index - 1):index],
            "context_after": ordered[index + 1:index + 2]}
```

实现约束：

- `list_document_manifests()` 只枚举 `document:*`。
- `put_document_manifest()` 将首次写入的 `created_at` 与每次更新的 `updated_at` 一并保存到 manifest JSON 内，避免详情 API 丢失 Hash 外层更新时间。
- 分块查询用 `document_id` 与 `chunk_type` 过滤，并按 `metadata.page`、`id` 稳定排序。
- `page_size` 限定 `1..100`。
- 返回值只含 `id`、`content` 和解析后的 `metadata`，不返回 `vector`。

- [ ] **Step 5: 定义知识浏览响应模型并注册兼容 API**

在 `schemas/response.py` 中新增：

```python
class KnowledgeDocumentManifest(BaseModel):
    document_id: str
    document_version: str | None = None
    status: str
    text_count: int = 0
    table_count: int = 0
    image_count: int = 0
    image_summary_count: int = 0

class KnowledgeChunkPageResponse(BaseResponse):
    items: list[dict]
    page: int
    page_size: int
    total: int
```

在 `api/main.py` 增加 Python 内部/迁移兼容读取入口：

```python
@app.get("/ai/knowledge/documents")
async def knowledge_documents():
    items = get_vector_service().list_document_manifests()
    return {"success": True, "data": items, "total": len(items)}

@app.get("/ai/knowledge/documents/{document_id}")
async def knowledge_document(document_id: str):
    manifest = get_vector_service().get_document_manifest(document_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"success": True, "data": manifest}

@app.get("/ai/knowledge/documents/{document_id}/chunks")
async def knowledge_chunks(document_id: str, type: str, page: int = 1, page_size: int = 20):
    return get_vector_service().list_document_chunks(document_id, type, page, page_size)

@app.get("/ai/knowledge/documents/{document_id}/chunks/{chunk_id}")
async def knowledge_chunk(document_id: str, chunk_id: str):
    try:
        return get_vector_service().get_document_chunk(document_id, chunk_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
```

- [ ] **Step 6: 固定现有搜索兼容行为**

在 `test_api_main.py` 中追加回归断言：

```python
assert search_item["metadata"]["chunk_type"] == "image"
assert search_item["metadata"]["source_chunk_type"] == "image_summary"
```

不改变 `/ai/knowledge/search` 的旧调用字段和响应外形。

- [ ] **Step 7: 运行 Python 门禁并提交**

Run:

```powershell
python manual_tests/run_automated.py test_knowledge_service test_vector_service test_api_main
```

Expected: PASS。

Commit:

```powershell
git add schemas/request.py schemas/response.py services/knowledge_service.py services/vector_service.py services/file_storage.py api/main.py manual_tests/test_knowledge_service.py manual_tests/test_vector_service.py manual_tests/test_api_main.py
git commit -m "feat: expose versioned knowledge evidence browsing"
```

## Task 2: 建立 Python RabbitMQ 索引 worker 与稳定文件定位读取

**Files:**
- Modify: `D:\FixAgent\requirements.txt`
- Modify: `D:\FixAgent\config\settings.py`
- Modify: `D:\FixAgent\schemas\request.py`
- Modify: `D:\FixAgent\schemas\response.py`
- Modify: `D:\FixAgent\services\file_storage.py`
- Create: `D:\FixAgent\services\knowledge_queue_consumer.py`
- Modify: `D:\FixAgent\api\main.py`
- Create: `D:\FixAgent\manual_tests\test_knowledge_queue_consumer.py`
- Modify: `D:\FixAgent\manual_tests\run_automated.py`

- [ ] **Step 1: 写入索引消息幂等和失败回传用例**

定义输入消息：

```python
payload = {
    "jobId": "idx-42-v2",
    "manualId": 42,
    "indexVersion": "v2",
    "documentId": "manual:42:v2",
    "file": {
        "storageBackend": "minio",
        "bucket": "weixiu-private-wendang",
        "objectKey": "abc.pdf",
        "contentType": "application/pdf",
    },
}
```

测试断言：

```python
assert published_result["jobId"] == "idx-42-v2"
assert published_result["status"] == "ready"
assert duplicate_delivery_did_not_reimport is True
assert failed_result["status"] == "failed"
```

- [ ] **Step 2: 运行测试确认尚无 worker**

Run:

```powershell
python manual_tests/run_automated.py test_knowledge_queue_consumer
```

Expected: FAIL，模块 `services.knowledge_queue_consumer` 不存在。

- [ ] **Step 3: 增加消息模型和配置**

在 `config/settings.py` 增加：

```python
rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
knowledge_import_queue = os.getenv("KNOWLEDGE_IMPORT_QUEUE", "knowledge.import.queue")
knowledge_exchange = os.getenv("KNOWLEDGE_EXCHANGE", "knowledge.exchange")
knowledge_result_routing_key = os.getenv("KNOWLEDGE_RESULT_ROUTING_KEY", "knowledge.result")
```

在 `schemas/request.py` 定义稳定定位模型：

```python
class KnowledgeIndexTask(BaseModel):
    job_id: str = Field(alias="jobId")
    manual_id: int = Field(alias="manualId")
    index_version: str = Field(alias="indexVersion")
    document_id: str = Field(alias="documentId")
    file: StableFileLocator
```

- [ ] **Step 4: 扩展文件存储适配器为 worker 提供本地可解析文件**

在 `services/file_storage.py` 新增：

```python
class StableFileLocator(BaseModel):
    storage_backend: str
    bucket: str | None = None
    object_key: str | None = None
    local_path: str | None = None
    content_type: str | None = None

def materialize_document(self, locator: StableFileLocator) -> str:
    """返回解析器可读取的临时或本地路径，不返回长期前端 URL。"""
```

`MinioStorage.materialize_document()` 以对象定位下载到临时文件；`LocalFileStorage` 验证本地路径后读取。`MinioStorage._upload()` 调用 `fput_object(..., content_type=...)`，保证 Python 自管上传的 PDF/图片 MIME 正确。

- [ ] **Step 5: 实现异步 consumer 与任务幂等**

使用 `aio-pika`，在 `requirements.txt` 增加：

```text
aio-pika>=9.4.0
```

`knowledge_queue_consumer.py` 提供：

```python
class KnowledgeQueueConsumer:
    async def process_message(self, payload: dict) -> dict:
        task = KnowledgeIndexTask.model_validate(payload)
        existing = self.vector_svc.get_document_manifest(task.document_id)
        if existing.get("job_id") == task.job_id and existing.get("status") == "ready":
            return self._ready_result_from_manifest(task, existing)
        local_source = self.file_storage.materialize_document(task.file)
        result = await self.knowledge_svc.import_document(
            file_url=local_source,
            document_id=task.document_id,
            document_version=task.index_version,
            replace_existing=False,
        )
        return self._ready_result(task, result)
```

发布结果必须包含 `jobId`、`manualId`、`indexVersion`、`documentId`、`status` 和统计/错误码。

- [ ] **Step 6: 绑定 FastAPI 生命周期**

`api/main.py` 使用 lifespan 启动/关闭 consumer；若 RabbitMQ 未配置或连接失败，日志明确记录，HTTP 搜索能力仍可启动用于本地调试。

- [ ] **Step 7: 运行 Python worker 测试并提交**

Run:

```powershell
python manual_tests/run_automated.py test_knowledge_queue_consumer test_knowledge_service test_api_main
```

Expected: PASS。

Commit:

```powershell
git add requirements.txt config/settings.py schemas/request.py schemas/response.py services/file_storage.py services/knowledge_queue_consumer.py api/main.py manual_tests/test_knowledge_queue_consumer.py manual_tests/run_automated.py
git commit -m "feat: consume versioned manual indexing jobs"
```

## Task 3: 将 Java 手册改为版本化索引主数据

**Files:**
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\resources\fix.sql`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\MaintenanceManual.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\ManualVersion.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\ManualIndexJob.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\ManualVersionMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\ManualIndexJobMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\ManualVersionCreateDTO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\KnowledgeIndexResultMessage.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\vo\ManualIndexStatusVO.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\MaintenanceManualService.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\MaintenanceManualServiceImpl.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mq\KnowledgeImportProducer.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mq\KnowledgeResultListener.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\controller\MaintenanceManualController.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\service\MaintenanceManualVersionServiceTest.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\mq\KnowledgeResultListenerTest.java`

- [ ] **Step 1: 写入“失败不替换服务版本”和“迟到结果不覆盖”测试**

`KnowledgeResultListenerTest` 使用 mock mapper/service，断言：

```java
assertThat(manual.getServingIndexVersion()).isEqualTo("v1");
listener.onResult(failedV2Message(), channel, 1L);
verify(manualService, never()).activateServingVersion(42L, "v2", "idx-v2");

listener.onResult(readyOldV1Message(), channel, 2L);
verify(manualService, never()).activateServingVersion(42L, "v1", "idx-old");
```

`MaintenanceManualVersionServiceTest` 断言首次上传在发消息前已经生成 `manualId`、版本、`jobId` 和稳定对象定位。

- [ ] **Step 2: 运行测试确认实体与服务方法不存在**

Run from Java root:

```powershell
mvn -Dtest=MaintenanceManualVersionServiceTest,KnowledgeResultListenerTest test
```

Expected: FAIL，新增实体或方法尚不存在。

- [ ] **Step 3: 增加数据库结构与实体**

`fix.sql` 添加：

```sql
ALTER TABLE maintenance_manual
    ADD COLUMN equipment_type VARCHAR(100) NULL,
    ADD COLUMN manual_type VARCHAR(50) NULL,
    ADD COLUMN serving_index_version VARCHAR(32) NULL,
    ADD COLUMN candidate_index_version VARCHAR(32) NULL,
    ADD COLUMN candidate_index_status VARCHAR(32) NULL;

CREATE TABLE manual_version (
    id BIGINT PRIMARY KEY,
    manual_id BIGINT NOT NULL,
    index_version VARCHAR(32) NOT NULL,
    storage_backend VARCHAR(20) NOT NULL,
    source_bucket VARCHAR(100) NOT NULL,
    source_object_key VARCHAR(500) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    created_at DATETIME NOT NULL,
    UNIQUE KEY uk_manual_version (manual_id, index_version)
);

CREATE TABLE manual_index_job (
    job_id VARCHAR(80) PRIMARY KEY,
    manual_id BIGINT NOT NULL,
    index_version VARCHAR(32) NOT NULL,
    document_id VARCHAR(160) NOT NULL,
    status VARCHAR(32) NOT NULL,
    text_count INT DEFAULT 0,
    table_count INT DEFAULT 0,
    image_count INT DEFAULT 0,
    image_summary_count INT DEFAULT 0,
    error_code VARCHAR(80) NULL,
    error_message VARCHAR(500) NULL,
    created_at DATETIME NOT NULL,
    completed_at DATETIME NULL
);
```

- [ ] **Step 4: 实现版本创建和消息契约**

新增服务方法：

```java
ManualIndexStatusVO createVersion(Long manualId, ManualVersionCreateDTO dto, MultipartFile file);
void recordIndexResult(KnowledgeIndexResultMessage result);
ManualVersion requireServingVersion(Long manualId);
```

`KnowledgeImportProducer` 的消息体必须使用：

```java
message.put("jobId", job.getJobId());
message.put("manualId", job.getManualId());
message.put("indexVersion", job.getIndexVersion());
message.put("documentId", job.getDocumentId());
message.put("file", Map.of(
        "storageBackend", "minio",
        "bucket", BucketEnum.PRIVATE.getName(),
        "objectKey", version.getSourceObjectKey(),
        "contentType", version.getContentType()
));
```

`KnowledgeResultListener` 不再只写 Redis 一小时缓存，而是调用事务服务更新 `manual_index_job`；仅当结果 `jobId` 匹配当前候选版本且 `status=ready` 时切换 `servingIndexVersion`。

- [ ] **Step 5: 保持既有路由并新增版本端点**

保留 `/weixiu/maintenance-manual/save` 作为首次手册创建入口，使其同步创建 `v1` 索引任务；新增：

```java
@PostMapping("/{id}/versions")
public Result<ManualIndexStatusVO> createVersion(@PathVariable Long id,
                                                   @ModelAttribute ManualVersionCreateDTO dto,
                                                   @RequestParam("file") MultipartFile file) {
    return Result.success(maintenanceManualService.createVersion(id, dto, file));
}

@GetMapping("/{id}/index-jobs/{jobId}")
public Result<ManualIndexStatusVO> getIndexJob(@PathVariable Long id,
                                                @PathVariable String jobId) {
    return Result.success(maintenanceManualService.getIndexJob(id, jobId));
}
```

`PUT /update` 的带文件行为改为委派创建新版本，避免原地删除旧证据源文件。

- [ ] **Step 6: 运行 Java 测试并提交**

Run:

```powershell
mvn -Dtest=MaintenanceManualVersionServiceTest,KnowledgeResultListenerTest test
```

Expected: `BUILD SUCCESS`。

Commit:

```powershell
git add src/main/resources/fix.sql src/main/java/ai/weixiu/entity src/main/java/ai/weixiu/mapper src/main/java/ai/weixiu/pojo src/main/java/ai/weixiu/service src/main/java/ai/weixiu/mq src/main/java/ai/weixiu/controller/MaintenanceManualController.java src/test/java/ai/weixiu/service/MaintenanceManualVersionServiceTest.java src/test/java/ai/weixiu/mq/KnowledgeResultListenerTest.java
git commit -m "feat: version maintenance manual indexing lifecycle"
```

## Task 4: 由 Java 对外聚合手册证据浏览与原 PDF 访问

**Files:**
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\vo\ManualChunkVO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\vo\ManualChunkPageVO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\vo\ManualSourceVO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\PythonKnowledgeClient.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\PythonKnowledgeClientImpl.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\MaintenanceManualService.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\MaintenanceManualServiceImpl.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\controller\MaintenanceManualController.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\controller\MaintenanceManualEvidenceControllerTest.java`

- [ ] **Step 1: 写入控制器失败测试**

使用 `MockMvc` 与 mock service 覆盖以下请求：

```java
mockMvc.perform(get("/weixiu/maintenance-manual/42/versions/v2/chunks")
        .param("type", "table").param("page", "1").param("page_size", "20"))
    .andExpect(status().isOk())
    .andExpect(jsonPath("$.data.items[0].tableRows[0][0]").value("名称"));

mockMvc.perform(get("/weixiu/maintenance-manual/42/versions/v2/source")
        .param("page", "12"))
    .andExpect(status().isOk())
    .andExpect(jsonPath("$.data.page").value(12));
```

- [ ] **Step 2: 运行测试固定缺少接口的失败**

Run:

```powershell
mvn -Dtest=MaintenanceManualEvidenceControllerTest test
```

Expected: FAIL，端点及 VO 不存在。

- [ ] **Step 3: 实现 Java 到 Python 的只读证据客户端**

`PythonKnowledgeClient` 只暴露版本固定读取方法：

```java
ManualChunkPageVO listChunks(String documentId, String type, int page, int pageSize);
ManualChunkVO getChunk(String documentId, String chunkId);
```

服务层用统一规则生成 `documentId`：

```java
private String documentId(Long manualId, String version) {
    return "manual:" + manualId + ":" + version;
}
```

Java 不暴露 Python 的 `document_id` 选择权给前端，先检查该版本属于手册且索引成功，再发内部读取请求。

- [ ] **Step 4: 增加证据与原文端点**

`MaintenanceManualController` 新增：

```java
@GetMapping("/{id}/versions/{version}/chunks")
public Result<ManualChunkPageVO> chunks(@PathVariable Long id,
                                        @PathVariable String version,
                                        @RequestParam String type,
                                        @RequestParam(defaultValue = "1") int page,
                                        @RequestParam(name = "page_size", defaultValue = "20") int pageSize) {
    return Result.success(maintenanceManualService.listChunks(id, version, type, page, pageSize));
}

@GetMapping("/{id}/versions/{version}/chunks/{chunkId}")
public Result<ManualChunkVO> chunk(@PathVariable Long id,
                                   @PathVariable String version,
                                   @PathVariable String chunkId) {
    return Result.success(maintenanceManualService.getChunk(id, version, chunkId));
}

@GetMapping("/{id}/versions/{version}/source")
public Result<ManualSourceVO> source(@PathVariable Long id,
                                     @PathVariable String version,
                                     @RequestParam(required = false) Integer page) {
    return Result.success(maintenanceManualService.getSource(id, version, page));
}
```

`source()` 从 `ManualVersion.sourceObjectKey` 生成新的短时预签名地址，并原样返回请求页码；不读取或缓存 Python manifest 的旧 `source_file_url`。

- [ ] **Step 5: 验证并提交**

Run:

```powershell
mvn -Dtest=MaintenanceManualEvidenceControllerTest test
```

Expected: `BUILD SUCCESS`。

Commit:

```powershell
git add src/main/java/ai/weixiu/pojo/vo src/main/java/ai/weixiu/service src/main/java/ai/weixiu/controller/MaintenanceManualController.java src/test/java/ai/weixiu/controller/MaintenanceManualEvidenceControllerTest.java
git commit -m "feat: proxy manual evidence browsing through java"
```

## Task 5: 建立 Java 权威图谱路径验证与 Python 内部客户端

**Files:**
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\config\AiIntegrationProperties.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\config\WebClientConfig.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\config\InternalApiInterceptor.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\config\WebMvcConfig.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\GraphValidationRequest.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\vo\GraphValidationVO.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\GraphQueryService.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\GraphQueryServiceImpl.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\controller\InternalGraphController.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\service\GraphValidationServiceTest.java`
- Modify: `D:\FixAgent\config\settings.py`
- Create: `D:\FixAgent\services\java_graph_client.py`
- Modify: `D:\FixAgent\tools\graph_query_tool.py`
- Modify: `D:\FixAgent\agents\review_agent.py`
- Create: `D:\FixAgent\manual_tests\test_java_graph_client.py`
- Modify: `D:\FixAgent\manual_tests\test_graph_query_tool.py`
- Modify: `D:\FixAgent\manual_tests\test_review_agent.py`
- Modify: `D:\FixAgent\manual_tests\test_fix_agent.py`

- [ ] **Step 1: 写入 Java 真实路径校验测试**

测试 mock `Neo4jClient` 的返回，要求同名节点存在但没有 `HAS_SOLUTION` 关系时不通过：

```java
GraphValidationVO result = service.validateFaultSolutions(request);
assertThat(result.getMatches()).isEmpty();
assertThat(result.getUnverifiedItems().getFirst().getReason()).isEqualTo("NO_GRAPH_PATH");
```

- [ ] **Step 2: 写入 Python 调 Java 与降级测试**

`test_java_graph_client.py` 断言：

```python
assert captured_headers["X-Internal-Token"] == "unit-test-token"
assert result["status"] == "verified"

client.post.side_effect = RuntimeError("timeout")
assert await graph_client.validate(request) == {"status": "unavailable", "matches": [], "unverifiedItems": []}
```

`test_review_agent.py` 改为断言 ReviewAgent 不调用 `services.graph_service` 的节点存在检查；`test_graph_query_tool.py` 和 `test_fix_agent.py` 覆盖文本、设备搜索与图片路径工具均通过 Java client 查询。

- [ ] **Step 3: 运行两端失败测试**

Run:

```powershell
mvn -Dtest=GraphValidationServiceTest test
```

Run from Python root:

```powershell
python manual_tests/run_automated.py test_java_graph_client test_graph_query_tool test_review_agent
```

Expected: FAIL，真实路径接口和 Python 客户端未实现。

- [ ] **Step 4: 实现 Java 内部鉴权与路径校验**

Java 新服务方法：

```java
GraphValidationVO validateFaultSolutions(GraphValidationRequest request);
```

核心 Cypher 必须匹配真实关系：

```java
String cypher = """
    MATCH (f:Fault)-[:HAS_SOLUTION]->(s:Solution)
    WHERE f.name CONTAINS $fault AND s.title CONTAINS $solution
    RETURN f.name AS fault, s.title AS solution LIMIT 1
    """;
```

内部端点同时提供计划校验和既有 Agent 所需的查询能力：

```java
@PostMapping("/internal/graph/diagnosis/validate")
public Result<GraphValidationVO> validate(@RequestBody GraphValidationRequest request) {
    return Result.success(graphQueryService.validateFaultSolutions(request));
}

@PostMapping("/internal/graph/diagnosis/search")
public Result<PageResult<DiagnosisPathVO>> search(@RequestBody DiagnosisSearchQuery request) {
    return Result.success(graphQueryService.searchDiagnosisPaths(request));
}

@GetMapping("/internal/graph/devices/search")
public Result<List<DeviceVO>> devices(@RequestParam String keyword, @RequestParam int limit) {
    return Result.success(graphQueryService.searchDevices(keyword, limit));
}
```

`InternalApiInterceptor` 校验 `X-Internal-Token`；空 token 在非测试运行中不允许访问 `/internal/**`。

- [ ] **Step 5: 实现 Python Java 图谱客户端并替换本地图谱权威用法**

`config/settings.py` 增加：

```python
java_internal_base_url = os.getenv("JAVA_INTERNAL_BASE_URL", "http://127.0.0.1:8080")
internal_service_token = os.getenv("AI_INTERNAL_TOKEN", "")
```

`java_graph_client.py`：

```python
class JavaGraphClient:
    async def validate(self, payload: dict) -> dict:
        response = await self.client.post(
            f"{self.base_url}/internal/graph/diagnosis/validate",
            json=payload,
            headers={"X-Internal-Token": self.token},
        )
        response.raise_for_status()
        return response.json()["data"]
```

`tools/graph_query_tool.py` 中的 `GraphQueryTool`、`GraphSearchDeviceTool` 与 `GraphImageSearchTool` 全部使用 `JavaGraphClient`；`agents/review_agent.py` 使用其验证结果。Java 不可达时返回 `GRAPH_VALIDATION_UNAVAILABLE` 语义，不把本地 Neo4j 节点检查当作通过依据。

- [ ] **Step 6: 验证并按仓库提交**

Run Java:

```powershell
mvn -Dtest=GraphValidationServiceTest test
```

Run Python:

```powershell
python manual_tests/run_automated.py test_java_graph_client test_graph_query_tool test_review_agent test_fix_agent
```

Expected: PASS。

Java commit:

```powershell
git add src/main/java/ai/weixiu/config src/main/java/ai/weixiu/pojo src/main/java/ai/weixiu/service src/main/java/ai/weixiu/controller/InternalGraphController.java src/test/java/ai/weixiu/service/GraphValidationServiceTest.java
git commit -m "feat: expose authoritative graph validation endpoint"
```

Python commit:

```powershell
git add config/settings.py services/java_graph_client.py tools/graph_query_tool.py agents/review_agent.py manual_tests/test_java_graph_client.py manual_tests/test_graph_query_tool.py manual_tests/test_review_agent.py manual_tests/test_fix_agent.py
git commit -m "feat: delegate graph validation to java"
```

## Task 6: Python 生成受证据约束的结构化计划草案

**Files:**
- Modify: `D:\FixAgent\schemas\request.py`
- Modify: `D:\FixAgent\schemas\response.py`
- Create: `D:\FixAgent\services\plan_draft_service.py`
- Modify: `D:\FixAgent\api\main.py`
- Create: `D:\FixAgent\manual_tests\test_plan_draft_service.py`
- Modify: `D:\FixAgent\manual_tests\test_api_main.py`
- Modify: `D:\FixAgent\manual_tests\run_automated.py`

- [ ] **Step 1: 写入证据门禁与图谱降级测试**

```python
draft = await service.generate(valid_request)
assert draft.steps[0].evidence_refs[0].manual_id == 42
assert draft.preparations
assert draft.safety_reminders
assert draft.acceptance_suggestions

try:
    await service.generate(request_with_no_search_hits)
    no_evidence_code = "missing_exception"
except NoUsableManualEvidence as exc:
    no_evidence_code = exc.code
assert no_evidence_code == "NO_USABLE_MANUAL_EVIDENCE"

warning_draft = await service.generate(request_when_graph_unavailable)
assert warning_draft.warnings[0].code == "GRAPH_VALIDATION_UNAVAILABLE"
```

- [ ] **Step 2: 运行测试确认计划服务未实现**

Run:

```powershell
python manual_tests/run_automated.py test_plan_draft_service test_api_main
```

Expected: FAIL，计划模型、服务和端点不存在。

- [ ] **Step 3: 定义内部请求与响应模型**

```python
class PlanManualScope(BaseModel):
    manual_id: int = Field(alias="manualId")
    index_version: str = Field(alias="indexVersion")
    document_id: str = Field(alias="documentId")

class PlanDraftRequest(BaseModel):
    task_id: str = Field(alias="taskId")
    request_id: str = Field(alias="requestId")
    maintenance_level: str = Field(alias="maintenanceLevel")
    fault_description: str = Field(alias="faultDescription")
    images: list[dict] = []
    manual_scope: list[PlanManualScope] = Field(alias="manualScope")

class PlanDraftResponse(BaseModel):
    task_id: str
    request_id: str
    preparations: list[str]
    steps: list[PlanDraftStep]
    safety_reminders: list[str]
    acceptance_suggestions: list[str]
    graph_validation: dict
    warnings: list[PlanWarning] = []
```

- [ ] **Step 4: 实现受限检索和草案生成服务**

`PlanDraftService.generate()` 必须：

1. 逐个 `manual_scope.document_id` 调用知识检索并只接纳返回中同 `document_id`、同版本的命中。
2. 无有效命中时抛出 `NoUsableManualEvidence`，API 返回可识别的业务失败，不输出纯 LLM 步骤。
3. 组织固定结构的生成提示：`preparations`、`steps`、`safety_reminders`、`acceptance_suggestions`。
4. 调用 `JavaGraphClient.validate()`；不可用时附加 `GRAPH_VALIDATION_UNAVAILABLE` 警告。
5. 每个步骤至少持有一个 `manualId + indexVersion + page + chunkId` 证据引用。

- [ ] **Step 5: 增加受内部令牌保护的计划端点**

```python
@app.post("/internal/maintenance/plans/draft", response_model=PlanDraftResponse)
async def create_plan_draft(request: PlanDraftRequest, _: None = Depends(require_internal_token)):
    return await get_plan_draft_service().generate(request)
```

服务令牌与 Java 内部图谱调用共用 `AI_INTERNAL_TOKEN` 环境变量。

- [ ] **Step 6: 验证并提交**

Run:

```powershell
python manual_tests/run_automated.py test_plan_draft_service test_api_main test_knowledge_retrieval_tool
```

Expected: PASS。

Commit:

```powershell
git add schemas/request.py schemas/response.py services/plan_draft_service.py api/main.py manual_tests/test_plan_draft_service.py manual_tests/test_api_main.py manual_tests/run_automated.py
git commit -m "feat: generate evidence-bound maintenance plan drafts"
```

## Task 7: 建立 Java 检修任务、计划审批与风险控制状态机

**Files:**
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\resources\fix.sql`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\MaintenanceTask.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\MaintenancePlan.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\MaintenanceStep.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\PlanEvidence.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\MaintenanceTaskMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\MaintenancePlanMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\MaintenanceStepMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\PlanEvidenceMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\enumerate\MaintenanceTaskStatus.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\enumerate\PlanApprovalStatus.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\enumerate\RiskLevel.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\MaintenanceTaskCreateDTO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\PlanApprovalDTO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\vo\MaintenanceTaskVO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\PythonPlanClient.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\PythonPlanClientImpl.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\MaintenanceTaskService.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\MaintenanceTaskServiceImpl.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\controller\MaintenanceTaskController.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\service\MaintenanceTaskPlanServiceTest.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\controller\MaintenanceTaskControllerTest.java`

- [ ] **Step 1: 写入状态机和风险规则失败测试**

覆盖规则：

```java
assertThat(service.createTask(dto).getStatus()).isEqualTo(CREATED);
assertThat(service.generatePlan(taskId).getStatus()).isEqualTo(PLAN_REVIEW_PENDING);
assertThat(savedStep.getRiskLevel()).isEqualTo(HIGH); // Java 规则提升 AI 低估风险

assertThatThrownBy(() -> service.approvePlan(planId, approvalWithoutWarningAck))
    .hasMessageContaining("GRAPH_VALIDATION_UNAVAILABLE");
```

控制器测试覆盖：

```java
post("/weixiu/maintenance-task")
post("/weixiu/maintenance-task/{taskId}/plans/generate")
post("/weixiu/maintenance-task/{taskId}/plans/{planId}/approve")
post("/weixiu/maintenance-task/{taskId}/plans/{planId}/reject")
```

- [ ] **Step 2: 运行测试确认状态机模块不存在**

Run:

```powershell
mvn -Dtest=MaintenanceTaskPlanServiceTest,MaintenanceTaskControllerTest test
```

Expected: FAIL。

- [ ] **Step 3: 新增任务、计划、步骤、证据表**

`fix.sql` 新增最小闭环字段：

```sql
CREATE TABLE maintenance_task (
    id BIGINT PRIMARY KEY,
    equipment_id VARCHAR(80) NOT NULL,
    fault_description TEXT NOT NULL,
    maintenance_level VARCHAR(32) NOT NULL,
    image_ids TEXT NULL,
    status VARCHAR(40) NOT NULL,
    created_by_id BIGINT NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE maintenance_plan (
    id BIGINT PRIMARY KEY,
    task_id BIGINT NOT NULL,
    revision INT NOT NULL,
    request_id VARCHAR(80) NOT NULL,
    approval_status VARCHAR(32) NOT NULL,
    diagnosis_summary TEXT NULL,
    warnings TEXT NULL,
    approved_by_id BIGINT NULL,
    approved_at DATETIME NULL,
    UNIQUE KEY uk_task_revision (task_id, revision),
    UNIQUE KEY uk_plan_request (request_id)
);

CREATE TABLE maintenance_step (
    id BIGINT PRIMARY KEY,
    plan_id BIGINT NOT NULL,
    sequence_no INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    instruction TEXT NOT NULL,
    ai_risk_hint VARCHAR(20) NULL,
    risk_level VARCHAR(20) NOT NULL,
    requires_confirmation TINYINT NOT NULL,
    status VARCHAR(32) NOT NULL
);

CREATE TABLE plan_evidence (
    id BIGINT PRIMARY KEY,
    step_id BIGINT NOT NULL,
    manual_id BIGINT NOT NULL,
    index_version VARCHAR(32) NOT NULL,
    page_no INT NULL,
    chunk_id VARCHAR(160) NOT NULL,
    graph_validation_id VARCHAR(80) NULL,
    graph_status VARCHAR(32) NULL
);
```

- [ ] **Step 4: 实现任务创建、计划生成和风险提升**

`MaintenanceTaskService`：

```java
MaintenanceTaskVO create(MaintenanceTaskCreateDTO dto);
MaintenancePlanVO generatePlan(Long taskId);
MaintenancePlanVO approvePlan(Long taskId, Long planId, PlanApprovalDTO dto);
void rejectPlan(Long taskId, Long planId, PlanApprovalDTO dto);
```

风险规则最小实现封装在独立私有方法或 `MaintenanceRiskPolicy` 类中：

```java
if (instruction.contains("断电") || instruction.contains("挂牌") || instruction.contains("拆卸")) {
    return RiskLevel.HIGH;
}
return aiHint;
```

规则结果与 AI hint 分开存储；返回的最终 `riskLevel` 不低于规则结果。

- [ ] **Step 5: 实现 Python 计划客户端和证据固化**

`PythonPlanClientImpl` 请求 `/internal/maintenance/plans/draft`，请求的 `manualScope` 仅来源于 `MaintenanceManual.servingIndexVersion`。保存计划时将 Python 返回的每条 `evidenceRefs` 写入 `plan_evidence`，而不是运行时重新检索。

- [ ] **Step 6: 实现审批门禁**

批准必须满足：

```java
if (plan.hasWarning("GRAPH_VALIDATION_UNAVAILABLE")
        && !dto.getAcknowledgedWarnings().contains("GRAPH_VALIDATION_UNAVAILABLE")) {
    throw new IllegalStateException("必须确认图谱校验不可用警告");
}
```

已有已批准计划时禁止另一版同时进入 `APPROVED`；一旦执行开始，旧步骤不允许原地修改。

- [ ] **Step 7: 运行测试并提交**

Run:

```powershell
mvn -Dtest=MaintenanceTaskPlanServiceTest,MaintenanceTaskControllerTest test
```

Expected: `BUILD SUCCESS`。

Commit:

```powershell
git add src/main/resources/fix.sql src/main/java/ai/weixiu/entity src/main/java/ai/weixiu/mapper src/main/java/ai/weixiu/enumerate src/main/java/ai/weixiu/pojo src/main/java/ai/weixiu/service src/main/java/ai/weixiu/controller/MaintenanceTaskController.java src/test/java/ai/weixiu/service/MaintenanceTaskPlanServiceTest.java src/test/java/ai/weixiu/controller/MaintenanceTaskControllerTest.java
git commit -m "feat: add maintenance task plan approval workflow"
```

## Task 8: 实现步骤执行、高风险确认与复测验收

**Files:**
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\resources\fix.sql`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\ExecutionEvidence.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\TaskAcceptance.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\ExecutionEvidenceMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\TaskAcceptanceMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\StepRiskConfirmationDTO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\StepCompletionDTO.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\TaskAcceptanceDTO.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\MaintenanceTaskService.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\MaintenanceTaskServiceImpl.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\controller\MaintenanceTaskController.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\service\MaintenanceExecutionServiceTest.java`

- [ ] **Step 1: 写入高风险拦截与验收失败用例**

```java
assertThatThrownBy(() -> service.completeStep(taskId, planId, highRiskStepId, completion))
    .hasMessageContaining("执行前确认");

service.confirmRisk(taskId, planId, highRiskStepId, confirmation);
service.completeStep(taskId, planId, highRiskStepId, completion);
assertThat(step.getStatus()).isEqualTo(COMPLETED);

service.accept(taskId, failedAcceptance);
assertThat(task.getStatus()).isEqualTo(ACCEPTANCE_PENDING);
```

- [ ] **Step 2: 运行失败测试**

Run:

```powershell
mvn -Dtest=MaintenanceExecutionServiceTest test
```

Expected: FAIL，执行和验收 API 尚不存在。

- [ ] **Step 3: 添加执行与验收数据表**

```sql
CREATE TABLE execution_evidence (
    id BIGINT PRIMARY KEY,
    step_id BIGINT NOT NULL,
    evidence_type VARCHAR(32) NOT NULL,
    content TEXT NULL,
    file_ids TEXT NULL,
    submitted_by_id BIGINT NOT NULL,
    submitted_at DATETIME NOT NULL
);

CREATE TABLE task_acceptance (
    id BIGINT PRIMARY KEY,
    task_id BIGINT NOT NULL,
    conclusion VARCHAR(32) NOT NULL,
    notes TEXT NULL,
    measurements TEXT NULL,
    image_ids TEXT NULL,
    accepted_by_id BIGINT NOT NULL,
    accepted_at DATETIME NOT NULL
);
```

- [ ] **Step 4: 实现高风险确认、步骤完成与验收接口**

新增服务方法：

```java
void confirmRisk(Long taskId, Long planId, Long stepId, StepRiskConfirmationDTO dto);
void completeStep(Long taskId, Long planId, Long stepId, StepCompletionDTO dto);
TaskAcceptanceVO accept(Long taskId, TaskAcceptanceDTO dto);
```

新增控制器路径：

```java
POST /weixiu/maintenance-task/{taskId}/plans/{planId}/steps/{stepId}/confirm-risk
POST /weixiu/maintenance-task/{taskId}/plans/{planId}/steps/{stepId}/complete
POST /weixiu/maintenance-task/{taskId}/acceptance
```

验收通过才将任务置为 `COMPLETED`；失败验收保存记录但任务停留 `ACCEPTANCE_PENDING`。

- [ ] **Step 5: 验证并提交**

Run:

```powershell
mvn -Dtest=MaintenanceExecutionServiceTest,MaintenanceTaskPlanServiceTest test
```

Expected: `BUILD SUCCESS`。

Commit:

```powershell
git add src/main/resources/fix.sql src/main/java/ai/weixiu/entity src/main/java/ai/weixiu/mapper src/main/java/ai/weixiu/pojo src/main/java/ai/weixiu/service src/main/java/ai/weixiu/controller/MaintenanceTaskController.java src/test/java/ai/weixiu/service/MaintenanceExecutionServiceTest.java
git commit -m "feat: enforce maintenance execution and acceptance gates"
```

## Task 9: 将案例沉淀改为草稿审核后发布

**Files:**
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\resources\fix.sql`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\entity\CaseRecordDraft.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\mapper\CaseRecordDraftMapper.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\pojo\dto\CaseDraftReviewDTO.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\CaseRecordService.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\CaseRecordServiceImpl.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\MaintenanceTaskServiceImpl.java`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\controller\CaseRecordController.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\PythonCaseIndexClient.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\main\java\ai\weixiu\service\impl\PythonCaseIndexClientImpl.java`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\service\CaseDraftPublicationServiceTest.java`
- Create: `D:\FixAgent\services\case_index_service.py`
- Modify: `D:\FixAgent\api\main.py`
- Create: `D:\FixAgent\manual_tests\test_case_index_service.py`

- [ ] **Step 1: 写入“未审核不入图”和“批准才索引”失败测试**

Java 断言：

```java
CaseRecordDraft draft = taskService.createDraftFromCompletedTask(taskId);
verify(caseRecordRepository, never()).save(any(CaseRecord.class));

caseRecordService.approveDraft(draft.getId(), reviewDto);
verify(caseRecordRepository).save(any(CaseRecord.class));
```

Python 断言仅处理状态为 `approved` 的案例索引消息：

```python
assert await service.index_approved_case({"status": "pending"}) == {"indexed": False}
assert (await service.index_approved_case(approved_payload))["indexed"] is True
```

- [ ] **Step 2: 运行失败测试**

Run Java:

```powershell
mvn -Dtest=CaseDraftPublicationServiceTest test
```

Run Python:

```powershell
python manual_tests/run_automated.py test_case_index_service
```

Expected: FAIL。

- [ ] **Step 3: 新增案例草稿表与生成逻辑**

```sql
CREATE TABLE case_record_draft (
    id BIGINT PRIMARY KEY,
    task_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    review_status VARCHAR(32) NOT NULL,
    review_comment TEXT NULL,
    reviewed_by_id BIGINT NULL,
    reviewed_at DATETIME NULL,
    published_case_id VARCHAR(80) NULL,
    UNIQUE KEY uk_case_draft_task (task_id)
);
```

任务验收通过后生成 `PENDING_REVIEW` 草稿，不调用当前 `CaseRecordServiceImpl.save()` 的 embedding/Neo4j 写入路径。

- [ ] **Step 4: 把正式发布约束放入案例服务**

`CaseRecordService` 增加：

```java
CaseRecordDraft approveDraft(Long draftId, CaseDraftReviewDTO review);
CaseRecordDraft rejectDraft(Long draftId, CaseDraftReviewDTO review);
```

仅 `approveDraft()` 构建正式 `CaseRecord`、生成 embedding 并写 Neo4j，然后由 `PythonCaseIndexClient` 调用受保护内部端点 `/internal/cases/index-approved`。

- [ ] **Step 5: 实现 Python 已批准案例索引接收能力**

`case_index_service.py` 验证输入 `status == "approved"` 后才将案例文字作为独立知识来源写入向量库；`api/main.py` 以 `require_internal_token` 保护内部入口：

```python
@app.post("/internal/cases/index-approved")
async def index_approved_case(request: ApprovedCaseIndexRequest, _: None = Depends(require_internal_token)):
    return await get_case_index_service().index_approved_case(request)
```

索引 metadata 至少包含：

```python
metadata = {
    "chunk_type": "case",
    "case_id": payload["caseId"],
    "source_task_id": payload["taskId"],
    "approval_status": "approved",
}
```

案例索引不能混同 `manualScope` 中的手册证据；第一阶段计划生成仍仅将 ready 手册作为强制作业依据。

- [ ] **Step 6: 验证并分仓提交**

Run Java:

```powershell
mvn -Dtest=CaseDraftPublicationServiceTest,MaintenanceExecutionServiceTest test
```

Run Python:

```powershell
python manual_tests/run_automated.py test_case_index_service
```

Expected: PASS。

Java commit:

```powershell
git add src/main/resources/fix.sql src/main/java/ai/weixiu/entity src/main/java/ai/weixiu/mapper src/main/java/ai/weixiu/pojo src/main/java/ai/weixiu/service src/main/java/ai/weixiu/controller/CaseRecordController.java src/test/java/ai/weixiu/service/CaseDraftPublicationServiceTest.java
git commit -m "feat: publish reviewed maintenance case drafts only"
```

Python commit:

```powershell
git add services/case_index_service.py api/main.py manual_tests/test_case_index_service.py manual_tests/run_automated.py
git commit -m "feat: index approved maintenance cases"
```

## Task 10: 联调回归、接口文档与演示闭环验收

**Files:**
- Modify: `D:\FixAgent\api\README.md`
- Modify: `D:\FixAgent\schemas\README.md`
- Modify: `D:\FixAgent\services\README.md`
- Create: `D:\FixAgent\manual_tests\test_maintenance_closed_loop_contract.py`
- Modify: `D:\FixAgent\manual_tests\run_automated.py`
- Modify: `C:\Users\27202\Desktop\weixiu\weixiu\HELP.md`
- Create: `C:\Users\27202\Desktop\weixiu\weixiu\src\test\java\ai\weixiu\integration\MaintenanceClosedLoopContractTest.java`

- [ ] **Step 1: 写入 Python 契约回归测试**

该测试用 mock Java graph client 和 mock Redis 验证以下固定契约：

```python
assert manual_scope_only_ready_versions_are_retrieved()
assert image_summary_search_mapping_remains_compatible()
assert graph_unavailable_yields_warning_not_verified_status()
assert table_chunk_response_contains_table_rows()
```

- [ ] **Step 2: 写入 Java 业务闭环契约测试**

使用 mock Python clients 和内存化 mapper 或 service mock 跑通：

```java
createManualAndReceiveReadyIndex();
createTaskAndGenerateEvidenceBoundPlan();
approvePlanWithAcknowledgedGraphWarning();
confirmHighRiskStepAndCompleteExecution();
acceptTaskAndApproveCaseDraft();
```

另加反向场景：

```java
failedCandidateIndexKeepsServingVersion();
highRiskStepCannotCompleteWithoutConfirmation();
rejectedCaseDraftDoesNotPublish();
```

- [ ] **Step 3: 运行契约测试确认文档/联调缺口**

Run Python:

```powershell
python manual_tests/run_automated.py test_maintenance_closed_loop_contract
```

Run Java:

```powershell
mvn -Dtest=MaintenanceClosedLoopContractTest test
```

Expected: 在文档和联调 fixture 尚未完善时 FAIL，随后通过最小补全转绿。

- [ ] **Step 4: 更新实际接口文档和演示步骤**

Python 文档明确：

- `/ai/knowledge/search` 仍为兼容搜索接口。
- `/ai/knowledge/documents/**` 是供 Java 聚合或迁移期调用的证据读取能力。
- `/internal/maintenance/plans/draft` 只接受内部令牌认证。
- Python 不修改审批、验收或案例审核状态。

Java `HELP.md` 明确真实外部路由：

```text
/weixiu/maintenance-manual/**
/weixiu/maintenance-task/**
/weixiu/case-record/**  (草稿审核与正式案例发布)
/internal/graph/diagnosis/validate  (内部端点)
```

并写出演示闭环：

```text
上传故障手册 -> 等待服务版本 ready -> 浏览手册证据
-> 创建检修任务并带现场图片生成计划 -> 审批警告/计划
-> 确认高风险步骤并上传执行结果 -> 复测验收
-> 审核案例草稿 -> 发布正式知识
```

- [ ] **Step 5: 运行完整门禁**

Python:

```powershell
python manual_tests/run_automated.py
```

Expected: 所有自动用例 PASS，进程 exit `0`。

Java:

```powershell
mvn test
```

Expected: `BUILD SUCCESS`；若现存依赖真实外部文件或服务的旧测试不适合作为自动门禁，必须先将其隔离为手动验证，不可忽略失败。

- [ ] **Step 6: 运行人工联调验收清单**

在配置好的测试环境中逐项保留请求 ID、任务 ID 和截图/日志证明：

1. 首次手册索引成功并可浏览文本、表格和 PDF 页码。
2. 新版本索引失败但旧服务版本仍可生成计划。
3. 图谱接口不可用时草案含警告，未经确认不能批准。
4. 高风险步骤未经确认不能提交结果。
5. 验收失败可重新生成计划版本。
6. 案例草稿驳回后不进入图谱和向量库。
7. 旧 `/ai/knowledge/search` 与已有图片会话链路继续工作。

- [ ] **Step 7: 分仓提交最终文档与验收测试**

Python:

```powershell
git add api/README.md schemas/README.md services/README.md manual_tests/test_maintenance_closed_loop_contract.py manual_tests/run_automated.py
git commit -m "docs: document maintenance evidence and plan contracts"
```

Java:

```powershell
git add HELP.md src/test/java/ai/weixiu/integration/MaintenanceClosedLoopContractTest.java
git commit -m "test: verify maintenance closed loop contract"
```

## 需求覆盖复核表

| 设计要求 | 覆盖任务 |
|---|---|
| 已跟踪凭据清理与轮换门禁 | Task 0 |
| Python 自动测试可作为门禁 | Task 0 |
| 文档列表、详情、分块分页、上下文 | Task 1, Task 4 |
| `table_rows` 与图片摘要搜索兼容 | Task 1 |
| 稳定文件定位、私有 PDF 与 MIME | Task 2, Task 3, Task 4 |
| RabbitMQ 异步手册索引、幂等和成功后切换 | Task 2, Task 3 |
| Java 图谱真实路径为唯一权威 | Task 5 |
| 图谱不可用生成带警告草案 | Task 5, Task 6, Task 7 |
| 限定 ready 手册证据的结构化计划 | Task 6, Task 7 |
| 任务、审批、计划版本与高风险确认 | Task 7, Task 8 |
| 复测验收失败不关闭任务 | Task 8 |
| 已验收任务生成草稿，审核后发布案例 | Task 9 |
| Java 图片列表既有实现的回归保护 | Task 10 |
| README/Schema/演示闭环同步 | Task 10 |

## 不在本计划中的事项

以下事项继续按已批准设计延期，不因实施过程中方便而顺手加入：

- 普通 AI 回答的通用人工采纳/修正/驳回知识沉淀流程。
- 记忆检索内部 `score` / `distance` 字段命名治理。
- 自动派工、排班、复杂权限矩阵与实时通知。
- 全事件驱动流程编排和前端工作台实现。

## 计划自检清单

- [ ] 所有功能修改均有对应失败测试、转绿命令和独立提交点。
- [ ] Python 和 Java 的共享名称使用 `jobId`、`documentId`、`indexVersion`、`requestId`、`validationId` 的同一语义。
- [ ] Java 外部路由遵循现有 `/weixiu/...` 前缀，Python 仅提供内部或兼容能力。
- [ ] 已存在实现的 `images` 列表与 Java MIME 上传未被误列为重建任务。
- [ ] 计划不包含真实密钥、永久私有文件 URL 或未经审核案例直接发布路径。
- [ ] 后续执行需再次获得用户明确编码批准。
