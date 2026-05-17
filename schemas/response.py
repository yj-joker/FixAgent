"""
Schemas响应模型模块

定义所有API的响应数据模型，包括分页、错误处理等。

【模块职责】
- 返回统一格式的响应数据
- 提供结构化的业务数据封装
- 包含分页、元信息等辅助数据

【使用顺序】
1. Agent/Chain 处理业务逻辑
2. 构建对应的 Response 模型
    3. FastAPI 自动序列化为 JSON
4. Java 端按 Response 类解析

【继承关系】
- 所有响应都继承 BaseResponse（success, message, code）
- Response 类扩展特定业务字段

【错误处理】
- 正常: success=True, code=200
- 错误: ErrorResponse（success=False, code=500/400）
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from schemas.models import (
    BaseResponse, PaginationMeta,
    DetectionResult, VectorSearchResult, GraphNode, GraphRelation
)


# ==================== 对话相关 ====================

class ChatStreamEvent(BaseModel):
    """
    对话流式事件模型

    【功能关联】SSE（Server-Sent Events）流式输出
    【何时用】启用 stream=True 的对话请求

    【event 类型】
    | event | 说明 | data 示例 |
    |-------|------|---------|
    | token | AI 输出的 token | {"content": "维修"} |
    | status | 状态更新 | {"stage": "检索知识库"} |
    | tool | 工具调用 | {"tool": "knowledge_retrieval"} |
    | done | 完成 | {} |
    | error | 错误 | {"message": "服务异常"} |

    【SSE 格式】
    ```
    event: token
    data: {"content": "维修"}

    event: done
    data: {}
    ```

    【Java 端处理示例】
    ```java
    EventSource eventSource = new EventSource(url, handler);
    eventSource.onEvent("token", event -> {
        String content = event.getData().get("content");
        // 实时显示
    });
    ```
    """
    event: str = Field(..., description="事件类型: token/status/tool/done/error")
    data: Any = Field(..., description="事件数据")

    class Config:
        json_schema_extra = {
            "example": {
                "event": "token",
                "data": {"content": "维修"}
            }
        }


class ChatResponse(BaseResponse):
    """
    对话响应模型（非流式）

    【功能关联】/ai/chat 及相关对话接口
    【何时用】用户发起对话请求且 stream=False 时

    【使用顺序】
    1. ChatRequest 进入 → OrchestratorAgent 处理
    2. Agent 识别意图、调用工具、生成回复
    3. 构建 ChatResponse 返回

    【字段说明】
    - session_id: 关联的会话 ID
    - message: AI 生成的回复内容
    - intention: 识别的用户意图（如 "troubleshoot"）
    - tools_used: 使用的工具列表（如 ["knowledge_retrieval", "graph_query"]）
    - latency_ms: 处理耗时（毫秒）

    【message 内容示例】
    ```
    根据您描述的电动机轴承过热现象，可能原因包括：
    1. 润滑不良 - 概率 85%
    2. 轴承磨损 - 概率 60%
    3. 负载过大 - 概率 40%

    建议检查顺序：...
    ```

    【Java 对应类】
    ```java
    public class ChatResponse extends BaseResponse {
        String sessionId;
        String message;
        String intention;           // 可选
        List<String> toolsUsed;     // 可选
        Integer latencyMs;          // 可选
    }
    ```
    """
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="AI回复")
    intention: Optional[str] = Field(default=None, description="识别到的意图")
    tools_used: Optional[List[str]] = Field(default=None, description="使用的工具列表")
    latency_ms: Optional[int] = Field(default=None, description="响应延迟(ms)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "电动机轴承过热可能由以下原因造成：1. 润滑不良...",
                "code": 200,
                "session_id": "sess_abc123",
                "intention": "troubleshoot",
                "tools_used": ["knowledge_retrieval", "graph_query"],
                "latency_ms": 1500
            }
        }


# ==================== 知识库相关 ====================

class KnowledgeItem(BaseModel):
    """
    知识条目模型

    【功能关联】知识库、案例库
    【何时用】返回单条知识详情时

    【字段说明】
    - id: 知识唯一标识
    - title: 知识标题
    - content: 知识正文
    - category: 分类（如 "motor"、"pump"）
    - tags: 标签列表
    - file_urls: 关联的文件 URL
    - status: 状态（draft/published/archived）
    - created_at: 创建时间
    - updated_at: 更新时间

    【Java 对应类】
    ```java
    public class KnowledgeItem {
        Integer id;
        String title;
        String content;
        String category;
        List<String> tags;
        List<String> fileUrls;
        String status;
        String createdAt;
        String updatedAt;
    }
    ```
    """
    id: int
    title: str
    content: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    file_urls: Optional[List[str]] = None
    status: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class KnowledgeListResponse(BaseResponse):
    """
    知识列表响应

    【功能关联】知识库列表查询
    【何时用】分页查询知识列表时

    【使用顺序】
    1. 用户请求知识列表（如第2页，每页10条）
    2. 查询数据库/向量库
    3. 返回 KnowledgeListResponse

    【字段说明】
    - data: 知识列表 List[KnowledgeItem]
    - meta: 分页信息 PaginationMeta

    【Java 对应类】
    ```java
    public class KnowledgeListResponse extends BaseResponse {
        List<KnowledgeItem> data;
        PaginationMeta meta;
    }
    ```
    """
    data: List[KnowledgeItem]
    meta: PaginationMeta


class KnowledgeDetailResponse(BaseResponse):
    """
    知识详情响应

    【功能关联】知识库详情查询
    【何时用】查看单条知识完整内容时

    【与 KnowledgeListResponse 的区别】
    - List: 返回列表，分页
    - Detail: 返回单条，完整内容
    """
    data: KnowledgeItem


class KnowledgeSearchResponse(BaseResponse):
    """
    知识检索响应

    【功能关联】向量检索、语义搜索
    【何时用】执行知识检索时

    【字段说明】
    - data: 检索结果列表（按相似度排序）
    - total: 符合条件的总数
    - query_time_ms: 检索耗时

    【VectorSearchResult 说明】
    - id: 知识 ID
    - score: 相似度分数（0~1，越高越相关）
    - content: 知识内容摘要
    - metadata: 附加信息

    【Java 对应类】
    ```java
    public class KnowledgeSearchResponse extends BaseResponse {
        List<VectorSearchResult> data;
        int total;
        int queryTimeMs;
    }
    ```
    """
    data: List[VectorSearchResult]
    total: int
    query_time_ms: int


class KnowledgeImportResponse(BaseResponse):
    """
    知识导入响应

    【功能关联】POST /ai/knowledge/import
    【何时用】文档解析并入库完成后，返回导入统计

    【字段说明】
    - file_name: 文档文件名
    - total_pages: PDF 总页数
    - text_count: 入库文本块数量
    - image_count: 入库图片数量（用图注向量化）
    - table_count: 入库表格数量
    - sections: 各章节统计摘要
    - extraction_summary: DocumentParserTool 的提取摘要
    - process_time_ms: 总耗时
    """
    file_name: str
    total_pages: int
    text_count: int
    image_count: int
    table_count: int
    sections: List[dict]
    extraction_summary: dict
    process_time_ms: int


# ==================== 案例相关 ====================

class CaseItem(BaseModel):
    """
    案例项模型

    【功能关联】案例库、故障案例管理
    【何时用】返回案例详情或列表时

    【字段说明】
    - id: 案例唯一标识
    - title: 案例标题
    - description: 详细故障描述
    - symptom: 故障现象
    - cause: 故障原因
    - solution: 解决方案
    - device_id / device_name: 关联设备信息
    - images: 故障图片列表
    - status: 案例状态（submitted/reviewing/approved/rejected）
    - submitter_id / submitter_name: 提交人信息
    - reviewer_id / reviewer_name: 审核人信息
    - reviewed_at: 审核时间
    - review_comment: 审核意见
    - created_at / updated_at: 时间戳

    【Java 对应类】
    ```java
    public class CaseItem {
        Integer id;
        String title;
        String description;
        String symptom;
        String cause;
        String solution;
        Integer deviceId;
        String deviceName;
        List<String> images;
        String status;
        Integer submitterId;
        String submitterName;
        Integer reviewerId;
        String reviewerName;
        String reviewedAt;
        String reviewComment;
        String createdAt;
        String updatedAt;
    }
    ```
    """
    id: int
    title: str
    description: str
    symptom: Optional[str] = None
    cause: Optional[str] = None
    solution: Optional[str] = None
    device_id: Optional[int] = None
    device_name: Optional[str] = None
    images: Optional[List[str]] = None
    status: str
    submitter_id: int
    submitter_name: Optional[str] = None
    reviewer_id: Optional[int] = None
    reviewer_name: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_comment: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class CaseListResponse(BaseResponse):
    """
    案例列表响应

    【功能关联】案例库列表查询
    【何时用】分页查询案例列表时

    【使用顺序】
    1. 用户请求案例列表
    2. 查询案例库（可按状态、设备等过滤）
    3. 返回 CaseListResponse

    【字段说明】
    - data: 案例列表 List[CaseItem]
    - meta: 分页信息 PaginationMeta
    """
    data: List[CaseItem]
    meta: PaginationMeta


class CaseDetailResponse(BaseResponse):
    """
    案例详情响应

    【功能关联】案例库详情查询
    【何时用】查看单条案例完整信息时
    """
    data: CaseItem


# ==================== 设备相关 ====================

class DeviceItem(BaseModel):
    """
    设备项模型

    【功能关联】设备管理
    【何时用】返回设备详情或列表时

    【字段说明】
    - id: 设备唯一标识
    - name: 设备名称
    - model: 设备型号
    - category: 设备类别
    - manufacturer: 制造商
    - specs: 规格参数字典
    - created_at / updated_at: 时间戳

    【Java 对应类】
    ```java
    public class DeviceItem {
        Integer id;
        String name;
        String model;
        String category;
        String manufacturer;
        Map<String, Object> specs;
        String createdAt;
        String updatedAt;
    }
    ```
    """
    id: int
    name: str
    model: Optional[str] = None
    category: Optional[str] = None
    manufacturer: Optional[str] = None
    specs: Optional[dict] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DeviceListResponse(BaseResponse):
    """
    设备列表响应

    【功能关联】设备管理
    【何时用】分页查询设备列表时
    """
    data: List[DeviceItem]
    meta: PaginationMeta


class DeviceDetailResponse(BaseResponse):
    """
    设备详情响应

    【功能关联】设备管理
    【何时用】查看单台设备详细信息时
    """
    data: DeviceItem


# ==================== 图谱相关 ====================

class GraphQueryResponse(BaseResponse):
    """
    图谱查询响应

    【功能关联】Neo4j 图数据库、知识图谱
    【何时用】查询图谱节点和关系时

    【字段说明】
    - nodes: 查询到的节点列表
    - relations: 查询到的关系列表
    - query_time_ms: 查询耗时

    【使用场景】
    - 查询故障现象的相关节点
    - 查询设备部件的关联关系
    - 图谱可视化展示

    【Java 对应类】
    ```java
    public class GraphQueryResponse extends BaseResponse {
        List<GraphNode> nodes;
        List<GraphRelation> relations;
        int queryTimeMs;
    }
    ```
    """
    nodes: List[GraphNode]
    relations: List[GraphRelation]
    query_time_ms: int


class GraphPathResponse(BaseResponse):
    """
    图谱路径查询响应

    【功能关联】Neo4j 图数据库、故障传播路径
    【何时用】查询两个实体间的最短路径或多条路径时

    【字段说明】
    - paths: 路径列表（每条路径是 GraphNode 列表）
    - total_paths: 找到的路径总数

    【paths 结构示例】
    ```json
    [
        [{"id": "1", "label": "轴承磨损"}, {"id": "2", "label": "振动过大"}, {"id": "3", "label": "停机"}],
        [{"id": "1", "label": "轴承磨损"}, {"id": "4", "label": "温度升高"}, {"id": "3", "label": "停机"}]
    ]
    ```

    【Java 对应类】
    ```java
    public class GraphPathResponse extends BaseResponse {
        List<List<GraphNode>> paths;
        int totalPaths;
    }
    ```
    """
    paths: List[List[GraphNode]]
    total_paths: int


class GraphStatsResponse(BaseResponse):
    """
    图谱统计响应

    【功能关联】Neo4j 图数据库、图谱概览
    【何时用】查看图谱整体统计信息时

    【字段说明】
    - total_nodes: 图谱中节点总数
    - total_relations: 图谱中关系总数
    - node_types: 各类型节点数量（如 {"Symptom": 100, "Device": 50}）
    - relation_types: 各类型关系数量（如 {"causes": 200, "belongs_to": 150}）

    【使用场景】
    - 图谱 Dashboard 展示
    - 知识图谱健康度检查

    【Java 对应类】
    ```java
    public class GraphStatsResponse extends BaseResponse {
        int totalNodes;
        int totalRelations;
        Map<String, Integer> nodeTypes;
        Map<String, Integer> relationTypes;
    }
    ```
    """
    total_nodes: int
    total_relations: int
    node_types: dict
    relation_types: dict


# ==================== 工具调用相关 ====================

class YoloDetectResponse(BaseResponse):
    """
    YOLO检测响应

    【功能关联】YOLO目标检测、故障部件识别
    【何时用】YOLO检测完成后

    【字段说明】
    - image_url: 被检测的图片 URL
    - detections: 检测结果列表
    - process_time_ms: 处理耗时

    【Java 对应类】
    ```java
    public class YoloDetectResponse extends BaseResponse {
        String imageUrl;
        List<DetectionResult> detections;
        int processTimeMs;
    }
    ```
    """
    image_url: str
    detections: List[DetectionResult]
    process_time_ms: int


class SamSegmentResponse(BaseResponse):
    """
    SAM分割响应

    【功能关联】SAM图像分割、精细化故障区域
    【何时用】SAM分割完成后

    【字段说明】
    - image_url: 被分割的图片 URL
    - masks: 分割掩码列表（每个 mask 是 dict）
    - labels: 每个 mask 对应的标签
    - process_time_ms: 处理耗时

    【masks 结构示例】
    ```json
    [
        {"mask": [[0,1,1], [1,1,0]], "bbox": [10, 20, 100, 150]},
        {"mask": [[1,0,0], [0,0,1]], "bbox": [50, 60, 120, 180]}
    ]
    ```
    """
    image_url: str
    masks: List[dict]
    labels: List[str]
    process_time_ms: int


class ClipEmbedResponse(BaseResponse):
    """
    CLIP向量化响应

    【功能关联】CLIP多模态模型、向量化
    【何时用】生成文本或图片向量后

    【字段说明】
    - embedding: 生成的向量（浮点数列表）
    - dimension: 向量维度（如 512、768）
    - model: 使用的模型名称

    【使用场景】
    - 文本向量化 → 存入 Redis 向量库
    - 图片向量化 → 用于相似图片检索

    【Java 对应类】
    ```java
    public class ClipEmbedResponse extends BaseResponse {
        List<Float> embedding;
        int dimension;
        String model;
    }
    ```
    """
    embedding: List[float]
    dimension: int
    model: str


class DocumentParseResponse(BaseResponse):
    """
    文档解析响应

    【功能关联】文档解析服务
    【何时用】PDF/Word 文档解析完成后

    【字段说明】
    - file_name: 文件名
    - total_pages: 总页数
    - pages: 每页内容列表（dict 列表）
    - tables: 提取的表格列表
    - images: 文档中的图片列表
    - process_time_ms: 处理耗时

    【pages 结构示例】
    ```json
    [
        {"page": 1, "content": "这是第1页的内容..."},
        {"page": 2, "content": "这是第2页的内容..."}
    ]
    ```

    【tables 结构示例】
    ```json
    [
        {"page": 1, "rows": 5, "cols": 3, "header": ["名称", "规格", "数量"], "data": [[...], [...]]}
    ]
    ```
    """
    file_name: str
    total_pages: int
    pages: List[dict]
    tables: List[dict]
    images: List[str]
    process_time_ms: int


# ==================== 任务相关 ====================

class TaskStatusResponse(BaseResponse):
    """
    任务状态响应

    【功能关联】异步任务、后台任务
    【何时用】查询长任务执行状态时

    【字段说明】
    - task_id: 任务唯一标识
    - status: 任务状态（pending/started/success/failure）
    - result: 任务结果（成功时返回）
    - error: 错误信息（失败时返回）
    - created_at: 任务创建时间
    - updated_at: 任务更新时间

    【status 生命周期】
    ```
    PENDING → STARTED → SUCCESS
                   ↘ FAILURE
    ```

    【Java 对应类】
    ```java
    public class TaskStatusResponse extends BaseResponse {
        String taskId;
        String status;
        Object result;      // 成功时
        String error;        // 失败时
        String createdAt;
        String updatedAt;
    }
    ```
    """
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ==================== 记忆整理相关 ====================

class FactSummary(BaseModel):
    """事实摘要子模型"""
    content: str = Field(description="事实描述")
    keywords: str = Field(default="", description="检索用关键词")
    source_seq_range: str = Field(default="", serialization_alias="sourceSeqRange", description="来源对话序号范围")


class PreferenceSummary(BaseModel):
    """
    偏好摘要子模型

    新增 sourceType 字段：区分偏好来源的可靠度
    - explicit: 用户直接说出来的指令，可信度高，Java端直接存为确认偏好
    - inferred: 从行为推断的，Java端存为候选偏好，需多次出现才升级
    """
    content: str = Field(description="偏好描述")
    category: str = Field(default="其他", description="分类: 交互风格|格式要求|工作习惯|关注领域|其他")
    preferenceCategory: int = Field(default=1, description="偏好类型: 0=用户级(所有对话公用), 1=会话级(单次会话公用)")
    sourceType: str = Field(default="inferred", serialization_alias="sourceType", description="explicit=用户明说, inferred=推断")


class UnresolvedSummary(BaseModel):
    """未完成事项子模型"""
    content: str = Field(description="待解决描述")
    type: str = Field(default="待办", description="类型: 未答复问题|进行中任务|用户待办")
    status: str = Field(default="active", description="状态: active=进行中, superseded=已放弃")


class MemorySummary(BaseModel):
    """
    记忆摘要模型

    【功能关联】MemoryAgent 整理输出
    【何时用】LLM 完成对话整理后，作为结构化摘要返回给 Java

    【字段说明】
    - new_facts: 新增/更新的事实（客观、已确认的信息）
    - superseded_ids: 本次整理覆盖掉的旧事实ID
    - updated_preferences: 合并后的用户偏好
    - updated_unresolved: 仍悬而未决的事项
    - resolved_item_ids: 本次解决的事项的数据库ID（用于精确标记）
    - brief_summary: 200字以内的整体摘要
    """
    new_facts: List[FactSummary] = Field(default_factory=list, serialization_alias="newFacts", description="新增事实列表")
    superseded_ids: List[str] = Field(default_factory=list, serialization_alias="supersededIds", description="被覆盖的旧事实ID")
    updated_preferences: List[PreferenceSummary] = Field(default_factory=list, serialization_alias="updatedPreferences", description="更新后的偏好列表（含sourceType）")
    updated_unresolved: List[UnresolvedSummary] = Field(default_factory=list, serialization_alias="updatedUnresolved", description="更新后的未完成事项")
    # 改为用数据库ID精确匹配，替代之前的content文本匹配
    resolved_item_ids: List[int] = Field(default_factory=list, serialization_alias="resolvedItemIds", description="已解决事项的数据库ID列表")
    brief_summary: str = Field(default="", serialization_alias="briefSummary", description="100字以内的渐进式摘要")
    # 向量库生成的doc_id列表，与new_facts一一对应，Java端用作MySQL的factId
    fact_ids: List[str] = Field(default_factory=list, serialization_alias="fact_ids", description="向量库doc_id列表，与newFacts一一对应")


class MemoryConsolidateResponse(BaseResponse):
    """
    记忆整理响应

    【功能关联】POST /ai/memory/consolidate
    【何时用】整理完成后返回结构化摘要给 Java

    【字段说明】
    - session_id: 关联的会话 ID
    - summary: 整理后的记忆摘要（含核心问题、结论等五个子字段）
    - original_count: 原始对话条数（用于 Java 端日志/统计）
    - consolidated_at: 整理时间（ISO格式）

    【Java 对应类】
    ```java
    public class MemoryConsolidateResponse extends BaseResponse {
        String sessionId;
        MemorySummary summary;
        int originalCount;
        String consolidatedAt;
    }
    ```
    """
    session_id: str = Field(..., serialization_alias="sessionId", description="会话ID")
    summary: MemorySummary = Field(..., description="整理后的记忆摘要")
    original_count: int = Field(..., serialization_alias="originalCount", description="原始对话条数")
    consolidated_at: str = Field(..., serialization_alias="consolidatedAt", description="整理时间")