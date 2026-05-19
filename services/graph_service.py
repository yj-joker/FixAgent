"""
图数据库服务

基于 Neo4j 实现设备检修知识图谱的只读查询。
查询逻辑与 Java 端 GraphQueryServiceImpl 保持一致。

【节点类型】
- Device: 设备（id, name, code, model, location, manufacturer）
- Component: 部件（id, name, partNumber, specification, supplier, lifecycle, embedding）
- Fault: 故障（id, code, name, description, severity, category, embedding）
- Solution: 解决方案（id, code, title, description, toolsRequired, estimatedTime, difficulty, verified）

【关系类型】
- OWNS: Device -> Component
- CAUSES: Component -> Fault
- HAS_SOLUTION: Fault -> Solution
- HAS_FAULT: Device -> Fault（历史故障记录）
"""

import logging

logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from config.settings import get_settings


class DiagnosisPath(BaseModel):
    """诊断路径模型，与 Java DiagnosisPathVO 字段对齐"""
    device_id: Optional[str] = Field(default=None)
    device_name: Optional[str] = Field(default=None)
    component_id: Optional[str] = Field(default=None)
    component_name: Optional[str] = Field(default=None)
    fault_id: Optional[str] = Field(default=None)
    fault_name: Optional[str] = Field(default=None)
    fault_severity: Optional[str] = Field(default=None)
    solution_id: Optional[str] = Field(default=None)
    solution_title: Optional[str] = Field(default=None)
    estimated_time: Optional[int] = Field(default=None)
    verified: Optional[bool] = Field(default=None)
    has_history: bool = Field(default=False, description="设备是否有该故障的历史记录")
    fault_score: Optional[float] = Field(default=None, description="故障向量匹配分数")
    component_score: Optional[float] = Field(default=None, description="部件向量匹配分数")
    path_text: Optional[str] = Field(default=None, description="可读路径文本")


class DeviceInfo(BaseModel):
    id: str
    name: str
    code: Optional[str] = None
    model: Optional[str] = None
    location: Optional[str] = None
    manufacturer: Optional[str] = None


class ComponentInfo(BaseModel):
    id: str
    name: str
    part_number: Optional[str] = None
    specification: Optional[str] = None
    supplier: Optional[str] = None
    lifecycle: Optional[str] = None
    unit_price: Optional[float] = None


class FaultInfo(BaseModel):
    id: str
    name: str
    code: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None


class SolutionInfo(BaseModel):
    id: str
    title: str
    code: Optional[str] = None
    description: Optional[str] = None
    tools_required: Optional[str] = None
    estimated_time: Optional[int] = None
    difficulty: Optional[str] = None
    verified: Optional[bool] = None


_FIELDS = ("deviceId,deviceName,componentId,componentName,"
           "faultId,faultName,faultSeverity,"
           "solutionId,solutionTitle,estimatedTime,verified,"
           "hasHistory")
_PATH_FIELDS = _FIELDS


def _map_path(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "device_id": record.get("deviceId"),
        "device_name": record.get("deviceName"),
        "component_id": record.get("componentId"),
        "component_name": record.get("componentName"),
        "fault_id": record.get("faultId"),
        "fault_name": record.get("faultName"),
        "fault_severity": record.get("faultSeverity"),
        "solution_id": record.get("solutionId"),
        "solution_title": record.get("solutionTitle"),
        "estimated_time": record.get("estimatedTime"),
        "verified": record.get("verified"),
        "has_history": bool(record.get("hasHistory", False)),
    }


def _build_path_text(row: Dict[str, Any]) -> str:
    parts = []
    if row.get("device_name"):
        parts.append(row["device_name"])
    if row.get("component_name"):
        parts.append(f"OWNS->{row['component_name']}" if parts else row["component_name"])
    if row.get("fault_name"):
        parts.append(f"CAUSES->{row['fault_name']}" if parts else row["fault_name"])
    if row.get("solution_title"):
        parts.append(f"HAS_SOLUTION->{row['solution_title']}" if parts else row["solution_title"])
    return " -> ".join(parts)


_BASE_CYPHER = """
MATCH (d:Device)-[:OWNS]->(c:Component)-[:CAUSES]->(f:Fault)
WHERE {where_clause}
OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(s:Solution)
OPTIONAL MATCH (d)-[hf:HAS_FAULT]->(f)
RETURN d.id AS deviceId, d.name AS deviceName,
       c.id AS componentId, c.name AS componentName,
       f.id AS faultId, f.name AS faultName, f.severity AS faultSeverity,
       s.id AS solutionId, s.title AS solutionTitle,
       s.estimated_time AS estimatedTime, s.verified AS verified,
       hf IS NOT NULL AS hasHistory
ORDER BY hasHistory DESC, s.verified DESC, s.estimated_time ASC
SKIP $skip LIMIT $limit
"""

_COUNT_CYPHER = """
MATCH (d:Device)-[:OWNS]->(c:Component)-[:CAUSES]->(f:Fault)
WHERE {where_clause}
OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(s:Solution)
RETURN count(*) AS total
"""


class GraphService:
    """Neo4j 图数据库只读查询服务，查询逻辑与 Java GraphQueryServiceImpl 对齐。"""

    def __init__(self):
        self.settings = get_settings()
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )
        self._database = self.settings.neo4j_database

    def close(self):
        if self.driver:
            self.driver.close()

    def _execute_query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self.driver.session(database=self._database) as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    def _execute_single(self, cypher: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self.driver.session(database=self._database) as session:
            result = session.run(cypher, params)
            record = result.single()
            return dict(record) if record else None

    # ==================== 向量检索 ====================

    def search_components_by_embedding(
        self, embedding: List[float], limit: int = 20, min_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """通过 Neo4j 向量索引检索部件。"""
        cypher = """
            CALL db.index.vector.queryNodes('component_embedding_index', $limit, $embedding)
            YIELD node AS c, score
            WHERE score >= $minScore
            RETURN c.id AS id, c.name AS name, c.part_number AS partNumber,
                   c.specification AS specification, c.supplier AS supplier,
                   c.lifecycle AS lifecycle, c.unit_price AS unitPrice, score
            ORDER BY score DESC
        """
        return self._execute_query(cypher, {
            "embedding": embedding, "limit": limit, "minScore": min_score
        })

    def search_faults_by_embedding(
        self, embedding: List[float], limit: int = 20, min_score: float = 0.80
    ) -> List[Dict[str, Any]]:
        """通过 Neo4j 向量索引检索故障。"""
        cypher = """
            CALL db.index.vector.queryNodes('fault_embedding_index', $limit, $embedding)
            YIELD node AS f, score
            WHERE score >= $minScore
            RETURN f.id AS id, f.name AS name, f.description AS description,
                   f.category AS category, f.severity AS severity, score
            ORDER BY score DESC
        """
        return self._execute_query(cypher, {
            "embedding": embedding, "limit": limit, "minScore": min_score
        })

    # ==================== 设备搜索 ====================

    def find_devices(
        self, keyword: str = None, skip: int = 0, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        cypher = """
            MATCH (d:Device)
            WHERE $keyword IS NULL OR $keyword = ''
               OR d.name CONTAINS $keyword OR d.code CONTAINS $keyword
               OR d.model CONTAINS $keyword OR d.location CONTAINS $keyword
            RETURN d.id AS id, d.name AS name, d.code AS code,
                   d.model AS model, d.location AS location, d.manufacturer AS manufacturer
            ORDER BY d.name ASC
            SKIP $skip LIMIT $limit
        """
        return self._execute_query(cypher, {
            "keyword": keyword or "", "skip": skip, "limit": limit
        })

    def get_device_overview(self, device_id: str) -> Optional[Dict[str, Any]]:
        cypher = """
            MATCH (d:Device {id: $deviceId})
            OPTIONAL MATCH (d)-[:OWNS]->(c:Component)
            WITH d, count(DISTINCT c) AS componentCount
            OPTIONAL MATCH (d)-[:HAS_FAULT]->(f:Fault)
            RETURN d.id AS deviceId, d.name AS deviceName,
                   d.code AS code, d.model AS model,
                   d.location AS location, d.manufacturer AS manufacturer,
                   componentCount, count(DISTINCT f) AS faultCount
        """
        return self._execute_single(cypher, {"deviceId": device_id})

    # ==================== 部件/故障 简单查询 ====================

    def find_components_by_device(
        self, device_id: str, component_name: str = None,
        skip: int = 0, limit: int = 10
    ) -> List[Dict[str, Any]]:
        cypher = """
            MATCH (d:Device {id: $deviceId})-[:OWNS]->(c:Component)
            WHERE $componentName IS NULL OR $componentName = ''
               OR c.name CONTAINS $componentName
            RETURN c.id AS id, c.name AS name, c.part_number AS partNumber,
                   c.specification AS specification, c.supplier AS supplier,
                   c.lifecycle AS lifecycle, c.unit_price AS unitPrice
            ORDER BY c.name ASC SKIP $skip LIMIT $limit
        """
        return self._execute_query(cypher, {
            "deviceId": device_id, "componentName": component_name or "",
            "skip": skip, "limit": limit
        })

    def find_faults_by_component(
        self, component_id: str, fault_name: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        cypher = """
            MATCH (c:Component {id: $componentId})-[:CAUSES]->(f:Fault)
            WHERE $faultName IS NULL OR $faultName = ''
               OR f.name CONTAINS $faultName
            RETURN f.id AS id, f.code AS code, f.name AS name,
                   f.description AS description, f.severity AS severity, f.category AS category
            LIMIT $limit
        """
        return self._execute_query(cypher, {
            "componentId": component_id, "faultName": fault_name or "", "limit": limit
        })

    def find_solutions_by_fault(
        self, fault_id: str, verified_only: bool = False, limit: int = 10
    ) -> List[Dict[str, Any]]:
        verify_clause = "WHERE s.verified = true" if verified_only else ""
        cypher = f"""
            MATCH (f:Fault {{id: $faultId}})-[:HAS_SOLUTION]->(s:Solution)
            {verify_clause}
            RETURN s.id AS id, s.code AS code, s.title AS title,
                   s.description AS description, s.tools_required AS toolsRequired,
                   s.estimated_time AS estimatedTime, s.difficulty AS difficulty,
                   s.verified AS verified
            ORDER BY s.verified DESC, s.estimated_time ASC
            LIMIT $limit
        """
        return self._execute_query(cypher, {"faultId": fault_id, "limit": limit})

    # ==================== 节点存在性检查 ====================

    def fault_node_exists(self, name: str) -> bool:
        """检查指定名称的 Fault 节点是否存在于图谱中（模糊匹配）。"""
        result = self._execute_query(
            "MATCH (f:Fault) WHERE f.name CONTAINS $name RETURN f.name LIMIT 1",
            {"name": name}
        )
        return len(result) > 0

    def solution_node_exists(self, title: str) -> bool:
        """检查指定标题的 Solution 节点是否存在于图谱中（模糊匹配）。"""
        result = self._execute_query(
            "MATCH (s:Solution) WHERE s.title CONTAINS $title RETURN s.title LIMIT 1",
            {"title": title}
        )
        return len(result) > 0

    # ==================== 核心：5分支诊断路径查询 ====================

    def find_diagnosis_paths(
        self,
        keyword: str = None,
        component_ids: List[str] = None,
        fault_ids: List[str] = None,
        component_score_map: Dict[str, float] = None,
        fault_score_map: Dict[str, float] = None,
        page: int = 0,
        size: int = 5
    ) -> Dict[str, Any]:
        """
        分页查询诊断路径，与 Java GraphQueryServiceImpl.findDiagnosisPaths() 对齐。

        调用方（工具层）负责：
        1. 将 component_description/fault_description 转为向量
        2. 调用 search_*_by_embedding() 获取匹配的 ID 和分数
        3. 将结果传入本方法

        5 分支路由：
        1. 设备 + 部件 + 故障
        2. 部件 + 故障
        3. 只有部件
        4. 设备 + 故障
        5. 只有故障

        Returns:
            {"records": [...], "total": N, "page": page, "size": size}
        """
        safe_page = max(page, 0)
        safe_size = max(size, 5)
        skip = safe_page * safe_size
        comp_ids = component_ids or []
        flt_ids = fault_ids or []
        comp_scores = component_score_map or {}
        flt_scores = fault_score_map or {}

        if not comp_ids and not flt_ids:
            return {"records": [], "total": 0, "page": safe_page, "size": safe_size}

        # 设备匹配
        device_ids = []
        if self._has_text(keyword):
            devices = self.find_devices(keyword=keyword)
            device_ids = [d["id"] for d in devices]

        has_device = len(device_ids) > 0
        has_component = len(comp_ids) > 0
        has_fault = len(flt_ids) > 0

        if not has_component and not has_fault:
            return {"records": [], "total": 0, "page": safe_page, "size": safe_size}

        # 5 分支路由
        if has_device and has_component and has_fault:
            records_raw, total = self._branch_device_component_fault(
                device_ids, comp_ids, flt_ids, skip, safe_size
            )
        elif has_component and has_fault:
            records_raw, total = self._branch_component_fault(
                comp_ids, flt_ids, skip, safe_size
            )
        elif has_component:
            records_raw, total = self._branch_component_only(
                comp_ids, skip, safe_size
            )
        elif has_device:
            records_raw, total = self._branch_device_fault(
                device_ids, flt_ids, skip, safe_size
            )
        else:
            records_raw, total = self._branch_fault_only(
                flt_ids, skip, safe_size
            )

        records: List[Dict[str, Any]] = []
        for r in records_raw:
            r["fault_score"] = flt_scores.get(r.get("fault_id"))
            r["component_score"] = comp_scores.get(r.get("component_id"))
            r["path_text"] = _build_path_text(r)
            records.append(r)

        logger.info(
            f"[graph] find_diagnosis_paths keyword={keyword} "
            f"comp_ids={len(comp_ids)} fault_ids={len(flt_ids)} "
            f"found={len(records)} total={total}"
        )
        return {"records": records, "total": total, "page": safe_page, "size": safe_size}

    # ---- 5 分支实现 ----

    def _branch_device_component_fault(
        self, device_ids, component_ids, fault_ids, skip, limit
    ):
        where = "d.id IN $deviceIds AND c.id IN $componentIds AND f.id IN $faultIds"
        cypher = _BASE_CYPHER.replace("{where_clause}", where)
        count_cypher = _COUNT_CYPHER.replace("{where_clause}", where)
        params = {"deviceIds": device_ids, "componentIds": component_ids,
                  "faultIds": fault_ids, "skip": skip, "limit": limit}
        records = self._execute_query(cypher, params)
        total = self._execute_single(count_cypher, params)
        return records, total["total"] if total else 0

    def _branch_component_fault(self, component_ids, fault_ids, skip, limit):
        where = "c.id IN $componentIds AND f.id IN $faultIds"
        cypher = _BASE_CYPHER.replace("{where_clause}", where)
        count_cypher = _COUNT_CYPHER.replace("{where_clause}", where)
        params = {"componentIds": component_ids, "faultIds": fault_ids,
                  "skip": skip, "limit": limit}
        records = self._execute_query(cypher, params)
        total = self._execute_single(count_cypher, params)
        return records, total["total"] if total else 0

    def _branch_component_only(self, component_ids, skip, limit):
        where = "c.id IN $componentIds"
        cypher = _BASE_CYPHER.replace("{where_clause}", where)
        count_cypher = _COUNT_CYPHER.replace("{where_clause}", where)
        params = {"componentIds": component_ids, "skip": skip, "limit": limit}
        records = self._execute_query(cypher, params)
        total = self._execute_single(count_cypher, params)
        return records, total["total"] if total else 0

    def _branch_device_fault(self, device_ids, fault_ids, skip, limit):
        where = "d.id IN $deviceIds AND f.id IN $faultIds"
        cypher = _BASE_CYPHER.replace("{where_clause}", where)
        count_cypher = _COUNT_CYPHER.replace("{where_clause}", where)
        params = {"deviceIds": device_ids, "faultIds": fault_ids,
                  "skip": skip, "limit": limit}
        records = self._execute_query(cypher, params)
        total = self._execute_single(count_cypher, params)
        return records, total["total"] if total else 0

    def _branch_fault_only(self, fault_ids, skip, limit):
        where = "f.id IN $faultIds"
        cypher = _BASE_CYPHER.replace("{where_clause}", where)
        count_cypher = _COUNT_CYPHER.replace("{where_clause}", where)
        params = {"faultIds": fault_ids, "skip": skip, "limit": limit}
        records = self._execute_query(cypher, params)
        total = self._execute_single(count_cypher, params)
        return records, total["total"] if total else 0

    def query_diagnosis_path(
        self, keyword: str = None, fault_name: str = None, limit: int = 10
    ) -> List[DiagnosisPath]:
        """
        简化版诊断路径查询（供 GraphQueryTool 使用）。
        按设备关键字 + 故障名称模糊匹配，返回 Device→Component→Fault→Solution 路径。
        """
        where_parts = []
        params = {"limit": limit}

        if self._has_text(keyword):
            where_parts.append(
                "(d.name CONTAINS $keyword OR d.code CONTAINS $keyword "
                "OR d.model CONTAINS $keyword OR d.location CONTAINS $keyword)"
            )
            params["keyword"] = keyword

        if self._has_text(fault_name):
            where_parts.append("f.name CONTAINS $faultName")
            params["faultName"] = fault_name

        where_clause = " AND ".join(where_parts) if where_parts else "true"

        cypher = f"""
            MATCH (d:Device)-[:OWNS]->(c:Component)-[:CAUSES]->(f:Fault)
            WHERE {where_clause}
            OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(s:Solution)
            WITH d, c, f, s
            ORDER BY s.verified DESC, s.estimated_time ASC
            LIMIT $limit
            RETURN d.id AS deviceId, d.name AS deviceName,
                   c.id AS componentId, c.name AS componentName,
                   f.id AS faultId, f.name AS faultName, f.severity AS faultSeverity,
                   s.id AS solutionId, s.title AS solutionTitle,
                   s.estimated_time AS estimatedTime, s.verified AS verified
        """
        rows = self._execute_query(cypher, params)
        return [DiagnosisPath(**_map_path(r)) for r in rows]

    def query_diagnosis_by_image_vector(
        self, vector: List[float], limit: int = 10, min_score: float = 0.5
    ) -> List[DiagnosisPath]:
        """
        用多模态向量检索故障，并展开关联的部件和解决方案路径。
        """
        cypher = """
            CALL db.index.vector.queryNodes('fault_multimodal_index', $limit, $vector)
            YIELD node AS f, score
            WHERE score >= $minScore
            OPTIONAL MATCH (c:Component)-[:CAUSES]->(f)
            OPTIONAL MATCH (f)-[:HAS_SOLUTION]->(s:Solution)
            OPTIONAL MATCH (d:Device)-[:OWNS]->(c)
            RETURN d.id AS deviceId, d.name AS deviceName,
                   c.id AS componentId, c.name AS componentName,
                   f.id AS faultId, f.name AS faultName, f.severity AS faultSeverity,
                   s.id AS solutionId, s.title AS solutionTitle,
                   s.estimated_time AS estimatedTime, s.verified AS verified,
                   score
            ORDER BY score DESC, s.verified DESC
            LIMIT $limit
        """
        rows = self._execute_query(cypher, {
            "vector": vector, "limit": limit, "minScore": min_score
        })
        results = []
        for r in rows:
            path = DiagnosisPath(**_map_path(r))
            path.fault_score = r.get("score")
            results.append(path)
        return results

    @staticmethod
    def _has_text(value: str) -> bool:
        return value is not None and value.strip() != ""


_graph_service: Optional[GraphService] = None


def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
