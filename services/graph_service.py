"""
图数据库服务

基于 Neo4j 实现设备检修知识图谱的查询和关系管理。

【节点类型】
- Device: 设备（id, name, code, model, location, manufacturer）
- Component: 部件（id, name, partNumber, specification, supplier, lifecycle）
- Fault: 故障（id, code, name, description, severity, category）
- Solution: 解决方案（id, code, title, description, toolsRequired, estimatedTime, difficulty, verified）
- CaseRecord: 案例记录（id, caseNumber, title, diagnosis, resolution, result）

【关系类型】
- OWNS: Device -> Component（设备拥有部件）
- HAS_FAULT: Device -> Fault（设备发生故障）
- CAUSES: Component -> Fault（部件导致故障）
- HAS_SOLUTION: Fault -> Solution（故障有解决方案）
- RECORDED: CaseRecord -> Fault（案例记录了故障）
"""

import logging

logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from config.settings import get_settings


class DiagnosisPath(BaseModel):
    """诊断路径模型 - 设备->部件->故障->解决方案"""
    component_id: Optional[str] = Field(default=None, description="部件ID")
    component_name: Optional[str] = Field(default=None, description="部件名称")
    fault_id: Optional[str] = Field(default=None, description="故障ID")
    fault_name: Optional[str] = Field(default=None, description="故障名称")
    fault_severity: Optional[str] = Field(default=None, description="故障严重程度")
    solution_id: Optional[str] = Field(default=None, description="解决方案ID")
    solution_title: Optional[str] = Field(default=None, description="解决方案标题")
    estimated_time: Optional[int] = Field(default=None, description="预计耗时（分钟）")
    verified: Optional[bool] = Field(default=None, description="是否已验证")


class DeviceInfo(BaseModel):
    """设备信息模型"""
    id: str = Field(description="设备ID")
    name: str = Field(description="设备名称")
    code: Optional[str] = Field(default=None, description="设备编码")
    model: Optional[str] = Field(default=None, description="设备型号")
    location: Optional[str] = Field(default=None, description="存放位置")
    manufacturer: Optional[str] = Field(default=None, description="制造商")


class ComponentInfo(BaseModel):
    """部件信息模型"""
    id: str = Field(description="部件ID")
    name: str = Field(description="部件名称")
    part_number: Optional[str] = Field(default=None, description="部件编号")
    specification: Optional[str] = Field(default=None, description="规格参数")
    supplier: Optional[str] = Field(default=None, description="供应商")
    lifecycle: Optional[str] = Field(default=None, description="生命周期")
    unit_price: Optional[float] = Field(default=None, description="单价")


class FaultInfo(BaseModel):
    """故障信息模型"""
    id: str = Field(description="故障ID")
    code: Optional[str] = Field(default=None, description="故障编码")
    name: str = Field(description="故障名称")
    description: Optional[str] = Field(default=None, description="故障描述")
    severity: Optional[str] = Field(default=None, description="严重程度")
    category: Optional[str] = Field(default=None, description="故障类别")


class SolutionInfo(BaseModel):
    """解决方案信息模型"""
    id: str = Field(description="解决方案ID")
    code: Optional[str] = Field(default=None, description="解决方案编码")
    title: str = Field(description="解决标题")
    description: Optional[str] = Field(default=None, description="详细描述")
    tools_required: Optional[str] = Field(default=None, description="所需工具")
    estimated_time: Optional[int] = Field(default=None, description="预计耗时（分钟）")
    difficulty: Optional[str] = Field(default=None, description="难度")
    verified: Optional[bool] = Field(default=None, description="是否已验证")


class GraphService:
    """
    Neo4j 图数据库服务

    提供知识图谱查询和关系管理功能
    """

    def __init__(self):
        self.settings = get_settings()
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

    def _execute_query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行查询并返回结果列表"""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    def _execute_single(self, cypher: str, params: Dict[str, Any]) -> Optional[Any]:
        """执行查询并返回单个结果"""
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher, params)
            record = result.single()
            return dict(record) if record else None

    # ==================== 核心诊断路径查询 ====================

    def query_diagnosis_path(
        self,
        keyword: str = None,
        fault_name: str = None,
        limit: int = 10
    ) -> List[DiagnosisPath]:
        """
        查询诊断路径

        根据设备名称关键字和故障名称，查询完整诊断路径：
        Device -> Component -> Fault -> Solution

        与 Java 端 GraphQueryServiceImpl.findDiagnosisPath() 对应。

        Args:
            keyword: 设备名称关键字（模糊匹配 name/code/model/location）
            fault_name: 故障名称关键字（模糊匹配）
            limit: 返回结果数量限制

        Returns:
            诊断路径列表
        """
        # 先搜索匹配的设备
        devices = self.find_devices(keyword=keyword, limit=limit)

        diagnosis_paths = []
        for device in devices:
            paths = self._query_path_for_device(device["id"], fault_name, limit)
            diagnosis_paths.extend(paths)

        return [DiagnosisPath(**p) for p in diagnosis_paths]

    def _query_path_for_device(
        self,
        device_id: str,
        fault_name: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查询指定设备的诊断路径

        对应 Java 端 GraphQueryServiceImpl.getList() 方法的 Cypher 查询。
        """
        cypher = """
            MATCH (d:Device {id: $deviceId})
                  -[:OWNS]->(c:Component)
                  -[:CAUSES]->(f:Fault)
                  -[:HAS_SOLUTION]->(s:Solution)
            WHERE $faultName IS NULL
               OR $faultName = ''
               OR f.name CONTAINS $faultName
            RETURN c.id AS componentId,
                   c.name AS componentName,
                   f.id AS faultId,
                   f.name AS faultName,
                   f.severity AS faultSeverity,
                   s.id AS solutionId,
                   s.title AS solutionTitle,
                   s.estimated_time AS estimatedTime,
                   s.verified AS verified
            ORDER BY s.verified DESC, s.estimated_time ASC
            LIMIT $limit
        """

        # 处理 estimatedTime 和 verified 可能为 None 的情况
        results = self._execute_query(cypher, {
            "deviceId": device_id,
            "faultName": fault_name or "",
            "limit": limit
        })

        # 标准化字段名和类型
        standardized = []
        for r in results:
            standardized.append({
                "component_id": r.get("componentId"),
                "component_name": r.get("componentName"),
                "fault_id": r.get("faultId"),
                "fault_name": r.get("faultName"),
                "fault_severity": r.get("faultSeverity"),
                "solution_id": r.get("solutionId"),
                "solution_title": r.get("solutionTitle"),
                "estimated_time": r.get("estimatedTime"),
                "verified": r.get("verified")
            })
        return standardized

    # ==================== 设备管理 ====================

    def find_devices(
        self,
        keyword: str = None,
        skip: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        按关键字搜索设备

        对应 Java 端 DeviceRepository.getDevices()。
        """
        cypher = """
            MATCH (d:Device)
            WHERE $keyword IS NULL
               OR $keyword = ''
               OR d.name CONTAINS $keyword
               OR d.code CONTAINS $keyword
               OR d.model CONTAINS $keyword
               OR d.location CONTAINS $keyword
            RETURN d.id AS id,
                   d.name AS name,
                   d.code AS code,
                   d.model AS model,
                   d.location AS location,
                   d.manufacturer AS manufacturer
            ORDER BY d.name ASC
            SKIP $skip
            LIMIT $limit
        """
        return self._execute_query(cypher, {
            "keyword": keyword or "",
            "skip": skip,
            "limit": limit
        })

    def get_device_overview(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        获取设备概览信息

        对应 Java 端 DeviceRepository.getDeviceOverview()。
        """
        cypher = """
            MATCH (d:Device {id: $deviceId})
            OPTIONAL MATCH (d)-[:OWNS]->(c:Component)
            WITH d, count(DISTINCT c) AS componentCount
            OPTIONAL MATCH (d)-[:HAS_FAULT]->(f:Fault)
            RETURN d.id AS deviceId,
                   d.name AS deviceName,
                   d.code AS code,
                   d.model AS model,
                   d.location AS location,
                   d.manufacturer AS manufacturer,
                   componentCount,
                   count(DISTINCT f) AS faultCount
        """
        return self._execute_single(cypher, {"deviceId": device_id})

    # ==================== 部件管理 ====================

    def find_components_by_device(
        self,
        device_id: str,
        component_name: str = None,
        skip: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查询设备拥有的部件

        对应 Java 端 DeviceRepository.getComponentRecords()。
        """
        cypher = """
            MATCH (d:Device {id: $deviceId})-[:OWNS]->(c:Component)
            WHERE $componentName IS NULL
               OR $componentName = ''
               OR c.name CONTAINS $componentName
            RETURN c.id AS id,
                   c.name AS name,
                   c.part_number AS partNumber,
                   c.specification AS specification,
                   c.supplier AS supplier,
                   c.lifecycle AS lifecycle,
                   c.unit_price AS unitPrice
            ORDER BY c.name ASC
            SKIP $skip
            LIMIT $limit
        """
        return self._execute_query(cypher, {
            "deviceId": device_id,
            "componentName": component_name or "",
            "skip": skip,
            "limit": limit
        })

    def find_faults_by_component(
        self,
        component_id: str,
        fault_name: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查询部件导致的故障

        用于分析：给定部件会出现哪些故障。
        """
        cypher = """
            MATCH (c:Component {id: $componentId})-[:CAUSES]->(f:Fault)
            WHERE $faultName IS NULL
               OR $faultName = ''
               OR f.name CONTAINS $faultName
            RETURN f.id AS id,
                   f.code AS code,
                   f.name AS name,
                   f.description AS description,
                   f.severity AS severity,
                   f.category AS category
            LIMIT $limit
        """
        return self._execute_query(cypher, {
            "componentId": component_id,
            "faultName": fault_name or "",
            "limit": limit
        })

    # ==================== 故障管理 ====================

    def find_solutions_by_fault(
        self,
        fault_id: str,
        verified_only: bool = False,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查询故障的解决方案

        对应 Java 端 Fault 实体的 solutions 关系。
        """
        cypher = """
            MATCH (f:Fault {id: $faultId})-[:HAS_SOLUTION]->(s:Solution)
            """ + ("WHERE s.verified = true" if verified_only else "") + """
            RETURN s.id AS id,
                   s.code AS code,
                   s.title AS title,
                   s.description AS description,
                   s.tools_required AS toolsRequired,
                   s.estimated_time AS estimatedTime,
                   s.difficulty AS difficulty,
                   s.verified AS verified
            ORDER BY s.verified DESC, s.estimated_time ASC
            LIMIT $limit
        """
        return self._execute_query(cypher, {
            "faultId": fault_id,
            "limit": limit
        })

# ==================== 单例模式 ====================
_graph_service: Optional[GraphService] = None


def get_graph_service() -> GraphService:
    """获取图数据库服务单例"""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
