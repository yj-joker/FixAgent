"""
图谱查询工具

封装 Neo4j 图数据库查询，为 Agent 提供设备检修知识图谱的查询能力。

【核心功能】
- 诊断路径查询：设备 → 部件 → 故障 → 解决方案
- 设备搜索：按关键字模糊搜索设备

【调用链】
Agent ReAct → GraphQueryTool._execute() → GraphService → Neo4j Cypher → 格式化结果

【关联】
- 上游：agents/diagnosis_agent.py（ReAct 循环中调用）
- 下游：services/graph_service.py（Neo4j 客户端）
- 继承：tools/base_tool.py 的 BaseTool
"""

from typing import List, Optional
import logging
from pydantic import BaseModel, Field

from tools.base_tool import BaseTool, ToolException
from services.graph_service import get_graph_service, DiagnosisPath, DeviceInfo

logger = logging.getLogger(__name__)


class DiagnosisPathResult(BaseModel):
    """诊断路径查询结果"""
    component_name: str = Field(description="部件名称")
    fault_name: str = Field(description="故障名称")
    fault_severity: Optional[str] = Field(default=None, description="故障严重程度")
    solution_title: str = Field(description="解决方案标题")
    estimated_time: Optional[int] = Field(default=None, description="预计耗时（分钟）")
    verified: Optional[bool] = Field(default=None, description="是否已验证")


class DeviceSearchResult(BaseModel):
    """设备搜索结果"""
    id: str = Field(description="设备ID")
    name: str = Field(description="设备名称")
    code: Optional[str] = Field(default=None, description="设备编码")
    model: Optional[str] = Field(default=None, description="设备型号")
    location: Optional[str] = Field(default=None, description="存放位置")
    manufacturer: Optional[str] = Field(default=None, description="制造商")


class GraphQueryTool(BaseTool):
    """
    图谱诊断路径查询工具

    根据设备名称关键字和故障描述，从知识图谱中查询完整的诊断路径。
    路径覆盖：设备 → 部件 → 故障 → 解决方案。

    对应 Java 端 GraphQueryServiceImpl.findDiagnosisPath() 和
    PathController.getPath() 端点。
    """

    @property
    def name(self) -> str:
        return "graph_query_diagnosis_path"

    @property
    def description(self) -> str:
        return (
            "从设备检修知识图谱中查询诊断路径。"
            "给定设备名称关键字和故障现象，返回完整的诊断链路："
            "哪些部件→导致什么故障→对应什么解决方案。"
            "结果按方案已验证优先、耗时少优先排序。"
            "适用场景：需要分析设备故障的因果关系、查找解决方案时使用。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "设备名称关键字，支持模糊匹配设备名称/编码/型号/位置"
                },
                "fault_name": {
                    "type": "string",
                    "description": "故障名称关键字，支持模糊匹配。如不确定可留空获取全部诊断路径"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量上限，默认10",
                    "default": 10
                }
            },
            "required": ["keyword"]
        }

    async def _execute(
        self,
        keyword: str,
        fault_name: str = None,
        limit: int = 10
    ) -> dict:
        """
        执行诊断路径查询

        Args:
            keyword: 设备名称关键字
            fault_name: 故障名称关键字（可选）
            limit: 返回数量上限

        Returns:
            {
                "devices_found": 匹配到的设备数量,
                "paths": [诊断路径列表]
            }

        Raises:
            ToolException: GRAPH_QUERY_FAILED
        """
        try:
            graph = get_graph_service()
            paths: List[DiagnosisPath] = graph.query_diagnosis_path(
                keyword=keyword,
                fault_name=fault_name,
                limit=limit
            )

            # 格式化结果
            formatted = []
            for p in paths:
                formatted.append({
                    "component_name": p.component_name or "未知部件",
                    "fault_name": p.fault_name or "未知故障",
                    "fault_severity": p.fault_severity,
                    "solution_title": p.solution_title or "暂无解决方案",
                    "estimated_time": p.estimated_time,
                    "verified": p.verified
                })

            return {
                "devices_found": len(formatted),
                "paths": formatted
            }

        except Exception as e:
            raise ToolException(
                code="GRAPH_QUERY_FAILED",
                message=f"图谱诊断路径查询失败: {e}"
            )


class GraphSearchDeviceTool(BaseTool):
    """
    图谱设备搜索工具

    按关键字搜索设备节点，返回匹配的设备基本信息列表。
    可用于诊断前缩小设备范围。
    """

    @property
    def name(self) -> str:
        return "graph_search_devices"

    @property
    def description(self) -> str:
        return (
            "从设备检修知识图谱中搜索设备。"
            "按设备名称/编码/型号/位置进行模糊匹配。"
            "适用场景：不确定设备全名时搜索设备列表，为诊断路径查询缩小范围。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键字，匹配设备名称/编码/型号/位置"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量上限，默认10",
                    "default": 10
                }
            },
            "required": ["keyword"]
        }

    async def _execute(
        self,
        keyword: str,
        limit: int = 10
    ) -> dict:
        """
        搜索设备

        Args:
            keyword: 搜索关键字
            limit: 返回数量上限

        Returns:
            {"count": 匹配数量, "devices": [设备列表]}

        Raises:
            ToolException: GRAPH_DEVICE_SEARCH_FAILED
        """
        try:
            graph = get_graph_service()
            devices = graph.find_devices(keyword=keyword, limit=limit)

            formatted = []
            for d in devices:
                formatted.append({
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "code": d.get("code"),
                    "model": d.get("model"),
                    "location": d.get("location"),
                    "manufacturer": d.get("manufacturer")
                })

            return {
                "count": len(formatted),
                "devices": formatted
            }

        except Exception as e:
            raise ToolException(
                code="GRAPH_DEVICE_SEARCH_FAILED",
                message=f"图谱设备搜索失败: {e}"
            )


class GraphImageSearchTool(BaseTool):
    """图片检索诊断路径工具"""

    @property
    def name(self) -> str:
        return "graph_image_search"

    @property
    def description(self) -> str:
        return (
            "通过上传的图片从知识图谱中检索诊断路径。"
            "将图片转为向量，在故障图片索引中搜索相似故障，"
            "返回关联的部件和解决方案。"
            "适用场景：用户上传了故障现场照片，需要识别故障并给出维修方案。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "图片 URL 列表（MinIO 地址），可选"
                },
                "text": {
                    "type": "string",
                    "description": "文字描述，可选。与图片一起融合为多模态向量检索"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量上限，默认10",
                    "default": 10
                }
            }
        }

    async def _execute(self, image_urls: list = None, text: str = "", limit: int = 10) -> dict:
        try:
            if not image_urls and not text:
                return {"paths": [], "message": "至少需要提供图片或文字描述"}

            import numpy as np

            text_vec = None
            img_vec = None

            if text:
                from embeddings.text_embedding import get_text_embedding
                text_emb = get_text_embedding()
                text_vec = np.array(await text_emb.embed(text))

            if image_urls:
                from embeddings.image_embedding import get_image_embedding
                img_emb = get_image_embedding()
                img_vecs = await img_emb.embed_batch(image_urls)
                img_vec = np.mean(img_vecs, axis=0)

            # 加权融合
            if text_vec is not None and img_vec is not None:
                fused = 0.3 * text_vec + 0.7 * img_vec
            elif text_vec is not None:
                fused = text_vec
            else:
                fused = img_vec

            # 归一化
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm

            avg_vector = fused.tolist()

            graph = get_graph_service()
            paths = graph.query_diagnosis_by_image_vector(
                vector=avg_vector, limit=limit, min_score=0.5
            )
            formatted = []
            for p in paths:
                formatted.append({
                    "component_name": p.component_name or "未知部件",
                    "fault_name": p.fault_name or "未知故障",
                    "fault_severity": p.fault_severity,
                    "solution_title": p.solution_title or "暂无解决方案",
                    "estimated_time": p.estimated_time,
                    "verified": p.verified,
                    "fault_score": p.fault_score
                })
            return {
                "input_text": bool(text),
                "input_images": len(image_urls or []),
                "paths_found": len(formatted),
                "paths": formatted
            }
        except Exception as e:
            raise ToolException(code="GRAPH_IMAGE_SEARCH_FAILED", message=f"多模态检索诊断路径失败: {e}")


# ==================== 单例 ====================

_query_tool: Optional[GraphQueryTool] = None
_search_tool: Optional[GraphSearchDeviceTool] = None
_image_search_tool: Optional[GraphImageSearchTool] = None


def get_graph_query_tool() -> GraphQueryTool:
    """获取图谱诊断路径查询工具单例"""
    global _query_tool
    if _query_tool is None:
        _query_tool = GraphQueryTool()
    return _query_tool


def get_graph_search_device_tool() -> GraphSearchDeviceTool:
    """获取图谱设备搜索工具单例"""
    global _search_tool
    if _search_tool is None:
        _search_tool = GraphSearchDeviceTool()
    return _search_tool


def get_graph_image_search_tool() -> GraphImageSearchTool:
    global _image_search_tool
    if _image_search_tool is None:
        _image_search_tool = GraphImageSearchTool()
    return _image_search_tool
