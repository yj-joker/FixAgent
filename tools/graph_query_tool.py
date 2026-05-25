"""
图谱设备搜索工具（通过 Java 端 HTTP 接口）

调用 Java 后端 /weixiu/device/search 接口搜索设备，
复用 Java 端 Neo4j 查询逻辑，Python 端不再直连 Neo4j。

【调用链】
FixAgent ReAct → GraphSearchDeviceTool._execute()
    → HTTP GET Java /weixiu/device/search?keyword=xxx&limit=10
    → Java DeviceService.searchDevices()
    → 返回 DeviceVO 列表

【关联】
- 上游：agents/fix_agent.py（ReAct 循环中调用）
- 下游：Java DeviceController /weixiu/device/search
- 继承：tools/base_tool.py 的 BaseTool
"""

from typing import Optional
import logging

import httpx

from tools.base_tool import BaseTool, ToolException
from config.settings import get_settings

logger = logging.getLogger(__name__)


class GraphSearchDeviceTool(BaseTool):
    """
    图谱设备搜索工具（通过 Java 后端）

    按关键字搜索设备节点，返回匹配的设备基本信息列表。
    """

    def __init__(self):
        self._settings = get_settings()
        self._base_url = self._settings.java_service_url

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
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/weixiu/device/search",
                    params={"keyword": keyword, "limit": limit}
                )
                resp.raise_for_status()
                result = resp.json()

            # Java 端返回格式: {"code": 1, "data": [...]}
            devices = result.get("data", [])

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

        except httpx.ConnectError:
            raise ToolException(
                code="JAVA_CONNECT_ERROR",
                message=f"无法连接 Java 后端服务: {self._base_url}"
            )
        except Exception as e:
            raise ToolException(
                code="GRAPH_DEVICE_SEARCH_FAILED",
                message=f"图谱设备搜索失败: {e}"
            )


# ==================== 单例 ====================

_search_tool: Optional[GraphSearchDeviceTool] = None


def get_graph_search_device_tool() -> GraphSearchDeviceTool:
    global _search_tool
    if _search_tool is None:
        _search_tool = GraphSearchDeviceTool()
    return _search_tool
