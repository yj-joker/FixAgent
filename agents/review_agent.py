"""
输出审核 Agent（ReviewAgent）

3层确定性校验，替代 LLM 自我审查。全部为确定性操作，零 LLM 调用。

设计原则：
- 第1层（_GroundingCheck）和第2层（_GraphCheck）只标记问题，不修改回答
- 第3层（_SafetyCheck）自动追加缺失的安全警告，是唯一会改输出的层
- 任一层异常时默认通过，确保不阻塞用户回复

调用链：api/main.py → FixAgent → ReviewAgent.review() → AgentOutput
"""

import re
import json
import math
import time
import logging
from typing import List, Dict, Any, Optional

from agents.base_agent import AgentOutput

logger = logging.getLogger(__name__)


# ====================================================================
# 第1层：检索依据校验（向量相似度）
# ====================================================================

class _GroundingCheck:
    """
    检查回答中的事实性陈述是否有检索结果支撑。

    算法：
    1. 拆分回答为句子，识别事实性陈述
    2. 从 react_trace 收集检索证据
    3. 批量向量化句子和证据，计算余弦相似度矩阵
    4. 相似度低于阈值的句子标记为"未验证"
    """

    THRESHOLD = 0.35

    _FACTUAL_KEYWORDS = [
        "建议", "需要", "必须", "检查", "更换", "维修",
        "原因", "导致", "造成", "引起", "可能", "一般",
        "型号", "规格", "参数", "温度", "压力", "电压",
        "步骤", "方法", "操作", "使用", "安装", "拆卸",
        "注意", "警告", "危险", "避免", "防止",
        "周期", "寿命", "频率", "次数", "时间",
    ]

    _FACTUAL_PATTERNS = [
        re.compile(r'\d+'),
        re.compile(r'[A-Z]+-\d+'),
        re.compile(r'[0-9]+°[CF]'),
        re.compile(r'[0-9]+V'),
        re.compile(r'[0-9]+[%％]'),
    ]

    _SKIP_PATTERNS = ["你好", "欢迎", "请问", "如需帮助", "以上是", "总结"]

    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        raw = re.split(r'[。；;\n]+', text)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    @classmethod
    def _is_factual_claim(cls, sentence: str) -> bool:
        if len(sentence) < 8:
            return False
        if any(p in sentence for p in cls._SKIP_PATTERNS):
            return False
        if any(kw in sentence for kw in cls._FACTUAL_KEYWORDS):
            return True
        if any(p.search(sentence) for p in cls._FACTUAL_PATTERNS):
            return True
        return False

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _collect_evidence(react_trace: List[Dict]) -> List[str]:
        texts = []
        for step in react_trace:
            if step.get("action") != "tool_call":
                continue
            for tc in step.get("tool_calls", []):
                args = tc.get("arguments", {})
                q = args.get("query", "") or args.get("keyword", "")
                if q:
                    texts.append(q)
                s = tc.get("result_summary", "")
                if s:
                    texts.append(s)
        return texts

    @classmethod
    async def run(cls, answer: str, react_trace: List[Dict]) -> Dict[str, Any]:
        factual = [s for s in cls._split_sentences(answer) if cls._is_factual_claim(s)]
        if not factual:
            return {"unverified_claims": [], "total_claims": 0, "verified_count": 0,
                    "unverified_count": 0, "threshold": cls.THRESHOLD}

        evidence = cls._collect_evidence(react_trace)
        if not evidence:
            return {"unverified_claims": [{"sentence": s, "max_similarity": 0.0} for s in factual],
                    "total_claims": len(factual), "verified_count": 0,
                    "unverified_count": len(factual), "threshold": cls.THRESHOLD,
                    "note": "无工具调用记录，无法验证"}

        try:
            from embeddings.text_embedding import get_text_embedding
            vecs = await get_text_embedding().embed_batch(factual + evidence)
            n = len(factual)
            sent_vecs, ev_vecs = vecs[:n], vecs[n:]
        except Exception as e:
            logger.warning(f"[grounding] 向量化失败: {e}")
            return {"unverified_claims": [], "total_claims": len(factual),
                    "verified_count": len(factual), "unverified_count": 0,
                    "threshold": cls.THRESHOLD, "error": str(e), "note": "向量化失败，默认通过"}

        unverified = []
        verified_count = 0
        for i, sv in enumerate(sent_vecs):
            sims = [cls._cosine(sv, ev) for ev in ev_vecs]
            best = max(sims) if sims else 0.0
            if best < cls.THRESHOLD:
                unverified.append({"sentence": factual[i], "max_similarity": round(best, 4)})
            else:
                verified_count += 1

        logger.info(f"[grounding] 总声明={n} 已验证={verified_count} 未验证={len(unverified)}")
        return {"unverified_claims": unverified, "total_claims": n,
                "verified_count": verified_count, "unverified_count": len(unverified),
                "threshold": cls.THRESHOLD}


# ====================================================================
# 第2层：图谱路径校验（Neo4j Cypher）
# ====================================================================

class _GraphCheck:
    """
    检查回答中的故障-方案对应关系是否在 Neo4j 图谱中真实存在。

    验证策略：
    1. 优先用 react_trace 中的图谱查询结果做 O(1) 匹配
    2. 未命中时用 Cypher 查询 Neo4j 确认故障/方案节点是否存在
    3. Neo4j 不可用时仅用 trace 结果，仍不行则标记未验证
    """

    @staticmethod
    def _parse_trace_results(react_trace: List[Dict]) -> List[Dict[str, str]]:
        paths = []
        for step in react_trace:
            if step.get("action") != "tool_call":
                continue
            for tc in step.get("tool_calls", []):
                if tc.get("name") not in ("graph_search_java", "graph_search_devices"):
                    continue
                summary = tc.get("result_summary", "")
                try:
                    parsed = json.loads(summary)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                paths.append({
                                    "fault_name": item.get("fault_name", ""),
                                    "solution_title": item.get("solution_title", ""),
                                })
                except (json.JSONDecodeError, TypeError):
                    fm = re.search(r'fault[_\s]?name["\']?\s*[:=]\s*["\']?([^"\'},\]]+)', summary, re.IGNORECASE)
                    sm = re.search(r'solution[_\s]?title["\']?\s*[:=]\s*["\']?([^"\'},\]]+)', summary, re.IGNORECASE)
                    if fm:
                        paths.append({
                            "fault_name": fm.group(1).strip(),
                            "solution_title": sm.group(1).strip() if sm else "",
                        })
        return paths

    @staticmethod
    def _extract_pairs(answer: str) -> List[Dict[str, str]]:
        pattern = re.compile(
            r'(?:^|\n)\s*(?:\d+[.、]|\-|\*)\s*'
            r'([^：:。\n]{3,30}?(?:故障|失效|损坏|断裂|磨损|过热|过载|短路|泄漏|异响|振动|腐蚀))'
            r'[：:，,\s]*'
            r'([^。\n]{5,50}?(?:更换|维修|修复|清洗|润滑|紧固|调整|校准|替换|加注|拆卸|检查))',
            re.MULTILINE
        )
        seen = set()
        pairs = []
        for m in pattern.finditer(answer):
            fn, st = m.group(1).strip(), m.group(2).strip()
            if len(fn) >= 2 and len(st) >= 2 and (fn, st) not in seen:
                seen.add((fn, st))
                pairs.append({"fault_name": fn, "solution_title": st})
        return pairs

    @classmethod
    async def run(cls, answer: str, react_trace: List[Dict]) -> Dict[str, Any]:
        claims = cls._extract_pairs(answer)
        trace_results = cls._parse_trace_results(react_trace)

        if not claims and not trace_results:
            return {"unverified_paths": [], "verified_paths": [], "total_paths": 0,
                    "verified_count": 0, "unverified_count": 0}

        known_faults = {r["fault_name"] for r in trace_results if r.get("fault_name")}
        known_pairs = {(r["fault_name"], r["solution_title"])
                       for r in trace_results if r.get("fault_name") and r.get("solution_title")}

        verified, unverified = [], []

        try:
            import httpx
            from config.settings import get_settings
            base_url = get_settings().java_service_url

            async with httpx.AsyncClient(timeout=10.0) as client:
                for c in claims:
                    fn, st = c["fault_name"], c["solution_title"]

                    if fn in known_faults and (not st or (fn, st) in known_pairs):
                        verified.append({"fault_name": fn, "solution_title": st, "verified_by": "trace"})
                        continue

                    try:
                        resp = await client.get(
                            f"{base_url}/weixiu/path/fault-exists",
                            params={"name": fn}
                        )
                        fault_exists = resp.json().get("data", False) if resp.status_code == 200 else False

                        if not fault_exists:
                            unverified.append({"fault_name": fn, "solution_title": st,
                                              "reason": "故障名不在图谱中"})
                            continue
                        if st:
                            resp = await client.get(
                                f"{base_url}/weixiu/path/solution-exists",
                                params={"title": st}
                            )
                            sol_exists = resp.json().get("data", False) if resp.status_code == 200 else False

                            if sol_exists:
                                verified.append({"fault_name": fn, "solution_title": st,
                                               "verified_by": "java_api"})
                            else:
                                unverified.append({"fault_name": fn, "solution_title": st,
                                                 "reason": "方案名不在图谱中"})
                        else:
                            verified.append({"fault_name": fn, "solution_title": "", "verified_by": "fault_only"})
                    except Exception:
                        unverified.append({"fault_name": fn, "solution_title": st, "reason": "查询执行异常"})

        except Exception as e:
            logger.warning(f"[验证] Java 图谱接口不可用: {e}")
            for c in claims:
                if c["fault_name"] in known_faults:
                    verified.append({**c, "verified_by": "trace_fallback"})
                else:
                    unverified.append({**c, "reason": "图谱接口不可用"})

        logger.info(f"[graph] 总路径={len(claims)} 已验证={len(verified)} 未验证={len(unverified)}")
        return {"unverified_paths": unverified, "verified_paths": verified,
                "total_paths": len(claims), "verified_count": len(verified),
                "unverified_count": len(unverified)}


# ====================================================================
# 第3层：安全规则引擎（关键词匹配）
# ====================================================================

class _SafetyCheck:
    """
    扫描回答中的危险操作关键词，检查是否有对应安全提醒。
    缺失则自动追加标准化警告文本。

    规则覆盖：高压电气 / 高温防护 / 化学品防护 / 重物吊装 /
              旋转部件 / 压力容器 / 电池电源

    注：此层为同步方法，纯 CPU 计算，无 I/O。
    """

    _RULES: List[Dict[str, Any]] = [
        {
            "name": "高压电气安全",
            "trigger": ["电压", "千伏", "kV", "通电", "电线", "电缆", "配电", "高压", "触电"],
            "required": ["断电", "验电"],
            "warning": "⚠️ 安全提醒：操作前必须切断电源并挂警示牌，用验电器确认无电压后方可作业。作业人员必须穿戴绝缘手套和绝缘鞋。"
        },
        {
            "name": "高温防护",
            "trigger": ["发动机", "排气", "冷却液", "高温", "过热", "涡轮", "锅炉", "蒸汽", "排气管", "气缸"],
            "required": ["冷却", "降温", "防烫"],
            "warning": "⚠️ 安全提醒：设备停机后需充分冷却（建议等待30分钟以上），操作时佩戴防烫手套。高温部件温度可达100°C以上，直接接触会造成严重烫伤。"
        },
        {
            "name": "化学品防护",
            "trigger": ["润滑油", "冷却液", "制动液", "溶剂", "清洗剂", "防冻液", "液压油", "机油", "燃油", "柴油", "汽油"],
            "required": ["防护手套", "护目镜", "手套", "通风"],
            "warning": "⚠️ 安全提醒：接触化学品时需佩戴防化手套和护目镜，确保操作区域通风良好。废液应按规定收集处理，禁止随意排放。"
        },
        {
            "name": "重物吊装",
            "trigger": ["吊装", "拆卸发动机", "变速箱", "起吊", "起重", "吊车", "千斤顶", "举升"],
            "required": ["起吊设备", "人员配合", "支撑", "固定"],
            "warning": "⚠️ 安全提醒：重物吊装前需检查吊具和索具完好性，确认载荷在设备额定范围内。作业时至少两人配合，无关人员需撤离作业区域。"
        },
        {
            "name": "旋转部件防护",
            "trigger": ["皮带", "齿轮", "风扇", "飞轮", "传动轴", "联轴器", "转子", "叶轮"],
            "required": ["停机", "断电", "防护罩"],
            "warning": "⚠️ 安全提醒：检查旋转部件前必须停机断电，确认部件完全停止转动。严禁在设备运行时将手或工具靠近旋转部件。"
        },
        {
            "name": "压力容器/管路安全",
            "trigger": ["气压", "液压", "压力容器", "气瓶", "压缩机", "高压油管", "蓄能器"],
            "required": ["泄压", "减压", "释放"],
            "warning": "⚠️ 安全提醒：拆卸压力管路或容器前必须先泄压，确认压力表归零。高压油液喷射可造成严重伤害，操作时必须佩戴护目镜。"
        },
        {
            "name": "电池/电源安全",
            "trigger": ["电池", "电瓶", "蓄电池", "锂电池", "充电"],
            "required": ["断开", "短路", "绝缘"],
            "warning": "⚠️ 安全提醒：操作电池前需先断开负极接线，工具手柄需做绝缘处理以防短路。电池短路会引起电弧、火灾或爆炸。"
        },
    ]

    @classmethod
    def run(cls, answer: str) -> Dict[str, Any]:
        triggered: List[str] = []
        missing: List[Dict] = []
        append_parts: List[str] = []

        for rule in cls._RULES:
            hits = [t for t in rule["trigger"] if t in answer]
            if not hits:
                continue
            triggered.append(rule["name"])
            lacked = [r for r in rule["required"] if r not in answer]
            if lacked:
                missing.append({"rule": rule["name"], "triggered_by": hits, "missing_keywords": lacked})
                append_parts.append(rule["warning"])

        appended = "\n\n".join(append_parts) if append_parts else ""
        logger.info(f"[safety] 触发规则={len(triggered)} 缺失警告={len(missing)}")
        return {
            "triggered_rules": triggered,
            "missing_warnings": missing,
            "appended_text": appended,
            "checked_rules": len(cls._RULES),
            "triggered_count": len(triggered),
            "missing_count": len(missing),
        }


# ====================================================================
# ReviewAgent 主体
# ====================================================================

class ReviewAgent:
    """
    输出审核 Agent — 3层确定性校验管线。

    不再调用 LLM 进行自我审查，而是通过：

    1. _GroundingCheck — 向量相似度验证检索依据
    2. _GraphCheck     — Neo4j Cypher 验证图谱路径
    3. _SafetyCheck    — 关键词规则引擎补全安全警告

    每层独立执行，异常时默认通过。仅第3层会修改输出内容（追加警告）。
    """

    @property
    def name(self) -> str:
        return "review_agent"

    @property
    def description(self) -> str:
        return "输出审核：3层确定性校验（检索依据/图谱路径/安全规则）"

    async def review(self, fix_output: AgentOutput) -> AgentOutput:
        """
        对 FixAgent 输出执行 3 层校验。

        Returns:
            AgentOutput，message 可能被第3层追加安全警告；
            metadata.verification 包含3层完整结果。
        """
        t0 = time.time()
        message = fix_output.message
        trace = fix_output.metadata.get("react_trace", [])

        # 第1层和第2层可并行（互相不依赖），第3层纯 CPU 可同步
        grounding = await _GroundingCheck.run(message, trace)
        graph = await _GraphCheck.run(message, trace)
        safety = _SafetyCheck.run(message)

        verification = {
            "grounding": grounding,
            "graph": graph,
            "safety": safety,
            "verification_latency_ms": int((time.time() - t0) * 1000),
        }

        has_issues = (
            grounding.get("unverified_count", 0) > 0 or
            graph.get("unverified_count", 0) > 0 or
            safety.get("missing_count", 0) > 0
        )

        final_message = message
        appended = safety.get("appended_text", "")
        if appended:
            final_message = f"{message}\n\n---\n{appended}"

        latency = verification["verification_latency_ms"]
        logger.info(
            f"[review] 依据校验={grounding.get('unverified_count', 0)}/"
            f"{grounding.get('total_claims', 0)} "
            f"图谱校验={graph.get('unverified_count', 0)}/"
            f"{graph.get('total_paths', 0)} "
            f"安全校验={safety.get('missing_count', 0)}/"
            f"{safety.get('triggered_count', 0)} "
            f"耗时={latency}ms 有问题={has_issues}"
        )

        return AgentOutput(
            agent_name="fix_agent",
            message=final_message,
            intention=fix_output.intention,
            tools_used=fix_output.tools_used,
            metadata={
                **fix_output.metadata,
                "verification": verification,
                "verification_has_issues": has_issues,
                "total_latency_ms": fix_output.latency_ms + latency,
            },
            latency_ms=fix_output.latency_ms + latency,
            raw_response=fix_output.raw_response,
        )


    def get_inline_markers(self, answer: str, verification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取内联验证标记的位置列表，供流式输出时插入。

        根据 grounding 和 graph 的校验结果，找出未验证内容在原文中的字符位置，
        返回按位置升序排列的标记列表。调用方在逐字流式输出时，
        当到达 marker["char_pos"] 时先发送 marker 事件再继续发 token。

        Returns:
            [{"char_pos": int, "text": str, "type": str}, ...]
        """
        markers: List[Dict[str, Any]] = []
        grounding = verification.get("grounding", {})
        graph = verification.get("graph", {})

        # grounding 未验证声明 → 在声明句首插入标记
        for claim in grounding.get("unverified_claims", []):
            sentence = claim.get("sentence", "")
            if not sentence:
                continue
            pos = answer.find(sentence)
            if pos < 0:
                continue
            sim = claim.get("max_similarity", 0.0)
            markers.append({
                "char_pos": pos,
                "text": f"⚠️[依据不足-相似度{sim:.2f}] ",
                "type": "grounding_unverified",
            })

        # graph 未验证路径 → 在故障名首次出现处插入标记
        for path in graph.get("unverified_paths", []):
            fault = path.get("fault_name", "")
            reason = path.get("reason", "")
            if not fault:
                continue
            pos = answer.find(fault)
            if pos < 0:
                continue
            if any(m["char_pos"] == pos for m in markers):
                continue
            label = f"⚠️[图谱:{reason}] " if reason else "⚠️[图谱未确认] "
            markers.append({
                "char_pos": pos,
                "text": label,
                "type": "graph_unverified",
            })

        markers.sort(key=lambda m: m["char_pos"])
        return markers


# ====================================================================
# 单例
# ====================================================================

_review_agent: Optional[ReviewAgent] = None


def get_review_agent() -> ReviewAgent:
    global _review_agent
    if _review_agent is None:
        _review_agent = ReviewAgent()
    return _review_agent
