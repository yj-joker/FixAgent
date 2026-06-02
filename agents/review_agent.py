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

    _MEASUREMENT_PATTERN = re.compile(
        r'\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*'
        r'(?:(?:[～~\-–—]|±)\s*\d+(?:\.\d+)?)?\s*'
        r'(?:N\s*[·.]?\s*m|N|mm|cm|km|公里|个月|月|小时|分钟|秒|h|min|'
        r'℃|°[CF]?|MPa|kPa|V|%|整?圈)',
        re.IGNORECASE,
    )
    _MODEL_PATTERN = re.compile(
        r'(?:型号|规格|推荐型号|推荐使用)[：:\s]*(?:为|使用)?\s*'
        r'([A-Z]{1,8}(?:[\s-]?[A-Z0-9]*\d[A-Z0-9-]*)+)',
        re.IGNORECASE,
    )
    _MODEL_REFERENCE_PATTERN = re.compile(
        r'(?:[A-Z]{2,}\s+)?[A-Z]+\d[A-Z0-9-]*(?:-[A-Z0-9-]+)?',
        re.IGNORECASE,
    )
    _SAFETY_WORDS = (
        "断开负极", "断开蓄电池", "切断电源", "断电", "验电", "泄压",
        "停止运行", "停机", "禁止启动", "佩戴护目镜", "佩戴防护手套",
    )

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
    def _normalize(text: str) -> str:
        value = text.upper().replace("，", ",").replace("·", ".").replace("～", "~")
        value = re.sub(r'[\s,]', '', value)
        return value

    @classmethod
    def _extract_critical_claims(cls, sentence: str) -> List[str]:
        claims: List[str] = []
        for match in cls._MEASUREMENT_PATTERN.finditer(sentence):
            value = match.group(0).strip()
            if value and value not in claims:
                claims.append(value)
        for match in cls._MODEL_PATTERN.finditer(sentence):
            value = match.group(1).strip()
            if value and value not in claims:
                claims.append(value)
        if any(word in sentence for word in ("型号", "规格", "匹配", "推荐")):
            for match in cls._MODEL_REFERENCE_PATTERN.finditer(sentence):
                value = match.group(0).strip()
                if value and value not in claims:
                    claims.append(value)
        for word in cls._SAFETY_WORDS:
            if word in sentence and word not in claims:
                claims.append(word)
        return claims

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @classmethod
    def _extract_result_text(cls, value: Any) -> List[str]:
        if isinstance(value, list):
            texts: List[str] = []
            for item in value:
                texts.extend(cls._extract_result_text(item))
            return texts
        if isinstance(value, dict):
            texts = []
            for key in ("content", "text", "summary", "caption", "image_summary"):
                content = value.get(key)
                if isinstance(content, str) and content.strip():
                    texts.append(content)
            if texts:
                return texts
            for child in value.values():
                texts.extend(cls._extract_result_text(child))
            return texts
        return []

    @classmethod
    def _collect_evidence(cls, react_trace: List[Dict]) -> List[str]:
        texts: List[str] = []
        for step in react_trace:
            if step.get("action") != "tool_call":
                continue
            for tc in step.get("tool_calls", []):
                if tc.get("name") != "knowledge_retrieval":
                    continue
                result_data = tc.get("result_data")
                if result_data is not None:
                    texts.extend(cls._extract_result_text(result_data))
                elif tc.get("result_summary"):
                    texts.append(tc["result_summary"])
        return texts

    @classmethod
    async def run(cls, answer: str, react_trace: List[Dict]) -> Dict[str, Any]:
        factual = [s for s in cls._split_sentences(answer) if cls._is_factual_claim(s)]
        if not factual:
            return {"unverified_claims": [], "total_claims": 0, "verified_count": 0,
                    "unverified_count": 0, "threshold": cls.THRESHOLD}

        evidence = cls._collect_evidence(react_trace)
        if not evidence:
            return {"unverified_claims": [
                        {"sentence": s, "max_similarity": 0.0,
                         "critical_claims": cls._extract_critical_claims(s)}
                        for s in factual
                    ],
                    "verified_claims": [],
                    "total_claims": len(factual), "verified_count": 0,
                    "unverified_count": len(factual), "threshold": cls.THRESHOLD,
                    "note": "无工具调用记录，无法验证"}

        evidence_text = "\n".join(evidence)
        normalized_evidence = cls._normalize(evidence_text)
        unverified = []
        verified_claims = []
        remaining_factual = []

        for sentence in factual:
            critical_claims = cls._extract_critical_claims(sentence)
            if not critical_claims:
                remaining_factual.append(sentence)
                continue
            unmatched = [
                claim for claim in critical_claims
                if cls._normalize(claim) not in normalized_evidence
            ]
            if unmatched:
                matched = [claim for claim in critical_claims if claim not in unmatched]
                unverified.append({
                    "sentence": sentence,
                    "critical_claims": critical_claims,
                    "matched_claims": matched,
                    "unmatched_claims": unmatched,
                    "reason": "关键内容未找到明确依据",
                })
            else:
                verified_claims.append({
                    "sentence": sentence,
                    "critical_claims": critical_claims,
                    "verified_by": "literal_evidence",
                })

        if not remaining_factual:
            verified_count = len(verified_claims)
            return {"unverified_claims": unverified, "verified_claims": verified_claims,
                    "total_claims": len(factual), "verified_count": verified_count,
                    "unverified_count": len(unverified), "threshold": cls.THRESHOLD}

        try:
            from embeddings.text_embedding import get_text_embedding
            vecs = await get_text_embedding().embed_batch(remaining_factual + evidence)
            n = len(remaining_factual)
            sent_vecs, ev_vecs = vecs[:n], vecs[n:]
        except Exception as e:
            logger.warning(f"[grounding] 向量化失败: {e}")
            unverified.extend({
                "sentence": sentence,
                "max_similarity": 0.0,
                "critical_claims": cls._extract_critical_claims(sentence),
                "reason": "审核服务暂不可用，无法确认",
            } for sentence in remaining_factual)
            return {"unverified_claims": unverified, "verified_claims": verified_claims,
                    "total_claims": len(factual), "verified_count": len(verified_claims),
                    "unverified_count": len(unverified), "threshold": cls.THRESHOLD,
                    "error": str(e), "note": "向量化失败，无法确认"}

        for i, sv in enumerate(sent_vecs):
            sims = [cls._cosine(sv, ev) for ev in ev_vecs]
            best = max(sims) if sims else 0.0
            if best < cls.THRESHOLD:
                unverified.append({"sentence": remaining_factual[i], "max_similarity": round(best, 4)})
            else:
                verified_claims.append({
                    "sentence": remaining_factual[i],
                    "max_similarity": round(best, 4),
                    "verified_by": "semantic_evidence",
                })

        verified_count = len(verified_claims)
        logger.info(f"[grounding] 总声明={len(factual)} 已验证={verified_count} 未验证={len(unverified)}")
        return {"unverified_claims": unverified, "verified_claims": verified_claims,
                "total_claims": len(factual),
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

    @staticmethod
    def _skipped_grounding() -> Dict[str, Any]:
        return {
            "skipped": True,
            "reason": "review_level skips grounding check",
            "unverified_claims": [],
            "verified_claims": [],
            "total_claims": 0,
            "verified_count": 0,
            "unverified_count": 0,
        }

    @staticmethod
    def _skipped_graph() -> Dict[str, Any]:
        return {
            "skipped": True,
            "reason": "review_level skips graph check",
            "unverified_paths": [],
            "verified_paths": [],
            "total_paths": 0,
            "verified_count": 0,
            "unverified_count": 0,
        }

    @staticmethod
    def _move_unverified_critical_lines(message: str, grounding: Dict[str, Any]) -> tuple[str, List[str]]:
        """Remove unsupported high-risk lines from formal guidance for separate display."""
        targets = [
            item.get("sentence", "").strip()
            for item in grounding.get("unverified_claims", [])
            if item.get("critical_claims") and item.get("sentence", "").strip()
        ]
        if not targets:
            return message, []

        normalized_targets = {
            re.sub(r'[。；;\s]+$', '', target).strip()
            for target in targets
        }
        kept_lines: List[str] = []
        removed: List[str] = []
        for line in message.splitlines():
            candidate = re.sub(r'[。；;\s]+$', '', line.strip()).strip()
            if any(target == candidate or target in candidate for target in normalized_targets):
                clean = line.strip().lstrip("-* ").strip()
                if clean not in removed:
                    removed.append(clean)
                continue
            kept_lines.append(line)

        formal_message = "\n".join(kept_lines).strip()
        if not formal_message:
            formal_message = "当前资料不足以形成可确认的正式操作指引。"
        return formal_message, removed

    @staticmethod
    def _confirmed_critical_values(grounding: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        for claim in grounding.get("verified_claims", []):
            for value in claim.get("critical_claims", []):
                if value not in values:
                    values.append(value)
        for claim in grounding.get("unverified_claims", []):
            for value in claim.get("matched_claims", []):
                if value not in values:
                    values.append(value)
        return values

    async def review(self, fix_output: AgentOutput, level: str = "full") -> AgentOutput:
        """
        对 FixAgent 输出执行 3 层校验。

        Returns:
            AgentOutput，message 可能被第3层追加安全警告；
            metadata.verification 包含3层完整结果。
        """
        t0 = time.time()
        message = fix_output.message
        trace = fix_output.metadata.get("react_trace", [])
        review_level = (level or "full").lower()

        if review_level == "light":
            grounding = self._skipped_grounding()
            graph = self._skipped_graph()
        elif review_level == "standard":
            grounding = await _GroundingCheck.run(message, trace)
            graph = self._skipped_graph()
        else:
            review_level = "full"
            grounding = await _GroundingCheck.run(message, trace)
            graph = await _GraphCheck.run(message, trace)
        safety = _SafetyCheck.run(message)

        verification = {
            "review_level": review_level,
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

        final_message, held_for_confirmation = self._move_unverified_critical_lines(message, grounding)
        sections = [final_message]
        confirmed_values = self._confirmed_critical_values(grounding)
        if confirmed_values:
            confirmed_lines = "\n".join(f"- {value}" for value in confirmed_values)
            sections.append(
                "## 已核对关键值\n"
                "以下数值或型号可在当前知识依据中找到明确匹配：\n"
                f"{confirmed_lines}"
            )
        if held_for_confirmation:
            pending_lines = "\n".join(f"- {text}" for text in held_for_confirmation)
            sections.append(
                "## 待确认信息\n"
                "以下关键内容未在当前知识依据中找到明确出处，已从正式指引中移出：\n"
                f"{pending_lines}"
            )

        appended = safety.get("appended_text", "")
        if appended:
            sections.append(
                "## 系统通用安全提醒\n"
                "以下提醒来自系统预设安全规则，不代表当前手册原文：\n\n"
                f"{appended}"
            )
        final_message = "\n\n".join(section for section in sections if section)

        latency = verification["verification_latency_ms"]
        logger.info(
            f"[review] level={review_level} "
            f"依据校验={grounding.get('unverified_count', 0)}/"
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
                "held_for_confirmation": held_for_confirmation,
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
            if claim.get("critical_claims"):
                continue
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
