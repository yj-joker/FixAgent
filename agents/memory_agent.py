"""
工作记忆整理 Agent

将多条原始对话记录压缩为结构化记忆摘要。
Java 端在对话达到阈值（如30条）时触发整理，调用本 Agent 提取关键信息。

采用单次 function calling 架构：
LLM 读取对话 → 提取候选事实 → 调用 search_similar_facts 检索 → 判断冲突 → 输出结果

【与其他模块的关系】
- 继承 BaseAgent，覆盖 run() 实现 function calling 流程
- 由 api/main.py 的 /ai/memory/consolidate 端点调用
- 调用 services/llm_service.py 的 chat_with_tools()
- 工具注册：tools/fact_retrieval_tool.py 提供向量检索能力
"""

import json
import logging
import re
import time

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from pydantic import ValidationError
from schemas.memory import MemorySummary

logger = logging.getLogger(__name__)


MEMORY_SYSTEM_PROMPT = """你是工作记忆整理助手。从对话记录中提取并整理记忆，输出结构化结果。

## ⚠️ 最重要的规则（违反此规则等于输出无效）
- 事实、偏好、未完成事项 **只能从【用户】发言中提取**
- 【助手】发言是AI生成的回复，**绝对不能**从中提取事实、偏好或待办
- 助手说的内容仅用于理解对话上下文，不代表用户的观点或需求
- 如果助手建议了某个方案/做法，除非用户明确认可，否则不算用户偏好或待办
- **绝对禁止将助手的建议/方案/步骤推断为用户的计划或待办！**
  ✗ 用户说"想吃蛋糕"，助手回复"如何制作蛋糕" → 不能记录"用户打算制作蛋糕"
  ✗ 用户问"电机异响怎么办"，助手回复"建议更换轴承" → 不能记录"用户计划更换轴承"
  ✓ 只有用户自己说"我准备去换轴承"才能记录为待办

## 可用工具
- **search_similar_facts**: 在已有事实库中批量搜索与候选事实语义相似的历史事实。
  用法：先读完所有对话，识别出全部候选事实，提取每条事实的核心关键词（设备型号、错误码、故障部位等），然后一次性批量调用。

## 分类标准

### 事实（客观、已确认、可独立理解的信息）

**属于事实：** 用户提到的设备型号/参数/配置、用户描述的诊断过程和结果、用户确认的技术结论、用户项目/系统的客观信息
**不属于事实（绝对不能记录）：**
- 主观评价、工作习惯、未完成的任务
- 助手/AI输出的任何内容：包括解释、建议、方案、步骤、知识性回答
  ✗ 助手说"蛋糕需要面粉和鸡蛋" → 不是用户的事实
  ✗ 助手说"建议用型号A" → 不是事实（除非用户说"好，就用型号A"）
  ✗ 助手解释了某个原理 → 不是事实
- 只有用户亲口陈述或明确确认的信息才是事实

**事实提取规则（非常重要）：**
1. 自包含：每条事实必须脱离对话上下文也能完整理解
   ✗ "他说用那个框架"
   ✓ "用户的项目使用 Spring Boot 3.2 框架"
2. 原子化：每条事实只描述一件事
   ✗ "用户在做维修系统，用Java和MySQL"
   ✓ "用户正在开发一个设备维修管理系统"
   ✓ "用户的后端技术栈是 Java"
   ✓ "用户使用 MySQL 作为数据库"
3. 时效标注：如果事实可能随时间改变，加上时间标记
   ✓ "用户当前正在调试登录模块的bug（2026-05）"
4. **重要度判断（宁缺毋滥！）** — 只记录有长期价值的事实：
   ✓ 值得记录：设备型号、技术架构、项目名称、确认的结论
   ✗ 不要记录：当前正在调试的临时状态、对话中的过渡性表述、用户的随口一提
   判断标准：**如果这条信息在下周还有用，就记录；如果只在今天有用，就不记录**
5. **重要度评分（importance, 1-10）：**
   - 1-3：临时信息（当前调试状态、过渡性表述）
   - 4-6：一般技术细节（某个配置项的值、一次性操作结果）
   - 7-9：重要信息（设备型号、系统架构、关键故障结论）
   - 10：核心信息（安全规程、多次确认的关键事实）
6. **置信度评分（confidence, 0-1）：**
   - 0.90-1.00：用户明确、反复确认的信息
   - 0.70-0.89：用户正常陈述，无矛盾
   - 0.50-0.69：从上下文推断，可能需要确认
   - < 0.50：不确定，最好不要提取
7. **业务维度标注（维修场景专用）：**
   如果事实明显与特定设备/场地/任务相关，请标注：
   - device_type: 设备类型名称（如"液压泵"、"曲轴"、"变速箱"）
   - equipment_id: 如果对话中提到了具体设备编号/ID
   - site_id: 如果对话中提到了具体场地编号/ID
   - task_id: 如果是在某个检修任务讨论中产生的事实
   如果事实是通用性的（不特定于某设备/场地），所有维度留空字符串。
   不确定时宁可留空，不要猜测。

### 偏好（用户主动表达的主观倾向，需要严格区分）

**【是偏好 —— 必须满足以下任一条件才能记录】**
1. 用户的明确指令："以后回答用中文"、"不要给我写注释"、"回复简洁一点"
2. 用户纠正AI行为后的隐含要求：AI用英文回复后用户说"说中文" → 偏好中文
3. 用户主动表达的工作习惯："我习惯先写测试再写代码"
4. 用户主动表达的好恶："我不喜欢用Lombok"、"我更喜欢函数式写法"

**【不是偏好 —— 绝对不要记录为偏好】**
- 助手/AI说的任何内容 → 绝不是用户偏好！（即使助手建议了某种方式）
- 用户正在讨论/使用的技术 ≠ 偏好该技术
  "帮我看看这个Java代码怎么改" → 不是偏好，只是当前任务涉及Java
  "用Python写个脚本" → 不是偏好，只是一次性任务需求
- 用户提到但未表达态度的事物
  "React的虚拟DOM是什么原理" → 不是偏好，只是在提问
- 对话的主题/领域
  一整段关于数据库优化的讨论 → 不代表偏好数据库，只是当前话题
- 助手的自我介绍或能力说明 → 不是事实也不是偏好

**【sourceType 标注】每条偏好必须标注来源类型：**
- "explicit"：用户直接说出来的指令或态度（如"不要写注释"、"我喜欢详细解释"）
- "inferred"：从用户反复出现的行为模式推断的（如用户多次追问细节→可能偏好详细回复）
  注意：单次行为不足以推断偏好，需要有多次一致的模式

**【preferenceCategory 判断规则】**
- 0（用户级）：涉及个人习惯、跨话题通用的偏好，如回复语言、风格习惯
- 1（会话级）：仅针对本次具体任务的临时偏好，如"这次用表格形式展示"

### 未完成事项（悬而未决的待办）

**【是待办 —— 用户自己明确表达的行动意图】**
- 用户说"我明天去修电机" → ✓ 用户计划明天修电机
- 用户说"我待会儿试试重启" → ✓ 用户打算重启设备
- 用户问了但没得到答案的问题 → ✓ 未答复问题

**【不是待办 —— 绝对不要记录】**
- 助手建议的方案/步骤 → ✗ 不是用户的计划！
  用户问"怎么办"，助手说"建议换轴承" → 不能记录"用户打算换轴承"
- 助手描述的操作流程 → ✗ 不是用户的待办！
  助手回复"第一步拆开外壳，第二步检查线路" → 不能记录为用户的行动计划
- 用户随口提到的愿望/想法（没有行动意图） → ✗
  "想吃蛋糕" ≠ "打算制作蛋糕"，除非用户明确说"我要自己做一个蛋糕"
- 助手推荐的任何东西 → ✗ 除非用户回复"好的我去做"

**核心判断标准：用户是否用自己的话表达了"我要做/我打算做/我准备做"？**
如果用户没有这样说，就不是待办。不要从助手的回复中推断用户意图。

注意：一旦事项在新对话中得到解决，应转为事实，并将该事项的 id 放入 resolved_item_ids

## 冲突判断规则（仅针对事实）
根据工具返回的相似事实判断：
- 无相似结果（score < 0.7）或结果为空 → 正常新增
- 有相似且内容相同 → 不重复添加
- 有相似、同话题但结论不同 → 以新对话中的结论为准，在 superseded_ids 中标记旧事实的 id
- 有相似且互相印证 → 合并为一条更完整的表述

偏好和未完成事项不调用工具，按以下规则处理：
- 同类别、同级别偏好有矛盾 → 以最新表述为准
- 未完成事项已解决 → 将其 id 放入 resolved_item_ids
- 已有偏好和未完成事项附带的字段说明：
  - preferenceCategory: 0=用户级（所有对话公用）, 1=会话级（仅本次会话有效）
  - status: active=进行中, superseded=已放弃。已放弃的事项无需处理
  - id: 数据库主键，用于精确标记已解决的事项

## 提取质量门控（最终检查清单）
输出前，请逐条检查每个提取的条目：
1. ✅ 这条信息是从【用户】发言中提取的吗？（如果是从助手发言推断的 → 删除）
2. ✅ 如果是事实：下周还有参考价值吗？（如果只是当前调试的临时状态 → 删除）
3. ✅ 如果是待办：用户是否亲口说了"我要做/打算做"？（如果是助手建议的 → 删除）
4. ✅ 如果是偏好：是持久性的还是一次性的？（"这次用英文" ≠ 永久偏好 → 删除）

**宁缺毋滥原则：** 如果不确定是否应该提取，就不要提取。
错误记忆比没有记忆危害更大。空的 new_facts/updated_preferences/updated_unresolved 是完全正常的。

## 摘要要求
brief_summary 是导航索引，不是信息源。100字以内，只需概括"这段对话聊了什么话题"。
具体细节已经被提取为事实/偏好/待办，摘要中不需要重复这些细节。
如果收到了"之前的对话背景"，请在此基础上生成渐进式摘要（即更新旧摘要，而非从零开始）。

## 输出格式
严格按以下 JSON 输出，不要输出其他内容：
```json
{
  "new_facts": [
    {"content": "自包含的事实描述", "keywords": "检索用关键词", "source_seq_range": "3-5", "importance": 7, "confidence": 0.85, "device_type": "", "equipment_id": "", "site_id": "", "task_id": ""}
  ],
  "superseded_ids": ["要标记为无效的旧事实ID"],
  "updated_preferences": [
    {"content": "偏好描述", "category": "交互风格|格式要求|工作习惯|关注领域|其他", "preferenceCategory": 0, "sourceType": "explicit"}
  ],
  "updated_unresolved": [
    {"content": "待解决描述", "type": "未答复问题|进行中任务|用户待办", "status": "active"}
  ],
  "resolved_item_ids": [12, 34],
  "brief_summary": "100字以内的话题概括"
}
```
"""


class MemoryAgent(BaseAgent):
    """
    工作记忆整理 Agent

    单次 function calling 架构：
    1. 构建消息（含已有偏好/未完成 + 新对话）
    2. 注册 search_similar_facts 工具
    3. 调用 LLM（自动处理工具调用循环）
    4. 解析 JSON 返回结构化数据
    """

    @property
    def name(self) -> str:
        return "memory_agent"

    @property
    def description(self) -> str:
        return "工作记忆整理Agent：将多条原始对话压缩为结构化摘要"

    def get_system_prompt(self) -> str:
        return MEMORY_SYSTEM_PROMPT

    def _format_conversations(self, conversations: list) -> str:
        """
        将对话列表格式化为 LLM 可读的文本块。
        使用明确的分隔符和标注，帮助LLM区分用户发言和助手发言。
        """
        lines = ["## 新对话记录"]
        lines.append("（⚠️ 只从【用户】发言中提取事实、偏好和待办。【助手】发言仅作为上下文参考，绝不从中提取任何内容。助手的建议/方案不代表用户意图！）\n")
        for item in conversations:
            seq = item.get("seq", "?")
            content = item.get("content", "")
            if item.get("role") == "user":
                lines.append(f"━━━ 第{seq}轮 ━━━")
                lines.append(f"【用户】{content}")
            else:
                lines.append(f"【助手】{content}")
                lines.append("")  # 轮次间空行
        return "\n".join(lines)

    def _build_messages(self, input_data: AgentInput) -> list:
        """
        构建消息列表，包含已有记忆上下文和待整理的新对话。

        组装顺序：
        1. 之前的对话背景（上一轮摘要，用于生成渐进式摘要）
        2. 已有偏好表格（让LLM知道哪些偏好已经存在，避免重复提取）
        3. 已有未完成事项表格（让LLM判断哪些已经在新对话中解决了）
        4. 新对话记录（本次需要整理的原始对话）
        """
        ctx = input_data.context or {}
        conversations = ctx.get("conversations", [])
        old_preferences = ctx.get("old_preferences", [])
        old_unresolved = ctx.get("old_unresolved", [])
        # 从Java端传来的上一轮整合产出的摘要，用于生成渐进式摘要
        previous_summary = ctx.get("previous_summary")

        parts = []

        # 如果有上一轮摘要，放在最前面作为对话背景
        # 这样LLM生成新摘要时会在旧摘要基础上更新，而非从零开始
        if previous_summary:
            parts.append("## 之前的对话背景（上一轮整合产出的摘要）\n")
            parts.append(previous_summary)
            parts.append("")
            parts.append("请在此基础上生成新的渐进式摘要，更新而非替换。\n")

        if old_preferences:
            parts.append("## 已有偏好（需与对话中的新偏好合并）\n")
            parts.append("| 偏好内容 | 分类 | 级别 |")
            parts.append("|----------|------|------|")
            for p in old_preferences:
                level = "用户级" if p.get('preferenceCategory') == 0 else "会话级"
                parts.append(f"| {p.get('content', '')} | {p.get('category', '其他')} | {level} |")
            parts.append("")

        if old_unresolved:
            # 带上id列，让LLM能通过id精确标记哪些事项已解决
            # 避免用content文本匹配导致的不精确问题
            parts.append("## 已有未完成事项（需根据新对话判断是否已解决，用id标记）\n")
            parts.append("| id | 事项内容 | 类型 | 状态 |")
            parts.append("|----|----------|------|------|")
            for u in old_unresolved:
                status_label = "进行中" if u.get('status') == 'active' else "已放弃"
                item_id = u.get('id', '?')
                parts.append(f"| {item_id} | {u.get('content', '')} | {u.get('type', '待办')} | {status_label} |")
            parts.append("")

        parts.append(self._format_conversations(conversations))

        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_content}
        ]

    def _extract_json(self, text: str) -> MemorySummary:
        """
        从 LLM 返回内容中提取 JSON（兼容 markdown 代码块包裹），Pydantic 校验

        防御措施：
        1. 去除 markdown 代码块包裹
        2. json.loads 后校验是否为 dict（JSON 规范允许纯数字/字符串，但我们只要对象）
        3. 如果 LLM 返回了嵌套结构（如把结果包在某个 key 下），尝试自动提取
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)

        # 防御：json.loads 对纯数字、字符串、数组都能成功解析，
        # 但 MemorySummary(**data) 需要 dict，这里显式检查
        if not isinstance(data, dict):
            raise ValueError(
                f"LLM返回的JSON类型为 {type(data).__name__}，期望 dict 对象。"
                f"原始内容前100字符: {text[:100]}"
            )

        return MemorySummary(**data)

    async def _store_facts_to_vector(self, facts: list[dict], session_id: str):
        """
        将提取的事实写入 Redis 向量库（带去重保护）

        写入前先做向量相似度检查：如果已存在高度相似（score < 0.15）的事实，
        则跳过写入，避免重复存储。这是整合时tool检索之外的第二道防线。

        Args:
            facts: LLM 输出的 new_facts 列表 [{"content": "", "keywords": "", "source_seq_range": ""}]
            session_id: 会话ID，用于 doc_id 前缀

        Returns:
            生成的 doc_id 列表，与 facts 一一对应（跳过或失败的返回空字符串占位）
        """
        if not facts:
            return []

        from services.vector_service import build_redis_filter, get_vector_service
        from embeddings.text_embedding import get_text_embedding

        vector_service = get_vector_service()
        embedding_service = get_text_embedding()
        fact_filter = build_redis_filter(record_type="fact", status="active")
        batch_ts = str(int(time.time() * 1000))
        generated_ids = []

        for i, fact in enumerate(facts):
            content = fact.get("content", "")
            keywords = fact.get("keywords", "")
            search_text = f"{keywords} {content}" if keywords else content

            try:
                vector = await embedding_service.embed(search_text)
            except Exception:
                generated_ids.append("")
                continue

            # ===== 去重保护：写入前检查是否已存在高度相似的事实 =====
            try:
                existing = vector_service.search(vector, top_k=1, filter=fact_filter)
                if existing:
                    top_match = existing[0]
                    top_meta = top_match.get("metadata", {})
                    # COSINE距离 < 0.15 表示几乎相同的事实已存在
                    if top_meta.get("type") == "fact" and top_match.get("score", 1) < 0.15:
                        logger.info(
                            f"[memory] 去重跳过: '{content[:40]}' 与已有事实相似度过高 "
                            f"(score={top_match.get('score', 0):.3f})"
                        )
                        # 返回已存在的doc_id，这样Java端也能正确对应
                        generated_ids.append(top_match.get("doc_id", ""))
                        continue
            except Exception:
                pass  # 去重检查失败不影响正常写入

            doc_id = f"fact:{session_id}:{batch_ts}_{i}"

            vector_service.add_vector(
                doc_id=doc_id,
                text=content,
                vector=vector,
                metadata={
                    "record_type": "fact",
                    "type": "fact",
                    "status": "active",
                    "session_id": session_id,
                    "keywords": keywords,
                    "source_seq_range": fact.get("source_seq_range", ""),
                    "importance": fact.get("importance", 5),
                    "confidence": fact.get("confidence", 0.80),
                    "device_type": fact.get("device_type", ""),
                    "equipment_id": fact.get("equipment_id", ""),
                    "site_id": fact.get("site_id", ""),
                    "task_id": fact.get("task_id", ""),
                    "created_at": batch_ts  # 记录创建时间，供未来衰减/清理使用
                }
            )
            generated_ids.append(doc_id)

        return generated_ids

    async def _call_llm_with_tools(self, messages, tools, tool_handlers, response_format):
        """
        封装 LLM 调用，供 run() 使用，方便重试时复用

        Returns:
            str: LLM 返回的文本内容
        """
        response = await self.llm_service.chat_with_tools(
            messages=messages,
            tools=tools,
            tool_handlers=tool_handlers,
            response_format=response_format
        )
        return response.get("content") or ""

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        执行记忆整理（function calling 模式），带自动重试

        流程：构建消息 → 注册工具 → chat_with_tools → 解析 JSON
        如果 LLM 返回垃圾数据（非 JSON 对象），自动重试1次。
        Qwen 模型偶尔会返回纯数字或乱码，重试通常能恢复。
        """
        start_time = time.time()
        max_retries = 1  # 最多重试1次（共2次调用）

        messages = self._build_messages(input_data)

        from tools.fact_retrieval_tool import get_fact_retrieval_tool
        fact_tool = get_fact_retrieval_tool()
        tools = [fact_tool.to_openai_schema()]
        async def fact_handler(**kwargs):
            result = await fact_tool.run(**kwargs)
            if result.success:
                return result.data if result.data is not None else {"result": "success"}
            return {"error": result.error.message if result.error else "unknown error"}

        tool_handlers = {"search_similar_facts": fact_handler}

        # ========== LLM 调用 + 重试 ==========
        content = ""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                content = await self._call_llm_with_tools(
                    messages, tools, tool_handlers,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                # LLM 调用本身失败（网络/超时等）
                last_error = e
                logger.warning(f"[memory] LLM调用失败 attempt={attempt+1}: {e}")
                if attempt < max_retries:
                    continue
                latency_ms = int((time.time() - start_time) * 1000)
                return AgentOutput(
                    agent_name=self.name,
                    message="记忆整理失败，请稍后重试",
                    intention=None,
                    tools_used=[],
                    metadata={
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_detail": str(e),
                        "latency_ms": latency_ms
                    },
                    latency_ms=latency_ms
                )

            # 尝试解析 JSON
            try:
                summary = self._extract_json(content)
                last_error = None
                break  # 解析成功，跳出重试循环
            except (json.JSONDecodeError, ValueError, ValidationError, AttributeError, TypeError) as e:
                last_error = e
                logger.warning(
                    f"[memory] JSON解析失败 attempt={attempt+1}/{max_retries+1}: {e}, "
                    f"raw content: {content[:100]}"
                )
                if attempt < max_retries:
                    # 重试前重新构建 messages（避免上一轮 tool_call 残留）
                    messages = self._build_messages(input_data)
                    continue

        latency_ms = int((time.time() - start_time) * 1000)

        # 所有重试用完仍失败
        if last_error is not None:
            return AgentOutput(
                agent_name=self.name,
                message="记忆整理失败：LLM返回格式异常，已重试仍失败",
                intention=None,
                tools_used=[],
                metadata={
                    "status": "error",
                    "error_type": "JsonParseError",
                    "error_detail": f"LLM返回内容无法解析为记忆摘要: {str(last_error)[:200]}",
                    "raw_content": content[:200] if content else "",
                    "latency_ms": latency_ms,
                    "attempts": max_retries + 1
                },
                latency_ms=latency_ms
            )

        # ========== 解析成功，存入向量库 ==========
        fact_ids = []
        if summary.new_facts:
            try:
                fact_ids = await self._store_facts_to_vector(
                    [f.model_dump() for f in summary.new_facts],
                    input_data.session_id
                )
            except Exception:
                logger.exception("Failed to store facts to Redis vector DB")

        # 将向量库生成的 doc_id 附加到 summary 输出中
        # Java 端用这些 ID 作为 MySQL 的 factId，确保两端 ID 一致
        summary_dict = summary.model_dump()
        summary_dict["fact_ids"] = fact_ids

        return AgentOutput(
            agent_name=self.name,
            message=summary.brief_summary,
            intention=None,
            tools_used=["search_similar_facts"] if summary.new_facts else [],
            metadata={
                "summary": summary_dict,
                "latency_ms": latency_ms
            },
            latency_ms=latency_ms
        )


# 单例
_memory_agent = None


def get_memory_agent() -> MemoryAgent:
    global _memory_agent
    if _memory_agent is None:
        from services.llm_service import get_llm_service
        _memory_agent = MemoryAgent(get_llm_service())
    return _memory_agent
