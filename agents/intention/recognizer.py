"""
意图识别核心模块

采用 LLM 轻量级识别 + 关键词兜底的混合方案：
1. 优先使用轻量级 LLM（qwen-turbo）进行意图识别
2. LLM 调用失败时降级到关键词匹配
3. 返回包含置信度和推理过程的详细信息
"""

import json
import httpx
from typing import Optional
from config.settings import get_settings
from schemas.models import IntentionType, IntentionResult

from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .fallback import fallback_recognize


class IntentionRecognizer:
    """
    意图识别器

    使用轻量级 LLM 进行意图识别，支持降级策略。
    """

    # 意图识别专用模型（轻量级）
    INTENTION_MODEL = "qwen-turbo"
    # 复杂推理时的备用模型
    COMPLEX_MODEL = "qwen-plus"

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.dashscope_api_key
        self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

    async def recognize(self, message: str, use_complex_model: bool = False) -> IntentionResult:
        """
        执行意图识别

        Args:
            message: 用户输入的消息
            use_complex_model: 是否使用复杂推理模型（qwen-plus）
                              默认 False 使用轻量模型（qwen-turbo）

        Returns:
            IntentionResult: 意图识别结果
        """
        model = self.COMPLEX_MODEL if use_complex_model else self.INTENTION_MODEL

        try:
            return await self._llm_recognize(message, model)
        except Exception as e:
            # LLM 失败时降级到关键词兜底
            print(f"[WARNING] LLM识别失败，使用兜底方案: {e}")
            return fallback_recognize(message)

    async def _llm_recognize(self, message: str, model: str) -> IntentionResult:
        """
        LLM 意图识别

        Args:
            message: 用户输入的消息
            model: 使用的模型名称

        Returns:
            IntentionResult: 意图识别结果
        """
        # 构建消息
        user_content = USER_TEMPLATE.format(message=message)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # 调用 LLM
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,  # 意图识别需要稳定输出，低温度
            "top_p": 0.9,
            "max_tokens": 500
        }

        response = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=params
        )
        response.raise_for_status()
        result = response.json()

        # 解析响应
        content = result["choices"][0]["message"]["content"]

        # 尝试解析 JSON
        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> IntentionResult:
        """
        解析 LLM 返回的 JSON 响应

        Args:
            content: LLM 返回的原始内容

        Returns:
            IntentionResult: 意图识别结果
        """
        # 尝试提取 JSON（处理可能存在的 markdown 代码块）
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        try:
            data = json.loads(json_str)

            # 验证并转换 intention 字段
            intention_str = data.get("intention", "")
            intention = self._str_to_intention(intention_str)

            return IntentionResult(
                intention=intention,
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", "LLM识别")
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # JSON 解析失败，抛出异常触发降级
            raise ValueError(f"JSON解析失败: {e}, content: {content}")

    def _str_to_intention(self, intention_str: str) -> IntentionType:
        """
        将字符串转换为 IntentionType

        Args:
            intention_str: 意图字符串

        Returns:
            IntentionType: 意图类型枚举
        """
        # 字符串到枚举的映射
        mapping = {
            "query_knowledge": IntentionType.QUERY_KNOWLEDGE,
            "troubleshoot": IntentionType.TROUBLESHOOT,
            "seek_guidance": IntentionType.SEEK_GUIDANCE,
            "submit_case": IntentionType.SUBMIT_CASE,
            "general_chat": IntentionType.GENERAL_CHAT,
        }

        intention_str = intention_str.strip().lower()

        if intention_str in mapping:
            return mapping[intention_str]

        # 无法识别，默认闲聊
        return IntentionType.GENERAL_CHAT


# 单例模式
_recognizer: Optional[IntentionRecognizer] = None


def get_intention_recognizer() -> IntentionRecognizer:
    """获取意图识别器单例"""
    global _recognizer
    if _recognizer is None:
        _recognizer = IntentionRecognizer()
    return _recognizer
