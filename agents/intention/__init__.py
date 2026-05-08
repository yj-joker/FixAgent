"""
Intention 模块

意图识别模块，采用 LLM 轻量级识别 + 关键词兜底的混合方案。

文件说明：
- recognizer.py - 意图识别器核心实现
- prompts.py - 识别提示词模板
- fallback.py - 关键词兜底策略

使用示例：
    from agents.intention import get_intention_recognizer

    recognizer = get_intention_recognizer()
    result = await recognizer.recognize("轴承过热是什么原因？")

    print(f"意图: {result.intention}")
    print(f"置信度: {result.confidence}")
    print(f"理由: {result.reasoning}")
"""

from .recognizer import IntentionRecognizer, get_intention_recognizer
from .fallback import fallback_recognize
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE

__all__ = [
    "IntentionRecognizer",
    "get_intention_recognizer",
    "fallback_recognize",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
]
