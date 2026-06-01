"""
事实多因子重排序服务

对 Redis KNN 粗筛的候选事实做精排，综合考虑：
- 语义相关度（向量距离，权重 0.40）
- 新近性衰减（created_at 越新越好，权重 0.20）
- 重要性（importance 1-10，权重 0.15）
- 使用频率（usage_count，权重 0.10）
- 置信度（confidence，权重 0.15）

公式：
finalScore = semantic * 0.40 + recency * 0.20 + importance * 0.15 + frequency * 0.10 + confidence * 0.15
"""

import logging
import math
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# 权重配置
WEIGHT_SEMANTIC = 0.40
WEIGHT_RECENCY = 0.20
WEIGHT_IMPORTANCE = 0.15
WEIGHT_FREQUENCY = 0.10
WEIGHT_CONFIDENCE = 0.15

# 新近性衰减半衰期（秒）：7天，即7天前的事实新近性分数为0.5
RECENCY_HALF_LIFE = 7 * 24 * 3600


def _semantic_score(raw_score: float) -> float:
    """
    将 COSINE 距离转为 0-1 相关度分数。
    COSINE 距离范围 [0, 2]，0 = 完全相同。
    """
    return max(0.0, 1.0 - raw_score / 2.0)


def _recency_score(created_at_str: str) -> float:
    """
    指数衰减：越新越接近 1.0，半衰期后为 0.5。
    """
    if not created_at_str:
        return 0.5  # 没有时间戳的默认中等
    try:
        created_ts = int(created_at_str)
        # created_at 是毫秒时间戳，转为秒
        if created_ts > 1e12:
            created_ts = created_ts / 1000.0
        age_seconds = time.time() - created_ts
        if age_seconds < 0:
            return 1.0
        decay = math.exp(-0.693 * age_seconds / RECENCY_HALF_LIFE)
        return decay
    except (ValueError, TypeError):
        return 0.5


def _importance_score(importance: int) -> float:
    """importance 1-10 线性归一化到 0-1"""
    return max(0.0, min(1.0, (importance - 1) / 9.0))


def _frequency_score(usage_count: int) -> float:
    """对数归一化，避免高频事实垄断。log(1+count)/log(1+50) 上限约1.0"""
    if usage_count <= 0:
        return 0.0
    return min(1.0, math.log(1 + usage_count) / math.log(51))


def _confidence_score(confidence: float) -> float:
    """直接使用，已经是 0-1 范围"""
    return max(0.0, min(1.0, confidence))


def rerank(candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    对候选事实做多因子重排序。

    Args:
        candidates: KNN 返回的候选列表，每条包含 score/metadata/text/doc_id
        top_k: 最终返回数量

    Returns:
        重排序后的 top_k 结果，每条增加 final_score 和 score_breakdown 字段
    """
    scored = []
    for c in candidates:
        metadata = c.get("metadata", {})

        semantic = _semantic_score(c.get("score", 1.0))
        recency = _recency_score(str(metadata.get("created_at", "")))
        importance = _importance_score(int(metadata.get("importance", 5)))
        frequency = _frequency_score(int(metadata.get("usage_count", 0)))
        confidence = _confidence_score(float(metadata.get("confidence", 0.80)))

        final_score = (
            semantic * WEIGHT_SEMANTIC
            + recency * WEIGHT_RECENCY
            + importance * WEIGHT_IMPORTANCE
            + frequency * WEIGHT_FREQUENCY
            + confidence * WEIGHT_CONFIDENCE
        )

        scored.append({
            **c,
            "final_score": round(final_score, 4),
            "score_breakdown": {
                "semantic": round(semantic, 4),
                "recency": round(recency, 4),
                "importance": round(importance, 4),
                "frequency": round(frequency, 4),
                "confidence": round(confidence, 4),
            }
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]
