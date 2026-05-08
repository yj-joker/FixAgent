"""
关键词兜底策略

当 LLM 不可用时，使用关键词匹配作为降级方案。
关键词覆盖设备维修领域的常见表达。
"""

from schemas.models import IntentionType, IntentionResult


# 意图关键词映射（设备维修领域扩展）
KEYWORD_MAP = {
    IntentionType.TROUBLESHOOT: [
        # 故障现象
        "坏", "坏了", "故障", "问题", "毛病",
        # 设备不工作
        "不转", "不动", "停止", "停转", "卡死", "堵转",
        # 温度异常
        "过热", "发烫", "温度高", "发热",
        # 声音异常
        "异响", "噪音", "响", "振动", "晃动",
        # 外观异常
        "冒烟", "火花", "漏油", "漏水", "漏气",
        # 性能下降
        "无力", "慢", "弱", "效率低",
        # 询问原因
        "原因", "为什么", "怎么回事", "啥问题", "什么情况",
        # 故障诊断类
        "诊断", "分析", "检查",
    ],
    IntentionType.SEEK_GUIDANCE: [
        # 维修询问
        "怎么修", "如何修", "维修", "检修", "修理",
        # 操作步骤
        "步骤", "流程", "顺序", "程序",
        # 指导请求
        "指引", "指导", "教程", "方法", "办法",
        # 操作类
        "操作", "使用", "调节", "调整", "设置",
        # 拆装类
        "拆卸", "安装", "拆装", "更换", "替换",
        # 保养类
        "保养", "维护", "维护", "润滑", "加油",
    ],
    IntentionType.QUERY_KNOWLEDGE: [
        # 概念询问
        "什么是", "什么叫", "含义", "定义",
        # 知识查询
        "知识", "介绍", "说明", "解释",
        # 原理类
        "原理", "机制", "结构", "组成",
        # 规格类
        "规格", "参数", "型号", "标准",
        # 学习类
        "了解", "学习", "知道", "清楚",
    ],
    IntentionType.SUBMIT_CASE: [
        # 提交案例
        "提交案例", "提交一个案例", "上传案例",
        # 分享经验
        "分享", "经验", "记录", "日志",
        # 上传类
        "上传", "提交记录", "案例分享",
    ],
}


def fallback_recognize(message: str) -> IntentionResult:
    """
    关键词兜底识别

    当 LLM 不可用时，通过扫描消息中的关键词来判断意图。

    Args:
        message: 用户输入的消息

    Returns:
        IntentionResult: 意图识别结果
    """
    message_lower = message.lower()

    # 遍历关键词映射表
    for intention, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in message_lower:
                return IntentionResult(
                    intention=intention,
                    confidence=0.6,  # 兜底方案置信度设为 0.6
                    reasoning=f"关键词命中: '{keyword}'"
                )

    # 无关键词匹配，默认闲聊
    return IntentionResult(
        intention=IntentionType.GENERAL_CHAT,
        confidence=0.5,
        reasoning="无关键词匹配，默认闲聊"
    )


def add_keyword(intention: IntentionType, keywords: list) -> None:
    """
    动态添加关键词（供外部调用扩展）

    Args:
        intention: 意图类型
        keywords: 关键词列表
    """
    if intention in KEYWORD_MAP:
        KEYWORD_MAP[intention].extend(keywords)
    else:
        KEYWORD_MAP[intention] = keywords
