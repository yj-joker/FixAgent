"""Shared user-visible output style rules.

These rules are source constraints for prompts and deterministic composers.
They are not a final response scrubber.
"""

USER_VISIBLE_PLAIN_TEXT_RULES = (
    "用户可见回答必须使用纯文本中文。"
    "禁止使用 emoji。"
    "禁止使用 Markdown。"
    "禁止使用 #、##、### 作为标题符号。"
    "禁止使用 -、*、+ 作为列表符号；需要分项时使用 1.、2.、3. 这种普通编号。"
    "禁止使用 **加粗**、*斜体*、反引号代码样式、Markdown 表格或 --- 分隔线。"
    "禁止输出内部技术标识和工具参数，例如 source=、doc_id、chunk_id、img:0000、image_url、top_k。"
    "只用自然段、普通换行和中文编号保证文本结构清晰。"
)

