"""MemoryAgent 结构化输出模型"""

from pydantic import BaseModel, Field


class FactItem(BaseModel):
    """事实条目"""
    content: str = Field(description="事实描述")
    keywords: str = Field(default="", description="检索用关键词")
    source_seq_range: str = Field(default="", description="来源对话序号范围")


class PreferenceItem(BaseModel):
    """偏好条目"""
    content: str = Field(description="偏好描述")
    category: str = Field(default="其他", description="分类：交互风格|格式要求|工作习惯|关注领域|其他")
    preferenceCategory: int = Field(default=0, description="0=用户级, 1=会话级")


class UnresolvedItem(BaseModel):
    """未完成事项条目"""
    content: str = Field(description="待解决描述")
    type: str = Field(default="待办", description="类型：未答复问题|进行中任务|用户待办")
    status: str = Field(default="active", description="active=进行中, superseded=已放弃")


class MemorySummary(BaseModel):
    """MemoryAgent 输出摘要"""
    new_facts: list[FactItem] = Field(default_factory=list)
    superseded_ids: list[str] = Field(default_factory=list)
    updated_preferences: list[PreferenceItem] = Field(default_factory=list)
    updated_unresolved: list[UnresolvedItem] = Field(default_factory=list)
    resolved_items: list[str] = Field(default_factory=list)
    brief_summary: str = Field(default="", description="200字以内的整体摘要")
