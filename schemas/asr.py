"""
ASR 语音识别响应模型

定义转写结果的 Pydantic 结构，供 FastAPI 自动序列化为 JSON。
"""

from typing import List
from pydantic import BaseModel, Field
from schemas.models import BaseResponse


class ASRSegment(BaseModel):
    """
    转写片段模型

    对应 faster-whisper 输出的单个 segment，
    包含起止时间和该段的识别文本。
    """
    start: float = Field(description="片段起始时间（秒）")
    end: float = Field(description="片段结束时间（秒）")
    text: str = Field(description="该片段识别文本")

    class Config:
        json_schema_extra = {
            "example": {
                "start": 0.0,
                "end": 2.5,
                "text": "电动机轴承过热"
            }
        }


class ASRResponse(BaseResponse):
    """
    语音转写完整响应

    继承 BaseResponse（success, message, code），
    附加识别的语言、时长、完整文本和时间轴片段列表。
    """
    language: str = Field(description="识别到的语言代码，如 'zh'")
    language_probability: float = Field(description="语言检测置信度")
    duration: float = Field(description="音频总时长（秒）")
    text: str = Field(description="完整转写文本")
    segments: List[ASRSegment] = Field(description="时间轴对齐的转写片段列表")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "转写完成",
                "code": 200,
                "language": "zh",
                "language_probability": 0.98,
                "duration": 12.5,
                "text": "电动机轴承过热可能是润滑不良引起的",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "电动机轴承过热"},
                    {"start": 2.5, "end": 5.0, "text": "可能是润滑不良引起的"}
                ]
            }
        }
