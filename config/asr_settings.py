"""
ASR 语音识别配置模块

通过环境变量配置 faster-whisper 模型参数，所有配置项均有默认值。
使用方式：
    from config.asr_settings import get_asr_settings
    settings = get_asr_settings()
"""

import os
from dotenv import load_dotenv

load_dotenv()


class AsrSettings:
    """ASR 配置类，所有值从环境变量读取"""

    model_name = os.getenv("ASR_MODEL_NAME", "medium")
    device = os.getenv("ASR_DEVICE", "cpu")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "int8")
    language = os.getenv("ASR_LANGUAGE", "zh")
    beam_size = int(os.getenv("ASR_BEAM_SIZE", "8"))
    best_of = int(os.getenv("ASR_BEST_OF", "8"))
    temperatures = [
        float(t) for t in os.getenv("ASR_TEMPERATURES", "0.0,0.2,0.4").split(",")
    ]
    condition_on_previous_text = os.getenv(
        "ASR_CONDITION_ON_PREVIOUS_TEXT", "true"
    ).lower() in ("true", "1", "yes")
    initial_prompt = os.getenv(
        "ASR_INITIAL_PROMPT",
        "以下是普通话音频的简体中文转写，请使用简体中文、中文标点，保留人名、地名、产品名和技术词汇。",
    )
    vad_filter = os.getenv("ASR_VAD_FILTER", "false").lower() in ("true", "1", "yes")
    vad_min_silence_ms = int(os.getenv("ASR_VAD_MIN_SILENCE_MS", "1000"))
    max_upload_mb = int(os.getenv("ASR_MAX_UPLOAD_MB", "50"))


_asr_settings = None


def get_asr_settings() -> AsrSettings:
    """获取 ASR 配置单例"""
    global _asr_settings
    if _asr_settings is None:
        _asr_settings = AsrSettings()
    return _asr_settings
