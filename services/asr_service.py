"""
ASR 语音识别服务模块

封装 faster-whisper 模型，提供同步转写和流式转写能力。
模型采用懒加载策略，首次调用时才下载和初始化。
"""

import logging
import asyncio
from typing import Optional, Dict, Any

from opencc import OpenCC

from config.asr_settings import get_asr_settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg"}


class AsrService:
    """
    faster-whisper 语音识别服务

    模型懒加载，首次转写请求时初始化 WhisperModel。
    同步的模型推理在独立线程中执行，避免阻塞 FastAPI 事件循环。
    """

    def __init__(self):
        self.settings = get_asr_settings()
        self._model = None
        self._opencc = None

    def _get_model(self):
        """懒加载 WhisperModel，首次调用时下载并初始化模型"""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(
                f"[asr] loading model={self.settings.model_name} "
                f"device={self.settings.device} compute_type={self.settings.compute_type}"
            )
            self._model = WhisperModel(
                self.settings.model_name,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
        return self._model

    def _get_opencc(self):
        """懒加载 OpenCC 繁→简转换器"""
        if self._opencc is None:
            self._opencc = OpenCC("t2s")
        return self._opencc

    def _to_simplified(self, text: str) -> str:
        """将文本转为简体中文"""
        return self._get_opencc().convert(text)

    def _build_transcribe_kwargs(self) -> Dict[str, Any]:
        """构造 model.transcribe() 的参数"""
        kwargs = {
            "language": self.settings.language,
            "task": "transcribe",
            "beam_size": self.settings.beam_size,
            "best_of": self.settings.best_of,
            "temperature": self.settings.temperatures,
            "condition_on_previous_text": self.settings.condition_on_previous_text,
            "initial_prompt": self.settings.initial_prompt,
            "vad_filter": self.settings.vad_filter,
        }
        if self.settings.vad_filter:
            kwargs["vad_parameters"] = {
                "min_silence_duration_ms": self.settings.vad_min_silence_ms,
            }
        return kwargs

    def _transcribe_sync(self, audio_path: str) -> Dict[str, Any]:
        """同步转写，在线程池中执行"""
        model = self._get_model()
        kwargs = self._build_transcribe_kwargs()
        logger.info(f"[asr] transcribe params: {kwargs}")

        segments, info = model.transcribe(audio_path, **kwargs)

        seg_list = []
        full_text_parts = []
        for seg in segments:
            text_simplified = self._to_simplified(seg.text.strip())
            seg_list.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text_simplified,
            })
            full_text_parts.append(text_simplified)

        full_text = self._to_simplified("".join(full_text_parts))

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "text": full_text,
            "segments": seg_list,
        }

    def _transcribe_stream_sync(self, audio_path: str, loop, queue: asyncio.Queue):
        """
        同步流式转写 worker。
        在线程中运行，通过 call_soon_threadsafe 将 segment 推入 asyncio.Queue。
        """
        try:
            model = self._get_model()
            kwargs = self._build_transcribe_kwargs()
            logger.info(f"[asr] transcribe-stream params: {kwargs}")

            segments, _info = model.transcribe(audio_path, **kwargs)

            for seg in segments:
                data = {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": self._to_simplified(seg.text.strip()),
                }
                loop.call_soon_threadsafe(queue.put_nowait, data)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """非流式转写，在线程池中执行"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio_path)

    async def transcribe_stream(self, audio_path: str):
        """
        流式转写 async generator。

        每识别出一个 segment 就 yield 一个 dict，
        供 SSE StreamingResponse 逐段推送给客户端。
        """
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        task = asyncio.create_task(
            asyncio.to_thread(self._transcribe_stream_sync, audio_path, loop, queue)
        )

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not task.done():
                task.cancel()
            await task


_asr_service: Optional[AsrService] = None


def get_asr_service() -> AsrService:
    """获取 ASR 服务单例"""
    global _asr_service
    if _asr_service is None:
        _asr_service = AsrService()
    return _asr_service
