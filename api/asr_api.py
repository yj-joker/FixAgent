"""
ASR 语音识别 API 路由

提供健康检查、音频转写（非流式）和 SSE 流式转写三个接口。
"""

import json
import logging
import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse

from config.asr_settings import get_asr_settings
from services.asr_service import get_asr_service, ALLOWED_EXTENSIONS
from schemas.asr import ASRResponse

logger = logging.getLogger(__name__)

asr_settings = get_asr_settings()
asr_router = APIRouter(prefix="/api/asr", tags=["ASR"])


def _validate_audio(upload_file: UploadFile, content_length: str | None) -> str:
    """
    验证上传音频文件，写入临时文件，返回临时文件路径。

    校验：
    1. 文件扩展名是否在支持列表中
    2. Content-Length 预检（如有）
    3. 实际文件大小是否超过限制
    """
    ext = os.path.splitext(upload_file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext}。支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    max_bytes = asr_settings.max_upload_mb * 1024 * 1024

    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超过限制 ({asr_settings.max_upload_mb}MB)",
                )
        except ValueError:
            pass

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    try:
        content = upload_file.file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制 ({asr_settings.max_upload_mb}MB)",
            )
        tmp.write(content)
        tmp.flush()
    except HTTPException:
        os.unlink(tmp_path)
        raise
    finally:
        tmp.close()

    return tmp_path


@asr_router.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model": asr_settings.model_name}


@asr_router.post("/transcribe", response_model=ASRResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    """
    音频转写（非流式）。

    上传音频文件，返回完整转写结果 JSON，
    包含识别语言、时长、完整文本和时间轴片段。
    """
    tmp_path = None
    try:
        content_length = request.headers.get("content-length")
        tmp_path = _validate_audio(file, content_length)
        logger.info(f"[asr] transcribe file={file.filename} size={os.path.getsize(tmp_path)}")

        service = get_asr_service()
        result = await service.transcribe(tmp_path)

        logger.info(
            f"[asr] transcribe done language={result['language']} "
            f"duration={result['duration']:.1f}s segments={len(result['segments'])}"
        )
        return ASRResponse(
            success=True,
            message="转写完成",
            code=200,
            language=result["language"],
            language_probability=result["language_probability"],
            duration=result["duration"],
            text=result["text"],
            segments=result["segments"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[asr] transcribe error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@asr_router.post("/transcribe-stream")
async def transcribe_stream(request: Request, file: UploadFile = File(...)):
    """
    音频转写（SSE 流式）。

    上传音频文件，通过 Server-Sent Events 按 segment 分段返回识别内容。
    每段为一个 JSON 行，格式: data: {"start": ..., "end": ..., "text": ...}
    结束事件: data: {"event": "done"}
    """
    tmp_path = None
    try:
        content_length = request.headers.get("content-length")
        tmp_path = _validate_audio(file, content_length)
        logger.info(f"[asr] transcribe-stream file={file.filename} size={os.path.getsize(tmp_path)}")

        service = get_asr_service()

        async def event_generator():
            try:
                async for segment in service.transcribe_stream(tmp_path):
                    yield f"data: {json.dumps(segment, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
            except Exception as e:
                logger.exception("[asr] stream error")
                yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}}, ensure_ascii=False)}\n\n"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[asr] transcribe-stream error")
        raise HTTPException(status_code=500, detail=str(e))
