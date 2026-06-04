"""Image semantic enrichment hooks for knowledge import."""

from __future__ import annotations

import json
from typing import Optional

from config.settings import get_settings
from services.llm_service import get_llm_service


class ImageSummaryService:
    """Generate retrieval-friendly text for an extracted image."""

    async def understand_user_image(self, image_url: str, user_message: str = "") -> dict:
        """Generate retrieval-friendly understanding for a user-uploaded chat image."""
        prompt = (
            "请识别用户上传的维修/设备图片，并返回 JSON。"
            "字段仅包含 image_title、image_summary、keywords。"
            "image_title 用一句话说明图中主体；"
            "image_summary 说明可见部件、标注、可能所属系统；"
            "keywords 是用于知识库检索的中文关键词数组。"
            f"\n用户文字：{user_message or '用户未输入文字，仅上传图片'}"
        )
        try:
            response = await get_llm_service().chat(
                [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                temperature=0.1,
                max_tokens=500,
                model=get_settings().vlm_model,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.get("content") or "{}")
            title = str(payload.get("image_title") or "").strip()
            summary = str(payload.get("image_summary") or "").strip()
            raw_keywords = payload.get("keywords") or []
            if isinstance(raw_keywords, str):
                keywords = [item.strip() for item in raw_keywords.replace("，", ",").split(",") if item.strip()]
            else:
                keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
            if title or summary or keywords:
                return {
                    "image_title": title or "用户上传图片",
                    "image_summary": summary or title,
                    "keywords": keywords,
                    "summary_source": "user_image_vlm",
                }
        except Exception:
            return {}
        return {}

    async def summarize(
        self,
        image_url: str,
        caption: str = "",
        context_before: str = "",
        context_after: str = "",
        section_title: str = "",
    ) -> dict:
        if image_url and get_settings().image_summary_llm_enabled:
            summary = await self._summarize_with_llm(
                image_url=image_url,
                caption=caption,
                context_before=context_before,
                context_after=context_after,
                section_title=section_title,
            )
            if summary:
                return summary
        return self._fallback_summary(caption, context_before, context_after, section_title)

    async def _summarize_with_llm(
        self,
        image_url: str,
        caption: str,
        context_before: str,
        context_after: str,
        section_title: str,
    ) -> dict:
        prompt = (
            "Summarize this maintenance-manual image for retrieval. "
            "Return JSON with image_title and image_summary only. "
            f"Section: {section_title}\nCaption: {caption}\n"
            f"Context before: {context_before[:500]}\nContext after: {context_after[:500]}"
        )
        try:
            response = await get_llm_service().chat(
                [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.get("content") or "{}")
            title = str(payload.get("image_title") or "").strip()
            summary = str(payload.get("image_summary") or "").strip()
            if title and summary:
                return {
                    "image_title": title,
                    "image_summary": summary,
                    "summary_source": "multimodal_llm",
                }
        except Exception:
            return {}
        return {}

    @staticmethod
    def _fallback_summary(
        caption: str,
        context_before: str,
        context_after: str,
        section_title: str,
    ) -> dict:
        caption = caption.strip()
        context = " ".join(part.strip() for part in (context_before, context_after) if part.strip())
        title = caption or section_title or "文档插图"
        summary = f"{title}。相关上下文：{context[:500]}" if context else title
        return {"image_title": title, "image_summary": summary, "summary_source": "fallback_context"}


_image_summary_service: Optional[ImageSummaryService] = None


def get_image_summary_service() -> ImageSummaryService:
    global _image_summary_service
    if _image_summary_service is None:
        _image_summary_service = ImageSummaryService()
    return _image_summary_service
