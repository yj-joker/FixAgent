"""Image semantic enrichment hooks for knowledge import."""

from __future__ import annotations

import json
from typing import Optional

from config.settings import get_settings
from services.llm_service import get_llm_service


class ImageSummaryService:
    """Generate retrieval-friendly text for an extracted image."""

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
