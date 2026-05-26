"""
知识入库服务

编排 文档解析 → 向量化 → Redis 向量库 的完整流程。
只做编排，不自己解析、不自己向量化、不自己写 Redis。

【执行流程】
1. DocumentParserTool 解析 PDF → sections
2. text_chunks → TextEmbedding.embed_batch() → VectorService.add_vector_batch()
3. tables → 转 markdown 文本 → TextEmbedding → VectorService
4. images → 优先用本地拆图路径做 ImageEmbedding，URL 仅用于持久化回显和兜底
5. 返回导入统计
"""

import time
import hashlib
import logging
from typing import List, Optional

from tools.document_tool import get_document_parser
from embeddings.text_embedding import get_text_embedding
from embeddings.image_embedding import get_image_embedding
from services.file_storage import get_file_storage
from services.image_summary_service import get_image_summary_service
from services.vector_service import get_vector_service

logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识入库服务"""

    # embed_batch 单批最大条数（百炼 API 限制）
    _BATCH_SIZE = 20

    def __init__(self):
        self.parser = get_document_parser()
        self.text_emb = get_text_embedding()
        self.image_emb = get_image_embedding()
        self.file_storage = get_file_storage()
        self.image_summary_svc = get_image_summary_service()
        self.vector_svc = get_vector_service()

    async def import_document(
        self,
        file_url: str,
        file_type: str = "pdf",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        device_type: Optional[str] = None,
        manual_type: Optional[str] = None,
        document_version: Optional[str] = None,
        replace_existing: bool = False,
        old_document_id: Optional[str] = None
    ) -> dict:
        try:
            return await self._import_document_impl(
                file_url=file_url,
                file_type=file_type,
                category=category,
                tags=tags,
                document_id=document_id,
                device_type=device_type,
                manual_type=manual_type,
                document_version=document_version,
                replace_existing=replace_existing,
                old_document_id=old_document_id,
            )
        except Exception as exc:
            if document_id:
                current = self.vector_svc.get_document_manifest(document_id) or {}
                if current.get("status") != "failed":
                    self.vector_svc.put_document_manifest(document_id, {
                        **current,
                        "document_id": document_id,
                        "status": "failed",
                        "error_message": str(exc),
                    })
            raise

    async def _import_document_impl(
        self,
        file_url: str,
        file_type: str = "pdf",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        device_type: Optional[str] = None,
        manual_type: Optional[str] = None,
        document_version: Optional[str] = None,
        replace_existing: bool = False,
        old_document_id: Optional[str] = None
    ) -> dict:
        """
        导入文档：解析 → 向量化 → 入库

        Returns:
            {
                "file_name": str,
                "total_pages": int,
                "text_count": int,       # 入库文本块数
                "image_count": int,      # 入库图片数
                "table_count": int,      # 入库表格数
                "sections": [...],       # 各章节统计摘要
                "extraction_summary": {...},
                "process_time_ms": int
            }
        """
        t0 = time.time()
        if document_id:
            self.vector_svc.put_document_manifest(document_id, {
                "document_id": document_id,
                "source_file_url": file_url,
                "device_type": device_type,
                "manual_type": manual_type,
                "document_version": document_version,
                "status": "parsing",
                "category": category,
                "tags": tags or [],
            })

        # 1. 解析文档
        parse_result = await self.parser._execute(file_url, file_type)
        file_name = parse_result["file_name"]
        total_pages = parse_result["total_pages"]
        sections = parse_result["sections"]
        extraction_summary = parse_result["extraction_summary"]
        source_file_url = self.file_storage.ensure_document_url(file_url)

        document_id = document_id or hashlib.md5(f"{file_name}|{file_url}".encode()).hexdigest()[:12]
        doc_prefix = hashlib.md5(document_id.encode()).hexdigest()[:8]
        common_metadata = {
            "file_name": file_name,
            "document_id": document_id,
            "source_file_url": source_file_url,
            "device_type": device_type,
            "manual_type": manual_type,
            "document_version": document_version,
        }
        if replace_existing and old_document_id:
            # 删除旧版本的向量数据，用旧版本的 document_id（而非当前新版本的）
            logger.info("删除旧版本向量: old_document_id=%s", old_document_id)
            self.vector_svc.delete_by_document(old_document_id)
        self.vector_svc.put_document_manifest(document_id, {
            **common_metadata,
            "status": "indexing",
            "category": category,
            "tags": tags or [],
        })

        text_count = 0
        image_count = 0
        table_count = 0
        image_summary_count = 0

        # 2. 逐 section 处理
        for sec_idx, section in enumerate(sections):
            section_title = section.get("section_title", f"第{sec_idx + 1}章")
            page_range = section.get("page_range", "")
            sec_category = category or section_title

            # 2a. 文本块 → 分批 embed_batch → 入库
            raw_chunks = section.get("text_chunks", [])
            valid_chunks = [
                self._normalize_text_chunk(chunk)
                for chunk in raw_chunks
                if len(self._normalize_text_chunk(chunk)["text"].strip()) >= 10
            ]
            for batch_start in range(0, len(valid_chunks), self._BATCH_SIZE):
                batch = valid_chunks[batch_start:batch_start + self._BATCH_SIZE]
                vectors = await self.text_emb.embed_batch([chunk["text"] for chunk in batch])
                docs = []
                for i, (chunk, vec) in enumerate(zip(batch, vectors)):
                    global_i = batch_start + i
                    chunk_id = f"{doc_prefix}:{sec_idx:02d}:txt:{global_i:04d}"
                    docs.append({
                        "doc_id": chunk_id,
                        "text": chunk["text"],
                        "vector": vec,
                        "category": sec_category,
                        "tags": tags,
                        "metadata": {
                            **common_metadata,
                            "section_title": section_title,
                            "page_range": page_range,
                            "chunk_type": "text",
                            "page": chunk.get("page"),
                            "chunk_label": chunk.get("chunk_label", "page"),
                            "context_before": chunk.get("context_before", ""),
                            "context_after": chunk.get("context_after", "")
                        }
                    })
                written = self.vector_svc.add_vector_batch(docs)
                if written != len(docs):
                    self._mark_failed_import(
                        document_id, common_metadata, category, tags,
                        text_count, image_count, table_count, image_summary_count,
                        "failed to write all text vector records",
                    )
                    raise RuntimeError("failed to write all text vector records")
                text_count += len(docs)

            # 2b. 表格 → 转文本 → 入库
            for t_idx, table in enumerate(section.get("tables", [])):
                table_text = self._table_to_text(table)
                if not table_text.strip():
                    continue
                vec = await self.text_emb.embed(table_text)
                table_id = f"{doc_prefix}:{sec_idx:02d}:tbl:{t_idx:04d}"
                table_written = self.vector_svc.add_vector(
                    doc_id=table_id,
                    text=table_text,
                    vector=vec,
                    category=sec_category,
                    tags=tags,
                    metadata={
                        **common_metadata,
                        "section_title": section_title,
                        "page_range": page_range,
                        "chunk_type": "table",
                        "page": table.get("page"),
                        "caption": table.get("caption", "")
                    }
                )
                if not table_written:
                    self._mark_failed_import(
                        document_id, common_metadata, category, tags,
                        text_count, image_count, table_count, image_summary_count,
                        "failed to write table vector record",
                    )
                    raise RuntimeError("failed to write table vector record")
                table_count += 1

            # 2c. 图片 → 图注文本向量化 → 入库
            for img_idx, img in enumerate(section.get("images", [])):
                caption = img.get("caption", "").strip()
                img_name = img.get("image_name", f"img_{img_idx}")
                local_path = img.get("local_path", "")
                image_url = self.file_storage.ensure_public_url(img)

                img_text = caption if caption else f"{section_title} 第{img.get('page', '?')}页插图"
                if local_path:
                    vec = await self.image_emb.embed(local_path)
                    embedding_source = "local_image"
                elif image_url:
                    vec = await self.image_emb.embed(image_url)
                    embedding_source = "image_url"
                else:
                    vec = await self.text_emb.embed(img_text)
                    embedding_source = "caption_text"
                img_id = f"{doc_prefix}:{sec_idx:02d}:img:{img_idx:04d}"
                image_written = self.vector_svc.add_vector(
                    doc_id=img_id,
                    text=img_text,
                    vector=vec,
                    category=sec_category,
                    tags=tags,
                    metadata={
                        **common_metadata,
                        "section_title": section_title,
                        "page_range": page_range,
                        "chunk_type": "image",
                        "page": img.get("page"),
                        "image_name": img_name,
                        "local_path": local_path,
                        "image_url": image_url,
                        "caption": caption,
                        "embedding_source": embedding_source
                    }
                )
                if not image_written:
                    self._mark_failed_import(
                        document_id, common_metadata, category, tags,
                        text_count, image_count, table_count, image_summary_count,
                        "failed to write image vector record",
                    )
                    raise RuntimeError("failed to write image vector record")
                summary = await self.image_summary_svc.summarize(
                    image_url=image_url,
                    caption=caption,
                    context_before=img.get("context_before", ""),
                    context_after=img.get("context_after", ""),
                    section_title=section_title,
                )
                summary_text = (summary.get("image_summary") or "").strip()
                if summary_text:
                    summary_vec = await self.text_emb.embed(summary_text)
                    summary_written = self.vector_svc.add_vector(
                        doc_id=f"{doc_prefix}:{sec_idx:02d}:ims:{img_idx:04d}",
                        text=summary_text,
                        vector=summary_vec,
                        category=sec_category,
                        tags=tags,
                        metadata={
                            **common_metadata,
                            "section_title": section_title,
                            "page_range": page_range,
                            "chunk_type": "image_summary",
                            "page": img.get("page"),
                            "image_name": img_name,
                            "image_url": image_url,
                            "image_title": summary.get("image_title", ""),
                            "image_summary": summary_text,
                            "summary_source": summary.get("summary_source", ""),
                            "retrieval_route": "image_summary",
                            "source_image_id": img_id,
                            "context_before": img.get("context_before", ""),
                            "context_after": img.get("context_after", ""),
                        }
                    )
                    if not summary_written:
                        self._mark_failed_import(
                            document_id, common_metadata, category, tags,
                            text_count, image_count, table_count, image_summary_count,
                            "failed to write image summary vector record",
                        )
                        raise RuntimeError("failed to write image summary vector record")
                    image_summary_count += 1
                image_count += 1

        t1 = time.time()
        self.vector_svc.put_document_manifest(document_id, {
            **common_metadata,
            "status": "ready",
            "category": category,
            "tags": tags or [],
            "total_pages": total_pages,
            "text_count": text_count,
            "image_count": image_count,
            "image_summary_count": image_summary_count,
            "table_count": table_count,
        })

        return {
            "file_name": file_name,
            "document_id": document_id,
            "document_version": document_version,
            "source_file_url": source_file_url,
            "total_pages": total_pages,
            "text_count": text_count,
            "image_count": image_count,
            "image_summary_count": image_summary_count,
            "table_count": table_count,
            "sections": [
                {
                    "section_title": s.get("section_title", ""),
                    "page_range": s.get("page_range", ""),
                    "text_chunks": len(s.get("text_chunks", [])),
                    "images": len(s.get("images", [])),
                    "tables": len(s.get("tables", []))
                }
                for s in sections
            ],
            "extraction_summary": extraction_summary,
            "process_time_ms": int((t1 - t0) * 1000)
        }

    @staticmethod
    def _table_to_text(table: dict) -> str:
        """将表格 dict 转为可向量化的 markdown 文本"""
        rows = table.get("rows", [])
        if not rows:
            return ""

        lines = []
        caption = table.get("caption", "")
        if caption:
            lines.append(f"表格：{caption}")

        for row in rows:
            if row and any(cell for cell in row):
                lines.append(" | ".join(str(cell).strip() for cell in row))

        return "\n".join(lines)

    @staticmethod
    def _normalize_text_chunk(chunk) -> dict:
        if isinstance(chunk, dict):
            return {
                "text": str(chunk.get("text", "")),
                "page": chunk.get("page"),
                "chunk_label": chunk.get("chunk_label", "page"),
                "context_before": chunk.get("context_before", ""),
                "context_after": chunk.get("context_after", ""),
            }
        return {
            "text": str(chunk),
            "page": None,
            "chunk_label": "page",
            "context_before": "",
            "context_after": "",
        }

    def _mark_failed_import(
        self,
        document_id: str,
        common_metadata: dict,
        category: Optional[str],
        tags: Optional[List[str]],
        text_count: int,
        image_count: int,
        table_count: int,
        image_summary_count: int,
        error_message: str,
    ) -> None:
        self.vector_svc.put_document_manifest(document_id, {
            **common_metadata,
            "status": "failed",
            "category": category,
            "tags": tags or [],
            "text_count": text_count,
            "image_count": image_count,
            "image_summary_count": image_summary_count,
            "table_count": table_count,
            "error_message": error_message,
        })


# 单例
_knowledge_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    """获取知识入库服务单例"""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
