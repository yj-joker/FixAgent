"""
文档解析工具

使用 pdfplumber + 可选 PyMuPDF，将 PDF/Word 等非结构化文档
拆分为结构化的文本块、图片和表格。

【与架构文档的对应关系】
- 位置：tools/document_tool.py
- 继承：tools/base_tool.py 的 BaseTool
- 下游：知识入库流程（/ai/knowledge/import 端点） → embedding 向量化 → Redis 向量库
- 上游：Java 后端在部署初始化时上传赛题提供的维修手册 PDF

【为什么需要这个工具】
赛题只给了一份《摩托车发动机维修手册》PDF 作为知识来源。
系统上线后必须先把 PDF 拆开、向量化、存入 Redis，检索功能才能跑起来。
这个工具负责"拆开"这一步。

【技术选型】
- pdfplumber（纯 Python）：提取文字和表格，龙芯 LoongArch 上直接能用
- PyMuPDF（可选）：提取图片时效果更好，但依赖 C 扩展库，龙芯上可能需要编译
  → 优先用 PyMuPDF 提图片，装不上则用 pdfplumber 记录图片位置，跳过实际提取

【和已实现模块的关系】
- 输入格式：接收文件路径或 URL（和 text_embedding/image_embedding 一样传 URL）
- 输出格式：结构化 dict，text/image/table 各自分好，下游直接消费
- 不负责入库：和 graph_query_tool 一样只做"获取数据"这一件事

【执行流程】
1. 校验 file_type（pdf/docx）
2. 本地文件直接读，远程文件先下载
3. pdfplumber 逐页提取文字 + 识别表格
4. PyMuPDF 逐页提取图片 → 保存为 PNG
5. 用"第X章"正则合并相邻页为章节
6. 返回结构化结果
"""

import os
import re
import hashlib
import asyncio
import logging
from typing import List, Optional

import httpx

from tools.base_tool import BaseTool, ToolException

logger = logging.getLogger(__name__)


class DocumentParserTool(BaseTool):
    """
    文档解析工具

    把 PDF/Word 拆成结构化内容：文字归文字、图片归图片、表格归表格。
    """

    @property
    def name(self) -> str:
        return "document_parser"

    @property
    def description(self) -> str:
        return (
            "解析 PDF/Word 文档，提取文本内容、图片和表格，输出按章节组织的结构化结果。"
            "适用场景：知识库初始化时批量导入维修手册、技术文档等。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "文档路径或 URL。本地文件用绝对路径如 C:/docs/manual.pdf，远程文件用 http/https URL"
                },
                "file_type": {
                    "type": "string",
                    "description": "文件类型，目前支持 pdf",
                    "default": "pdf"
                },
                "output_image_dir": {
                    "type": "string",
                    "description": "提取图片的保存目录，默认为文档同目录下的 manual_images/ 子目录"
                }
            },
            "required": ["file_url"]
        }

    async def _execute(
        self,
        file_url: str,
        file_type: str = "pdf",
        output_image_dir: Optional[str] = None
    ) -> dict:
        """
        解析文档，返回按章节组织的结构化内容。

        Args:
            file_url: 文档路径或 URL
            file_type: 文件类型，目前仅支持 "pdf"
            output_image_dir: 图片输出目录，默认自动生成

        Returns:
            {
                "file_name": "摩托车发动机维修手册.pdf",
                "total_pages": 45,
                "sections": [
                    {
                        "section_title": "第二章 发动机结构",
                        "page_range": "8-15",
                        "text_chunks": ["段落1", "段落2", ...],
                        "images": [{"page": 9, "image_name": "...", "caption": "...", "local_path": "..."}],
                        "tables": [{"page": 11, "caption": "...", "headers": [...], "rows": [[...], ...]}]
                    }
                ],
                "extraction_summary": {
                    "text_chunks_total": 230,
                    "images_total": 68,
                    "tables_total": 15,
                    "image_extraction_method": "pymupdf" | "metadata_only"
                }
            }

        Raises:
            ToolException: UNSUPPORTED_FILE_TYPE / FILE_NOT_FOUND / PDF_PARSE_FAILED
        """
        if file_type not in ("pdf",):
            raise ToolException(
                code="UNSUPPORTED_FILE_TYPE",
                message=f"不支持的文件类型: {file_type}，目前仅支持 pdf"
            )

        local_path = await self._resolve_file(file_url)

        if output_image_dir is None:
            output_image_dir = os.path.join(
                os.path.dirname(local_path) or ".",
                f"{os.path.splitext(os.path.basename(local_path))[0]}_images"
            )
        os.makedirs(output_image_dir, exist_ok=True)

        file_name = os.path.basename(local_path)

        try:
            result = await asyncio.to_thread(
                self._parse_pdf, local_path, output_image_dir
            )
            result["file_name"] = file_name
            return result
        except Exception as e:
            raise ToolException(
                code="PDF_PARSE_FAILED",
                message=f"文档解析失败: {e}"
            )

    # ==================== 文件解析入口 ====================

    def _parse_pdf(self, file_path: str, image_dir: str) -> dict:
        """
        用 pdfplumber 逐页解析 PDF 文本和表格，
        用 PyMuPDF 提取图片（降级到 metadata_only）。
        """
        import pdfplumber

        pages_data = []
        image_extraction_method = "pymupdf"

        # 尝试加载 PyMuPDF 用于图片提取
        fitz = self._try_import_fitz()
        if fitz is not None:
            fitz_doc = fitz.open(file_path)
        else:
            fitz_doc = None
            image_extraction_method = "metadata_only"

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                # 1. 提取文字
                text = page.extract_text() or ""

                # 2. 提取表格
                raw_tables = page.extract_tables()
                tables = self._clean_tables(raw_tables)

                # 3. 提取图片
                images = []
                if fitz_doc is not None:
                    images = self._extract_images_fitz(
                        fitz_doc, page_num, image_dir
                    )
                else:
                    images = self._record_image_positions(page, page_num)

                for image in images:
                    image.setdefault("context_before", text[:300].strip())
                    image.setdefault("context_after", text[-300:].strip())

                pages_data.append({
                    "page": page_num,
                    "text": text,
                    "tables": tables,
                    "images": images
                })

        if fitz_doc is not None:
            fitz_doc.close()

        # 将逐页数据合并为章节
        sections = self._group_into_sections(pages_data)

        return {
            "total_pages": total_pages,
            "sections": sections,
            "extraction_summary": {
                "text_chunks_total": sum(len(s["text_chunks"]) for s in sections),
                "images_total": sum(len(s["images"]) for s in sections),
                "tables_total": sum(len(s["tables"]) for s in sections),
                "image_extraction_method": image_extraction_method
            }
        }

    # ==================== 图片提取（PyMuPDF） ====================

    @staticmethod
    def _try_import_fitz():
        """尝试导入 PyMuPDF，失败返回 None"""
        try:
            import fitz
            return fitz
        except ImportError:
            return None

    def _extract_images_fitz(self, fitz_doc, page_num: int, image_dir: str) -> list:
        """
        用 PyMuPDF 从指定页提取图片，保存为 PNG。

        PDF 中嵌入的图片可能是 CMYK 色彩空间，需要转为 RGB 再保存。
        """
        import fitz as fitz_module

        images = []
        page = fitz_doc[page_num - 1]

        for img_index, img_info in enumerate(page.get_images(full=True), start=1):
            xref = img_info[0]
            try:
                base_image = fitz_doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                image_name = f"page_{page_num:03d}_img_{img_index:02d}.{ext}"
                image_path = os.path.join(image_dir, image_name)

                # CMYK 转 RGB
                if base_image.get("colorspace") == 4:  # CMYK
                    pix = fitz_module.Pixmap(fitz_doc, xref)
                    if pix.n >= 4:
                        pix = fitz_module.Pixmap(fitz_module.csRGB, pix)
                    pix.save(image_path)
                else:
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                images.append({
                    "page": page_num,
                    "image_name": image_name,
                    "local_path": image_path,
                    "width": base_image.get("width"),
                    "height": base_image.get("height"),
                    "format": ext
                })
            except Exception:
                continue

        # 尝试从页面文字中匹配图注
        page_text = page.get_text("text")
        self._attach_captions(images, page_text)

        return images

    def _record_image_positions(self, page, page_num: int) -> list:
        """
        降级方案：用 pdfplumber 记录图片位置，不提取实际数据。
        调用方可以基于 page 和坐标手动截图。
        """
        images = []
        for img in page.images:
            images.append({
                "page": page_num,
                "image_name": None,
                "local_path": None,
                "x0": img.get("x0"),
                "top": img.get("top"),
                "x1": img.get("x1"),
                "bottom": img.get("bottom"),
                "width": img.get("width"),
                "height": img.get("height"),
                "note": "图片数据未提取（PyMuPDF 不可用），请手动截取"
            })
        return images

    # ==================== 图注匹配 ====================

    @staticmethod
    def _attach_captions(images: list, page_text: str) -> None:
        """
        从页面文字中找图注（图X-X 格式），按文字位置匹配图片。

        规则：一本书里的插图图注通常在图片正下方，
        文本中 "图2-1 ..." 出现在图片坐标下方且最近的文字即为图注。
        由于我们没有精确的文字坐标，这里用简单策略：
        按图片在页面上从上到下的顺序，匹配文本中出现的图注顺序。
        """
        caption_pattern = re.compile(r'图\s*\d+[-–—]\s*\d+\s*[：:，,\s]*(.+?)(?:\n|图\s*\d+|\Z)', re.DOTALL)
        captions = caption_pattern.findall(page_text)

        if not captions or not images:
            return

        # 按从上到下排列图片（如果有坐标信息）
        sorted_images = sorted(
            images,
            key=lambda x: (x.get("top") if x.get("top") is not None else 9999)
        )

        for i, img in enumerate(sorted_images):
            if i < len(captions):
                img["caption"] = captions[i].strip()

    # ==================== 表格清理 ====================

    @staticmethod
    def _clean_tables(raw_tables: list) -> list:
        """清理 pdfplumber 提取的表格：去 None、去空行、剥离空白"""
        cleaned = []
        for table in raw_tables:
            rows = []
            for row in table:
                if row is None:
                    continue
                cleaned_row = [cell.strip() if isinstance(cell, str) else (str(cell) if cell is not None else "") for cell in row]
                if any(cleaned_row):
                    rows.append(cleaned_row)
            if rows:
                cleaned.append(rows)
        return cleaned

    @staticmethod
    def _split_page_text(text: str, page_num: int) -> list:
        """Split a page into structured step chunks when numbered steps exist."""
        page_text = text.strip()
        if not page_text:
            return []

        step_pattern = re.compile(r'(?m)^\s*(\d+[.、]\s+[^\n]+)')
        matches = list(step_pattern.finditer(page_text))
        if not matches:
            return [{
                "text": page_text,
                "page": page_num,
                "chunk_label": "page",
                "context_before": "",
                "context_after": "",
            }]

        prefix = page_text[:matches[0].start()].strip()
        chunks = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(page_text)
            chunk_text = page_text[start:end].strip()
            if prefix:
                chunk_text = f"{prefix}\n{chunk_text}"
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "chunk_label": "step",
                "context_before": prefix,
                "context_after": page_text[end:end + 300].strip(),
            })
        return chunks

    # ==================== 章节合并 ====================

    @staticmethod
    def _group_into_sections(pages_data: list) -> list:
        """
        将逐页数据按"第X章"标题合并为章节。

        在一页里发现章节标题 → 新建 section。
        后续页跟在当前 section 里，直到下一个章节标题出现。
        """
        chapter_pattern = re.compile(r'第[一二三四五六七八九十\d]+章')

        sections = []
        current_section = {
            "section_title": "前言",
            "page_range": "",
            "text_chunks": [],
            "images": [],
            "tables": []
        }
        start_page = 1
        sections.append(current_section)

        for page_data in pages_data:
            page_num = page_data["page"]
            text = page_data["text"]

            # 检测章节标题
            match = chapter_pattern.search(text)
            if match and page_num > 1:
                current_section["page_range"] = f"{start_page}-{page_num - 1}"
                start_page = page_num
                current_section = {
                    "section_title": match.group(),
                    "page_range": "",
                    "text_chunks": [],
                    "images": [],
                    "tables": []
                }
                sections.append(current_section)

            # 将当前页内容归入当前章节
            if text.strip():
                current_section["text_chunks"].extend(
                    DocumentParserTool._split_page_text(text, page_num)
                )
            current_section["images"].extend(page_data["images"])
            for table in page_data["tables"]:
                label = f"第{page_num}页表格"
                current_section["tables"].append({
                    "page": page_num,
                    "caption": label,
                    "rows": table
                })

        # 最后一个章节的页码范围
        if sections:
            last_page = pages_data[-1]["page"] if pages_data else 1
            for sec in sections:
                if not sec["page_range"]:
                    sec["page_range"] = f"{start_page}-{last_page}"

        # 过滤掉空章节
        return [
            s for s in sections
            if s["text_chunks"] or s["images"] or s["tables"]
        ]

    # ==================== 文件下载 ====================

    async def _resolve_file(self, file_url: str) -> str:
        """
        如果是 HTTP URL 则下载到临时目录，否则直接返回本地路径。

        下载的文件以 URL hash 命名，放在系统临时目录下。
        """
        file_url = file_url.strip().strip('"')
        if file_url.startswith(("http://", "https://")):
            import tempfile

            parsed = hashlib.md5(file_url.encode()).hexdigest()[:12]
            ext = ".pdf"
            local_path = os.path.join(tempfile.gettempdir(), f"docparser_{parsed}{ext}")

            if os.path.exists(local_path):
                return local_path

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(file_url)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)
            return local_path

        # 本地路径
        if not os.path.exists(file_url):
            raise ToolException(
                code="FILE_NOT_FOUND",
                message=f"文件不存在: {file_url}"
            )
        return file_url


# 单例
_document_parser: Optional[DocumentParserTool] = None


def get_document_parser() -> DocumentParserTool:
    """获取文档解析工具单例"""
    global _document_parser
    if _document_parser is None:
        _document_parser = DocumentParserTool()
    return _document_parser
