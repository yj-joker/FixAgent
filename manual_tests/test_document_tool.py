from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, run_async, run_auto_cases, run_menu


def auto_test():
    from tools.document_tool import DocumentParserTool

    async def unsupported_type():
        result = await DocumentParserTool().run(file_url="manual.docx", file_type="docx")
        return result.model_dump()

    async def missing_file():
        result = await DocumentParserTool().run(file_url="Z:/not_exists.pdf", file_type="pdf")
        return result.model_dump()

    def clean_tables():
        return DocumentParserTool._clean_tables([[[" A ", None], ["", ""]], [["B", "C"]]])

    def group_sections():
        pages = [
            {"page": 1, "text": "前言内容", "tables": [], "images": []},
            {"page": 2, "text": "第一章 发动机\n正文", "tables": [[["A"]]], "images": [{"page": 2}]},
        ]
        return DocumentParserTool._group_into_sections(pages)

    run_auto_cases([
        {
            "name": "不支持的文件类型返回 UNSUPPORTED_FILE_TYPE",
            "input": "file_type=docx",
            "expected": {"error.code": "UNSUPPORTED_FILE_TYPE"},
            "run": lambda: run_async(unsupported_type()),
            "check": lambda x: x["success"] is False and x["error"]["code"] == "UNSUPPORTED_FILE_TYPE",
        },
        {
            "name": "本地文件不存在返回 FILE_NOT_FOUND",
            "input": "Z:/not_exists.pdf",
            "expected": {"error.code": "FILE_NOT_FOUND"},
            "run": lambda: run_async(missing_file()),
            "check": lambda x: x["success"] is False and x["error"]["code"] == "FILE_NOT_FOUND",
        },
        {
            "name": "_clean_tables 清理 None 和空行",
            "input": "含 None、空行、空白单元格",
            "expected": "保留有效表格",
            "run": clean_tables,
            "check": lambda x: x == [[["A", ""]], [["B", "C"]]],
        },
        {
            "name": "_group_into_sections 按章节合并文本、表格、图片",
            "input": "前言 + 第一章",
            "expected": "至少一个章节包含表格和图片",
            "run": group_sections,
            "check": lambda x: any(s["tables"] and s["images"] for s in x),
        },
    ])


def manual_test():
    from tools.document_tool import DocumentParserTool

    file_url = ask("请输入 PDF 文件路径或 URL", "manual.pdf")
    result = run_async(DocumentParserTool().run(file_url=file_url, file_type="pdf"))
    print_json(result.model_dump())


if __name__ == "__main__":
    run_menu("tools/document_tool.py", auto_test, manual_test)
