from test_runner import run_auto_cases, run_menu


def auto_test():
    from tools.document_tool import DocumentParserTool

    run_auto_cases([
        {
            "name": "step-aware page splitter keeps repair steps",
            "input": "chapter + two numbered steps",
            "expected": "two structured chunks",
            "run": lambda: DocumentParserTool._split_page_text(
                "3.2 拆卸发动机\n1. 排放机油\n拆下放油螺栓。\n2. 拆下水管\n松开卡箍。",
                page_num=6,
            ),
            "check": lambda x: len(x) == 2
            and x[0]["page"] == 6
            and "排放机油" in x[0]["text"]
            and x[1]["chunk_label"] == "step",
        },
    ])


def manual_test():
    auto_test()


if __name__ == "__main__":
    run_menu("tools/document_tool.py chunking", auto_test, manual_test)
