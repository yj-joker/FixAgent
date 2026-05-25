from test_runner import ask, print_json, run_auto_cases, run_menu


def auto_test():
    from services.retrieval_policy import (
        cosine_distance_to_relevance,
        detect_query_intent,
        diversify_candidates,
        summarize_confidence,
    )

    candidates = [
        {"doc_id": "txt1", "metadata": {"chunk_type": "text"}, "relevance_score": 0.95},
        {"doc_id": "txt2", "metadata": {"chunk_type": "text"}, "relevance_score": 0.91},
        {"doc_id": "img1", "metadata": {"chunk_type": "image"}, "relevance_score": 0.90},
        {"doc_id": "tbl1", "metadata": {"chunk_type": "table"}, "relevance_score": 0.89},
    ]

    run_auto_cases([
        {
            "name": "cosine distance 统一为高分相关度",
            "input": [0.0, 0.4, 1.2],
            "expected": [1.0, 0.6, 0.0],
            "run": lambda: [
                cosine_distance_to_relevance(0.0),
                cosine_distance_to_relevance(0.4),
                cosine_distance_to_relevance(1.2),
            ],
            "check": lambda x: x == [1.0, 0.6, 0.0],
        },
        {
            "name": "查询意图区分文本图片表格和混合",
            "input": ["发动机怎么拆", "发动机拆卸图", "发动机扭矩参数", "发动机"],
            "expected": ["text", "image", "table", "mixed"],
            "run": lambda: [
                detect_query_intent("发动机怎么拆"),
                detect_query_intent("发动机拆卸图"),
                detect_query_intent("发动机扭矩参数"),
                detect_query_intent("发动机"),
            ],
            "check": lambda x: x == ["text", "image", "table", "mixed"],
        },
        {
            "name": "混合查询保留不同结果类型",
            "input": "text,text,image,table",
            "expected": "top3 包含 text/image/table",
            "run": lambda: diversify_candidates(candidates, top_k=3, intent="mixed"),
            "check": lambda x: {item["metadata"]["chunk_type"] for item in x} == {"text", "image", "table"},
        },
        {
            "name": "图片双路命中提高检索置信度",
            "input": "image query + dual route hit",
            "expected": "high",
            "run": lambda: summarize_confidence(
                [{"metadata": {"chunk_type": "image"}, "relevance_score": 0.93, "routes": ["image_vector", "image_summary"]}],
                intent="image",
            ),
            "check": lambda x: x["confidence"] == "high" and "image" in x["matched_types"],
        },
    ])


def manual_test():
    from services.retrieval_policy import detect_query_intent

    query = ask("query", "发动机拆卸图")
    print_json({"intent": detect_query_intent(query)})


if __name__ == "__main__":
    run_menu("services/retrieval_policy.py", auto_test, manual_test)
