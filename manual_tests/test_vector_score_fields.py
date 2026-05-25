from unittest.mock import MagicMock, patch

from test_runner import run_auto_cases, run_menu


def vec():
    return [0.1] * 1024


def auto_test():
    from schemas.models import VectorSearchResult
    import services.vector_service as module

    def schema_case():
        return VectorSearchResult(
            id="doc2",
            score=0.2,
            raw_score=0.2,
            raw_score_type="cosine_distance",
            relevance_score=0.8,
            retrieval_route="semantic",
            content="engine text",
        ).model_dump()

    def filter_case():
        return module.build_redis_filter(document_id="manual-v1", chunk_type="image")

    def search_case():
        with patch("services.vector_service.redis.Redis") as redis_cls:
            client = MagicMock()
            redis_cls.return_value = client

            def execute_command(*args):
                if args[0] == "FT.INFO":
                    return ["num_docs", "1"]
                if args[0] == "FT.SEARCH":
                    return [
                        1,
                        b"doc:doc1",
                        [
                            b"id",
                            b"doc1",
                            b"text",
                            "发动机".encode("utf-8"),
                            b"score",
                            b"0.12",
                            b"metadata",
                            '{"chunk_type":"text"}'.encode("utf-8"),
                        ],
                    ]
                return "OK"

            client.execute_command.side_effect = execute_command
            return module.VectorService().search(vec(), top_k=1)

    run_auto_cases([
        {
            "name": "VectorSearchResult carries score metadata",
            "input": "raw score contract",
            "expected": "raw_score/relevance_score/retrieval_route",
            "run": schema_case,
            "check": lambda x: x["raw_score"] == 0.2
            and x["raw_score_type"] == "cosine_distance"
            and x["relevance_score"] == 0.8
            and x["retrieval_route"] == "semantic",
        },
        {
            "name": "Redis filter supports document and chunk type",
            "input": "manual-v1 image",
            "expected": "filter expression",
            "run": filter_case,
            "check": lambda x: "@document_id:{manual-v1}" in x and "@chunk_type:{image}" in x,
        },
        {
            "name": "Redis search exposes normalized score",
            "input": "cosine distance 0.12",
            "expected": "relevance score 0.88",
            "run": search_case,
            "check": lambda x: x[0]["raw_score"] == 0.12
            and x[0]["raw_score_type"] == "cosine_distance"
            and x[0]["relevance_score"] == 0.88,
        },
    ])


def manual_test():
    auto_test()


if __name__ == "__main__":
    run_menu("vector score fields", auto_test, manual_test)
