import asyncio
from unittest.mock import MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def fake_vector(seed=0.1):
    return [float(seed)] * 1024


def build_service():
    with patch("embeddings.text_embedding.redis.Redis"):
        from embeddings.text_embedding import TextEmbedding
        svc = TextEmbedding()
    svc.redis = MagicMock()
    svc.redis.get.return_value = None
    return svc


def auto_test():
    from embeddings.text_embedding import get_text_embedding
    import embeddings.text_embedding as module

    async def single_embed():
        svc = build_service()
        svc._call_api_sync = MagicMock(return_value=[fake_vector()])
        result = await svc.embed("电动机轴承过热")
        return {"dimension": len(result), "first": result[0], "api_calls": svc._call_api_sync.call_count}

    async def batch_embed():
        svc = build_service()
        svc._call_api_sync = MagicMock(side_effect=[[[float(i)] * 1024] for i in range(25)])
        result = await svc.embed_batch([f"文本{i}" for i in range(25)])
        return {
            "count": len(result),
            "dimension": len(result[0]),
            "last_first": result[-1][0],
            "api_calls": svc._call_api_sync.call_count,
        }

    async def cache_hit():
        svc = build_service()
        svc.redis.get.return_value = __import__("pickle").dumps(fake_vector(0.8))
        svc._call_api_sync = MagicMock(return_value=[fake_vector()])
        result = await svc.embed("相同文本")
        return {
            "first": result[0],
            "api_calls": svc._call_api_sync.call_count,
            "key": svc._get_cache_key("same")[:18],
        }

    def singleton():
        with patch("embeddings.text_embedding.redis.Redis"):
            module._text_embedding = None
            return get_text_embedding() is get_text_embedding()

    run_auto_cases([
        {
            "name": "单条文本向量化返回 1024 维 float 列表",
            "input": "电动机轴承过热",
            "expected": {"dimension": 1024, "api_calls": 1},
            "run": lambda: run_async(single_embed()),
            "check": lambda x: x["dimension"] == 1024 and x["api_calls"] == 1,
        },
        {
            "name": "批量 25 条文本逐条调用 API 并返回 25 个 1024 维向量且顺序一致",
            "input": "25 条文本",
            "expected": {"count": 25, "dimension": 1024, "api_calls": 25},
            "run": lambda: run_async(batch_embed()),
            "check": lambda x: x["count"] == 25 and x["dimension"] == 1024 and x["last_first"] == 24.0 and x["api_calls"] == 25,
        },
        {
            "name": "Redis 缓存命中时不调用 DashScope API",
            "input": "缓存已有相同文本向量",
            "expected": {"api_calls": 0, "key_prefix": "cache:emb:text:v2:"},
            "run": lambda: run_async(cache_hit()),
            "check": lambda x: x["first"] == 0.8 and x["api_calls"] == 0 and x["key"] == "cache:emb:text:v2:",
        },
        {
            "name": "get_text_embedding() 多次调用返回同一实例",
            "input": "连续调用 get_text_embedding()",
            "expected": True,
            "run": singleton,
            "check": lambda x: x is True,
        },
    ])


def manual_test():
    from embeddings.text_embedding import get_text_embedding

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    text = ask("请输入要向量化的文本", "电动机轴承过热")
    svc = get_text_embedding()
    svc.redis.ping()
    vector = asyncio.run(svc.embed(text))
    print_json({"dimension": len(vector), "preview": vector[:5]})


if __name__ == "__main__":
    run_menu("embeddings/text_embedding.py", auto_test, manual_test)
