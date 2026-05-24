import asyncio
from unittest.mock import MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def fake_vector(seed=0.2):
    return [float(seed)] * 1024


def build_service():
    with patch("embeddings.image_embedding.redis.Redis"):
        from embeddings.image_embedding import ImageEmbedding
        svc = ImageEmbedding()
    svc.redis = MagicMock()
    svc.redis.get.return_value = None
    return svc


def auto_test():
    import embeddings.image_embedding as module
    from embeddings.image_embedding import get_image_embedding

    async def single_embed():
        svc = build_service()
        svc._call_api_sync = MagicMock(return_value=[fake_vector()])
        result = await svc.embed("https://example.com/fault.jpg")
        return {"dimension": len(result), "api_calls": svc._call_api_sync.call_count}

    async def batch_embed():
        svc = build_service()
        svc._call_api_sync = MagicMock(return_value=[fake_vector(i) for i in range(3)])
        result = await svc.embed_batch(["u1", "u2", "u3"])
        return {"count": len(result), "dimension": len(result[0]), "last_first": result[-1][0]}

    async def cache_hit():
        import pickle
        svc = build_service()
        svc.redis.get.return_value = pickle.dumps(fake_vector(0.9))
        svc._call_api_sync = MagicMock(return_value=[fake_vector()])
        result = await svc.embed("https://example.com/fault.jpg")
        return {"first": result[0], "api_calls": svc._call_api_sync.call_count, "key": svc._get_cache_key("https://example.com/fault.jpg")[:19]}

    def singleton():
        with patch("embeddings.image_embedding.redis.Redis"):
            module._image_embedding = None
            return get_image_embedding() is get_image_embedding()

    run_auto_cases([
        {
            "name": "单张图片 URL 向量化返回 1024 维 float 列表",
            "input": "https://example.com/fault.jpg",
            "expected": {"dimension": 1024},
            "run": lambda: run_async(single_embed()),
            "check": lambda x: x["dimension"] == 1024 and x["api_calls"] == 1,
        },
        {
            "name": "批量图片 URL 返回 N 个 1024 维向量且顺序一致",
            "input": "3 张图片 URL",
            "expected": {"count": 3, "dimension": 1024},
            "run": lambda: run_async(batch_embed()),
            "check": lambda x: x["count"] == 3 and x["last_first"] == 2.0,
        },
        {
            "name": "Redis 缓存命中时不调用 API，key 前缀为 img_emb:v2:",
            "input": "缓存已有相同图片 URL",
            "expected": {"api_calls": 0, "key_prefix": "cache:emb:image:v2:"},
            "run": lambda: run_async(cache_hit()),
            "check": lambda x: x["first"] == 0.9 and x["api_calls"] == 0 and x["key"] == "cache:emb:image:v2:",
        },
        {
            "name": "get_image_embedding() 多次调用返回同一实例",
            "input": "连续调用 get_image_embedding()",
            "expected": True,
            "run": singleton,
            "check": lambda x: x is True,
        },
    ])


def manual_test():
    from embeddings.image_embedding import get_image_embedding

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    url = ask("请输入图片 URL", "https://example.com/fault.jpg")
    svc = get_image_embedding()
    svc.redis.ping()
    vector = asyncio.run(svc.embed(url))
    print_json({"dimension": len(vector), "preview": vector[:5]})


if __name__ == "__main__":
    run_menu("embeddings/image_embedding.py", auto_test, manual_test)
