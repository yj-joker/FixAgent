from unittest.mock import AsyncMock, MagicMock, patch

from test_runner import ask, print_json, require_env_value, require_real_dependency, run_async, run_auto_cases, run_menu


def vec(value):
    return [float(value)] * 1024


def build_service():
    with patch("embeddings.multimodal_embedding.get_text_embedding") as get_text, patch("embeddings.multimodal_embedding.get_image_embedding") as get_image:
        from embeddings.multimodal_embedding import MultimodalEmbedding
        text = MagicMock()
        image = MagicMock()
        text.embed = AsyncMock(return_value=vec(1))
        text.embed_batch = AsyncMock(return_value=[vec(1), vec(2)])
        image.embed_batch = AsyncMock(return_value=[vec(3), vec(5)])
        get_text.return_value = text
        get_image.return_value = image
        svc = MultimodalEmbedding()
    return svc


def auto_test():
    import embeddings.multimodal_embedding as module
    from embeddings.multimodal_embedding import get_multimodal_embedding

    async def text_only():
        result = await build_service().embed(text="电动机故障")
        return {"text_dim": len(result["text_vector"]), "image_vectors": result["image_vectors"]}

    async def image_only():
        result = await build_service().embed(image_urls=["url1"])
        return {"text_vector": result["text_vector"], "image_count": len(result["image_vectors"]), "image_dim": len(result["image_vectors"][0])}

    async def mixed():
        result = await build_service().embed(text="轴承", image_urls=["url1", "url2"])
        return {"text_dim": len(result["text_vector"]), "image_count": len(result["image_vectors"])}

    async def empty():
        return await build_service().embed(text=None, image_urls=[])

    def singleton():
        with patch("embeddings.multimodal_embedding.get_text_embedding") as get_text, patch("embeddings.multimodal_embedding.get_image_embedding") as get_image:
            get_text.return_value = MagicMock()
            get_image.return_value = MagicMock()
            module._multimodal_embedding = None
            return get_multimodal_embedding() is get_multimodal_embedding()

    run_auto_cases([
        {
            "name": "纯文本查询返回 text_vector 和空 image_vectors",
            "input": "text='电动机故障'",
            "expected": {"text_dim": 1024, "image_vectors": []},
            "run": lambda: run_async(text_only()),
            "check": lambda x: x["text_dim"] == 1024 and x["image_vectors"] == [],
        },
        {
            "name": "纯图片查询返回 image_vectors，text_vector=None",
            "input": "image_urls=['url1']",
            "expected": {"text_vector": None, "image_count": 2},
            "run": lambda: run_async(image_only()),
            "check": lambda x: x["text_vector"] is None and x["image_count"] == 2 and x["image_dim"] == 1024,
        },
        {
            "name": "图文混合查询同时返回文本向量和图片向量",
            "input": "text + 2 images",
            "expected": {"text_dim": 1024, "image_count": 2},
            "run": lambda: run_async(mixed()),
            "check": lambda x: x["text_dim"] == 1024 and x["image_count"] == 2,
        },
        {
            "name": "空输入返回 None 和空列表",
            "input": "text=None, image_urls=[]",
            "expected": {"text_vector": None, "image_vectors": []},
            "run": lambda: run_async(empty()),
            "check": lambda x: x["text_vector"] is None and x["image_vectors"] == [],
        },
        {
            "name": "get_multimodal_embedding() 返回同一实例",
            "input": "连续调用",
            "expected": True,
            "run": singleton,
            "check": lambda x: x is True,
        },
    ])


def manual_test():
    from embeddings.multimodal_embedding import get_multimodal_embedding

    require_real_dependency("redis", "pip install redis")
    require_real_dependency("dashscope", "pip install dashscope")
    require_env_value("DASHSCOPE_API_KEY", '请先设置 $env:DASHSCOPE_API_KEY="你的key"')
    text = ask("请输入文本（可空）", "轴承过热")
    urls = ask("请输入图片 URL，多个用英文逗号分隔（可空）", "")
    image_urls = [u.strip() for u in urls.split(",") if u.strip()]
    result = run_async(get_multimodal_embedding().embed(text=text or None, image_urls=image_urls))
    print_json({
        "text_dim": len(result["text_vector"]) if result["text_vector"] else None,
        "image_dims": [len(v) for v in result["image_vectors"]],
        "dimensions": result["dimensions"],
    })


if __name__ == "__main__":
    run_menu("embeddings/multimodal_embedding.py", auto_test, manual_test)
