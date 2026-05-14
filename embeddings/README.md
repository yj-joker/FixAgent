# Embeddings 模块

## 模块概述

Embeddings 模块负责将**文本**和**图片**转换为向量表示，支撑向量检索和跨模态匹配能力。

| 类型 | 模型 | 来源 | 维度 |
|-----|------|------|------|
| 文本向量 | text-embedding-v4 | 阿里云百炼 | 1024 |
| 图片向量 | multimodal-embedding-v1 | 阿里云百炼 | 1024 |

> **注意**：两个模型维度一致（1024），可直接混合检索，无需向量转换。百炼 API 自行处理跨模态对齐。

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| 文本嵌入 | text-embedding-v4 | 国产、中文优化、API 稳定 |
| 图片嵌入 | multimodal-embedding-v1 | 国产、统一向量空间、文本图片共用 API |
| 向量库 | Redis Search (FT.SEARCH KNN) | 国产、高性能 |

## 项目实现

### text_embedding.py — 文本向量化

调用百炼 text-embedding-v4，将文本转为 1024 维向量：

```python
from embeddings.text_embedding import get_text_embedding

emb = get_text_embedding()

# 单条
vec = await emb.embed("电动机轴承过热原因")

# 批量（自动缓存命中）
vecs = await emb.embed_batch(["文本1", "文本2"])
```

**特性**：
- Redis 缓存（MD5 key，TTL 可配），重复文本不调 API
- 批量接口（最多 25 条/批，百炼限制）
- 单例模式：`get_text_embedding()`

### image_embedding.py — 图片向量化

调用百炼 multimodal-embedding-v1，将图片 URL 转为 1024 维向量：

```python
from embeddings.image_embedding import get_image_embedding

emb = get_image_embedding()

# 单张（百炼 API 自行下载图片）
vec = await emb.embed("https://cdn.example.com/bearing_fault.jpg")

# 批量
vecs = await emb.embed_batch(["url1", "url2"])
```

**特性**：
- 百炼 API 直接接受 URL，无需本地下载
- Redis 缓存（URL 的 MD5 作为 key）
- 与 text_embedding 共用同一 API Key、同一端点、同一缓存方案

### multimodal_embedding.py — 图文统一向量

封装 text_embedding + image_embedding，实现跨模态检索：

```python
from embeddings.multimodal_embedding import get_multimodal_embedding

multi = get_multimodal_embedding()

# 文本→向量
text_vec = await multi.embed_text("轴承磨损")

# 图片→向量
img_vec = await multi.embed_image("https://cdn.example.com/fault.jpg")

# 图文混合查询（向量加权融合）
query_vec = await multi.embed_query(
    query_text="轴承区域异常",
    query_images=["user_upload/fault.jpg"]
)
```

**与 services/vector_service.py 的关系**：
- `vector_service.search_by_text()` 直接用 text_embedding，自动处理向量化
- 后续可扩展 `search_by_image()` 接入 image_embedding

## 与 Java 后端交互

| Python 模块 | Java 对应 | 说明 |
|------------|----------|------|
| text_embedding | Python Tool 透传 | 检索时 Python 内部处理，Java 不感知 |
| image_embedding | Python Tool 透传 | 故障图片向量化 |
| multimodal_embedding | Python Tool 透传 | 图文混合查询 |

## 文件结构

```
embeddings/
├── __init__.py
├── README.md                    # 本文件
├── text_embedding.py            # 文本向量化（text-embedding-v4，1024维）
├── image_embedding.py           # 图片向量化（multimodal-embedding-v1，1024维，百炼API）
└── multimodal_embedding.py      # 图文统一向量（加权融合）
```

## 注意事项

1. **维度一致性**：text-embedding-v4 和 multimodal-embedding-v1 输出维度相同，检索时无需转换
2. **API Key 复用**：三个 embedding 服务共用 `DASHSCOPE_API_KEY`，无需单独配置
3. **缓存隔离**：文本缓存 key 以文本内容 MD5 计算，图片缓存 key 以 URL MD5 计算，互不干扰
4. **向量归一化**：百炼 API 返回的向量已是归一化结果，可直接用于余弦相似度检索
5. **图片处理**：百炼 multimodal API 直接接受图片 URL，无需本地下载再上传
