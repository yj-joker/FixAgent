import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_top_p = float(os.getenv("LLM_TOP_P", "0.9"))
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # Redis 配置
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    redis_db = int(os.getenv("REDIS_DB", "0"))
    redis_ttl = int(os.getenv("REDIS_TTL", "86400"))  # 缓存默认24小时过期

    # Neo4j 图数据库配置
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings