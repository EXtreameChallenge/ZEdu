"""
应用配置 — 使用pydantic-settings从环境变量加载
所有密钥通过.env文件注入，代码中不硬编码
"""

import json

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ZEdu Duo"
    app_version: str = "4.0.41"
    app_env: str = "development"
    debug: bool = True
    db_echo: bool = False
    dev_reload: bool = False

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    database_url: str = "sqlite+aiosqlite:///./data/zedu_duo.db"

    jwt_secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    zhipu_api_keys: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4-flash"
    zhipu_thinking_model: str = "glm-4.5-air"

    search_provider: str = "auto"
    tavily_api_key: str = ""
    search_max_results: int = 5

    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-turbo"

    xinghuo_api_key: str = ""
    xinghuo_base_url: str = "https://spark-api-open.xf-yun.com/v1"
    xinghuo_model: str = "generalv3.5"

    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "embedding-3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 16
    embedding_timeout: float = 30.0

    upload_dir: str = "./data/uploads"
    static_dist: str = "./frontend/dist"
    serve_frontend: bool = False
    strict_routes: bool = True

    content_safety_enabled: bool = True
    sensitive_words_file: str = "./backend/content_safety/sensitive_words.txt"

    rate_limit_per_minute: int = 300
    trust_proxy_headers: bool = True

    min_age_required: int = 14
    privacy_policy_url: str = "/privacy-policy"

    demo_mode: bool = False

    mcp_servers: str = ""

    @property
    def mcp_server_configs(self) -> list[dict]:
        if not self.mcp_servers:
            return []
        try:
            data = json.loads(self.mcp_servers)
        except Exception as e:
            logger.warning(f"MCP_SERVERS 不是合法 JSON，已忽略：{e}")
            return []
        if not isinstance(data, list):
            logger.warning("MCP_SERVERS 顶层不是数组，已忽略")
            return []
        return [item for item in data if isinstance(item, dict)]

    @property
    def zhipu_key_list(self) -> list[str]:
        if not self.zhipu_api_keys:
            return []
        return [k.strip() for k in self.zhipu_api_keys.split(",") if k.strip()]


settings = Settings()
