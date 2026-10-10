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

    # 应用
    app_name: str = "ZEdu Duo"
    app_version: str = "4.0.42"
    app_env: str = "development"
    debug: bool = True
    # SQL 回显与热重载必须能单独关：debug=true 顺手打开它们会造成
    # ① 每条语句写一次日志（登录/对话热路径直接变慢）
    # ② uvicorn 监视 CWD，而 data/ 下的 chroma 与 -wal 每次业务写都触发自重启
    db_echo: bool = False
    dev_reload: bool = False

    # 服务器
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS：逗号分隔的来源白名单；生产环境必须改为具体域名（如 https://zedu.example.com）
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # 数据库
    database_url: str = "sqlite+aiosqlite:///./data/zedu_duo.db"

    # JWT
    jwt_secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24小时

    # 智谱AI
    zhipu_api_keys: str = ""  # 逗号分隔的多个Key
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    zhipu_model: str = "glm-4-flash"  # 日常对话默认模型（快速、免费）
    zhipu_thinking_model: str = "glm-4.5-air"  # 深度思考时自动切换的模型（支持thinking推理链）

    # 联网搜索配置
    search_provider: str = "auto"  # auto(自动降级:360>DDG>搜狗>必应>百度) / 360 / duckduckgo / tavily(需key)
    tavily_api_key: str = ""  # Tavily API Key（可选，质量更高）
    search_max_results: int = 5

    # 降级模型（通义千问）
    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-turbo"

    # 讯飞星火（可选通道；软件杯要求 AI 相关工具选用讯飞系）
    xinghuo_api_key: str = ""
    xinghuo_base_url: str = "https://spark-api-open.xf-yun.com/v1"
    xinghuo_model: str = "generalv3.5"

    # 向量库（嵌入计算走智谱 API，本地不再驻留 torch/sentence-transformers）
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "embedding-3"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 16
    embedding_timeout: float = 30.0

    # 桌面化：上传归档目录与前端静态资源目录（entry.py 在冻结态改写到 runtime_root）
    upload_dir: str = "./data/uploads"
    static_dist: str = "./frontend/dist"
    serve_frontend: bool = False  # 桌面形态由后端托管 SPA（Electron loadURL 同一端口）
    strict_routes: bool = True

    # 内容安全
    content_safety_enabled: bool = True
    sensitive_words_file: str = "./backend/content_safety/sensitive_words.txt"

    # 限流
    rate_limit_per_minute: int = 300
    trust_proxy_headers: bool = True

    # 合规
    min_age_required: int = 14
    privacy_policy_url: str = "/privacy-policy"

    # 演示模式（无API Key时返回模拟回复）
    demo_mode: bool = False

    # ===== MCP 外部工具接入（P3）=====
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
