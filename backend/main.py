"""
FastAPI应用入口
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings
from backend.database import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库 + 后台暖机，关闭时收干净句柄"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    if settings.app_env == "production" and settings.jwt_secret_key == "change-this-in-production":
        raise RuntimeError("生产环境必须通过 JWT_SECRET_KEY 环境变量设置强随机密钥，拒绝以默认密钥启动")
    await init_db()
    logger.info("Database initialized")
    if FAILED_ROUTES:
        logger.error(
            f"{len(FAILED_ROUTES)} 个路由注册失败，对应功能整片 404（前端会显示成空态）："
            + "; ".join(f"{name}: {err}" for name, err in FAILED_ROUTES)
        )
    from backend.tools import register_builtin_tools
    register_builtin_tools()

    from backend.scheduler.engine import scheduler_engine
    scheduler_engine.start()

    from backend.tools.mcp_manager import mcp_manager
    app.state.mcp_manager = mcp_manager
    if settings.mcp_server_configs:
        asyncio.create_task(mcp_manager.startup(settings.mcp_server_configs))

    app.state.warm_task = asyncio.create_task(_warm_heavy_state(app))
    yield
    app.state.warm_task.cancel()
    logger.info("Shutting down...")
    if hasattr(app.state, "mcp_manager"):
        try:
            await app.state.mcp_manager.shutdown()
        except Exception as e:
            logger.debug(f"MCP shutdown 失败: {e}")
    try:
        from backend.scheduler.engine import scheduler_engine
        scheduler_engine.shutdown(wait=True)
    except Exception as e:
        logger.debug(f"scheduler shutdown 失败: {e}")
    from backend.agents.llm_client import llm_client
    from backend.rag.embeddings import embedder
    for closeable in (llm_client, embedder):
        try:
            await closeable.close()
        except Exception as e:
            logger.debug(f"关闭 {closeable.__class__.__name__} 失败: {e}")
    try:
        await engine.dispose()
    except Exception as e:
        logger.debug(f"engine.dispose 失败: {e}")


async def _warm_heavy_state(app: FastAPI) -> None:
    app.state.chroma_ready = False
    app.state.chroma_count = 0
    try:
        from backend.rag.vectorstore import vector_store
        app.state.chroma_count = await vector_store.warm()
        app.state.chroma_ready = True
        logger.info(f"向量库已暖机（{app.state.chroma_count} 条向量）")
    except Exception as e:
        logger.error(f"向量库暖机失败，知识库检索将退回首次请求现建: {e}")

    from backend.personas.engine import persona_engine
    from backend.skills.engine import skills_engine
    for label, loader in (("人格", persona_engine.load_personas), ("技能", skills_engine.load_skills)):
        try:
            items = await asyncio.to_thread(loader)
            logger.info(f"{label}词典已预载（{len(items or [])} 项）")
        except Exception as e:
            logger.warning(f"{label}词典预加载失败: {e}")

    await _prewarm_egress(app)


async def _prewarm_egress(app: FastAPI) -> None:
    from backend.agents.llm_client import llm_client
    from backend.rag.embeddings import embedder

    for label, holder in (("对话", llm_client), ("嵌入", embedder)):
        try:
            client = await holder._get_client()
            base = getattr(holder, "base_url", None) or "https://open.bigmodel.cn"
            origin = base.split("/api")[0]
            await client.head(origin, timeout=5.0)
            logger.info(f"{label}通道连接已预热（{origin}）")
        except Exception as e:
            logger.info(f"{label}通道预热跳过（{type(e).__name__}），首条消息会稍慢")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="双端双角色多智能体AI教育平台",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.security.middleware import SecurityMiddleware
app.add_middleware(SecurityMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"},
    )


@app.get("/api/health")
async def health_check():
    from backend.rag.knowledge_graph import knowledge_graph
    state = app.state
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "routers_failed": [name for name, _ in FAILED_ROUTES],
        "chroma_ready": getattr(state, "chroma_ready", False),
        "vectors": getattr(state, "chroma_count", 0),
        "knowledge_points": knowledge_graph.point_count,
    }


@app.get("/privacy-policy")
async def privacy_policy():
    return {
        "title": "隐私政策",
        "content": """
        ZEdu Duo 隐私政策
        
        1. 我们收集的信息：用户名、邮箱、学习记录、错题数据
        2. 我们如何使用信息：提供个性化学习服务、改进教学效果
        3. 数据安全：我们采用加密存储和传输，保护您的数据
        4. 用户权利：您可以随时查看、导出或删除您的个人数据
        5. 未成年人保护：本产品面向14周岁以上用户，不收集未成年人数据
        6. 联系我们：如有疑问请通过应用内反馈渠道联系
        """,
        "last_updated": "2026-09-11",
    }


ROUTE_TABLE = [
    ("backend.auth.routes", "/api/auth", "认证"),
    ("backend.chat.routes", "/api/chat", "对话"),
    ("backend.rag.routes", "/api/knowledge", "知识库"),
    ("backend.tutor.routes", "/api/tutor", "辅导"),
    ("backend.teacher.routes", "/api/teacher", "教师端"),
    ("backend.admin.routes", "/api/admin", "管理后台"),
    ("backend.profile.routes", "/api/profile", "个性化配置"),
    ("backend.plan.routes", "/api/plans", "学习计划"),
    ("backend.personas.routes", "/api/personas", "人格系统"),
    ("backend.evolution.routes", "/api/evolution", "画像自进化"),
    ("backend.memory.routes", "/api/memory", "双维度记忆"),
    ("backend.tools.routes", "/api/tools", "工具系统"),
    ("backend.skills.routes", "/api/skills", "技能系统"),
    ("backend.notification.routes", "/api/notifications", "通知系统"),
    ("backend.settings.routes", "/api/settings", "系统设置"),
    ("backend.resources.routes", "/api/resources", "个性化资源中心"),
    ("backend.approval.routes", "/api/approval", "审批系统"),
    ("backend.dsh.routes", "", "DeepSeek Harness"),
]

FAILED_ROUTES: list[tuple[str, str]] = []


def register_routes() -> None:
    """逐个注册路由，一个坏不掉一串"""
    import importlib

    for module_path, prefix, tag in ROUTE_TABLE:
        try:
            module = importlib.import_module(module_path)
            app.include_router(module.router, prefix=prefix, tags=[tag])
        except Exception as e:
            name = module_path.rsplit(".", 2)[-2]
            FAILED_ROUTES.append((name, f"{type(e).__name__}: {e}"))
            logger.error(f"路由 {module_path} 注册失败（{prefix} 整片不可用）: {e}")
    if FAILED_ROUTES and settings.strict_routes:
        raise RuntimeError(
            "以下路由注册失败（strict_routes=true 时直接拒绝启动）："
            + "; ".join(f"{n} → {r}" for n, r in FAILED_ROUTES)
        )


register_routes()
if not FAILED_ROUTES:
    logger.info("All routes registered")


ASSET_CACHE = "public, max-age=31536000, immutable"
DOC_CACHE = "no-cache"


def mount_frontend(app: FastAPI) -> None:
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from backend.desktop.env import resource_path

    dist = resource_path(settings.static_dist)
    if not (dist / "index.html").is_file():
        logger.warning(f"serve_frontend 开启但找不到 {dist}/index.html，跳过静态托管")
        return
    assets = dist / "assets"
    if assets.is_dir():

        class _ImmutableStatic(StaticFiles):
            def file_response(self, *args, **kwargs):
                response = super().file_response(*args, **kwargs)
                response.headers["Cache-Control"] = ASSET_CACHE
                return response

        app.mount("/assets", _ImmutableStatic(directory=str(assets)), name="spa-assets")

    index = dist / "index.html"
    dist_resolved = dist.resolve()
    reserved = ("api", "docs", "redoc", "openapi.json")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.split("/", 1)[0] in reserved:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (dist_resolved / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(dist_resolved):
            return FileResponse(candidate, headers={"Cache-Control": DOC_CACHE})
        return FileResponse(index, headers={"Cache-Control": DOC_CACHE})


if settings.serve_frontend:
    mount_frontend(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.dev_reload,
        reload_dirs=["backend"] if settings.dev_reload else None,
    )
