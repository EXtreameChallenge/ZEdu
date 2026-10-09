"""
启动脚本 — python run.py
"""
import uvicorn

from backend.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.dev_reload,
        reload_dirs=["backend"] if settings.dev_reload else None,
        log_level="info",
    )
