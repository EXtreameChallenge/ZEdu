"""
数据库配置 — SQLite异步引擎
"""
import asyncio

from loguru import logger
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

write_semaphore = asyncio.Semaphore(3)


class Base(DeclarativeBase):
    """ORM模型基类"""
    pass


_CONNECT_PRAGMAS = (
    ("journal_mode", "WAL"),
    ("busy_timeout", "4000"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("temp_store", "MEMORY"),
)


def _apply_pragmas(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    try:
        for key, value in _CONNECT_PRAGMAS:
            cursor.execute(f"PRAGMA {key}={value}")
            cursor.fetchall()
    except Exception as e:
        logger.warning(f"SQLite PRAGMA 设置失败（{e}），将退回 SQLite 默认行为")
    finally:
        cursor.close()


engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,
    connect_args={
        "timeout": 5,
        "check_same_thread": False,
    },
)
event.listen(engine.sync_engine, "connect", _apply_pragmas)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """初始化数据库：建表 + 落实并复核连接级设置"""
    import os
    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    import backend.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    from backend.migrations import ensure_indexes, missing_indexes
    async with engine.begin() as conn:
        failed = await ensure_indexes(conn)
        still_missing = await missing_indexes(conn)
    if not failed and not still_missing:
        logger.info("必需索引已全部就绪")

    from sqlalchemy import text
    async with engine.connect() as conn:
        live = {
            key: str((await conn.execute(text(f"PRAGMA {key}"))).scalar())
            for key in ("journal_mode", "synchronous", "busy_timeout", "foreign_keys")
        }
    logger.info(
        "SQLite 运行期实测: journal_mode={journal_mode} synchronous={synchronous} "
        "busy_timeout={busy_timeout} foreign_keys={foreign_keys}".format(**live)
    )
    if live.get("foreign_keys") != "1":
        logger.error("foreign_keys 未生效（实测 {}）：级联约束形同虚设".format(live.get("foreign_keys")))
    if live.get("journal_mode", "").lower() != "wal":
        logger.error("WAL 未生效（实测 {}）：并发读写会互相阻塞".format(live.get("journal_mode")))


async def get_db() -> AsyncSession:
    """FastAPI依赖注入：获取数据库会话"""
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_write_db() -> AsyncSession:
    """写操作专用：通过信号量限制并发写"""
    async with write_semaphore:
        async with async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


class write_txn:
    """只把短提交块圈进写信号量的上下文管理器"""

    def __init__(self):
        self._session = None

    async def __aenter__(self) -> AsyncSession:
        await write_semaphore.acquire()
        self._session = async_session()
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is not None and self._session is not None:
                await self._session.rollback()
            return await self._session.__aexit__(exc_type, exc, tb)
        finally:
            write_semaphore.release()
