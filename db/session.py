# utils for session, engine

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import quote_plus
from pydantic import BaseModel
from core.config import ServiceSettings
from enum import Enum
from common.logger import logger

class DBSettings(BaseModel):
    """
    Portable DB settings class used to pass in configs for generic Postgres connection.
    Provides validation layer for database configuration.
    """
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    POOL_RECYCLE: int = 1800
    USER: str
    PW: str
    HOST: str
    PORT: str
    NAME: str

class DBType(str, Enum):
    """
    Enum types for different DBs.
    For now, just main db.
    """
    MainDB = "MAIN_DB"

def parse_db_settings_from_service(settings: ServiceSettings, db_type: DBType = DBType.MainDB) -> DBSettings:
    """
    Helper to extract and validate database settings from ServiceSettings.
    Specify the db_type to parse accordingly, defaults to MAIN_DB.
    """
    if db_type == DBType.MainDB:
        return DBSettings(
            POOL_SIZE=settings.MAIN_DB_POOL_SIZE,
            MAX_OVERFLOW=settings.MAIN_DB_MAX_OVERFLOW,
            POOL_TIMEOUT=settings.MAIN_DB_POOL_TIMEOUT,
            POOL_RECYCLE=settings.MAIN_DB_POOL_RECYCLE,
            USER=settings.MAIN_DB_USER,
            HOST=settings.MAIN_DB_HOST,
            PW=settings.MAIN_DB_PW,
            PORT=settings.MAIN_DB_PORT,
            NAME=settings.MAIN_DB_NAME
        )
    else:
        logger.error(f"Unknown DB type: {db_type}, shutting down service. Please double check!")
        raise ValueError(f"Unknown DB type: {db_type}")

def build_supabase_url(db_settings: DBSettings) -> str:
    """
    Build async PostgreSQL connection URL for Supabase.
    Uses asyncpg driver and enforces SSL connection.
    """
    # URL-encode credentials to handle special characters
    user = quote_plus(db_settings.USER)
    password = quote_plus(db_settings.PW)

    # Supabase host format: db.{project_ref}.supabase.co
    host = db_settings.HOST
    port = db_settings.PORT
    db_name = db_settings.NAME

    # Build connection URL with SSL requirement for Supabase
    # NOTE: asyncpg uses 'ssl=require' to enforce SSL connections
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}?ssl=require"

@asynccontextmanager
async def create_db_engine_context(
    db_settings: DBSettings
) -> AsyncGenerator[AsyncEngine, None]:
    """
    Async context manager for creating Supabase database engine and managing its lifecycle.
    Automatically disposes of the engine when exiting the context.
    """
    database_url = build_supabase_url(db_settings)

    # Create async engine with connection pooling
    engine = create_async_engine(
        database_url,
        pool_size=db_settings.POOL_SIZE,
        max_overflow=db_settings.MAX_OVERFLOW,
        pool_timeout=db_settings.POOL_TIMEOUT,
        pool_recycle=db_settings.POOL_RECYCLE,
        echo=False, # Set to True for SQL query logging during development
        pool_pre_ping=True, # Verify connections before using them
    )

    try:
        # Yield the engine for use within the 'async with' block in lifespan
        yield engine
    finally:
        # Cleanup: dispose of the engine and close all connections
        await engine.dispose()

def get_async_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Creates a session maker factory for the given async engine.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession, # explicitly declare type
        expire_on_commit=False,
        autoflush=False,
        autocommit=False
    )
