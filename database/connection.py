"""
SQL Server Database Connection Module

Provides async SQLAlchemy engine and session for SQL Server 2022.
Uses aioodbc driver for async operations.
"""

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import sys
sys.path.append('..')
from config import (
    MSSQL_HOST, 
    MSSQL_PORT, 
    MSSQL_USER, 
    MSSQL_PASSWORD, 
    MSSQL_DATABASE
)

# Connection string for SQL Server with aioodbc
# aioodbc requires DSN-style connection string passed via connect_kwargs
# Format uses SERVER=host,port (comma separated, not colon)
CONNECTION_STRING = (
    f"mssql+aioodbc:///?odbc_connect="
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={MSSQL_HOST},{MSSQL_PORT};"
    f"DATABASE={MSSQL_DATABASE};"
    f"UID={MSSQL_USER};"
    f"PWD={MSSQL_PASSWORD};"
    f"TrustServerCertificate=yes;"
    f"Connection Timeout=30"
)

# Singleton engine instance
_engine: AsyncEngine = None


def get_async_engine() -> AsyncEngine:
    """
    Get or create async SQLAlchemy engine.
    
    Uses connection pooling for better performance.
    
    Returns:
        AsyncEngine: SQLAlchemy async engine
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            CONNECTION_STRING,
            echo=False,  # Set True for SQL debugging
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections after 30 minutes
        )
        print(f"🔌 SQL Server engine created: {MSSQL_HOST}:{MSSQL_PORT}/{MSSQL_DATABASE}")
    return _engine


# Async session factory
_async_session_factory = None


def get_session_factory():
    """Get async session factory (creates if not exists)."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database session.
    
    Usage in FastAPI:
        @app.get("/")
        async def endpoint(session: AsyncSession = Depends(get_async_session)):
            ...
    
    Yields:
        AsyncSession: Database session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful
    """
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.fetchone()
            if row and row[0] == 1:
                print("✅ SQL Server connection successful!")
                return True
    except Exception as e:
        print(f"❌ SQL Server connection failed: {e}")
        return False
    return False


async def init_database():
    """
    Initialize database - create tables if not exist.
    
    Should be called on application startup.
    """
    from .models import Base
    
    engine = get_async_engine()
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables initialized")


async def close_database():
    """
    Close database engine.
    
    Should be called on application shutdown.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        print("🔌 SQL Server connection closed")
