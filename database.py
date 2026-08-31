from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Create the engine and its local connection pool.
#
# Neon may close an idle connection when its compute suspends.  pre_ping checks a
# connection immediately before handing it to a request and transparently
# replaces it when the underlying socket is no longer usable.  recycle is based
# on total connection age (not idle time) and is enforced on the next checkout;
# it therefore does not run a background task or keep the database awake.
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=120,
    pool_timeout=10,
    pool_use_lifo=True,
    connect_args={"connect_timeout": 10},
    echo=False  # Set True to see raw SQL in logs (useful for debugging)
)

# Session factory — like a DB transaction manager
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
class Base(DeclarativeBase):
    pass

def get_db():
    """
    Dependency injection for DB sessions.
    FastAPI calls this automatically for routes that need DB.
    Like middleware in Express.
    """
    db = SessionLocal()
    try:
        yield db  # 'yield' = give the session to the route
    finally:
        db.close()  # Always close, even if error occurs

def test_connection():
    """Call this on startup to verify DB is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
