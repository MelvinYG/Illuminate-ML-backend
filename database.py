from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine — like creating a DB connection pool
# pool_size=5 means 5 concurrent connections max
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
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