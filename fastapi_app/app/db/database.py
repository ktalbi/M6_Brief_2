import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

def _sqlite_url(path: str) -> str:
    # SQLite URL needs 3 slashes for absolute paths: sqlite:////abs/path.db
    if path.startswith("/"):
        return f"sqlite:////{path.lstrip('/')}"
    return f"sqlite:///{path}"

DB_PATH = os.getenv("FEEDBACK_DB_PATH", "/app/data/feedback.db")
DATABASE_URL = os.getenv("DATABASE_URL", _sqlite_url(DB_PATH))

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

class Base(DeclarativeBase):
    pass
