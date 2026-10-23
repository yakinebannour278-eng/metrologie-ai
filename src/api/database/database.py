"""
database/database.py
=====================
SQLite persistence for the LOCAL PROTOTYPE ONLY. This is prediction
history for the metrologist's own reference while demoing the MVP.

It is explicitly NOT the production database: the target architecture is
SQL Server behind the .NET backend (see README.md). Do not extend this to
store anything beyond local prediction history, and do not treat it as a
source of truth.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.api.core.config import get_settings

settings = get_settings()
DATABASE_URL = f"sqlite:///{settings.database_path.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from src.api.database import models  # noqa: F401  (register models on Base)

    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
