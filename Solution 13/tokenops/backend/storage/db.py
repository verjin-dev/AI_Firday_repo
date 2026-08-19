"""SQLAlchemy engine/session plus a fast bulk-insert path for the simulator."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings
from backend.storage.models import Base

_engine = None
_SessionLocal = None


def _resolve_url(url: str) -> str:
    if url.startswith("sqlite:///./"):
        path = Path(url.replace("sqlite:///./", "")).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"
    return url


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        s = get_settings()
        # the live engine writes from a background thread, so SQLite must
        # not enforce same-thread access
        _engine = create_engine(
            _resolve_url(s.SQLITE_URL), future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(_engine)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def bulk_insert(table: str, rows: Iterable[Dict[str, Any]], chunk: int = 5000) -> int:
    """Core-level bulk insert. 300k ledger rows in a few seconds."""
    engine = get_engine()
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    stmt = text(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"
    )
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk):
            conn.execute(stmt, rows[i : i + chunk])
    return len(rows)


def query_df(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Read helper — every analytics module goes through this."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {})


def reset_database() -> None:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def table_counts() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in ["llm_calls", "outcomes", "alerts", "router_state"]:
        try:
            out[t] = int(query_df(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0])
        except Exception:
            out[t] = 0
    return out


def has_data() -> bool:
    return table_counts().get("llm_calls", 0) > 0


__all__ = [
    "get_engine",
    "session_scope",
    "bulk_insert",
    "query_df",
    "reset_database",
    "table_counts",
    "has_data",
    "List",
]
