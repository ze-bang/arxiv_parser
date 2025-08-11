import sqlite3
from pathlib import Path
from typing import Iterable

from .config import config


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS subscribers (
    email TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS seen_preprints (
    arxiv_id TEXT PRIMARY KEY,
    published DATE,
    title TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn() -> sqlite3.Connection:
    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


def add_subscriber(email: str) -> bool:
    with get_conn() as conn:
        try:
            conn.execute("INSERT INTO subscribers(email) VALUES (?)", (email,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def remove_subscriber(email: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM subscribers WHERE email = ?", (email,))
        conn.commit()
        return cur.rowcount > 0


def list_subscribers() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT email FROM subscribers ORDER BY email").fetchall()
        return [r[0] for r in rows]


def mark_seen(arxiv_ids: Iterable[str], title_by_id: dict[str, str], published_by_id: dict[str, str]) -> None:
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_preprints(arxiv_id, published, title) VALUES (?, ?, ?)",
            [(aid, published_by_id.get(aid), title_by_id.get(aid)) for aid in arxiv_ids],
        )
        conn.commit()


def filter_unseen(arxiv_ids: Iterable[str]) -> list[str]:
    ids = list(arxiv_ids)
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT arxiv_id FROM seen_preprints WHERE arxiv_id IN ({placeholders})",
            ids,
        ).fetchall()
    seen = {r[0] for r in rows}
    return [aid for aid in ids if aid not in seen]
