# services/recommender_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple

DB_PATH = Path(__file__).resolve().parents[1] / "models" / "recommender.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            month INTEGER NOT NULL,
            price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            destination TEXT NOT NULL,
            value INTEGER NOT NULL,             -- +1 or -1
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, destination)
        );
        """)
        conn.commit()


def log_search(user_id: str, origin: str, destination: str, month: int, price: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO searches(user_id, origin, destination, month, price) VALUES (?, ?, ?, ?, ?)",
            (user_id, origin, destination, month, float(price))
        )
        conn.commit()


def get_searches(limit: int = 500) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT user_id, origin, destination, month, price FROM searches ORDER BY id DESC LIMIT ?",
            (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_feedback(user_id: str, destination: str, value: int) -> None:
    value = 1 if int(value) > 0 else -1
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO feedback(user_id, destination, value)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, destination)
            DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, destination, value)
        )
        conn.commit()


def get_feedback_summary(user_id: str) -> Dict[str, int]:
    """
    Returns dict destination -> value (+1 or -1)
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT destination, value FROM feedback WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return {r["destination"]: int(r["value"]) for r in rows}


def get_popularity() -> Dict[str, int]:
    """
    Returns dict destination -> count
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT destination, COUNT(*) AS c FROM searches GROUP BY destination"
        ).fetchall()
    return {r["destination"]: int(r["c"]) for r in rows}
