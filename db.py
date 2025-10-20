import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "notes.db")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              character_id TEXT NOT NULL,
              character_name TEXT NOT NULL,
              content TEXT NOT NULL,
              embedding BLOB,
              created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_notes_character ON notes(character_id);"
        )


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def add_note(character_id: str, character_name: str, content: str, embedding: Optional[bytes] = None) -> int:
    created_at = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes(character_id, character_name, content, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            (character_id, character_name, content, embedding, created_at),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_notes_by_character(character_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, content, created_at FROM notes WHERE character_id = ? ORDER BY id DESC",
            (character_id,),
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "content": r[1], "created_at": r[2]}
            for r in rows
        ]


def list_all_notes() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, character_id, character_name, content, created_at FROM notes ORDER BY id DESC"
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "character_id": r[1],
                "character_name": r[2],
                "content": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
