import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List


class OnlinePreferenceStore:
    """
    Thread-safe sqlite store for online preference pairs.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_group_key TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    chosen_text TEXT NOT NULL,
                    rejected_text TEXT NOT NULL,
                    score_gap REAL NOT NULL,
                    task TEXT,
                    part TEXT,
                    phase TEXT,
                    sample_id TEXT,
                    metadata_json TEXT,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    batch_tag TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_preferences_pending
                ON preferences(consumed, score_gap, created_at)
                """
            )
            conn.commit()

    def insert_many(self, pairs: Iterable[Dict[str, Any]]) -> int:
        rows = list(pairs)
        if not rows:
            return 0
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO preferences (
                    node_group_key, prompt, chosen_text, rejected_text, score_gap,
                    task, part, phase, sample_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(r.get("node_group_key", "")),
                        str(r.get("prompt", "")),
                        str(r.get("chosen", "")),
                        str(r.get("rejected", "")),
                        float(r.get("score_gap", 0.0)),
                        str(r.get("task", "")),
                        str(r.get("part", "")),
                        str(r.get("phase", "")),
                        str(r.get("sample_id", "")),
                        json.dumps(r.get("metadata", {}), ensure_ascii=False),
                        now,
                    )
                    for r in rows
                ],
            )
            conn.commit()
        return len(rows)

    def count_pending(self, min_score_gap: float = 0.0) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM preferences WHERE consumed = 0 AND score_gap >= ?",
                (float(min_score_gap),),
            ).fetchone()
        return int(row["c"]) if row else 0

    def fetch_pending(self, limit: int, min_score_gap: float = 0.0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM preferences
                WHERE consumed = 0 AND score_gap >= ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (float(min_score_gap), int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "node_group_key": row["node_group_key"],
                    "prompt": row["prompt"],
                    "chosen": row["chosen_text"],
                    "rejected": row["rejected_text"],
                    "score_gap": float(row["score_gap"]),
                    "task": row["task"],
                    "part": row["part"],
                    "phase": row["phase"],
                    "sample_id": row["sample_id"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "created_at": float(row["created_at"]),
                }
            )
        return out

    def mark_consumed(self, ids: List[int], batch_tag: str) -> int:
        if not ids:
            return 0
        placeholders = ",".join(["?"] * len(ids))
        params: List[Any] = [str(batch_tag)] + [int(x) for x in ids]
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE preferences SET consumed=1, batch_tag=? WHERE id IN ({placeholders})",
                tuple(params),
            )
            changed = conn.total_changes
            conn.commit()
        return int(changed)

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(1) AS c FROM preferences").fetchone()
            pending = conn.execute(
                "SELECT COUNT(1) AS c FROM preferences WHERE consumed = 0"
            ).fetchone()
            consumed = conn.execute(
                "SELECT COUNT(1) AS c FROM preferences WHERE consumed = 1"
            ).fetchone()
        return {
            "db_path": self.db_path,
            "total": int(total["c"]) if total else 0,
            "pending": int(pending["c"]) if pending else 0,
            "consumed": int(consumed["c"]) if consumed else 0,
        }
