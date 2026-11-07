"""
lexuz.api.store

Хранилище результатов обработки батчей:
- Гарантирует идемпотентность (повторный запрос с тем же batch_id возвращает кэшированный результат)
- Детектирует конфликты (тот же batch_id, но другое тело запроса -> HTTP 409)
- Поддерживает async polling через GET /v1/rag/batch/{batch_id}
- Сохраняет историю в SQLite (срок хранения >= 30 дней)
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("lexuz.api.store")


class BatchConflictError(Exception):
    """Исключение при повторной отправке batch_id с другим содержимым."""
    pass


class BatchStore:
    """SQLite-хранилище батчей с поддержкой идемпотентности и истории."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            default_dir = Path("data")
            default_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(default_dir / "rag_batch_store.db")
        else:
            self.db_path = db_path

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batches_run_id ON batches(run_id);")
            conn.commit()

    @staticmethod
    def compute_request_hash(data: Dict[str, Any]) -> str:
        """Вычисляет детерминированный SHA256 хеш содержимого запроса."""
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def check_or_register_batch(
        self, batch_id: str, run_id: str, request_data: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Проверяет batch_id:
        - Если не существует: регистрирует со статусом 'processing' и возвращает ('new', None)
        - Если существует с тем же хешем:
            - если completed: возвращает ('completed', cached_response)
            - если processing: возвращает ('processing', None)
        - Если существует с другим хешем: поднимает BatchConflictError (HTTP 409)
        """
        req_hash = self.compute_request_hash(request_data)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT request_hash, status, response_json FROM batches WHERE batch_id = ?",
                (batch_id,),
            )
            row = cur.fetchone()

            if row is not None:
                stored_hash = row["request_hash"]
                stored_status = row["status"]
                stored_response = row["response_json"]

                if stored_hash != req_hash:
                    logger.warning(
                        "Batch conflict detected for batch_id=%s. Existing hash=%s, New hash=%s",
                        batch_id,
                        stored_hash,
                        req_hash,
                    )
                    raise BatchConflictError(
                        f"Batch '{batch_id}' already exists with different request contents."
                    )

                if stored_status == "completed" and stored_response:
                    return "completed", json.loads(stored_response)
                return stored_status, None

            # Новый батч - регистрируем со статусом processing
            cur.execute(
                """
                INSERT INTO batches (batch_id, run_id, request_hash, request_json, response_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, NULL, 'processing', ?, ?)
                """,
                (
                    batch_id,
                    run_id,
                    req_hash,
                    json.dumps(request_data, ensure_ascii=False),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
            return "new", None

    def save_completed_batch(self, batch_id: str, response_data: Dict[str, Any]) -> None:
        """Сохраняет финальный результат выполнения батча."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE batches
                SET status = 'completed',
                    response_json = ?,
                    updated_at = ?
                WHERE batch_id = ?
                """,
                (json.dumps(response_data, ensure_ascii=False), now_iso, batch_id),
            )
            conn.commit()

    def mark_batch_failed(self, batch_id: str, error_message: str) -> None:
        """Помечает батч как failed."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE batches
                SET status = 'failed',
                    response_json = ?,
                    updated_at = ?
                WHERE batch_id = ?
                """,
                (json.dumps({"error": error_message}), now_iso, batch_id),
            )
            conn.commit()

    def cleanup_old_batches(self, retention_days: int = 30) -> int:
        """
        Удаляет записи батчей старше retention_days (по умолчанию 30 дней) согласно разделу 5.
        Возвращает количество удаленных записей.
        """
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = cutoff_dt.isoformat()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM batches WHERE created_at < ?", (cutoff_iso,))
            deleted = cur.rowcount
            conn.commit()
            return deleted

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о батче по batch_id."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT batch_id, run_id, status, response_json, created_at, updated_at FROM batches WHERE batch_id = ?",
                (batch_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            result = {
                "batch_id": row["batch_id"],
                "run_id": row["run_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            if row["response_json"]:
                try:
                    result["response"] = json.loads(row["response_json"])
                except Exception:
                    result["response"] = None
            else:
                result["response"] = None
            return result

    def list_recent_batches(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Возвращает последние обработанные батчи для дашборда."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT batch_id, run_id, status, created_at, updated_at, response_json
                FROM batches
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            output = []
            for r in rows:
                item = {
                    "batch_id": r["batch_id"],
                    "run_id": r["run_id"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                if r["response_json"]:
                    try:
                        resp = json.loads(r["response_json"])
                        item["summary"] = resp.get("summary")
                        item["total_ms"] = resp.get("timing", {}).get("total_ms")
                    except Exception:
                        pass
                output.append(item)
            return output
