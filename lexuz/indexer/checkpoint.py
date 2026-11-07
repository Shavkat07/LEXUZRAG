"""
lexuz.indexer.checkpoint

Менеджер контрольных точек (Checkpointing) для конвейера векторизации.
Обеспечивает сохранение прогресса каждые N чанков и безопасный resume
в случае прерывания процесса (SIGINT / аварийное завершение).
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("lexuz.indexer.checkpoint")

DEFAULT_CHECKPOINT_FILE = Path("data/indexing_state.json")
DEFAULT_SAVE_INTERVAL = 1000


class CheckpointManager:
    """Управляет сохранением и восстановлением состояния индексации."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_FILE,
        save_interval: int = DEFAULT_SAVE_INTERVAL,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.save_interval = save_interval
        self.last_saved_count = 0
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Загружает сохраненное состояние из JSON файла, если он существует."""
        if not self.checkpoint_path.exists():
            return {
                "last_chunk_id": None,
                "processed_count": 0,
                "elapsed_seconds": 0.0,
                "updated_at": None,
            }

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(
                    "Loaded checkpoint from %s: last_chunk_id=%s, processed_count=%d",
                    self.checkpoint_path,
                    data.get("last_chunk_id"),
                    data.get("processed_count", 0),
                )
                self.last_saved_count = data.get("processed_count", 0)
                return data
        except Exception as e:
            logger.warning(
                "Failed to parse checkpoint file %s: %s. Starting from scratch.",
                self.checkpoint_path,
                e,
            )
            return {
                "last_chunk_id": None,
                "processed_count": 0,
                "elapsed_seconds": 0.0,
                "updated_at": None,
            }

    @property
    def last_chunk_id(self) -> Optional[str]:
        return self.state.get("last_chunk_id")

    @property
    def processed_count(self) -> int:
        return self.state.get("processed_count", 0)

    @property
    def elapsed_seconds(self) -> float:
        return self.state.get("elapsed_seconds", 0.0)

    def should_save(self, current_count: int) -> bool:
        """Проверяет, наступил ли интервал для сохранения."""
        return (current_count - self.last_saved_count) >= self.save_interval

    def save(
        self,
        last_chunk_id: str,
        processed_count: int,
        elapsed_seconds: float,
        force: bool = False,
    ) -> bool:
        """
        Атомарно сохраняет состояние в JSON файл.
        Если force=False, сохранение происходит только при достижении save_interval.
        """
        if not force and not self.should_save(processed_count):
            return False

        self.state = {
            "last_chunk_id": last_chunk_id,
            "processed_count": processed_count,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "updated_at": datetime.now().isoformat(),
        }

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Атомарная запись через временный файл в той же директории
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.checkpoint_path.parent,
            prefix="idx_ckpt_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.checkpoint_path)
            self.last_saved_count = processed_count
            logger.info(
                "Checkpoint saved: last_chunk_id=%s, processed=%d, elapsed=%.1fs",
                last_chunk_id,
                processed_count,
                elapsed_seconds,
            )
            return True
        except Exception as e:
            logger.error("Failed to write checkpoint atomically: %s", e)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return False

    def reset(self) -> None:
        """Сбрасывает сохраненное состояние."""
        self.state = {
            "last_chunk_id": None,
            "processed_count": 0,
            "elapsed_seconds": 0.0,
            "updated_at": None,
        }
        self.last_saved_count = 0
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
                logger.info("Checkpoint file %s removed on reset.", self.checkpoint_path)
            except Exception as e:
                logger.warning("Failed to delete checkpoint file: %s", e)
