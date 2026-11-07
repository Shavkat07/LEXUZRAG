"""
lexuz.indexer package

Модуль векторизации и пакетной заливки чанков в Qdrant:
- BGEM3Embedder (dense + sparse embedding)
- CheckpointManager (отказоустойчивое сохранение прогресса)
- Ingestion Pipeline
"""

from lexuz.indexer.embedder import BGEM3Embedder, MockEmbedder
from lexuz.indexer.checkpoint import CheckpointManager
from lexuz.indexer.pipeline import run_indexing

__all__ = [
    "BGEM3Embedder",
    "MockEmbedder",
    "CheckpointManager",
    "run_indexing",
]
