"""
lexuz.indexer.pipeline

Конвейер пакетной векторизации и заливки правового корпуса в Qdrant:
- Потоковое чтение compressed JSONL.GZ без распаковки на диск
- Динамический батчинг (64 - 128 чанков)
- Генерация Dense + Sparse представлений через BGEM3
- Отказоустойчивое сохранение прогресса (CheckpointManager)
- Безопасная обработка прерываний (SIGINT / Ctrl+C)
"""

import gzip
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from qdrant_client import models
from tqdm import tqdm

from lexuz.indexer.checkpoint import (
    CheckpointManager,
    DEFAULT_CHECKPOINT_FILE,
    DEFAULT_SAVE_INTERVAL,
)
from lexuz.indexer.embedder import BGEM3Embedder, MockEmbedder
from lexuz.storage.qdrant_manager import (
    QdrantManager,
    chunk_id_to_uuid,
    DEFAULT_COLLECTION_NAME,
)

logger = logging.getLogger("lexuz.indexer.pipeline")

DEFAULT_BATCH_SIZE = 64

DEFAULT_INPUT_FILES = [
    Path("data/chunks_archetype_a.jsonl.gz"),
    Path("data/chunks_archetype_b.jsonl.gz"),
]


def stream_chunks_from_files(
    files: List[Path],
) -> Generator[Dict[str, Any], None, None]:
    """
    Потоковый генератор чанков из списка файлов (.jsonl или .jsonl.gz).
    Не загружает весь датасет в оперативную память.
    """
    for file_path in files:
        if not file_path.exists():
            logger.warning("File %s not found, skipping.", file_path)
            continue

        logger.info("Streaming chunks from %s...", file_path)
        is_gz = file_path.suffix == ".gz" or str(file_path).endswith(".jsonl.gz")
        open_fn = gzip.open if is_gz else open

        with open_fn(file_path, "rt", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError as e:
                    logger.warning("JSON decode error in %s at line %d: %s", file_path, line_idx, e)


def build_embedding_text(chunk: Dict[str, Any]) -> str:
    """
    Формирует контекстно-обогащенный текст для векторизации:
    [Хлебные крошки и заголовок] + [Текст нормы/статьи].
    """
    breadcrumbs = chunk.get("breadcrumb", [])
    content = chunk.get("content", "")

    header_parts = []
    if breadcrumbs:
        header_parts.append(" > ".join(breadcrumbs))

    if header_parts:
        header_str = "\n".join(header_parts)
        return f"{header_str}\n\n{content}".strip()
    return content.strip()


def chunk_to_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Преобразует запись чанка в словарь полезной нагрузки (payload) Qdrant."""
    metadata = chunk.get("metadata", {})
    structural_unit = chunk.get("structural_unit", {})

    citations = metadata.get("citations", [])
    citations_count = len(citations) if isinstance(citations, list) else 0

    arch = str(chunk.get("archetype", "")).strip().upper()
    if arch and not arch.startswith("ARCHETYPE_"):
        arch = f"ARCHETYPE_{arch}"

    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": str(chunk.get("doc_id", "")),
        "lang": chunk.get("lang", "uz"),
        "doc_number": chunk.get("doc_number", ""),
        "doc_title": chunk.get("doc_title", ""),
        "archetype": arch,
        "unit_type": structural_unit.get("type", ""),
        "unit_number": str(structural_unit.get("number", "")),
        "unit_title": structural_unit.get("title", ""),
        "breadcrumb": chunk.get("breadcrumb", []),
        "okoz_codes": metadata.get("okoz_codes", []),
        "has_table": bool(metadata.get("has_table", False)),
        "original_script": metadata.get("original_script", "lat"),
        "anchors": metadata.get("anchors", []),
        "citations_count": citations_count,
        "content": chunk.get("content", ""),
    }


def run_indexing(
    input_files: Optional[List[Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    recreate: bool = False,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_FILE,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    limit: Optional[int] = None,
    embedder: Optional[Any] = None,
    qdrant_manager: Optional[QdrantManager] = None,
    max_length: Optional[int] = None,
    encode_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Основной пайплайн пакетной индексации корпуса документов в Qdrant.
    """
    if input_files is None:
        input_files = DEFAULT_INPUT_FILES

    # 1. Хранилище Qdrant
    if qdrant_manager is None:
        qdrant_manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)

    if not qdrant_manager.healthcheck():
        raise RuntimeError("Qdrant healthcheck failed! Ensure Qdrant is running.")

    # 2. Инициализация чекпоинта
    checkpoint = CheckpointManager(
        checkpoint_path=checkpoint_path,
        save_interval=save_interval,
    )

    if recreate:
        logger.info("Recreating collection '%s' and clearing checkpoint...", collection_name)
        checkpoint.reset()
        qdrant_manager.create_collection(collection_name=collection_name, recreate=True)
    else:
        qdrant_manager.create_collection(collection_name=collection_name, recreate=False)

    # 3. Векторизатор
    if embedder is None:
        embedder_kwargs: Dict[str, Any] = {"device": device, "use_fp16": True}
        if max_length is not None:
            embedder_kwargs["max_length"] = max_length
        if encode_batch_size is not None:
            embedder_kwargs["batch_size"] = encode_batch_size
        embedder = BGEM3Embedder(**embedder_kwargs)

    # 4. Обработка сигналов прерывания
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        logger.warning("Interrupt signal received (%s). Gracefully stopping...", signum)
        interrupted = True

    prev_sigint = signal.signal(signal.SIGINT, handle_interrupt)
    prev_sigterm = signal.signal(signal.SIGTERM, handle_interrupt)

    start_time = time.time()
    last_chunk_id = checkpoint.last_chunk_id
    processed_count = checkpoint.processed_count
    accumulated_time = checkpoint.elapsed_seconds

    if limit is not None and processed_count >= limit:
        logger.info(
            "Already reached limit of %d chunks (processed=%d). Nothing to index.",
            limit,
            processed_count,
        )
        return {
            "processed_count": processed_count,
            "session_indexed": 0,
            "elapsed_seconds": round(accumulated_time, 2),
            "speed_chunks_per_sec": 0.0,
            "interrupted": False,
        }

    resuming = last_chunk_id is not None
    if resuming:
        logger.info(
            "Resuming indexing from checkpoint: skipping until chunk_id='%s' (already processed %d)",
            last_chunk_id,
            processed_count,
        )

    # 5. Итерация по потоку чанков
    current_batch: List[Dict[str, Any]] = []
    matched_resume_point = not resuming
    batch_index = 0
    total_indexed_in_session = 0
    last_processed_chunk_id: Optional[str] = last_chunk_id
    success = False

    try:
        chunk_stream = stream_chunks_from_files(input_files)

        with tqdm(
            desc="Indexing to Qdrant",
            initial=processed_count,
            unit="chunks",
            dynamic_ncols=True,
        ) as pbar:
            for chunk in chunk_stream:
                if interrupted:
                    break

                cid = chunk.get("chunk_id", "")
                if not matched_resume_point:
                    if cid == last_chunk_id:
                        matched_resume_point = True
                    continue

                current_batch.append(chunk)

                reached_limit = (
                    limit is not None and (processed_count + len(current_batch)) >= limit
                )
                should_flush = (len(current_batch) >= batch_size) or reached_limit

                if should_flush:
                    _process_batch(
                        batch=current_batch,
                        embedder=embedder,
                        qdrant_manager=qdrant_manager,
                        collection_name=collection_name,
                        wait=True,
                    )
                    processed_count += len(current_batch)
                    total_indexed_in_session += len(current_batch)
                    pbar.update(len(current_batch))

                    last_processed_chunk_id = current_batch[-1].get("chunk_id", "")
                    elapsed_now = accumulated_time + (time.time() - start_time)
                    checkpoint.save(
                        last_chunk_id=last_processed_chunk_id,
                        processed_count=processed_count,
                        elapsed_seconds=elapsed_now,
                        force=reached_limit,
                    )

                    current_batch = []
                    batch_index += 1

                    if reached_limit:
                        logger.info("Reached limit of %d chunks. Stopping.", limit)
                        break

            # Проверка, был ли найден чекпоинт при resume
            if resuming and not matched_resume_point and not interrupted:
                raise RuntimeError(
                    f"Checkpoint resume failed: chunk_id '{last_chunk_id}' was not found in the input files. "
                    f"The input data may have changed. Use --recreate to restart indexing from scratch."
                )

            # Финальный остаток батча
            if current_batch and not interrupted:
                if limit is not None:
                    max_take = max(0, limit - processed_count)
                    current_batch = current_batch[:max_take]
                if current_batch:
                    _process_batch(
                        batch=current_batch,
                        embedder=embedder,
                        qdrant_manager=qdrant_manager,
                        collection_name=collection_name,
                        wait=True,
                    )
                    processed_count += len(current_batch)
                    total_indexed_in_session += len(current_batch)
                    pbar.update(len(current_batch))
                    last_processed_chunk_id = current_batch[-1].get("chunk_id", "")

            if last_processed_chunk_id:
                elapsed_now = accumulated_time + (time.time() - start_time)
                checkpoint.save(
                    last_chunk_id=last_processed_chunk_id,
                    processed_count=processed_count,
                    elapsed_seconds=elapsed_now,
                    force=True,
                )
        success = True
    except Exception as e:
        logger.error("Indexing failed with exception: %s", e)
        # Emergency checkpoint save for work completed before the failing batch
        if last_processed_chunk_id:
            elapsed_now = accumulated_time + (time.time() - start_time)
            checkpoint.save(
                last_chunk_id=last_processed_chunk_id,
                processed_count=processed_count,
                elapsed_seconds=elapsed_now,
                force=True,
            )
            logger.info(
                "Progress checkpoint saved at chunk_id='%s' (processed=%d) before exiting.",
                last_processed_chunk_id,
                processed_count,
            )
        raise
    finally:
        # Восстановление сигналов
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

        total_elapsed = accumulated_time + (time.time() - start_time)
        session_elapsed = max(time.time() - start_time, 0.001)
        speed = total_indexed_in_session / session_elapsed

        if interrupted:
            logger.warning("Pipeline safely stopped after interrupt.")
            if last_processed_chunk_id:
                elapsed_now = accumulated_time + (time.time() - start_time)
                checkpoint.save(
                    last_chunk_id=last_processed_chunk_id,
                    processed_count=processed_count,
                    elapsed_seconds=elapsed_now,
                    force=True,
                )
        elif success:
            logger.info("Indexing finished successfully.")
        else:
            logger.error("Indexing stopped due to error.")
            if last_processed_chunk_id:
                elapsed_now = accumulated_time + (time.time() - start_time)
                checkpoint.save(
                    last_chunk_id=last_processed_chunk_id,
                    processed_count=processed_count,
                    elapsed_seconds=elapsed_now,
                    force=True,
                )

        logger.info(
            "Summary: total_processed=%d, session_indexed=%d, time=%.2fs, speed=%.1f chunks/s",
            processed_count,
            total_indexed_in_session,
            session_elapsed,
            speed,
        )

    return {
        "processed_count": processed_count,
        "session_indexed": total_indexed_in_session,
        "elapsed_seconds": round(total_elapsed, 2),
        "speed_chunks_per_sec": round(speed, 2),
        "interrupted": interrupted,
    }


def _process_batch(
    batch: List[Dict[str, Any]],
    embedder: Any,
    qdrant_manager: QdrantManager,
    collection_name: str,
    wait: bool = True,
) -> None:
    """Векторизует батч и вставляет точки в Qdrant."""
    texts = [build_embedding_text(c) for c in batch]
    embeddings = embedder.encode_documents(texts)

    dense_vectors = embeddings["dense"]
    sparse_vectors = embeddings["sparse"]

    points: List[models.PointStruct] = []
    for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
        cid = chunk["chunk_id"]
        pid = chunk_id_to_uuid(cid)
        payload = chunk_to_payload(chunk)

        point = models.PointStruct(
            id=pid,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec,
            },
            payload=payload,
        )
        points.append(point)

    qdrant_manager.upsert_points(points=points, collection_name=collection_name, wait=wait)

    if hasattr(embedder, "clear_cache"):
        embedder.clear_cache()
