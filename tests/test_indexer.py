"""
tests/test_indexer.py

Тесты для модулей индексации lexuz.indexer:
- CheckpointManager (атомарное сохранение, resume, сброс)
- Helper функции pipeline (stream_chunks, build_embedding_text, chunk_to_payload)
- Сквозной тест run_indexing с MockEmbedder и Qdrant
"""

import gzip
import json
import tempfile
from pathlib import Path
import pytest

from lexuz.indexer.checkpoint import CheckpointManager
from lexuz.indexer.embedder import MockEmbedder
from lexuz.indexer.pipeline import (
    build_embedding_text,
    chunk_to_payload,
    stream_chunks_from_files,
    run_indexing,
)
from lexuz.storage.qdrant_manager import QdrantManager

TEST_INDEX_COLLECTION = "test_indexing_collection"


def test_checkpoint_manager_lifecycle(tmp_path: Path):
    ckpt_file = tmp_path / "test_state.json"
    ckpt = CheckpointManager(checkpoint_path=ckpt_file, save_interval=10)

    # 1. Initial state
    assert ckpt.last_chunk_id is None
    assert ckpt.processed_count == 0
    assert ckpt.elapsed_seconds == 0.0

    # 2. Save interval check
    assert ckpt.should_save(5) is False
    assert ckpt.should_save(10) is True

    # 3. Save force
    saved = ckpt.save(last_chunk_id="chunk_1", processed_count=1, elapsed_seconds=2.5, force=True)
    assert saved is True
    assert ckpt_file.exists()

    # 4. Resume in new instance
    ckpt2 = CheckpointManager(checkpoint_path=ckpt_file, save_interval=10)
    assert ckpt2.last_chunk_id == "chunk_1"
    assert ckpt2.processed_count == 1
    assert ckpt2.elapsed_seconds == 2.5

    # 5. Reset
    ckpt2.reset()
    assert ckpt2.last_chunk_id is None
    assert not ckpt_file.exists()


def test_checkpoint_manager_corrupted_file(tmp_path: Path):
    ckpt_file = tmp_path / "corrupt_state.json"
    ckpt_file.write_text("{invalid_json: true", encoding="utf-8")

    ckpt = CheckpointManager(checkpoint_path=ckpt_file)
    assert ckpt.last_chunk_id is None
    assert ckpt.processed_count == 0


def test_build_embedding_text():
    chunk = {
        "doc_title": "O'zbekiston Respublikasi Konstitutsiyasi",
        "breadcrumb": ["1-bob", "1-modda"],
        "content": "O'zbekiston — suveren respublika.",
    }
    text = build_embedding_text(chunk)
    assert "1-bob > 1-modda" in text
    assert "O'zbekiston — suveren respublika." in text


def test_chunk_to_payload():
    chunk = {
        "chunk_id": "-3492199_uz_art_1",
        "doc_id": "-3492199",
        "lang": "uz",
        "doc_number": "ЗРУ-505",
        "doc_title": "Qonun",
        "archetype": "ARCHETYPE_A",
        "structural_unit": {"type": "article", "number": "1", "title": "Maqsad"},
        "breadcrumb": ["Bob 1", "Modda 1"],
        "content": "Matn mazmuni",
        "metadata": {
            "okoz_codes": ["01.00.00.00"],
            "has_table": False,
            "original_script": "lat",
            "anchors": ["-123"],
            "citations": [{"target_doc_id": "-999", "target_anchor": "1", "raw_ref_text": "ref"}],
        },
    }
    payload = chunk_to_payload(chunk)
    assert payload["chunk_id"] == "-3492199_uz_art_1"
    assert payload["doc_id"] == "-3492199"
    assert payload["unit_type"] == "article"
    assert payload["unit_number"] == "1"
    assert payload["citations_count"] == 1
    assert payload["has_table"] is False
    assert payload["original_script"] == "lat"


def test_run_indexing_end_to_end(tmp_path: Path):
    # 1. Create a dummy gzipped chunks file
    chunks_file = tmp_path / "sample_chunks.jsonl.gz"
    sample_data = []
    for i in range(1, 11):
        sample_data.append({
            "chunk_id": f"chunk_{i}",
            "doc_id": f"doc_{i}",
            "lang": "uz",
            "doc_number": f"N-{i}",
            "doc_title": f"Test Document {i}",
            "archetype": "ARCHETYPE_A",
            "structural_unit": {"type": "clause", "number": str(i), "title": f"Clause {i}"},
            "breadcrumb": [f"Test Document {i}", f"{i}-band"],
            "content": f"Mazkur qoidalar {i}-band talablarini belgilaydi.",
            "metadata": {
                "okoz_codes": ["01.01.00.00"],
                "has_table": False,
                "original_script": "lat",
                "anchors": [f"-{i}"],
                "citations": [],
            },
        })

    with gzip.open(chunks_file, "wt", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 2. Qdrant manager and mock embedder
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)
    embedder = MockEmbedder(dense_dim=1024)
    ckpt_file = tmp_path / "test_indexing_ckpt.json"

    # Clean collection
    if manager.collection_exists(TEST_INDEX_COLLECTION):
        manager.client.delete_collection(TEST_INDEX_COLLECTION)

    # 3. Run indexing with limit=6
    res1 = run_indexing(
        input_files=[chunks_file],
        collection_name=TEST_INDEX_COLLECTION,
        batch_size=3,
        recreate=True,
        checkpoint_path=ckpt_file,
        save_interval=2,
        limit=6,
        embedder=embedder,
        qdrant_manager=manager,
    )

    assert res1["processed_count"] == 6
    assert res1["session_indexed"] == 6
    count1 = manager.count(TEST_INDEX_COLLECTION)
    assert count1 == 6

    # 4. Resume indexing to completion (remaining 4 chunks)
    res2 = run_indexing(
        input_files=[chunks_file],
        collection_name=TEST_INDEX_COLLECTION,
        batch_size=3,
        recreate=False,
        checkpoint_path=ckpt_file,
        save_interval=2,
        limit=None,
        embedder=embedder,
        qdrant_manager=manager,
    )

    assert res2["processed_count"] == 10
    assert res2["session_indexed"] == 4
    count2 = manager.count(TEST_INDEX_COLLECTION)
    assert count2 == 10

    # 5. Test exact limit enforcement not multiple of batch_size (limit=7, batch_size=5)
    ckpt_limit_file = tmp_path / "test_limit_ckpt.json"
    res_limit = run_indexing(
        input_files=[chunks_file],
        collection_name=TEST_INDEX_COLLECTION,
        batch_size=5,
        recreate=True,
        checkpoint_path=ckpt_limit_file,
        limit=7,
        embedder=embedder,
        qdrant_manager=manager,
    )
    assert res_limit["processed_count"] == 7
    assert manager.count(TEST_INDEX_COLLECTION) == 7

    # 6. Test early return when already at or past limit
    res_noop = run_indexing(
        input_files=[chunks_file],
        collection_name=TEST_INDEX_COLLECTION,
        batch_size=5,
        recreate=False,
        checkpoint_path=ckpt_limit_file,
        limit=5,
        embedder=embedder,
        qdrant_manager=manager,
    )
    assert res_noop["session_indexed"] == 0

    # 7. Test resume failure when checkpoint chunk ID is missing in dataset
    corrupt_resume_ckpt = tmp_path / "test_missing_id_ckpt.json"
    corrupt_resume_ckpt.write_text(
        json.dumps({"last_chunk_id": "nonexistent_chunk_id_xyz", "processed_count": 5}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Checkpoint resume failed"):
        run_indexing(
            input_files=[chunks_file],
            collection_name=TEST_INDEX_COLLECTION,
            batch_size=3,
            recreate=False,
            checkpoint_path=corrupt_resume_ckpt,
            embedder=embedder,
            qdrant_manager=manager,
        )

    # Clean up
    manager.client.delete_collection(TEST_INDEX_COLLECTION)


def test_embedder_defaults_and_interface():
    from lexuz.indexer.embedder import BGEM3Embedder, DEFAULT_MAX_LENGTH, DEFAULT_ENCODE_BATCH_SIZE

    embedder = BGEM3Embedder(device="cpu")
    assert embedder.max_length == 1024
    assert embedder.max_length == DEFAULT_MAX_LENGTH
    assert embedder.batch_size == 32
    assert embedder.batch_size == DEFAULT_ENCODE_BATCH_SIZE

    # Test clear_cache does not fail on CPU or GPU
    embedder.clear_cache()

    # Empty inputs
    assert embedder.encode_documents([]) == {"dense": [], "sparse": []}
    assert embedder.encode_queries([]) == {"dense": [], "sparse": []}


def test_embedder_oom_recovery_simulation(monkeypatch):
    from lexuz.indexer.embedder import BGEM3Embedder
    import numpy as np

    embedder = BGEM3Embedder(device="cpu", batch_size=32)

    call_history = []

    class DummyModel:
        def encode(self, texts, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
            call_history.append(batch_size)
            if batch_size > 16:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.14 GiB.")
            # Successful retry with batch_size=16
            return {
                "dense_vecs": np.zeros((len(texts), 1024), dtype=np.float32),
                "lexical_weights": [{"123": 1.5} for _ in texts],
            }

    monkeypatch.setattr(embedder, "_model", DummyModel())

    texts = [f"Text norm {i}" for i in range(20)]
    res = embedder.encode_documents(texts, batch_size=32)

    assert len(res["dense"]) == 20
    assert len(res["sparse"]) == 20
    # First call had batch_size=32 (failed), second call had batch_size=16 (succeeded)
    assert call_history == [32, 16]
    # Verify embedder adapted internal batch_size
    assert embedder.batch_size == 16

    # Subsequent call should use adapted batch_size=16 directly without OOM
    call_history.clear()
    embedder.encode_documents(texts)
    assert call_history == [16]

    # Test recovery down to batch_size=1
    embedder2 = BGEM3Embedder(device="cpu", batch_size=8)
    history2 = []

    class TightMemoryModel:
        def encode(self, texts, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
            history2.append(batch_size)
            if batch_size > 1:
                raise RuntimeError("CUDA out of memory.")
            return {
                "dense_vecs": np.zeros((len(texts), 1024), dtype=np.float32),
                "lexical_weights": [{"1": 1.0} for _ in texts],
            }

    monkeypatch.setattr(embedder2, "_model", TightMemoryModel())
    res2 = embedder2.encode_documents(["Single chunk text"])
    assert len(res2["dense"]) == 1
    assert history2 == [8, 4, 2, 1]
    assert embedder2.batch_size == 1

    # Test encode_queries OOM recovery
    embedder_q = BGEM3Embedder(device="cpu", batch_size=16)
    q_history = []

    class QueryOOMModel:
        def encode(self, queries, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
            q_history.append(batch_size)
            if batch_size > 8:
                raise RuntimeError("CUDA out of memory.")
            return {
                "dense_vecs": np.zeros((len(queries), 1024), dtype=np.float32),
                "lexical_weights": [{"2": 2.0} for _ in queries],
            }

    monkeypatch.setattr(embedder_q, "_model", QueryOOMModel())
    queries_list = [f"query {i}" for i in range(20)]
    res_q = embedder_q.encode_queries(queries_list)
    assert len(res_q["dense"]) == 20
    assert q_history == [16, 8]
    assert embedder_q.batch_size == 8


def test_reranker_oom_recovery_simulation(monkeypatch):
    from lexuz.retrieval.reranker import BGEReranker
    from lexuz.retrieval.models import SearchResult

    reranker = BGEReranker(device="cpu")
    rerank_history = []

    class DummyRerankModel:
        def compute_score(self, pairs, batch_size, max_length, normalize):
            rerank_history.append(batch_size)
            if batch_size > 16:
                raise RuntimeError("CUDA out of memory.")
            return [0.9] * len(pairs)

    monkeypatch.setattr(reranker, "_model", DummyRerankModel())

    dummy_results = [
        SearchResult(
            point_id=f"p_{i}",
            chunk_id=f"c_{i}",
            score=0.95,
            doc_id=f"d_{i}",
            lang="uz",
            doc_number=f"N-{i}",
            doc_title=f"Doc {i}",
            archetype="ARCHETYPE_A",
            unit_type="clause",
            unit_number=str(i),
            unit_title=f"C {i}",
            breadcrumb=["Law"],
            okoz_codes=[],
            has_table=False,
            original_script="lat",
            anchors=[],
            citations_count=0,
            content=f"Content {i}",
        )
        for i in range(20)
    ]

    res = reranker.rerank(query="soliq", results=dummy_results, top_k=5, batch_size=32)
    assert len(res) == 5
    assert rerank_history == [32, 16]


def test_pipeline_emergency_checkpoint_on_failure(tmp_path: Path):
    chunks_file = tmp_path / "failing_chunks.jsonl.gz"
    sample_data = [
        {
            "chunk_id": f"chunk_{i}",
            "doc_id": f"doc_{i}",
            "lang": "uz",
            "doc_number": f"N-{i}",
            "doc_title": f"Doc {i}",
            "archetype": "ARCHETYPE_A",
            "structural_unit": {"type": "clause", "number": str(i), "title": f"C {i}"},
            "breadcrumb": [f"Doc {i}"],
            "content": f"Clause {i} text",
            "metadata": {"okoz_codes": [], "has_table": False, "original_script": "lat", "anchors": [], "citations": []},
        }
        for i in range(1, 11)
    ]
    with gzip.open(chunks_file, "wt", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")

    ckpt_file = tmp_path / "emergency_ckpt.json"
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)

    # Custom embedder that fails on the second batch
    class FailsOnSecondBatchEmbedder(MockEmbedder):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def encode_documents(self, texts, batch_size=None):
            self.call_count += 1
            if self.call_count >= 2:
                raise RuntimeError("Simulated crash in batch 2")
            return super().encode_documents(texts, batch_size)

    failing_embedder = FailsOnSecondBatchEmbedder()

    # batch_size=3 -> Batch 1 processes chunks 1..3, Batch 2 fails on chunks 4..6
    with pytest.raises(RuntimeError, match="Simulated crash in batch 2"):
        run_indexing(
            input_files=[chunks_file],
            collection_name="test_emergency_col",
            batch_size=3,
            recreate=True,
            checkpoint_path=ckpt_file,
            save_interval=1000,  # save_interval is large, would not save normally
            embedder=failing_embedder,
            qdrant_manager=manager,
        )

    # Verify emergency checkpoint was saved for Batch 1 (chunk_3, processed_count=3)
    assert ckpt_file.exists()
    ckpt_data = json.loads(ckpt_file.read_text(encoding="utf-8"))
    assert ckpt_data["last_chunk_id"] == "chunk_3"
    assert ckpt_data["processed_count"] == 3

    # Clean up
    if manager.collection_exists("test_emergency_col"):
        manager.client.delete_collection("test_emergency_col")

