"""
tests/test_storage.py

Тесты для модуля lexuz.storage.qdrant_manager:
- Проверка генерации UUID v5
- Healthcheck Qdrant
- Создание гибридной коллекции с dense + sparse + int8-квантованием
- Проверка создания payload-индексов
- Вставка и валидация тестовых точек
"""

import uuid
import pytest
from qdrant_client import models

from lexuz.storage.qdrant_manager import (
    QdrantManager,
    chunk_id_to_uuid,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEXED_FIELDS,
)

TEST_COLLECTION = "test_lexuz_collection"


def test_chunk_id_to_uuid():
    cid = "-3492199_uz_art_1"
    uid = chunk_id_to_uuid(cid)
    assert isinstance(uid, str)
    # Validate UUID syntax
    parsed = uuid.UUID(uid)
    assert parsed.version == 5
    # Determinism
    assert chunk_id_to_uuid(cid) == uid
    # Uniqueness
    assert chunk_id_to_uuid("-3492199_uz_art_2") != uid


def test_qdrant_healthcheck():
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)
    assert manager.healthcheck() is True


def test_create_collection_and_indexes():
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)
    
    # Clean up before test
    if manager.collection_exists(TEST_COLLECTION):
        manager.client.delete_collection(TEST_COLLECTION)

    # 1. Create collection
    created = manager.create_collection(
        collection_name=TEST_COLLECTION,
        recreate=True,
        dense_dim=1024,
    )
    assert created is True
    assert manager.collection_exists(TEST_COLLECTION) is True

    # 2. Verify collection metadata & configs
    info = manager.get_collection_info(TEST_COLLECTION)
    vectors_cfg = info.config.params.vectors
    assert "dense" in vectors_cfg
    assert vectors_cfg["dense"].size == 1024
    assert vectors_cfg["dense"].distance == models.Distance.COSINE
    assert vectors_cfg["dense"].on_disk is True

    sparse_cfg = info.config.params.sparse_vectors
    assert "sparse" in sparse_cfg
    assert sparse_cfg["sparse"].modifier == models.Modifier.IDF

    quant_cfg = info.config.quantization_config
    assert quant_cfg is not None
    assert quant_cfg.scalar.type == models.ScalarType.INT8
    assert quant_cfg.scalar.always_ram is True

    # 3. Verify payload indexes
    payload_schema = info.payload_schema
    for field in DEFAULT_INDEXED_FIELDS:
        assert field in payload_schema
        assert payload_schema[field].data_type == models.PayloadSchemaType.KEYWORD

    # 4. Upsert a point
    point_id = chunk_id_to_uuid("-3492199_uz_art_1")
    point = models.PointStruct(
        id=point_id,
        vector={
            "dense": [0.01] * 1024,
            "sparse": models.SparseVector(
                indices=[10, 25, 100],
                values=[0.5, 0.8, 0.3],
            ),
        },
        payload={
            "chunk_id": "-3492199_uz_art_1",
            "doc_id": "-3492199",
            "lang": "uz",
            "unit_type": "article",
            "doc_number": "ЗРУ-505",
            "okoz_codes": ["01.03.00.00"],
            "archetype": "ARCHETYPE_A",
            "content": "Mazkur Qonunning maqsadi...",
        },
    )

    manager.upsert_points([point], collection_name=TEST_COLLECTION, wait=True)
    count = manager.count(collection_name=TEST_COLLECTION)
    assert count == 1

    # Clean up
    manager.client.delete_collection(TEST_COLLECTION)
