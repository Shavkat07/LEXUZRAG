"""
tests/test_retrieval.py

Тесты для поискового движка lexuz.retrieval:
- Построение фильтров Qdrant (HybridRetriever.build_filter)
- Гибридный поиск по MockEmbedder (Dense + Sparse + RRF)
- Фильтрация по метаданным (lang, doc_id, okoz_code, unit_type)
- Реранкинг результатов (MockReranker)
- Сборка контекста родительского документа (ParentDocumentContextBuilder)
- Сквозной вызов LegalSearchService
"""

import pytest
from qdrant_client import models

from lexuz.indexer.embedder import MockEmbedder
from lexuz.retrieval.context_builder import ParentDocumentContextBuilder
from lexuz.retrieval.hybrid import HybridRetriever
from lexuz.retrieval.models import SearchResult
from lexuz.retrieval.reranker import MockReranker
from lexuz.retrieval.service import LegalSearchService
from lexuz.storage.qdrant_manager import QdrantManager, chunk_id_to_uuid

TEST_RETRIEVAL_COLLECTION = "test_retrieval_collection"


@pytest.fixture(scope="module")
def populated_collection():
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)
    embedder = MockEmbedder(dense_dim=1024)

    # Recreate collection
    manager.create_collection(TEST_RETRIEVAL_COLLECTION, recreate=True)

    # Insert 3 diverse points
    points_data = [
        {
            "chunk_id": "-1001_uz_art_1",
            "doc_id": "-1001",
            "lang": "uz",
            "doc_number": "ЗРУ-100",
            "doc_title": "Mehnat kodeksi",
            "archetype": "ARCHETYPE_A",
            "unit_type": "article",
            "unit_number": "1",
            "unit_title": "Mehnat to'g'risidagi qonunchilik",
            "breadcrumb": ["Mehnat kodeksi", "1-bob", "1-modda"],
            "okoz_codes": ["05.01.00.00"],
            "has_table": False,
            "original_script": "lat",
            "anchors": ["-5001"],
            "citations_count": 0,
            "content": "Mehnat to'g'risidagi qonunchilikning asosiy prinsiplari mehnat qilish erkinligi.",
        },
        {
            "chunk_id": "-1001_uz_art_2",
            "doc_id": "-1001",
            "lang": "uz",
            "doc_number": "ЗРУ-100",
            "doc_title": "Mehnat kodeksi",
            "archetype": "ARCHETYPE_A",
            "unit_type": "article",
            "unit_number": "2",
            "unit_title": "Mehnat munosabatlari",
            "breadcrumb": ["Mehnat kodeksi", "1-bob", "2-modda"],
            "okoz_codes": ["05.01.00.00"],
            "has_table": False,
            "original_script": "lat",
            "anchors": ["-5002"],
            "citations_count": 1,
            "content": "Mehnat munosabatlari xodim bilan ish beruvchi o'rtasidagi kelishuv asosida yuzaga keladi.",
        },
        {
            "chunk_id": "2002_ru_clause_1",
            "doc_id": "2002",
            "lang": "ru",
            "doc_number": "ПП-200",
            "doc_title": "О мерах по развитию малого бизнеса",
            "archetype": "ARCHETYPE_A",
            "unit_type": "clause",
            "unit_number": "1",
            "unit_title": "",
            "breadcrumb": ["О мерах по развитию малого бизнеса", "Пункт 1"],
            "okoz_codes": ["09.00.00.00"],
            "has_table": True,
            "original_script": "cyr",
            "anchors": ["6001"],
            "citations_count": 0,
            "content": "Установить льготный порядок налогообложения для субъектов предпринимательства.",
        },
    ]

    points: list[models.PointStruct] = []
    for p in points_data:
        text = f"{' > '.join(p['breadcrumb'])}\n\n{p['content']}"
        enc = embedder.encode_documents([text])
        point = models.PointStruct(
            id=chunk_id_to_uuid(p["chunk_id"]),
            vector={
                "dense": enc["dense"][0],
                "sparse": enc["sparse"][0],
            },
            payload=p,
        )
        points.append(point)

    manager.upsert_points(points, collection_name=TEST_RETRIEVAL_COLLECTION, wait=True)

    yield manager, embedder

    # Cleanup
    manager.client.delete_collection(TEST_RETRIEVAL_COLLECTION)


def test_build_filter():
    retriever = HybridRetriever()
    assert retriever.build_filter() is None

    flt = retriever.build_filter(
        lang="uz",
        doc_id="-1001",
        okoz_code="05.01.00.00",
        unit_type="article",
        archetype="a",
    )
    assert flt is not None
    assert len(flt.must) == 5
    # Verify doc_id variants
    doc_cond = flt.must[1]
    assert doc_cond.key == "doc_id"
    assert set(doc_cond.match.any) == {"-1001", "1001"}
    # Verify archetype normalized
    arch_cond = flt.must[4]
    assert arch_cond.key == "archetype"
    assert arch_cond.match.value == "ARCHETYPE_A"


def test_hybrid_search(populated_collection):
    manager, embedder = populated_collection
    retriever = HybridRetriever(
        qdrant_manager=manager,
        embedder=embedder,
        collection_name=TEST_RETRIEVAL_COLLECTION,
    )

    # Empty query
    assert retriever.search("") == []

    # Search for mehnat
    results = retriever.search("mehnat qonunchilik", limit=10)
    assert len(results) > 0
    first = results[0]
    assert isinstance(first, SearchResult)
    assert first.chunk_id in ["-1001_uz_art_1", "-1001_uz_art_2"]
    assert first.doc_id == "-1001"
    assert "Mehnat kodeksi" in first.doc_title


def test_hybrid_search_with_filters(populated_collection):
    manager, embedder = populated_collection
    retriever = HybridRetriever(
        qdrant_manager=manager,
        embedder=embedder,
        collection_name=TEST_RETRIEVAL_COLLECTION,
    )

    # Filter by lang='ru'
    ru_results = retriever.search("бизнес", limit=10, lang="ru")
    assert len(ru_results) == 1
    assert ru_results[0].chunk_id == "2002_ru_clause_1"
    assert ru_results[0].lang == "ru"

    # Filter by doc_id without minus sign: "1001" matches "-1001"
    doc_results = retriever.search("mehnat", limit=10, doc_id="1001")
    assert len(doc_results) == 2
    assert all(r.doc_id == "-1001" for r in doc_results)

    # Filter by unit_type='clause'
    clause_results = retriever.search("qonunchilik", limit=10, unit_type="clause")
    assert len(clause_results) == 0 or all(r.unit_type == "clause" for r in clause_results)

    # Filter by archetype="a"
    arch_results = retriever.search("mehnat", limit=10, archetype="a")
    assert len(arch_results) == 3
    assert all(r.archetype == "ARCHETYPE_A" for r in arch_results)


def test_search_nonexistent_collection():
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)
    retriever = HybridRetriever(
        qdrant_manager=manager,
        embedder=MockEmbedder(),
        collection_name="nonexistent_collection_12345",
    )
    with pytest.raises(ValueError, match="does not exist in Qdrant"):
        retriever.search("test query")


def test_context_builder():
    builder = ParentDocumentContextBuilder()

    url = builder.build_lexuz_url(doc_id="-3492199", anchors=["-5001"])
    assert url == "https://lex.uz/docs/3492199#-5001"

    url_pos = builder.build_lexuz_url(doc_id="12345", anchors=["6789"])
    assert url_pos == "https://lex.uz/docs/12345#6789"

    res = SearchResult(
        point_id="abc",
        chunk_id="-1001_uz_art_1",
        score=0.95,
        doc_id="-1001",
        lang="uz",
        doc_number="ЗРУ-100",
        doc_title="Mehnat kodeksi",
        archetype="ARCHETYPE_A",
        unit_type="article",
        unit_number="1",
        unit_title="Mehnat qonunchiligi",
        breadcrumb=["Mehnat kodeksi", "1-bob", "1-modda"],
        okoz_codes=["05.01.00.00"],
        has_table=False,
        original_script="lat",
        anchors=["-5001"],
        citations_count=0,
        content="Normativ matn.",
    )

    norm_block = builder.format_single_norm(res, rerank_score=0.9876, rank=1)
    assert "Mehnat kodeksi (№ ЗРУ-100)" in norm_block
    assert "Mehnat kodeksi > 1-bob > 1-modda" in norm_block
    assert "https://lex.uz/docs/1001#-5001" in norm_block
    assert "Normativ matn." in norm_block

    # Test fallback header when doc_title is missing
    res_no_title = SearchResult(
        point_id="def",
        chunk_id="-2002_ru_clause_1",
        score=0.9,
        doc_id="-2002",
        lang="ru",
        doc_number="",
        doc_title="",
        archetype="ARCHETYPE_A",
        unit_type="clause",
        unit_number="1",
        unit_title="",
        breadcrumb=[],
        okoz_codes=[],
        has_table=False,
        original_script="cyr",
        anchors=[],
        citations_count=0,
        content="Текст.",
    )
    norm_no_title = builder.format_single_norm(res_no_title)
    assert "Акт -2002" in norm_no_title


def test_legal_search_service_end_to_end(populated_collection):
    manager, embedder = populated_collection
    retriever = HybridRetriever(
        qdrant_manager=manager,
        embedder=embedder,
        collection_name=TEST_RETRIEVAL_COLLECTION,
    )
    reranker = MockReranker()
    service = LegalSearchService(
        retriever=retriever,
        reranker=reranker,
    )

    out = service.search(
        query="mehnat munosabatlari",
        top_k=2,
        candidates_k=10,
    )
    assert out["total_candidates"] > 0
    assert len(out["reranked_results"]) <= 2
    assert "ПРАВОВОЙ КОНТЕКСТ ДЛЯ ОТВЕТА" in out["rag_context"]
    assert "Mehnat kodeksi" in out["rag_context"]
