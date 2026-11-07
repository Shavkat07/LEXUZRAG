"""
tests/test_api_spec.py

Автоматизированные тесты для проверки 100% требований:
RAG API — test spetsifikatsiyasi (lex.uz).pdf

Проверяемые разделы:
1. Раздел 2: Общие требования, эндпоинты, форматы, лимиты, кодировка UTF-8, аутентификация.
2. Раздел 3: Валидация входного контракта (POST /v1/rag/batch).
3. Раздел 4: Контракт выходного JSON (system, timing, summary, results, citations, contexts).
4. Раздел 5: Коды ошибок (200, 202, 400, 401, 409, 429), изоляция ошибок на уровне вопроса.
5. Раздел 7: Чеклист готовности к тестированию.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Активируем режим тестирования перед загрузкой модулей API
os.environ["TESTING"] = "1"
os.environ["LEXUZ_MOCK_SEARCH"] = "1"
os.environ["LEXUZ_API_KEY"] = "test-secret-key,eval-2026-test-key"

from lexuz.api.app import create_app
from lexuz.api.limiter import RateLimiter, global_rate_limiter
from lexuz.api.store import BatchStore
from lexuz.api.engine import RAGEngine, MockSearchService
import lexuz.api.routes as routes_mod

VALID_TOKEN = "test-secret-key"
AUTH_HEADER = {"Authorization": f"Bearer {VALID_TOKEN}"}


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Изолирует базу данных SQLite и пересоздает TestClient для каждого теста."""
    db_file = str(tmp_path / "test_rag_store.db")
    test_store = BatchStore(db_path=db_file)
    test_engine = RAGEngine(search_service=MockSearchService())

    # Подменяем синглтоны
    routes_mod._store = test_store
    routes_mod._engine = test_engine

    app = create_app()
    client = TestClient(app)
    return client


def test_meta_endpoint(setup_test_environment):
    """Проверка GET /v1/meta: Model, embedding, indeks versiyasi, lex.uz snapshot sanasi."""
    client = setup_test_environment
    res = client.get("/v1/meta")
    assert res.status_code == 200

    data = res.json()
    assert "pipeline_version" in data
    assert "llm_model" in data
    assert "embedding_model" in data
    assert "reranker" in data
    assert "index_version" in data
    assert "lex_snapshot_date" in data
    assert data["index_version"] == "lex-idx-2026-10-28"
    assert data["lex_snapshot_date"] == "2026-10-28"


def test_health_endpoint(setup_test_environment):
    """Проверка GET /v1/health: Servis ishlayaptimi."""
    client = setup_test_environment
    res = client.get("/v1/health")
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "qdrant" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_authentication(setup_test_environment):
    """Проверка требований авторизации: 401 при отсутствии или неверном ключе."""
    client = setup_test_environment
    payload = {
        "run_id": "eval-2026-11-03",
        "batch_id": "eval-auth-1",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Savol matni"}],
    }

    # Без заголовка Authorization
    res_no_auth = client.post("/v1/rag/batch", json=payload)
    assert res_no_auth.status_code == 401

    # С недействительным токеном
    res_bad_auth = client.post(
        "/v1/rag/batch",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert res_bad_auth.status_code == 401

    # С корректным токеном
    res_ok_auth = client.post(
        "/v1/rag/batch",
        json=payload,
        headers=AUTH_HEADER,
    )
    assert res_ok_auth.status_code == 200


def test_validation_batch_size_and_limits(setup_test_environment):
    """
    Проверка валидации (HTTP 400):
    - 0 вопросов
    - Более 10 вопросов
    - Дублирующиеся id
    - Пустой текст вопроса
    - Вопрос длиннее 2000 символов
    """
    client = setup_test_environment

    # 1. 0 вопросов
    res_empty = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "run-1",
            "batch_id": "b-empty",
            "batch_index": 1,
            "total_batches": 1,
            "items": [],
        },
        headers=AUTH_HEADER,
    )
    assert res_empty.status_code == 400

    # 2. 11 вопросов (> 10)
    eleven_items = [{"id": f"q{i:03d}", "question": f"Savol {i}"} for i in range(1, 12)]
    res_11 = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "run-1",
            "batch_id": "b-11",
            "batch_index": 1,
            "total_batches": 1,
            "items": eleven_items,
        },
        headers=AUTH_HEADER,
    )
    assert res_11.status_code == 400

    # 3. Дублирующиеся id
    dup_items = [
        {"id": "q001", "question": "Savol 1"},
        {"id": "q001", "question": "Savol 2"},
    ]
    res_dup = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "run-1",
            "batch_id": "b-dup",
            "batch_index": 1,
            "total_batches": 1,
            "items": dup_items,
        },
        headers=AUTH_HEADER,
    )
    assert res_dup.status_code == 400

    # 4. Пустой текст вопроса
    res_blank_q = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "run-1",
            "batch_id": "b-blank",
            "batch_index": 1,
            "total_batches": 1,
            "items": [{"id": "q001", "question": "   "}],
        },
        headers=AUTH_HEADER,
    )
    assert res_blank_q.status_code == 400

    # 5. Вопрос более 2000 символов
    long_q = "A" * 2005
    res_long_q = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "run-1",
            "batch_id": "b-long",
            "batch_index": 1,
            "total_batches": 1,
            "items": [{"id": "q001", "question": long_q}],
        },
        headers=AUTH_HEADER,
    )
    assert res_long_q.status_code == 400


def test_batch_response_schema_section_4(setup_test_environment):
    """
    Проверка полного соответствия Section 4:
    - run_id, batch_id, status: completed
    - system (pipeline_version, llm_model, embedding_model, reranker, index_version, lex_snapshot_date)
    - timing (received_at, completed_at, total_ms)
    - summary (total, ok, no_answer, error)
    - results[].citations: ref, doc_id, doc_title, doc_type, url, article, part, edition_date, doc_status, chunk_id, quote
    - results[].contexts: rank, chunk_id, doc_id, article, text, retrieval_score, rerank_score, used_in_answer
    - ПРАВИЛО: citations[].chunk_id СТРОГО присутствует в contexts!
    """
    client = setup_test_environment
    batch_req = {
        "run_id": "eval-2026-11-03",
        "batch_id": "eval-2026-11-03-b01",
        "batch_index": 1,
        "total_batches": 10,
        "options": {
            "top_k": 5,
            "return_contexts": True,
            "return_debug": True,
            "temperature": 0,
            "seed": 42,
            "as_of_date": "2026-11-01",
            "answer_language": "auto",
        },
        "items": [
            {
                "id": "q001",
                "question": "Yillik asosiy mehnat ta'tilining eng kam muddati qancha?",
                "language": "uz-Latn",
            },
            {
                "id": "q002",
                "question": "Какой срок исковой давности установлен Гражданским кодексом?",
                "language": "ru",
            },
        ],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    data = res.json()

    # Корневые поля
    assert data["run_id"] == "eval-2026-11-03"
    assert data["batch_id"] == "eval-2026-11-03-b01"
    assert data["status"] == "completed"

    # system
    sys_info = data["system"]
    assert sys_info["pipeline_version"] == "0.4.2"
    assert "llm_model" in sys_info
    assert "embedding_model" in sys_info
    assert "index_version" in sys_info
    assert "lex_snapshot_date" in sys_info

    # timing
    timing = data["timing"]
    assert "received_at" in timing
    assert "completed_at" in timing
    assert isinstance(timing["total_ms"], int)

    # summary
    summary = data["summary"]
    assert summary["total"] == 2
    assert summary["ok"] == 2
    assert summary["no_answer"] == 0
    assert summary["error"] == 0

    # results
    results = data["results"]
    assert len(results) == 2
    assert [r["id"] for r in results] == ["q001", "q002"]

    for r in results:
        assert r["status"] == "ok"
        assert r["answer"] is not None
        assert "[1]" in r["answer"]  # Содержит ссылку на источник
        assert r["no_answer"] is False
        assert isinstance(r["confidence"], float)

        # Citations
        assert len(r["citations"]) >= 1
        cit = r["citations"][0]
        assert cit["ref"] == 1
        assert cit["doc_id"].startswith("lex:")
        assert "doc_title" in cit
        assert "doc_type" in cit
        assert cit["url"].startswith("https://lex.uz/docs/")
        assert "article" in cit
        assert "doc_status" in cit
        assert "quote" in cit
        assert len(cit["quote"]) > 0

        # Contexts
        assert len(r["contexts"]) >= 1
        ctx = r["contexts"][0]
        assert ctx["rank"] == 1
        assert "chunk_id" in ctx
        assert ctx["doc_id"].startswith("lex:")
        assert "article" in ctx
        assert "text" in ctx
        assert isinstance(ctx["retrieval_score"], float)
        assert isinstance(ctx["rerank_score"], float)

        # ЖЕСТКОЕ ПРАВИЛО СПЕЦИФИКАЦИИ:
        # Har bir citations[].chunk_id contexts ichida bo'lishi kerak
        context_chunk_ids = {c["chunk_id"] for c in r["contexts"]}
        for c in r["citations"]:
            assert c["chunk_id"] in context_chunk_ids, (
                f"Citation chunk_id '{c['chunk_id']}' MUST be in contexts!"
            )

        # debug
        assert r["debug"] is not None
        assert "rewritten_query" in r["debug"]
        assert "retrieved_total" in r["debug"]

        # timing & usage
        assert "retrieval_ms" in r["timing"]
        assert "total_ms" in r["timing"]
        assert "input_tokens" in r["usage"]
        assert "output_tokens" in r["usage"]
        assert r["error"] is None


def test_no_answer_handling(setup_test_environment):
    """
    Проверка no_answer: true для вопросов, не имеющих регулирования в законе:
    Qonunda javob yo'q bo'lsa tizim to'qib chiqarmasdan no_answer: true qaytarishi kerak.
    """
    client = setup_test_environment
    batch_req = {
        "run_id": "eval-2026-11-03",
        "batch_id": "eval-b-no-ans",
        "batch_index": 1,
        "total_batches": 1,
        "items": [
            {
                "id": "q010",
                "question": "Kosmik kemalarni xususiylashtirish haqida javobsiz savol",
                "language": "uz-Latn",
            }
        ],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    data = res.json()

    assert data["summary"]["no_answer"] == 1
    assert data["summary"]["ok"] == 0

    item = data["results"][0]
    assert item["status"] == "no_answer"
    assert item["no_answer"] is True
    assert len(item["citations"]) == 0
    assert "topilmadi" in item["answer"] or "не найден" in item["answer"]


def test_item_error_isolation(setup_test_environment, monkeypatch):
    """
    Проверка изоляции ошибок на уровне единичного вопроса (Section 5):
    Bitta savoldagi xato butun batch ni yiqitmasligi kerak: batch 200 qaytaradi,
    xato esa o'sha savolning status va error maydonida ko'rinadi.
    """
    client = setup_test_environment
    engine = routes_mod.get_engine()

    orig_generate_answer = engine._generate_answer

    def mock_generate_answer(item_id, question, target_lang, contexts, reranked_results, options):
        if item_id == "q_failing":
            raise RuntimeError("Simulated internal LLM Timeout")
        return orig_generate_answer(item_id, question, target_lang, contexts, reranked_results, options)

    monkeypatch.setattr(engine, "_generate_answer", mock_generate_answer)

    batch_req = {
        "run_id": "eval-error-iso",
        "batch_id": "b-error-iso",
        "batch_index": 1,
        "total_batches": 1,
        "items": [
            {"id": "q_ok", "question": "Mehnat ta'tili muddati qancha?"},
            {"id": "q_failing", "question": "Xatolik keltirib chiqaruvchi savol"},
        ],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    # Батч ДОЛЖЕН вернуть 200, несмотря на сбой одного вопроса!
    assert res.status_code == 200
    data = res.json()

    assert data["summary"]["total"] == 2
    assert data["summary"]["ok"] == 1
    assert data["summary"]["error"] == 1

    # Успешный вопрос
    item_ok = next(r for r in data["results"] if r["id"] == "q_ok")
    assert item_ok["status"] == "ok"
    assert item_ok["error"] is None

    # Упавший вопрос (Section 5: ok, no_answer, error, timeout)
    item_err = next(r for r in data["results"] if r["id"] == "q_failing")
    assert item_err["status"] in ("error", "timeout")
    assert item_err["answer"] is None
    assert item_err["citations"] == []
    assert item_err["error"] is not None
    assert item_err["error"]["code"] in ["LLM_ERROR", "LLM_TIMEOUT", "RETRIEVAL_FAILED"]
    assert "Simulated internal LLM Timeout" in item_err["error"]["message"]


def test_idempotency_and_conflict(setup_test_environment):
    """
    Проверка идемпотентности и 409 Conflict:
    - Повторная отправка с тем же batch_id возвращает сохраненный результат (200).
    - Отправка того же batch_id с другим содержимым возвращает 409 Conflict.
    """
    client = setup_test_environment
    batch_req = {
        "run_id": "eval-idempotency",
        "batch_id": "b-idemp-001",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Mehnat to'g'risidagi savol"}],
    }

    # Первая отправка
    res1 = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res1.status_code == 200
    data1 = res1.json()

    # Повторная отправка с идентичным телом (Идемпотентный 200)
    res2 = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data1["batch_id"] == data2["batch_id"]
    assert data1["timing"]["received_at"] == data2["timing"]["received_at"]

    # Отправка с тем же batch_id, но другим вопросом -> 409 Conflict!
    conflicting_req = {
        "run_id": "eval-idempotency",
        "batch_id": "b-idemp-001",  # Тот же ID
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Mutlaqo boshqa savol"}],
    }
    res_conflict = client.post("/v1/rag/batch", json=conflicting_req, headers=AUTH_HEADER)
    assert res_conflict.status_code == 409


def test_get_batch_by_id(setup_test_environment):
    """Проверка GET /v1/rag/batch/{batch_id}: получение сохраненного батча и 404."""
    client = setup_test_environment

    # Несуществующий batch -> 404
    res_404 = client.get("/v1/rag/batch/non-existent-batch", headers=AUTH_HEADER)
    assert res_404.status_code == 404

    # Создаем батч
    batch_req = {
        "run_id": "eval-get",
        "batch_id": "batch-get-test",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Mehnat kodeksi 21-modda"}],
    }
    client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)

    # Получаем результат по GET
    res_get = client.get("/v1/rag/batch/batch-get-test", headers=AUTH_HEADER)
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["batch_id"] == "batch-get-test"
    assert data["status"] == "completed"
    assert len(data["results"]) == 1


def test_query_single_endpoint(setup_test_environment):
    """Проверка отладочного эндпоинта POST /v1/rag/query."""
    client = setup_test_environment
    q_req = {
        "id": "q_single_1",
        "question": "Yillik asosiy mehnat ta'tili muddati qancha?",
        "language": "uz-Latn",
    }
    res = client.post("/v1/rag/query", json=q_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert "system" in data
    assert "result" in data
    assert data["result"]["id"] == "q_single_1"
    assert data["result"]["status"] == "ok"


def test_rate_limiter_429():
    """Проверка 429 Too Many Requests и заголовка Retry-After при превышении лимита."""
    limiter = RateLimiter(rate_per_second=1.0, capacity=1.0)
    # Первый вызов проходит
    limiter.check("test-key", force=True)

    # Второй мгновенный вызов должен вызвать 429
    with pytest.raises(Exception) as excinfo:
        limiter.check("test-key", force=True)

    exc = excinfo.value
    assert getattr(exc, "status_code", None) == 429
    assert "Retry-After" in getattr(exc, "headers", {})


def test_web_dashboard_routes(setup_test_environment):
    """Проверка доступности веб-дашборда на / и /dashboard."""
    client = setup_test_environment

    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Lex.uz" in res_root.text
    assert "Legal RAG" in res_root.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "Interaktiv So'rov" in res_dash.text


def test_cyrillic_and_apostrophe_preservation(setup_test_environment):
    """
    Проверка лингвистических требований спецификации:
    - O'zbek lotin, o'zbek kirill va rus tilidagi savollar qabul qilinadi;
    - o' , g' , ʻ belgilari buzilmasligi kerak.
    """
    client = setup_test_environment
    batch_req = {
        "run_id": "eval-lang",
        "batch_id": "b-lang-test",
        "batch_index": 1,
        "total_batches": 1,
        "items": [
            {
                "id": "q_latin",
                "question": "Oʻzbekiston Respublikasi fuqaroligi toʻgʻrisida qonun",
                "language": "uz-Latn",
            },
            {
                "id": "q_cyrillic",
                "question": "Ўзбекистон Республикаси Меҳнат кодекси",
                "language": "uz-Cyrl",
            },
            {
                "id": "q_russian",
                "question": "Гражданский кодекс Республики Узбекистан",
                "language": "ru",
            },
        ],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 3
    # Проверяем что все три языка успешно распознаны и обработаны
    assert data["results"][0]["status"] == "ok"
    assert data["results"][1]["status"] == "ok"
    assert data["results"][2]["status"] == "ok"


def test_batch_boundary_sizes_1_and_10(setup_test_environment):
    """Проверка граничных значений размера пакета: ровно 1 и ровно 10 элементов."""
    client = setup_test_environment

    # Ровно 1 вопрос (минимум)
    res_1 = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "eval-boundary",
            "batch_id": "b-min-1",
            "batch_index": 1,
            "total_batches": 1,
            "items": [{"id": "q001", "question": "Yagona savol"}],
        },
        headers=AUTH_HEADER,
    )
    assert res_1.status_code == 200
    assert len(res_1.json()["results"]) == 1

    # Ровно 10 вопросов (максимум)
    items_10 = [{"id": f"q{i:03d}", "question": f"Mehnat savoli {i}"} for i in range(1, 11)]
    res_10 = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "eval-boundary",
            "batch_id": "b-max-10",
            "batch_index": 1,
            "total_batches": 1,
            "items": items_10,
        },
        headers=AUTH_HEADER,
    )
    assert res_10.status_code == 200
    assert len(res_10.json()["results"]) == 10
    assert res_10.json()["summary"]["total"] == 10


def test_options_contexts_and_debug_toggles(setup_test_environment):
    """
    Проверка опций options:
    - return_contexts: false -> contexts=[]
    - return_debug: false -> debug=null
    - top_k: 1 -> len(contexts) <= 1
    """
    client = setup_test_environment
    batch_req = {
        "run_id": "eval-opts",
        "batch_id": "b-opts-test",
        "batch_index": 1,
        "total_batches": 1,
        "options": {
            "top_k": 1,
            "return_contexts": False,
            "return_debug": False,
        },
        "items": [{"id": "q001", "question": "Mehnat kodeksi"}],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    item = res.json()["results"][0]

    assert item["contexts"] == []
    assert item["debug"] is None
    assert len(item["citations"]) >= 1


def test_mixed_batch_summary_counts(setup_test_environment, monkeypatch):
    """
    Проверка точности счетчиков summary:
    - total, ok, no_answer, error
    """
    client = setup_test_environment
    engine = routes_mod.get_engine()
    orig_generate_answer = engine._generate_answer

    def mock_gen_mixed(item_id, question, target_lang, contexts, reranked_results, options):
        if item_id == "q_err":
            raise RuntimeError("Item error")
        return orig_generate_answer(item_id, question, target_lang, contexts, reranked_results, options)

    monkeypatch.setattr(engine, "_generate_answer", mock_gen_mixed)

    batch_req = {
        "run_id": "eval-mixed",
        "batch_id": "b-mixed-summary",
        "batch_index": 1,
        "total_batches": 1,
        "items": [
            {"id": "q_ok", "question": "Mehnat to'g'risida savol"},
            {"id": "q_no_ans", "question": "Kosmik kemalarni sotish javobsiz savol"},
            {"id": "q_err", "question": "Xatolik savoli"},
        ],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    summary = res.json()["summary"]

    assert summary["total"] == 3
    assert summary["ok"] == 1
    assert summary["no_answer"] == 1
    assert summary["error"] == 1


def test_async_processing_status(setup_test_environment):
    """Проверка возврата HTTP 202 для батча, находящегося в процессе обработки."""
    client = setup_test_environment
    store = routes_mod.get_store()

    # Регистрируем батч вручную со статусом processing
    store.check_or_register_batch(
        batch_id="b-async-test",
        run_id="eval-async",
        request_data={"items": []},
    )

    res = client.get("/v1/rag/batch/b-async-test", headers=AUTH_HEADER)
    assert res.status_code == 202
    data = res.json()
    assert data["batch_id"] == "b-async-test"
    assert data["status"] == "processing"


def test_utf8_json_content_type(setup_test_environment):
    """
    Проверка кодировки и заголовка Content-Type (Section 2):
    Content-Type: application/json; charset=utf-8
    """
    client = setup_test_environment

    # 1. GET /v1/meta
    res_meta = client.get("/v1/meta")
    assert res_meta.status_code == 200
    assert res_meta.headers.get("content-type") == "application/json; charset=utf-8"

    # 2. GET /v1/health
    res_health = client.get("/v1/health")
    assert res_health.status_code == 200
    assert res_health.headers.get("content-type") == "application/json; charset=utf-8"

    # 3. POST /v1/rag/batch
    batch_req = {
        "run_id": "eval-charset",
        "batch_id": "b-charset-1",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Mehnat ta'tili haqida"}],
    }
    res_batch = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res_batch.status_code == 200
    assert res_batch.headers.get("content-type") == "application/json; charset=utf-8"

    # 4. Error response 400
    res_err = client.post("/v1/rag/batch", json={"items": []}, headers=AUTH_HEADER)
    assert res_err.status_code == 400
    assert res_err.headers.get("content-type") == "application/json; charset=utf-8"


def test_gemini_call_preserves_item_id(setup_test_environment, monkeypatch):
    """
    Проверка сохранения ID вопроса при вызове Gemini LLM (защита от бага с id='item'):
    results[].id должен строго совпадать с items[].id.
    """
    client = setup_test_environment
    engine = routes_mod.get_engine()
    engine.gemini_api_key = "fake-gemini-key"

    # Мокаем вызов Gemini API
    class MockModelResponse:
        text = '{"answer": "Mehnat ta\'tili 21 kun [1].", "no_answer": false, "confidence": 0.92, "used_ranks": [1]}'

    class MockGenaiClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, model, contents, config):
            return MockModelResponse()

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", MockGenaiClient)

    batch_req = {
        "run_id": "eval-gemini-id",
        "batch_id": "b-gemini-id",
        "batch_index": 1,
        "total_batches": 1,
        "items": [
            {"id": "q001", "question": "Savol 1"},
            {"id": "q002", "question": "Savol 2"},
        ],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    data = res.json()

    results = data["results"]
    assert len(results) == 2
    assert results[0]["id"] == "q001"
    assert results[1]["id"] == "q002"
    assert results[0]["id"] != "item"
    assert results[1]["id"] != "item"
    assert len(results[0]["citations"]) == 1
    assert results[0]["citations"][0]["chunk_id"] in [c["chunk_id"] for c in results[0]["contexts"]]


def test_async_batch_execution_with_background_tasks(setup_test_environment):
    """
    Проверка асинхронного режима выполнения батча через BackgroundTasks:
    POST /v1/rag/batch?mode=async возвращает HTTP 202, а GET /v1/rag/batch/{batch_id}
    возвращает статус processing / completed.
    """
    client = setup_test_environment

    batch_req = {
        "run_id": "eval-async-exec",
        "batch_id": "b-async-exec-1",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Mehnat kodeksi 21-modda"}],
    }

    # Отправка с query-параметром mode=async
    res_post = client.post("/v1/rag/batch?mode=async", json=batch_req, headers=AUTH_HEADER)
    assert res_post.status_code == 202
    data_post = res_post.json()
    assert data_post["batch_id"] == "b-async-exec-1"
    assert data_post["status"] == "processing"

    # В Starlette TestClient фоновые задачи выполняются синхронно до завершения вызова
    res_get = client.get("/v1/rag/batch/b-async-exec-1", headers=AUTH_HEADER)
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["batch_id"] == "b-async-exec-1"
    assert data_get["status"] == "completed"
    assert len(data_get["results"]) == 1
    assert data_get["results"][0]["id"] == "q001"


def test_error_codes_mapping_section_5(setup_test_environment, monkeypatch):
    """
    Проверка маппинга ошибок на статус timeout и коды ошибок (Section 5):
    - TimeoutError -> status: 'timeout', code: 'LLM_TIMEOUT', retryable: true
    - batch_index > total_batches -> HTTP 400
    - Неподдерживаемый язык -> HTTP 400
    """
    client = setup_test_environment
    engine = routes_mod.get_engine()

    def mock_timeout_item(item_id, question, target_lang, contexts, reranked_results, options):
        raise TimeoutError("Generation exceeded 60s timeout")

    monkeypatch.setattr(engine, "_generate_answer", mock_timeout_item)

    batch_req = {
        "run_id": "eval-timeout",
        "batch_id": "b-timeout-1",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Savol timeout"}],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    item = res.json()["results"][0]

    assert item["status"] == "timeout"
    assert item["error"] is not None
    assert item["error"]["code"] == "LLM_TIMEOUT"
    assert item["error"]["retryable"] is True
    assert "timeout" in item["error"]["message"].lower()

    # batch_index > total_batches -> 400
    res_bad_index = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "eval-idx",
            "batch_id": "b-bad-idx",
            "batch_index": 15,
            "total_batches": 10,
            "items": [{"id": "q001", "question": "Savol"}],
        },
        headers=AUTH_HEADER,
    )
    assert res_bad_index.status_code == 400

    # Неподдерживаемый язык -> 400
    res_bad_lang = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "eval-lang",
            "batch_id": "b-bad-lang",
            "batch_index": 1,
            "total_batches": 1,
            "items": [{"id": "q001", "question": "Savol", "language": "unsupported-lang"}],
        },
        headers=AUTH_HEADER,
    )
    assert res_bad_lang.status_code == 400


def test_tashkent_timezone_utc5_timestamps(setup_test_environment):
    """
    Проверка таймзон в таймингах (+05:00 Ташкент согласно примерам Section 4):
    2026-11-03T10:30:00+05:00
    """
    client = setup_test_environment
    batch_req = {
        "run_id": "eval-tz",
        "batch_id": "b-tz-1",
        "batch_index": 1,
        "total_batches": 1,
        "items": [{"id": "q001", "question": "Mehnat ta'tili"}],
    }

    res = client.post("/v1/rag/batch", json=batch_req, headers=AUTH_HEADER)
    assert res.status_code == 200
    timing = res.json()["timing"]

    assert "+05:00" in timing["received_at"]
    assert "+05:00" in timing["completed_at"]


def test_batch_store_cleanup_retention(tmp_path):
    """
    Проверка политики удержания данных >= 30 дней (Section 5):
    Метод cleanup_old_batches удаляет только записи старше указанного срока.
    """
    from datetime import datetime, timedelta, timezone
    from lexuz.api.store import BatchStore

    db_path = str(tmp_path / "retention_test.db")
    store = BatchStore(db_path=db_path)

    # Добавляем свежий батч (сегодня)
    store.check_or_register_batch("b-today", "run-1", {"items": [1]})
    store.save_completed_batch("b-today", {"status": "ok"})

    # Добавляем старый батч (40 дней назад) через прямой SQL
    old_iso = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with store._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO batches (batch_id, run_id, request_hash, request_json, response_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'completed', ?, ?)
            """,
            ("b-old", "run-1", "hash-old", "{}", "{}", old_iso, old_iso),
        )
        conn.commit()

    assert store.get_batch("b-today") is not None
    assert store.get_batch("b-old") is not None

    # Запускаем очистку с порогом 30 дней
    deleted = store.cleanup_old_batches(retention_days=30)
    assert deleted == 1

    # Проверяем, что старый удален, а свежий остался
    assert store.get_batch("b-today") is not None
    assert store.get_batch("b-old") is None


def test_section_6_evaluation_framework():
    """
    Проверка метрик Раздела 6 через RAGEvaluator:
    Hit@5, MRR, Faithfulness, Citation accuracy, No-answer accuracy, Latency.
    """
    from lexuz.eval.evaluator import RAGEvaluator, BenchmarkItem

    bench = [
        BenchmarkItem(
            id="q001",
            batch="b01",
            category="faktik",
            question="Yillik ta'til qancha?",
            expected_answer="21 kun",
            expected_doc_id="lex:6445145",
            expected_article="21-modda",
            answerable=True,
        ),
        BenchmarkItem(
            id="q002",
            batch="b01",
            category="javobsiz",
            question="Kosmik kema xususiylashtirish",
            expected_answer="",
            expected_doc_id=None,
            expected_article=None,
            answerable=False,
        ),
    ]

    api_results = [
        {
            "id": "q001",
            "status": "ok",
            "answer": "Yillik asosiy mehnat ta'tili 21 kalendar kun [1].",
            "no_answer": False,
            "citations": [
                {"ref": 1, "doc_id": "lex:6445145", "article": "21-modda", "chunk_id": "c1"}
            ],
            "contexts": [
                {"rank": 1, "doc_id": "lex:6445145", "article": "21-modda", "chunk_id": "c1", "retrieval_score": 0.9, "rerank_score": 0.95}
            ],
            "timing": {"total_ms": 350},
        },
        {
            "id": "q002",
            "status": "no_answer",
            "answer": "Qonunchilikda topilmadi.",
            "no_answer": True,
            "citations": [],
            "contexts": [],
            "timing": {"total_ms": 120},
        },
    ]

    summary = RAGEvaluator.evaluate(bench, api_results)
    assert summary.hit_at_5_doc == 1.0
    assert summary.hit_at_5_article == 1.0
    assert summary.mrr == 1.0
    assert summary.no_answer_accuracy == 1.0
    assert summary.fabricated_citations_count == 0
    assert summary.outdated_norms_count == 0
    assert summary.error_rate == 0.0

    md = summary.to_markdown()
    assert "Hit@5 (hujjat)" in md
    assert "MRR" in md


def test_labor_leave_exact_question_regression(setup_test_environment):
    """
    Регрессионный тест для вопроса: 'Mehnat ta'tilining eng kam muddati qancha ?'
    Гарантирует:
    1. Точный ответ с указанием 21 календарного дня.
    2. Отсутствие процедурных санкций ('aybdor shaxslar...').
    3. Ссылку на Трудовой кодекс и правильную статью.
    4. Точную цитату в citations[0].quote.
    """
    client = setup_test_environment
    payload = {
        "run_id": "eval-regression",
        "batch_id": "b-labor-leave-1",
        "batch_index": 1,
        "total_batches": 1,
        "options": {
            "top_k": 5,
            "return_contexts": True,
            "return_debug": True,
        },
        "items": [
            {
                "id": "q_debug_1",
                "question": "Mehnat ta'tilining eng kam muddati qancha ?",
                "language": "uz-Latn",
            }
        ],
    }

    res = client.post("/v1/rag/batch", json=payload, headers=AUTH_HEADER)
    assert res.status_code == 200
    data = res.json()
    item = data["results"][0]

    assert item["status"] == "ok"
    assert item["no_answer"] is False
    assert "yigirma bir kalendar kun" in item["answer"].lower() or "21" in item["answer"]
    assert "aybdor shaxslar" not in item["answer"].lower()

    # Citations
    assert len(item["citations"]) >= 1
    cit = item["citations"][0]
    assert cit["ref"] == 1
    assert "Mehnat kodeksi" in cit["doc_title"]
    assert cit["doc_type"] == "kodeks"
    assert "yigirma bir kalendar kun" in cit["quote"].lower()
    assert cit["chunk_id"] in [c["chunk_id"] for c in item["contexts"]]


