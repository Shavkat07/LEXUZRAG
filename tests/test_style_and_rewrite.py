"""
tests/test_style_and_rewrite.py

Тесты для нового функционала:
1. Переписывание неформальных/разговорных запросов в строгие юридические (Query Rewriting & Legal Formalization).
2. Выбор стиля ответа (Dual Response Mode: 'simple' / 'oddiy' vs 'formal' / 'rasmiy').
3. Валидация стилей в OptionsRequest.
4. Интеграция с CLI (main.py ask --style).
5. Работа API (/v1/rag/query, /v1/rag/batch) с выбором стиля.
"""

import subprocess
import sys
import pytest
from pydantic import ValidationError

from lexuz.api.engine import RAGEngine, MockSearchService
from lexuz.api.schemas import OptionsRequest, BatchRequest


def test_is_colloquial_or_informal():
    """Проверка детекции разговорного, сленгового или неформального языка."""
    # Неформальные узбекские запросы
    assert RAGEngine.is_colloquial_or_informal("otpuska qancha kun beriladi ozi") is True
    assert RAGEngine.is_colloquial_or_informal("ishdan haydab yuborsa nima qilaman") is True
    assert RAGEngine.is_colloquial_or_informal("uvolit qisa bo'ladimi") is True
    assert RAGEngine.is_colloquial_or_informal("oylik bermayapti nima qilay") is True
    assert RAGEngine.is_colloquial_or_informal("prava olish tartibi qanaqa") is True
    assert RAGEngine.is_colloquial_or_informal("отпуска қанча кун берилади ози") is True

    # Неформальные русские запросы
    assert RAGEngine.is_colloquial_or_informal("меня уволили без предупреждения че делать", lang="ru") is True
    assert RAGEngine.is_colloquial_or_informal("шеф не платит деньги что делать", lang="ru") is True
    assert RAGEngine.is_colloquial_or_informal("отобрали права на дороге", lang="ru") is True

    # Строгие формальные запросы (не должны детектироваться как разговорные)
    assert RAGEngine.is_colloquial_or_informal("Yillik asosiy mehnat ta'tilining eng kam muddati qancha?") is False
    assert RAGEngine.is_colloquial_or_informal("Какой срок исковой давности установлен Гражданским кодексом?", lang="ru") is False
    assert RAGEngine.is_colloquial_or_informal("Kosmik kemalarni shaxsiy xususiylashtirish tartibi") is False


def test_query_rewriting_heuristic_uzbek():
    """Тестирование смарт-эвристического переписывания неформальных запросов на узбекском языке."""
    engine = RAGEngine(search_service=MockSearchService())

    # Отпуск (otpuska / tatil)
    rw1 = engine.rewrite_query("otpuska qancha kun beriladi ozi", lang="uz-Latn")
    assert "mehnat ta'tili" in rw1
    assert "muddati" in rw1 or "davomiyligi" in rw1

    # Увольнение (ishdan haydash / uvolit)
    rw2 = engine.rewrite_query("ishdan haydab yuborsa nima qilaman", lang="uz-Latn")
    assert "mehnat shartnomasi" in rw2
    assert "bekor qilish" in rw2

    rw3 = engine.rewrite_query("uvolit qisa bo'ladimi", lang="uz-Latn")
    assert "mehnat shartnomasi" in rw3
    assert "bekor qilish" in rw3

    # Декретный отпуск
    rw4 = engine.rewrite_query("dekret puli qanday to'lanadi", lang="uz-Latn")
    assert "homiladorlik" in rw4 or "tug'ish" in rw4

    # Задержка зарплаты
    rw5 = engine.rewrite_query("oylik bermasa qayerga murojaat qilish kerak", lang="uz-Latn")
    assert "ish haqi" in rw5
    assert "to'lash" in rw5


def test_query_rewriting_heuristic_russian():
    """Тестирование смарт-эвристического переписывания неформальных запросов на русском языке."""
    engine = RAGEngine(search_service=MockSearchService())

    # Увольнение без предупреждения
    rw1 = engine.rewrite_query("меня уволили без предупреждения че делать", lang="ru")
    assert "расторжение трудового договора" in rw1
    assert "работодателя" in rw1

    # Длительность отпуска
    rw2 = engine.rewrite_query("отпуск сколько дней положено по закону", lang="ru")
    assert "ежегодного основного трудового отпуска" in rw2

    # Зарплата
    rw3 = engine.rewrite_query("не платят зарплату что делать", lang="ru")
    assert "заработной платы" in rw3


def test_query_rewriting_preserves_formal_and_unanswerable():
    """Формальные и заведомо нерегулируемые (unanswerable) запросы не должны искажаться."""
    engine = RAGEngine(search_service=MockSearchService())

    # Формальный запрос
    formal_q = "Yillik asosiy mehnat ta'tilining eng kam muddati qancha?"
    assert engine.rewrite_query(formal_q, lang="uz-Latn") == formal_q

    # Заведомо javobsiz
    no_ans_q = "Kosmik kemalarni shaxsiy xususiylashtirish tartibi haqida javobsiz savol"
    assert engine.rewrite_query(no_ans_q, lang="uz-Latn") == no_ans_q


def test_query_rewriting_llm_mock(monkeypatch):
    """Проверка переписывания через LLM при активном API-ключе."""
    engine = RAGEngine(search_service=MockSearchService())
    engine.gemini_api_key = "test-api-key"

    class MockModelResponse:
        text = "xodimning yillik asosiy mehnat ta'tili muddati va davomiyligi"

    class MockGenaiClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, model, contents, config):
            return MockModelResponse()

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", MockGenaiClient)

    rw = engine.rewrite_query("otpuska qancha kun beriladi ozi", lang="uz-Latn")
    assert rw == "xodimning yillik asosiy mehnat ta'tili muddati va davomiyligi"


def test_options_request_style_validation():
    """Проверка валидации и нормализации поля style в OptionsRequest."""
    # Default is simple
    opt_def = OptionsRequest()
    assert opt_def.style == "simple"

    # Explicit simple and aliases
    assert OptionsRequest(style="simple").style == "simple"
    assert OptionsRequest(style="oddiy").style == "simple"
    assert OptionsRequest(style="простой").style == "simple"
    assert OptionsRequest(style="plain").style == "simple"
    assert OptionsRequest(style=None).style == "simple"

    # Explicit formal and aliases
    assert OptionsRequest(style="formal").style == "formal"
    assert OptionsRequest(style="rasmiy").style == "formal"
    assert OptionsRequest(style="официальный").style == "formal"

    # Invalid style raises ValueError
    with pytest.raises(ValidationError):
        OptionsRequest(style="invalid_style_123")


def test_grounded_answer_style_simple_vs_formal():
    """
    Проверка детерминированного синтеза ответов в стилях 'simple' и 'formal':
    1. Оба стиля сохраняют сноски [1], название документа и статью.
    2. 'simple' содержит человеческие пояснения (Oddiy tilda / Простыми словами).
    3. 'formal' содержит строгие юридические формулировки.
    """
    engine = RAGEngine(search_service=MockSearchService())

    # --- Simple Style (Uzbek) ---
    res_simple = engine.process_item(
        item_id="q_simple_1",
        question="Mehnat ta'tilining eng kam muddati qancha?",
        item_language="uz-Latn",
        options=OptionsRequest(style="simple", return_debug=True),
    )
    assert res_simple.status == "ok"
    assert "[1]" in res_simple.answer
    assert "Oddiy tilda" in res_simple.answer
    assert "21" in res_simple.answer or "yigirma bir kalendar kun" in res_simple.answer.lower()
    assert len(res_simple.citations) == 1
    assert res_simple.citations[0].chunk_id in [c.chunk_id for c in res_simple.contexts]

    # --- Formal Style (Uzbek) ---
    res_formal = engine.process_item(
        item_id="q_formal_1",
        question="Mehnat ta'tilining eng kam muddati qancha?",
        item_language="uz-Latn",
        options=OptionsRequest(style="formal", return_debug=True),
    )
    assert res_formal.status == "ok"
    assert "[1]" in res_formal.answer
    assert "Oddiy tilda" not in res_formal.answer
    assert "21" in res_formal.answer or "yigirma bir kalendar kun" in res_formal.answer.lower()
    assert len(res_formal.citations) == 1
    assert res_formal.citations[0].chunk_id in [c.chunk_id for c in res_formal.contexts]

    # --- Simple Style (Russian) ---
    res_ru_simple = engine.process_item(
        item_id="q_ru_simple",
        question="Какой срок исковой давности установлен Гражданским кодексом?",
        item_language="ru",
        options=OptionsRequest(style="simple", return_debug=True),
    )
    assert res_ru_simple.status == "ok"
    assert "[1]" in res_ru_simple.answer
    assert "Простыми словами" in res_ru_simple.answer

    # --- Formal Style (Russian) ---
    res_ru_formal = engine.process_item(
        item_id="q_ru_formal",
        question="Какой срок исковой давности установлен Гражданским кодексом?",
        item_language="ru",
        options=OptionsRequest(style="formal", return_debug=True),
    )
    assert res_ru_formal.status == "ok"
    assert "[1]" in res_ru_formal.answer
    assert "Простыми словами" not in res_ru_formal.answer


def test_colloquial_query_rewritten_in_debug():
    """Проверяет, что переписанный запрос сохраняется в ItemDebug.rewritten_query."""
    engine = RAGEngine(search_service=MockSearchService())
    res = engine.process_item(
        item_id="q_colloq_1",
        question="otpuska qancha kun beriladi ozi",
        item_language="uz-Latn",
        options=OptionsRequest(style="simple", return_debug=True),
    )
    assert res.status == "ok"
    assert res.debug is not None
    assert res.debug.rewritten_query is not None
    assert "mehnat ta'tili" in res.debug.rewritten_query


@pytest.fixture
def setup_test_environment(tmp_path):
    """Изолирует базу данных SQLite и пересоздает TestClient для каждого теста."""
    from fastapi.testclient import TestClient
    from lexuz.api.app import create_app
    from lexuz.api.store import BatchStore
    import lexuz.api.routes as routes_mod

    db_file = str(tmp_path / "test_style_store.db")
    test_store = BatchStore(db_path=db_file)
    test_engine = RAGEngine(search_service=MockSearchService())

    routes_mod._store = test_store
    routes_mod._engine = test_engine

    app = create_app()
    client = TestClient(app)
    return client


def test_cli_ask_with_style_flag():
    """Проверяет CLI команду 'main.py ask' с флагом --style simple и --style formal."""
    import os
    env_mock = dict(os.environ, LEXUZ_FORCE_LOCAL_LLM="1")

    # Test --style simple
    cmd_simple = [
        sys.executable,
        "main.py",
        "ask",
        "mehnat ta'tili muddati qancha?",
        "--mock",
        "--style",
        "simple",
    ]
    res_s = subprocess.run(cmd_simple, capture_output=True, text=True, env=env_mock)
    assert res_s.returncode == 0
    assert "Uslub: SIMPLE" in res_s.stdout
    assert "Oddiy tilda" in res_s.stdout

    # Test --style formal
    cmd_formal = [
        sys.executable,
        "main.py",
        "ask",
        "mehnat ta'tili muddati qancha?",
        "--mock",
        "--style",
        "formal",
    ]
    res_f = subprocess.run(cmd_formal, capture_output=True, text=True, env=env_mock)
    assert res_f.returncode == 0
    assert "Uslub: FORMAL" in res_f.stdout
    assert "Oddiy tilda" not in res_f.stdout


def test_api_query_and_batch_style_support(setup_test_environment):
    """Проверяет работу API эндпоинтов /v1/rag/query и /v1/rag/batch с полем style."""
    client = setup_test_environment
    auth_header = {"Authorization": "Bearer lexuz-test-api-key"}

    # 1. POST /v1/rag/query with style="simple"
    res_q_simple = client.post(
        "/v1/rag/query",
        json={
            "question": "Mehnat ta'tili muddati qancha?",
            "options": {"style": "simple", "return_debug": True},
        },
        headers=auth_header,
    )
    assert res_q_simple.status_code == 200
    data_s = res_q_simple.json()
    assert data_s["result"]["status"] == "ok"
    assert "Oddiy tilda" in data_s["result"]["answer"]

    # 2. POST /v1/rag/query with style="formal"
    res_q_formal = client.post(
        "/v1/rag/query",
        json={
            "question": "Mehnat ta'tili muddati qancha?",
            "options": {"style": "formal", "return_debug": True},
        },
        headers=auth_header,
    )
    assert res_q_formal.status_code == 200
    data_f = res_q_formal.json()
    assert data_f["result"]["status"] == "ok"
    assert "Oddiy tilda" not in data_f["result"]["answer"]

    # 3. POST /v1/rag/query with invalid style returns 400
    res_q_bad = client.post(
        "/v1/rag/query",
        json={
            "question": "Mehnat ta'tili muddati qancha?",
            "options": {"style": "bad_style_name"},
        },
        headers=auth_header,
    )
    assert res_q_bad.status_code == 400

    # 4. POST /v1/rag/batch with style="simple"
    res_batch = client.post(
        "/v1/rag/batch",
        json={
            "run_id": "eval-style-test",
            "batch_id": "b-style-1",
            "batch_index": 1,
            "total_batches": 1,
            "options": {"style": "simple", "return_debug": True},
            "items": [{"id": "q001", "question": "otpuska qancha kun beriladi ozi"}],
        },
        headers=auth_header,
    )
    assert res_batch.status_code == 200
    data_batch = res_batch.json()
    item_res = data_batch["results"][0]
    assert item_res["status"] == "ok"
    assert "mehnat ta'tili" in item_res["debug"]["rewritten_query"]
    assert "Oddiy tilda" in item_res["answer"]


def test_mixed_slang_and_loanwords():
    """Проверка переписывания смешанных сленговых и разговорных запросов."""
    engine = RAGEngine(search_service=MockSearchService())

    # Смешанный русско-узбекский сленг: увольнение без предупреждения
    q_mixed = "mene bez preduprejdeniya ishdan haydavorishsa nima qilaman"
    assert engine.is_colloquial_or_informal(q_mixed) is True
    rw_mixed = engine.rewrite_query(q_mixed)
    assert "mehnat shartnomasi" in rw_mixed
    assert "ogohlantirish" in rw_mixed

    # Разговорный русский сленг (шеф, насяльника, бабки)
    q_ru1 = "шеф не платит бабки"
    assert engine.is_colloquial_or_informal(q_ru1, lang="ru") is True
    rw_ru1 = engine.rewrite_query(q_ru1, lang="ru")
    assert "заработной платы" in rw_ru1

    q_ru2 = "насяльника не платит бабки"
    assert engine.is_colloquial_or_informal(q_ru2, lang="ru") is True
    rw_ru2 = engine.rewrite_query(q_ru2, lang="ru")
    assert "заработной платы" in rw_ru2


def test_dismissal_and_wages_grounded_answer():
    """
    Проверяет, что запросы по увольнению и задержке зарплаты не путаются с отпуском:
    1. 'ishdan haydab yuborsa nima qilaman' возвращает статью 161 (а не статью 21 об отпуске).
    2. 'simple' режим дает понятное гражданам объяснение гарантий.
    3. 'formal' режим дает строгую формулировку статьи.
    4. Запросы по зарплате возвращают статью 253.
    """
    engine = RAGEngine(search_service=MockSearchService())

    # 1. Увольнение Simple (Uzbek)
    res_dismissal_simple = engine.process_item(
        item_id="q_dismissal_simple",
        question="ishdan haydab yuborsa nima qilaman",
        item_language="uz-Latn",
        options=OptionsRequest(style="simple", return_debug=True),
    )
    assert res_dismissal_simple.status == "ok"
    assert "161-modda" in res_dismissal_simple.citations[0].article
    assert "21-modda" not in res_dismissal_simple.citations[0].article
    assert "Oddiy tilda" in res_dismissal_simple.answer
    assert "ogohlantir" in res_dismissal_simple.answer.lower() or "kompensatsiya" in res_dismissal_simple.answer.lower()

    # 2. Увольнение Formal (Uzbek)
    res_dismissal_formal = engine.process_item(
        item_id="q_dismissal_formal",
        question="ishdan haydab yuborsa nima qilaman",
        item_language="uz-Latn",
        options=OptionsRequest(style="formal", return_debug=True),
    )
    assert res_dismissal_formal.status == "ok"
    assert "161-modda" in res_dismissal_formal.citations[0].article
    assert "Oddiy tilda" not in res_dismissal_formal.answer

    # 3. Увольнение Simple (Russian)
    res_dismissal_ru = engine.process_item(
        item_id="q_dismissal_ru",
        question="меня уволили без предупреждения че делать",
        item_language="ru",
        options=OptionsRequest(style="simple", return_debug=True),
    )
    assert res_dismissal_ru.status == "ok"
    assert "161-modda" in res_dismissal_ru.citations[0].article
    assert "Простыми словами" in res_dismissal_ru.answer
    assert "предупрежд" in res_dismissal_ru.answer.lower() or "компенсаци" in res_dismissal_ru.answer.lower()

    # 4. Зарплата Simple (Uzbek)
    res_wage_simple = engine.process_item(
        item_id="q_wage_simple",
        question="oylik bermasa qayerga murojaat qilish kerak",
        item_language="uz-Latn",
        options=OptionsRequest(style="simple", return_debug=True),
    )
    assert res_wage_simple.status == "ok"
    assert "253-modda" in res_wage_simple.citations[0].article
    assert "Oddiy tilda" in res_wage_simple.answer
    assert "to'lanishi" in res_wage_simple.answer.lower() or "yarim oyda" in res_wage_simple.answer.lower()


def test_gemini_thinking_budget_and_model_fallback(monkeypatch):
    """
    Проверяет, что при вызове Gemini LLM:
    1. Передается thinking_config(thinking_budget=0) для исключения переполнения токенов.
    2. При ошибке модели (например, 503 или 404) срабатывает резервная модель gemini-3.5-flash.
    """
    engine = RAGEngine(search_service=MockSearchService(), llm_model="gemini-custom-model")
    engine.gemini_api_key = "test-key"

    captured_configs = []
    attempted_models = []

    class MockModelResponse:
        text = "xodimning yillik asosiy mehnat ta'tili muddati va davomiyligi"

    class MockGenaiClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, model, contents, config):
            attempted_models.append(model)
            captured_configs.append(config)
            # Симулируем 503 на первой модели и успех на fallback модели
            if model == "gemini-custom-model":
                raise Exception("503 Server Unavailable")
            return MockModelResponse()

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", MockGenaiClient)

    rw = engine.rewrite_query("otpuska qancha kun beriladi ozi", lang="uz-Latn")
    assert rw == "xodimning yillik asosiy mehnat ta'tili muddati va davomiyligi"
    assert "gemini-custom-model" in attempted_models
    assert "gemini-3.5-flash" in attempted_models
    # Проверяем, что thinking_budget=0 был установлен
    assert captured_configs[0].thinking_config.thinking_budget == 0

