"""
tests/test_multi_llm.py

Unit tests for multi-LLM architecture:
- DeepSeek (chat, reasoner)
- OpenAI (gpt-4o, gpt-4o-mini)
- Anthropic Claude (claude-3-5-sonnet, claude-3-5-haiku)
- Custom / Local LLM (OpenRouter, Groq, Ollama)
- Auto-detection logic
- JSON extraction resilience
- CLI arguments & Engine options override
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lexuz.api.engine import MockSearchService, RAGEngine
from lexuz.api.llm import (
    call_anthropic,
    call_custom,
    call_deepseek,
    call_llm,
    call_openai,
    detect_provider,
    extract_json_from_llm_response,
    get_base_url_for_provider,
    get_default_model,
    is_valid_api_key,
    normalize_provider,
)
from lexuz.api.schemas import OptionsRequest


# ==============================================================================
# 1. Provider Resolution & Auto-Detection Tests
# ==============================================================================

def test_normalize_provider():
    assert normalize_provider("deepseek") == "deepseek"
    assert normalize_provider("Deep-Seek") == "deepseek"
    assert normalize_provider("openai") == "openai"
    assert normalize_provider("chatgpt") == "openai"
    assert normalize_provider("claude") == "anthropic"
    assert normalize_provider("Anthropic") == "anthropic"
    assert normalize_provider("gemini") == "gemini"
    assert normalize_provider("google-genai") == "gemini"
    assert normalize_provider("ollama") == "custom"
    assert normalize_provider("openrouter") == "custom"
    assert normalize_provider("local") == "custom"


def test_detect_provider_precedence(monkeypatch):
    # Case 1: Explicit LLM_PROVIDER has highest priority
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert detect_provider() == "deepseek"

    # Case 2: No LLM_PROVIDER -> DEEPSEEK_API_KEY takes precedence
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert detect_provider() == "deepseek"

    # Case 3: No DeepSeek key -> OPENAI_API_KEY takes precedence
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert detect_provider() == "openai"

    # Case 4: No OpenAI key -> ANTHROPIC_API_KEY takes precedence
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert detect_provider() == "anthropic"

    # Case 5: No Anthropic key -> GEMINI_API_KEY
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert detect_provider() == "gemini"

    # Case 6: No Gemini key -> LLM_BASE_URL gives custom
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert detect_provider() == "custom"

    # Case 7: Nothing configured -> default fallback to gemini
    monkeypatch.delenv("LLM_BASE_URL")
    assert detect_provider() == "gemini"


def test_get_default_model():
    assert get_default_model("deepseek") == "deepseek-chat"
    assert get_default_model("openai") == "gpt-4o"
    assert get_default_model("anthropic") == "claude-3-5-sonnet-20241022"
    assert get_default_model("gemini") == "gemini-3.5-flash"


def test_get_base_url_for_provider():
    assert get_base_url_for_provider("deepseek") == "https://api.deepseek.com"
    assert get_base_url_for_provider("openai") == "https://api.openai.com/v1"
    assert get_base_url_for_provider("anthropic") == "https://api.anthropic.com/v1"
    assert get_base_url_for_provider("custom", "http://custom-host:8080/v1") == "http://custom-host:8080/v1"


# ==============================================================================
# 2. JSON Extraction Resilience Tests
# ==============================================================================

def test_extract_json_from_llm_response_clean():
    payload = {"answer": "Mehnat ta'tili 21 kun [1].", "no_answer": False, "confidence": 0.95, "used_ranks": [1]}
    res = extract_json_from_llm_response(json.dumps(payload))
    assert res["answer"] == "Mehnat ta'tili 21 kun [1]."
    assert res["confidence"] == 0.95
    assert res["used_ranks"] == [1]


def test_extract_json_from_llm_response_markdown_fence():
    raw = """
Mana sizga javob:
```json
{
  "answer": "Yillik ta'til 21 kalendar kunidan kam bo'lmasligi lozim [1].",
  "no_answer": false,
  "confidence": 0.91,
  "used_ranks": [1]
}
```
Umid qilaman yordam berdi.
"""
    res = extract_json_from_llm_response(raw)
    assert "21 kalendar" in res["answer"]
    assert res["confidence"] == 0.91
    assert res["no_answer"] is False


def test_extract_json_from_llm_response_plain_text_fallback():
    raw = "Mehnat kodeksining 216-moddasiga ko'ra ta'til 21 kun [1]."
    res = extract_json_from_llm_response(raw)
    assert res["answer"] == raw
    assert res["no_answer"] is False
    assert res["confidence"] == 0.85
    assert res["used_ranks"] == [1]


# ==============================================================================
# 3. Provider Call Mocks (DeepSeek, OpenAI, Anthropic, Custom)
# ==============================================================================

def test_call_deepseek_mock(monkeypatch):
    captured_requests = []

    def mock_post(url, headers, json, timeout):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": json_payload_str,
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    json_payload_str = json.dumps({
        "answer": "DeepSeek javobi: 21 kun [1].",
        "no_answer": False,
        "confidence": 0.94,
        "used_ranks": [1],
    })

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    resp = call_deepseek(
        prompt="Savol: Ta'til necha kun?",
        model="deepseek-chat",
        api_key="sk-deepseek-test",
        json_mode=True,
    )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "https://api.deepseek.com/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-deepseek-test"
    assert req["json"]["model"] == "deepseek-chat"
    assert req["json"]["response_format"] == {"type": "json_object"}
    assert resp.input_tokens == 120
    assert resp.output_tokens == 45
    assert "DeepSeek javobi" in resp.text


def test_call_openai_mock(monkeypatch):
    captured_requests = []

    def mock_post(url, headers, json, timeout):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "OpenAI javobi [1].", "no_answer": false, "confidence": 0.98, "used_ranks": [1]}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 85, "completion_tokens": 30},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    resp = call_openai(
        prompt="Savol: OpenAI orqali",
        model="gpt-4o",
        api_key="sk-openai-test",
        json_mode=True,
    )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "https://api.openai.com/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-openai-test"
    assert req["json"]["model"] == "gpt-4o"
    assert "OpenAI javobi" in resp.text


def test_call_anthropic_mock(monkeypatch):
    captured_requests = []

    def mock_post(url, headers, json, timeout):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        response_data = {
            "content": [
                {
                    "type": "text",
                    "text": '{"answer": "Claude javobi [1].", "no_answer": false, "confidence": 0.96, "used_ranks": [1]}',
                }
            ],
            "usage": {"input_tokens": 150, "output_tokens": 50},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    resp = call_anthropic(
        prompt="Savol: Claude orqali",
        model="claude-3-5-sonnet-20241022",
        api_key="sk-ant-test",
        json_mode=True,
    )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-ant-test"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert "Claude javobi" in resp.text
    assert resp.input_tokens == 150
    assert resp.output_tokens == 50


def test_call_custom_local_llm_mock(monkeypatch):
    captured_requests = []

    def mock_post(url, headers, json, timeout):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "Ollama javobi [1].", "no_answer": false, "confidence": 0.89, "used_ranks": [1]}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 20},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    resp = call_custom(
        prompt="Savol: Local Ollama orqali",
        model="llama3",
        base_url="http://localhost:11434/v1",
        api_key=None,
        json_mode=True,
    )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "http://localhost:11434/v1/chat/completions"
    assert req["json"]["model"] == "llama3"
    assert "Ollama javobi" in resp.text


# ==============================================================================
# 4. End-to-End RAG Engine with Multi-LLM Providers
# ==============================================================================

def test_engine_process_item_with_deepseek(monkeypatch):
    def mock_post(url, headers, json, timeout):
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "DeepSeek: Mehnat ta\'tili eng kamida 21 kalendar kuni [1].", "no_answer": false, "confidence": 0.95, "used_ranks": [1]}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="deepseek",
        llm_api_key="sk-test-deepseek-key",
    )
    options = OptionsRequest(
        top_k=3,
        return_contexts=True,
        return_debug=True,
        provider="deepseek",
        model="deepseek-chat",
    )

    res = engine.process_item(
        item_id="q_deepseek_1",
        question="Ta'til muddati qancha?",
        item_language="uz-Latn",
        options=options,
    )

    assert res.status == "ok"
    assert "DeepSeek: Mehnat ta'tili" in res.answer
    assert res.llm_provider == "deepseek"
    assert res.llm_model == "deepseek-chat"
    assert res.debug is not None
    assert res.debug.llm_provider == "deepseek"
    assert len(res.citations) >= 1


def test_engine_process_item_with_claude(monkeypatch):
    def mock_post(url, headers, json, timeout):
        response_data = {
            "content": [
                {
                    "type": "text",
                    "text": '{"answer": "Claude: Ta\'til muddati 21 kun qilib belgilangan [1].", "no_answer": false, "confidence": 0.97, "used_ranks": [1]}',
                }
            ],
            "usage": {"input_tokens": 110, "output_tokens": 35},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="anthropic",
        llm_api_key="sk-ant-test-key",
    )
    options = OptionsRequest(
        top_k=3,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
    )

    res = engine.process_item(
        item_id="q_claude_1",
        question="Ta'til muddati qancha?",
        item_language="uz-Latn",
        options=options,
    )

    assert res.status == "ok"
    assert "Claude: Ta'til" in res.answer
    assert res.llm_provider == "anthropic"
    assert res.llm_model == "claude-3-5-sonnet-20241022"


def test_engine_query_rewriting_with_deepseek(monkeypatch):
    def mock_post(url, headers, json, timeout):
        # Plain text search query for rewrite
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "xodimning yillik asosiy mehnat tatili davomiyligi",
                    }
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="deepseek",
        llm_api_key="sk-deepseek-test",
    )
    rw = engine.rewrite_query("otpuska qancha kun beriladi ozi", lang="uz-Latn")
    assert rw == "xodimning yillik asosiy mehnat tatili davomiyligi"


def test_engine_system_info_reflects_active_provider():
    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="deepseek",
        llm_model="deepseek-chat",
    )
    info = engine.get_system_info()
    assert info.llm_provider == "deepseek"
    assert info.llm_model == "deepseek-chat"


def test_engine_process_item_with_openai(monkeypatch):
    def mock_post(url, headers, json, timeout):
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "OpenAI: Срок давности 3 года [1].", "no_answer": false, "confidence": 0.95, "used_ranks": [1]}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="openai",
        llm_api_key="sk-test-openai-key",
    )
    options = OptionsRequest(provider="openai", model="gpt-4o-mini")
    res = engine.process_item(
        item_id="q_oai_1",
        question="Срок давности?",
        item_language="ru",
        options=options,
    )

    assert res.status == "ok"
    assert "OpenAI: Срок" in res.answer
    assert res.llm_provider == "openai"
    assert res.llm_model == "gpt-4o-mini"
    assert len(res.citations) >= 1


def test_engine_process_item_with_custom_endpoint(monkeypatch):
    captured_urls = []

    def mock_post(url, headers, json, timeout):
        captured_urls.append(url)
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "Custom vLLM: javob [1].", "no_answer": false, "confidence": 0.88, "used_ranks": [1]}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 25},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    custom_url = "http://localhost:8000/v1"
    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="custom",
        llm_base_url=custom_url,
    )
    options = OptionsRequest(provider="custom", base_url=custom_url, model="meta-llama-3")
    res = engine.process_item(
        item_id="q_cust_1",
        question="Savol?",
        item_language="uz-Latn",
        options=options,
    )

    assert res.status == "ok"
    assert res.llm_provider == "custom"
    assert "Custom vLLM" in res.answer
    assert len(captured_urls) == 1
    assert captured_urls[0] == f"{custom_url}/chat/completions"


def test_llm_failure_triggers_grounded_fallback(monkeypatch):
    def mock_failing_post(self, url, **kwargs):
        raise httpx.ConnectTimeout("Connection to LLM provider timed out")

    monkeypatch.setattr(httpx.Client, "post", mock_failing_post)

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="deepseek",
        llm_api_key="sk-deepseek-invalid",
    )

    res = engine.process_item(
        item_id="q_fail_safe_1",
        question="Mehnat ta'tili muddati qancha?",
        item_language="uz-Latn",
        options=OptionsRequest(provider="deepseek"),
    )

    assert res.status == "ok"
    assert "[1]" in res.answer
    assert len(res.citations) >= 1
    assert res.citations[0].chunk_id in [c.chunk_id for c in res.contexts]


def test_options_request_provider_sync():
    opt1 = OptionsRequest(provider="deepseek", model="deepseek-chat")
    assert opt1.provider == "deepseek"
    assert opt1.model == "deepseek-chat"

    opt2 = OptionsRequest(llm_provider="claude", llm_model="claude-3-5-haiku-20241022")
    assert opt2.provider == "claude"
    assert opt2.model == "claude-3-5-haiku-20241022"


def test_cli_ask_with_provider_flag():
    import sys
    import subprocess
    cmd = [
        sys.executable,
        "main.py",
        "ask",
        "mehnat ta'tili muddati qancha?",
        "--mock",
        "--provider",
        "deepseek",
        "--model",
        "deepseek-chat",
    ]
    env_mock = dict(os.environ, LEXUZ_FORCE_LOCAL_LLM="1")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env_mock)
    assert res.returncode == 0
    assert "LLM: deepseek (deepseek-chat)" in res.stdout
    assert "Oddiy tilda" in res.stdout


# ==============================================================================
# 5. Robustness & Edge Cases Tests
# ==============================================================================

def test_is_valid_api_key():
    assert is_valid_api_key("sk-1234567890") is True
    assert is_valid_api_key(None) is False
    assert is_valid_api_key("") is False
    assert is_valid_api_key("   ") is False
    assert is_valid_api_key("YOUR_DEEPSEEK_API_KEY_HERE") is False
    assert is_valid_api_key("YOUR_OPENAI_API_KEY_HERE") is False
    assert is_valid_api_key("your_key_if_needed") is False


def test_deepseek_reasoner_payload_cleanliness(monkeypatch):
    """Проверка, что для deepseek-reasoner не отправляются temperature и response_format."""
    captured_payloads = []

    def mock_post(url, headers, json, timeout):
        captured_payloads.append(json)
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "R1 javobi [1].", "no_answer": false, "confidence": 0.99, "used_ranks": [1]}',
                        "reasoning_content": "Fikrlash jarayoni...",
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 100},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    resp = call_deepseek(
        prompt="Savol: Da'vo muddati?",
        model="deepseek-reasoner",
        api_key="sk-deepseek-r1",
        temperature=0.7,
        json_mode=True,
    )

    assert len(captured_payloads) == 1
    p = captured_payloads[0]
    assert p["model"] == "deepseek-reasoner"
    assert "temperature" not in p
    assert "response_format" not in p
    assert "R1 javobi" in resp.text


def test_empty_choices_resilience(monkeypatch):
    """Проверка защиты от IndexError при возврате пустого списка choices upstream API."""
    def mock_empty_choices(url, headers, json, timeout):
        return httpx.Response(200, json={"choices": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_empty_choices(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    resp_ds = call_deepseek("prompt", "deepseek-chat", "sk-test")
    assert resp_ds.text == ""

    resp_oai = call_openai("prompt", "gpt-4o", "sk-test")
    assert resp_oai.text == ""

    resp_cust = call_custom("prompt", "local-model", "sk-test")
    assert resp_cust.text == ""


def test_extract_json_python_literal_eval_fallback():
    """Проверка извлечения словаря с одинарными кавычками (Python format)."""
    raw = "{'answer': 'Qonuniy tatil 21 kun [1].', 'no_answer': False, 'confidence': 0.9, 'used_ranks': [1]}"
    res = extract_json_from_llm_response(raw)
    assert res["answer"] == "Qonuniy tatil 21 kun [1]."
    assert res["confidence"] == 0.9
    assert res["no_answer"] is False
    assert res["used_ranks"] == [1]


def test_engine_handles_non_list_used_ranks_and_null_confidence(monkeypatch):
    """Проверка безопасной обработки некорректных форматов confidence и used_ranks из LLM."""
    def mock_post(url, headers, json, timeout):
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "Tahlil natijasi [1].", "no_answer": false, "confidence": null, "used_ranks": "1"}',
                    }
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 20},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="deepseek",
        llm_api_key="sk-test",
    )
    res = engine.process_item(
        item_id="q_ranks_test",
        question="Ta'til muddati?",
        item_language="uz-Latn",
        options=OptionsRequest(provider="deepseek"),
    )
    assert res.status == "ok"
    assert res.confidence == 0.85
    assert len(res.citations) >= 1
    assert "Tahlil natijasi" in res.answer


def test_query_rewriting_cleans_markdown_and_prefix(monkeypatch):
    """Проверка очистки префиксов 'Rewritten Legal Query:' и markdown из ответа LLM."""
    def mock_post(url, headers, json, timeout):
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "Rewritten Legal Query: **xodimning yillik asosiy mehnat ta'tili davomiyligi**\nIzoh: Bu rasmiy norma.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15},
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, kwargs.get("headers"), kwargs.get("json"), kwargs.get("timeout")))

    engine = RAGEngine(
        search_service=MockSearchService(),
        llm_provider="deepseek",
        llm_api_key="sk-test",
    )
    rw = engine.rewrite_query("otpuska necha kun ozi", lang="uz-Latn")
    assert rw == "xodimning yillik asosiy mehnat ta'tili davomiyligi"


def test_cli_case_insensitive_provider():
    import sys
    import subprocess
    cmd = [
        sys.executable,
        "main.py",
        "ask",
        "ta'til muddati",
        "--mock",
        "--provider",
        "DeepSeek",
    ]
    env_mock = dict(os.environ, LEXUZ_FORCE_LOCAL_LLM="1")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env_mock)
    assert res.returncode == 0
    assert "deepseek" in res.stdout.lower()
