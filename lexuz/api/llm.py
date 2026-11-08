"""
lexuz.api.llm

Универсальный интерфейс и клиент для множества LLM провайдеров:
- DeepSeek (https://api.deepseek.com/chat/completions)
- OpenAI (https://api.openai.com/v1/chat/completions)
- Anthropic Claude (https://api.anthropic.com/v1/messages)
- Google Gemini (google-genai SDK)
- Custom / Local LLM (OpenRouter, Groq, Ollama, vLLM через OpenAI-совместимый API)
"""

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class LLMProvider:
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CLAUDE = "claude"  # alias for anthropic
    GEMINI = "gemini"
    CUSTOM = "custom"


PROVIDER_DEEPSEEK = LLMProvider.DEEPSEEK
PROVIDER_OPENAI = LLMProvider.OPENAI
PROVIDER_ANTHROPIC = LLMProvider.ANTHROPIC
PROVIDER_GEMINI = LLMProvider.GEMINI
PROVIDER_CUSTOM = LLMProvider.CUSTOM


DEFAULT_MODELS: Dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-3.5-flash",
    "custom": "gpt-4o-mini",
}

DEFAULT_BASE_URLS: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "",
    "custom": "http://localhost:11434/v1",
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


def is_valid_api_key(key: Optional[str]) -> bool:
    """Проверяет валидность ключа (не пустой, не None и не дефолтный плейсхолдер)."""
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    if not k or k.startswith("YOUR_") or k.startswith("your_"):
        return False
    return True


def normalize_provider(name: Optional[str]) -> str:
    """Нормализует имя провайдера к каноническому виду."""
    if not name:
        return ""
    n = name.strip().lower()
    if n in ("claude", "anthropic"):
        return "anthropic"
    if n in ("deepseek", "deep-seek"):
        return "deepseek"
    if n in ("openai", "chatgpt"):
        return "openai"
    if n in ("gemini", "google", "google-genai"):
        return "gemini"
    if n in ("custom", "local", "ollama", "vllm", "openrouter", "groq"):
        return "custom"
    return n


def detect_provider() -> str:
    """
    Автоматическое определение активного провайдера:
    1. Переменная окружения LLM_PROVIDER.
    2. Приоритет наличия ключей: DEEPSEEK_API_KEY -> OPENAI_API_KEY -> ANTHROPIC_API_KEY -> GEMINI_API_KEY -> LLM_BASE_URL.
    3. Fallback: gemini.
    """
    env_prov = os.environ.get("LLM_PROVIDER")
    if env_prov and env_prov.strip():
        norm = normalize_provider(env_prov.strip())
        if norm:
            return norm

    if is_valid_api_key(os.environ.get("DEEPSEEK_API_KEY")):
        return "deepseek"
    if is_valid_api_key(os.environ.get("OPENAI_API_KEY")):
        return "openai"
    if is_valid_api_key(os.environ.get("ANTHROPIC_API_KEY")) or is_valid_api_key(os.environ.get("CLAUDE_API_KEY")):
        return "anthropic"
    if is_valid_api_key(os.environ.get("GEMINI_API_KEY")):
        return "gemini"
    if os.environ.get("LLM_BASE_URL"):
        return "custom"

    return "gemini"


def get_default_model(provider: str) -> str:
    """Возвращает рекомендуемую модель по умолчанию для провайдера."""
    p = normalize_provider(provider)
    env_model = os.environ.get("LLM_MODEL") or os.environ.get("LEXUZ_LLM_MODEL")
    if env_model:
        # Для Gemini нормализуем устаревшие модели
        if p == "gemini" and ("2.5" in env_model or "1.5" in env_model or not env_model):
            return "gemini-3.5-flash"
        return env_model
    return DEFAULT_MODELS.get(p, "deepseek-chat")


def get_env_api_key(provider: str) -> Optional[str]:
    """Извлекает валидный API ключ из переменных окружения для указанного провайдера."""
    p = normalize_provider(provider)
    raw_key = None
    if p == "deepseek":
        raw_key = os.environ.get("DEEPSEEK_API_KEY")
    elif p == "openai":
        raw_key = os.environ.get("OPENAI_API_KEY")
    elif p == "anthropic":
        raw_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    elif p == "gemini":
        raw_key = os.environ.get("GEMINI_API_KEY")
    elif p == "custom":
        raw_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return raw_key.strip() if is_valid_api_key(raw_key) else None


def get_base_url_for_provider(provider: str, override_url: Optional[str] = None) -> str:
    """Возвращает базовый URL для провайдера."""
    if override_url and override_url.strip():
        return override_url.strip()
    p = normalize_provider(provider)
    env_base = os.environ.get("LLM_BASE_URL")
    if p == "custom":
        return env_base or DEFAULT_BASE_URLS["custom"]
    elif p == "deepseek":
        return os.environ.get("DEEPSEEK_BASE_URL") or env_base or DEFAULT_BASE_URLS["deepseek"]
    elif p == "openai":
        return os.environ.get("OPENAI_BASE_URL") or env_base or DEFAULT_BASE_URLS["openai"]
    elif p == "anthropic":
        return os.environ.get("ANTHROPIC_BASE_URL") or env_base or DEFAULT_BASE_URLS["anthropic"]
    return ""


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """
    Надежное извлечение JSON из ответа LLM:
    1. Прямой парсинг json.loads.
    2. Извлечение из markdown блока ```json ... ``` или ``` ... ```.
    3. Поиск первого { и последнего }.
    4. Безопасный ast.literal_eval (для словарей в Python-нотации с одинарными кавычками).
    5. Fallback-словарь при невозможности парсинга.
    """
    raw_text = (text or "").strip()
    if not raw_text:
        return {
            "answer": "",
            "no_answer": True,
            "confidence": 0.0,
            "used_ranks": [],
        }

    # 1. Прямой json.loads
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Markdown блок ```json ... ``` или ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3. Скобки { ... }
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        snippet = raw_text[first_brace : last_brace + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 4. Попытка ast.literal_eval (если модель ответила в синтаксисе Python dict)
        try:
            eval_data = ast.literal_eval(snippet)
            if isinstance(eval_data, dict):
                return eval_data
        except Exception:
            pass

    # 5. Fallback
    logger.warning("Could not parse JSON from LLM response. Using fallback structure.")
    return {
        "answer": raw_text,
        "no_answer": False,
        "confidence": 0.85,
        "used_ranks": [1],
    }


def call_deepseek(
    prompt: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = False,
    timeout: float = 60.0,
) -> LLMResponse:
    """Вызов DeepSeek Chat API."""
    url = (base_url or "https://api.deepseek.com").rstrip("/")
    endpoint = url if url.endswith("/chat/completions") else f"{url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    mdl = model or "deepseek-chat"
    is_reasoner = "reasoner" in mdl.lower()
    payload: Dict[str, Any] = {
        "model": mdl,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
    }
    # Модели deepseek-reasoner (R1) не поддерживают temperature и response_format
    if not is_reasoner:
        payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        # Если модель отклонила response_format или temperature (400 Bad Request)
        if resp.status_code == 400:
            modified = False
            if "response_format" in payload:
                payload.pop("response_format", None)
                modified = True
            if "temperature" in payload:
                payload.pop("temperature", None)
                modified = True
            if modified:
                resp = client.post(endpoint, headers=headers, json=payload)

        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    text = msg.get("content") or choice.get("text") or msg.get("reasoning_content") or ""
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        raw=data,
    )


def call_openai(
    prompt: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    json_mode: bool = False,
    timeout: float = 60.0,
) -> LLMResponse:
    """Вызов OpenAI API (gpt-4o, gpt-4o-mini)."""
    url = (base_url or "https://api.openai.com/v1").rstrip("/")
    if url.endswith("/chat/completions"):
        endpoint = url
    elif url.endswith("/v1"):
        endpoint = f"{url}/chat/completions"
    else:
        endpoint = f"{url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    mdl = model or "gpt-4o"
    messages: List[Dict[str, str]] = []
    if json_mode:
        messages.append({
            "role": "system",
            "content": "You are a professional legal assistant for Uzbekistan legislation (Lex.uz). Always respond strictly with a valid JSON object.",
        })
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": mdl,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if seed is not None:
        payload["seed"] = seed

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        if resp.status_code == 400 and json_mode and "response_format" in payload:
            payload.pop("response_format", None)
            resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    text = msg.get("content") or choice.get("text") or ""
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        raw=data,
    )


def call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = False,
    timeout: float = 60.0,
) -> LLMResponse:
    """Вызов Anthropic Claude API (claude-3-5-sonnet, claude-3-5-haiku)."""
    url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
    if url.endswith("/messages"):
        endpoint = url
    elif url.endswith("/v1"):
        endpoint = f"{url}/messages"
    else:
        endpoint = f"{url}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    system_prompt = (
        "You are a professional legal assistant for Uzbekistan legislation (Lex.uz). Always output strictly a valid JSON object without any additional markdown formatting or preamble."
        if json_mode
        else "You are a professional legal assistant for Uzbekistan legislation (Lex.uz)."
    )

    payload: Dict[str, Any] = {
        "model": model or "claude-3-5-sonnet-20241022",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "system": system_prompt,
        "temperature": temperature,
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    if data.get("type") == "error":
        err_msg = data.get("error", {}).get("message", "Anthropic Claude API returned an error")
        raise RuntimeError(f"Anthropic error: {err_msg}")

    text_blocks = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    text = "".join(text_blocks)
    if not text and "text" in data:
        text = data["text"]

    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        raw=data,
    )


def call_custom(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    json_mode: bool = False,
    timeout: float = 60.0,
) -> LLMResponse:
    """Вызов Custom/Local LLM через OpenAI-совместимый API (Ollama, vLLM, OpenRouter, Groq)."""
    url = (base_url or "http://localhost:11434/v1").rstrip("/")
    endpoint = url if url.endswith("/chat/completions") else f"{url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    mdl = model or "gpt-4o-mini"
    messages: List[Dict[str, str]] = []
    if json_mode:
        messages.append({
            "role": "system",
            "content": "You are a professional legal assistant for Uzbekistan legislation (Lex.uz). Always respond strictly with a valid JSON object.",
        })
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": mdl,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if seed is not None:
        payload["seed"] = seed

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        if resp.status_code == 400 and json_mode and "response_format" in payload:
            payload.pop("response_format", None)
            resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    text = msg.get("content") or choice.get("text") or ""
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        raw=data,
    )


def call_gemini(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Вызов Google Gemini API через google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"
    if seed is not None:
        cfg_kwargs["seed"] = seed

    config = types.GenerateContentConfig(**cfg_kwargs)

    candidate_models = [model]
    if model != "gemini-3.5-flash":
        candidate_models.append("gemini-3.5-flash")

    last_err = None
    for mdl in candidate_models:
        try:
            resp = client.models.generate_content(
                model=mdl,
                contents=prompt,
                config=config,
            )
            if resp and resp.text:
                return LLMResponse(text=resp.text)
        except Exception as ex:
            last_err = ex
            logger.warning("Gemini attempt with model %s failed: %s", mdl, str(ex))
            continue

    if last_err:
        raise last_err
    raise RuntimeError(f"Gemini generation returned empty response for models {candidate_models}")


def call_llm(
    provider: str,
    prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    json_mode: bool = False,
    timeout: float = 60.0,
) -> LLMResponse:
    """
    Главная диспетчерская функция для вызова любой подключенной LLM.
    """
    p = normalize_provider(provider) or detect_provider()
    key = api_key or get_env_api_key(p)
    mdl = model or get_default_model(p)
    b_url = get_base_url_for_provider(p, base_url)

    if p != "custom" and not key:
        raise ValueError(
            f"API key is not configured for LLM provider '{p}'. "
            f"Please set {p.upper()}_API_KEY in your environment or provide it in options."
        )

    if p == "deepseek":
        return call_deepseek(
            prompt=prompt,
            model=mdl,
            api_key=key,  # type: ignore
            base_url=b_url,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif p == "openai":
        return call_openai(
            prompt=prompt,
            model=mdl,
            api_key=key,  # type: ignore
            base_url=b_url,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif p == "anthropic":
        return call_anthropic(
            prompt=prompt,
            model=mdl,
            api_key=key,  # type: ignore
            base_url=b_url,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif p == "gemini":
        return call_gemini(
            prompt=prompt,
            model=mdl,
            api_key=key,  # type: ignore
            temperature=temperature,
            seed=seed,
            json_mode=json_mode,
        )
    elif p == "custom":
        return call_custom(
            prompt=prompt,
            model=mdl,
            api_key=key,
            base_url=b_url,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            json_mode=json_mode,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")
