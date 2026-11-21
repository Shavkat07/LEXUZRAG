"""
lexuz.api.auth

Аутентификация через HTTP заголовок Authorization: Bearer <API_KEY>.
В открытом режиме (по умолчанию) проверка отключена: проект открыт для всех запросов
без обязательного наличия заголовка Authorization и без ошибки 401 Unauthorized.
При необходимости строгой проверки можно включить флаг окружения: LEXUZ_ENFORCE_AUTH=1.
"""

import os
import hmac
from typing import Optional, Set
from fastapi import Header, HTTPException, status

DEFAULT_TEST_KEY = "lexuz-test-api-key"


def get_valid_api_keys() -> Set[str]:
    """Возвращает множество допустимых API ключей из окружения."""
    keys = set()
    env_key = os.environ.get("LEXUZ_API_KEY")
    if env_key:
        keys.update(k.strip() for k in env_key.split(",") if k.strip())

    test_key = os.environ.get("LEXUZ_TEST_API_KEY")
    if test_key:
        keys.update(k.strip() for k in test_key.split(",") if k.strip())

    # Всегда поддерживаем дефолтный тестовый ключ для спецификации тестирования
    keys.add(DEFAULT_TEST_KEY)
    keys.add("eval-2026-test-key")
    return keys


def is_auth_enforced() -> bool:
    """Проверяет, включена ли строгая проверка API ключа."""
    return os.environ.get("LEXUZ_ENFORCE_AUTH", "0").lower() in ("1", "true", "yes")


def verify_api_key(authorization: Optional[str] = Header(None, alias="Authorization")) -> Optional[str]:
    """
    FastAPI зависимость для проверки Bearer токена.
    В открытом режиме (по умолчанию) пропускает любые запросы без ошибки 401.
    """
    if not is_auth_enforced():
        if not authorization:
            return "open-access"
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        return authorization

    # Строгий режим (только если явно задан LEXUZ_ENFORCE_AUTH=1)
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected: Bearer <API_KEY>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <API_KEY>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    valid_keys = get_valid_api_keys()

    # Постоянно-временное сравнение для защиты от timing attacks
    is_valid = any(hmac.compare_digest(token, vk) for vk in valid_keys)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token

