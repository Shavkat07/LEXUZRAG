"""
lexuz.api.limiter

Ограничитель частоты запросов (Rate Limiter):
- Требование спецификации: "kamida 1 batch/soniya; oshib ketsa 429 + Retry-After"
- Возвращает HTTP 429 Too Many Requests со стандартным заголовком Retry-After
"""

import os
import time
import threading
from typing import Dict, Tuple
from fastapi import HTTPException, Request, status


class RateLimiter:
    """Потокобезопасный ограничитель на базе алгоритма Token Bucket."""

    def __init__(self, rate_per_second: float = 2.0, capacity: float = 5.0):
        self.rate = rate_per_second
        self.capacity = capacity
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._lock = threading.Lock()

    def check(self, key: str = "global", force: bool = False) -> None:
        """Проверяет возможность выполнения запроса. При превышении бросает 429."""
        if not force:
            if os.environ.get("LEXUZ_DISABLE_RATE_LIMIT") == "1":
                return
            if os.environ.get("TESTING") == "1" and os.environ.get("LEXUZ_ENABLE_TEST_RATE_LIMIT") != "1":
                return

        now = time.monotonic()
        with self._lock:
            tokens, last_update = self._buckets.get(key, (self.capacity, now))
            # Пополняем токены со временем
            elapsed = now - last_update
            tokens = min(self.capacity, tokens + elapsed * self.rate)

            if tokens < 1.0:
                retry_after = max(1, int((1.0 - tokens) / self.rate) + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Minimum capacity is {self.rate} req/s. Retry after {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Списываем 1 токен
            self._buckets[key] = (tokens - 1.0, now)


global_rate_limiter = RateLimiter(
    rate_per_second=float(os.environ.get("LEXUZ_RATE_LIMIT_RPS", "2.0")),
    capacity=float(os.environ.get("LEXUZ_RATE_LIMIT_BURST", "5.0")),
)


def check_rate_limit(request: Request) -> None:
    """Зависимость FastAPI для проверки лимита на входящие запросы."""
    client_ip = request.client.host if request.client else "unknown"
    global_rate_limiter.check(f"ip:{client_ip}")
