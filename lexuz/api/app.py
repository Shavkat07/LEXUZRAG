"""
lexuz.api.app

FastAPI фабрика приложения Legal RAG Lex.uz:
- Полное соответствие спецификации test spetsifikatsiyasi.pdf
- Преобразование ошибок валидации Pydantic в HTTP 400 (согласно требованиям раздела 2 и 5)
- UTF-8 кодирование ответов без экранирования кириллицы и апострофов
- Встроенный веб-дашборд и OpenAPI документация
"""

import logging
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lexuz.api.dashboard import get_dashboard_html
from lexuz.api.routes import router

logger = logging.getLogger("lexuz.api.app")


class UTF8JSONResponse(JSONResponse):
    """Кастомный JSONResponse с явным указанием UTF-8 кодировки в заголовке Content-Type."""
    media_type = "application/json; charset=utf-8"


def create_app() -> FastAPI:
    """Создает и настраивает экземпляр приложения FastAPI."""
    app = FastAPI(
        title="Lex.uz Legal RAG API",
        description="Официальный RAG API для интеллектуального поиска и генерации ответов по законодательству Узбекистана.",
        version="0.4.2",
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=UTF8JSONResponse,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Спецификация строго требует HTTP 400 для ошибок валидации (10+ вопросов, дубликаты, пусто):
    # Раздел 2: "Batch hajmi: items 1–10 ta. 10 dan ko'p bo'lsa 400"
    # Раздел 5: "400: Validatsiya: 10 dan ko'p savol, takroriy id, bo'sh question"
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        clean_errors = []
        error_msgs = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            error_msgs.append(f"{loc}: {msg}")
            clean_errors.append({
                "loc": list(err.get("loc", [])),
                "msg": msg,
                "type": err.get("type", "value_error"),
            })

        detail_str = "; ".join(error_msgs)
        logger.warning("Validation error on %s: %s", request.url.path, detail_str)

        return UTF8JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": detail_str,
                "errors": clean_errors,
            },
        )

    # Подключение маршрутов API
    app.include_router(router)

    # Веб-дашборд
    @app.get("/", include_in_schema=False)
    def root_dashboard():
        return get_dashboard_html()

    @app.get("/dashboard", include_in_schema=False)
    def web_dashboard():
        return get_dashboard_html()

    return app


# Инстанс для Uvicorn (uvicorn lexuz.api.app:app)
app = create_app()
