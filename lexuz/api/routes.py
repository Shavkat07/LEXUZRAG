"""
lexuz.api.routes

Маршруты REST API для Legal RAG Lex.uz.
Реализует все требования test spetsifikatsiyasi.pdf:
- POST /v1/rag/batch
- GET  /v1/rag/batch/{batch_id}
- POST /v1/rag/query
- GET  /v1/meta
- GET  /v1/health
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

# Uzbekistan Time Zone (UTC+5) as specified in test specification examples
UZ_TZ = timezone(timedelta(hours=5))

from lexuz.api.auth import verify_api_key
from lexuz.api.engine import RAGEngine
from lexuz.api.limiter import check_rate_limit
from lexuz.api.schemas import (
    AsyncBatchStatusResponse,
    BatchRequest,
    BatchResponse,
    BatchSummary,
    BatchTiming,
    HealthResponse,
    ItemResult,
    QueryRequest,
    QueryResponse,
    SystemInfo,
)
from lexuz.api.store import BatchConflictError, BatchStore

logger = logging.getLogger("lexuz.api.routes")

router = APIRouter()

# Синглтоны хранилища и движка
_store: Optional[BatchStore] = None
_engine: Optional[RAGEngine] = None
_start_time = time.monotonic()


def get_store() -> BatchStore:
    global _store
    if _store is None:
        _store = BatchStore()
    return _store


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


@router.get("/v1/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """GET /v1/health - Проверка доступности сервиса и компонентов."""
    engine = get_engine()
    qdrant_info: Dict[str, Any] = {"status": "unknown"}

    try:
        if hasattr(engine.search_service, "retriever") and hasattr(engine.search_service.retriever, "qdrant_manager"):
            mgr = engine.search_service.retriever.qdrant_manager
            collections = mgr.client.get_collections().collections
            names = [c.name for c in collections]
            coll_info = mgr.client.get_collection("lexuz_collection")
            qdrant_info = {
                "status": "connected",
                "collections": names,
                "points_count": getattr(coll_info, "points_count", None),
            }
        else:
            qdrant_info = {"status": "connected", "mode": "in_memory_or_mock"}
    except Exception as e:
        qdrant_info = {"status": "error", "error": str(e)}

    uptime = time.monotonic() - _start_time
    health_status = "ok" if qdrant_info.get("status") in ("connected", "ok", "unknown") else "degraded"
    return HealthResponse(
        status=health_status,
        version="0.4.2",
        qdrant=qdrant_info,
        gemini_configured=bool(engine.gemini_api_key),
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(UZ_TZ).isoformat(),
    )


@router.get("/v1/meta", response_model=SystemInfo, tags=["System"])
def get_meta() -> SystemInfo:
    """GET /v1/meta - Версии моделей, индекса и снапшота Lex.uz."""
    engine = get_engine()
    return engine.get_system_info()


def run_batch_worker(
    req: BatchRequest,
    store: BatchStore,
    engine: RAGEngine,
    received_at_str: str,
    t_start: float,
) -> None:
    """Фоновый воркер для асинхронной обработки пакета (HTTP 202)."""
    try:
        results: List[ItemResult] = []
        summary_ok = 0
        summary_no_ans = 0
        summary_err = 0

        for item in req.items:
            res = engine.process_item(
                item_id=item.id,
                question=item.question,
                item_language=item.language,
                options=req.options,
            )
            results.append(res)
            if res.status == "ok":
                summary_ok += 1
            elif res.status == "no_answer":
                summary_no_ans += 1
            else:
                summary_err += 1

        completed_at_dt = datetime.now(UZ_TZ)
        completed_at_str = completed_at_dt.isoformat()
        total_elapsed_ms = int((time.monotonic() - t_start) * 1000)

        batch_resp = BatchResponse(
            run_id=req.run_id,
            batch_id=req.batch_id,
            status="completed",
            system=engine.get_system_info(),
            timing=BatchTiming(
                received_at=received_at_str,
                completed_at=completed_at_str,
                total_ms=total_elapsed_ms,
            ),
            summary=BatchSummary(
                total=len(req.items),
                ok=summary_ok,
                no_answer=summary_no_ans,
                error=summary_err,
            ),
            results=results,
        )
        store.save_completed_batch(req.batch_id, batch_resp.model_dump())
    except Exception as e:
        logger.exception("Async worker failed for batch_id=%s: %s", req.batch_id, str(e))
        store.mark_batch_failed(req.batch_id, str(e))


@router.post(
    "/v1/rag/batch",
    response_model=BatchResponse,
    responses={
        200: {"model": BatchResponse, "description": "Пакет успешно обработан"},
        202: {"model": AsyncBatchStatusResponse, "description": "Пакет принят в асинхронную обработку"},
        400: {"description": "Ошибка валидации (10+ вопросов, дубликаты, пусто)"},
        401: {"description": "Недействительный или отсутствующий API ключ"},
        409: {"description": "Конфликт batch_id (существует с другим содержимым)"},
        429: {"description": "Превышен rate limit"},
    },
    tags=["RAG"],
)
def process_batch(
    req: BatchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    mode: Optional[str] = Query(None, description="Режим выполнения: 'sync' или 'async'"),
    async_mode: Optional[bool] = Query(None, alias="async", description="Флаг асинхронного режима"),
    api_key: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
) -> Any:
    """POST /v1/rag/batch - Пакетная обработка 1-10 вопросов."""
    received_at_dt = datetime.now(UZ_TZ)
    received_at_str = received_at_dt.isoformat()
    t_start = time.monotonic()

    store = get_store()
    engine = get_engine()

    # 1. Проверка идемпотентности и регистрация в хранилище
    try:
        status_in_store, cached_data = store.check_or_register_batch(
            batch_id=req.batch_id,
            run_id=req.run_id,
            request_data=req.model_dump(),
        )
    except BatchConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    # Если уже вычислен ранее - возвращаем сохраненный результат (HTTP 200 Idempotent)
    if status_in_store == "completed" and cached_data:
        return cached_data

    # Если батч уже в процессе обработки - возвращаем HTTP 202
    if status_in_store == "processing":
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"batch_id": req.batch_id, "status": "processing"},
        )

    # Проверка на запрос асинхронного режима
    is_async = (
        async_mode is True
        or mode == "async"
        or request.headers.get("Prefer") == "respond-async"
    )

    if is_async:
        background_tasks.add_task(
            run_batch_worker,
            req=req,
            store=store,
            engine=engine,
            received_at_str=received_at_str,
            t_start=t_start,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"batch_id": req.batch_id, "status": "processing"},
        )

    # 2. Синхронная последовательная обработка вопросов батча с изоляцией ошибок
    results: List[ItemResult] = []
    summary_ok = 0
    summary_no_ans = 0
    summary_err = 0

    for item in req.items:
        res = engine.process_item(
            item_id=item.id,
            question=item.question,
            item_language=item.language,
            options=req.options,
        )
        results.append(res)
        if res.status == "ok":
            summary_ok += 1
        elif res.status == "no_answer":
            summary_no_ans += 1
        else:
            summary_err += 1

    completed_at_dt = datetime.now(UZ_TZ)
    completed_at_str = completed_at_dt.isoformat()
    total_elapsed_ms = int((time.monotonic() - t_start) * 1000)

    # 3. Сборка полного ответа Section 4
    batch_resp = BatchResponse(
        run_id=req.run_id,
        batch_id=req.batch_id,
        status="completed",
        system=engine.get_system_info(),
        timing=BatchTiming(
            received_at=received_at_str,
            completed_at=completed_at_str,
            total_ms=total_elapsed_ms,
        ),
        summary=BatchSummary(
            total=len(req.items),
            ok=summary_ok,
            no_answer=summary_no_ans,
            error=summary_err,
        ),
        results=results,
    )

    # 4. Сохранение в SQLite хранилище
    store.save_completed_batch(req.batch_id, batch_resp.model_dump())

    return batch_resp


@router.get(
    "/v1/rag/batch/{batch_id}",
    responses={
        200: {"model": BatchResponse, "description": "Результат завершенного пакета"},
        202: {"model": AsyncBatchStatusResponse, "description": "Пакет еще в процессе обработки"},
        401: {"description": "Неавторизован"},
        404: {"description": "Пакет не найден"},
    },
    tags=["RAG"],
)
def get_batch_by_id(
    batch_id: str,
    api_key: str = Depends(verify_api_key),
) -> Any:
    """GET /v1/rag/batch/{batch_id} - Получение сохраненного результата или статуса батча."""
    store = get_store()
    record = store.get_batch(batch_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found.",
        )

    current_status = record["status"]
    if current_status == "processing":
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"batch_id": batch_id, "status": "processing"},
        )
    elif current_status == "completed":
        return record["response"]
    elif current_status == "failed":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"batch_id": batch_id, "status": "failed", "data": record.get("response")},
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"batch_id": batch_id, "status": current_status, "data": record.get("response")},
        )


@router.post("/v1/rag/query", response_model=QueryResponse, tags=["RAG"])
def query_single(
    req: QueryRequest,
    api_key: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit),
) -> QueryResponse:
    """POST /v1/rag/query - Отладочный одиночный запрос."""
    engine = get_engine()
    from lexuz.api.schemas import OptionsRequest
    opts = req.options if req.options else OptionsRequest()

    res = engine.process_item(
        item_id=req.id or "q_debug",
        question=req.question,
        item_language=req.language,
        options=opts,
    )

    return QueryResponse(
        system=engine.get_system_info(),
        result=res,
    )


@router.get("/api/dashboard/recent-batches", tags=["Dashboard"])
def get_recent_batches(
    limit: int = Query(20, ge=1, le=100),
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Вспомогательный эндпоинт для отображения недавних пакетов на дашборде."""
    store = get_store()
    return store.list_recent_batches(limit=limit)
