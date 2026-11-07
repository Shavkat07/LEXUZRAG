"""
lexuz.api.schemas

Pydantic схемы данных для REST API RAG Lex.uz.
Полное соответствие документу: RAG API — test spetsifikatsiyasi (lex.uz).
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OptionsRequest(BaseModel):
    """Опции выполнения RAG запроса."""

    model_config = ConfigDict(extra="ignore")

    top_k: int = Field(default=5, ge=1, le=50, description="Количество возвращаемых контекстов")
    return_contexts: bool = Field(default=True, description="Возвращать ли текст найденных фрагментов")
    return_debug: bool = Field(default=False, description="Возвращать ли отладочную информацию")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Температура генерации")
    seed: Optional[int] = Field(default=None, description="Seed для детерминированной генерации")
    as_of_date: Optional[str] = Field(default=None, description="Дата актуальности редакции (YYYY-MM-DD)")
    answer_language: Optional[str] = Field(
        default="auto",
        description="auto = язык вопроса; uz-Latn, uz-Cyrl, ru",
    )
    style: Optional[str] = Field(
        default="simple",
        description="Стиль ответа: 'simple' (простой, понятный гражданам) или 'formal' (строгий юридический)",
    )

    @field_validator("answer_language")
    @classmethod
    def validate_answer_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("auto", "uz-Latn", "uz-Cyrl", "ru"):
            raise ValueError("answer_language must be one of: 'auto', 'uz-Latn', 'uz-Cyrl', 'ru'")
        return v

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "simple"
        val = v.strip().lower()
        if val in ("simple", "oddiy", "простой", "plain"):
            return "simple"
        if val in ("formal", "rasmiy", "официальный"):
            return "formal"
        raise ValueError("style must be one of: 'simple', 'formal' (or aliases 'oddiy', 'rasmiy')")


class ItemRequest(BaseModel):
    """Единичный вопрос в пакете (Batch Item)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., min_length=1, max_length=100, description="Уникальный идентификатор вопроса (q001)")
    question: str = Field(..., min_length=1, max_length=2000, description="Текст вопроса")
    language: Optional[str] = Field(default=None, description="Язык вопроса (uz-Latn, uz-Cyrl, ru)")

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace only")
        return stripped

    @field_validator("language")
    @classmethod
    def validate_item_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("uz-Latn", "uz-Cyrl", "ru"):
            raise ValueError("language must be one of: 'uz-Latn', 'uz-Cyrl', 'ru'")
        return v


class BatchRequest(BaseModel):
    """Пакетный запрос (POST /v1/rag/batch)."""

    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(..., min_length=1, description="Идентификатор сессии тестирования (eval-2026-11-03)")
    batch_id: str = Field(..., min_length=1, description="Уникальный ID пакета / ключ идемпотентности")
    batch_index: int = Field(..., ge=1, description="Номер пакета в сессии (1..10)")
    total_batches: int = Field(..., ge=1, description="Всего пакетов в сессии")
    options: OptionsRequest = Field(default_factory=OptionsRequest, description="Общие настройки")
    items: List[ItemRequest] = Field(..., description="Список вопросов (1-10)")

    @model_validator(mode="after")
    def validate_batch_items(self) -> "BatchRequest":
        if self.batch_index > self.total_batches:
            raise ValueError(f"batch_index ({self.batch_index}) cannot exceed total_batches ({self.total_batches})")

        if not self.items or len(self.items) < 1:
            raise ValueError("items must contain at least 1 question")
        if len(self.items) > 10:
            raise ValueError(f"Batch items count ({len(self.items)}) exceeds maximum limit of 10")

        # Проверка уникальности id внутри батча
        seen_ids = set()
        duplicates = set()
        for item in self.items:
            if item.id in seen_ids:
                duplicates.add(item.id)
            seen_ids.add(item.id)

        if duplicates:
            raise ValueError(f"Duplicate item IDs found in batch: {sorted(list(duplicates))}")

        return self


class Citation(BaseModel):
    """Ссылка на нормативный источник Lex.uz."""

    ref: int = Field(..., description="Номер ссылки в ответе [1], [2]")
    doc_id: str = Field(..., description="ID документа в формате lex:XXXXXXX")
    doc_title: str = Field(..., description="Название нормативного акта")
    doc_type: str = Field(..., description="Тип акта (kodeks, qonun, qaror, farmon, buyruq, etc.)")
    url: str = Field(..., description="Прямой URL на портал Lex.uz")
    article: str = Field(..., description="Статья или пункт (XXX-modda, XXX-band)")
    part: Optional[str] = Field(default=None, description="Часть нормы (1-qism)")
    edition_date: Optional[str] = Field(default=None, description="Дата редакции (YYYY-MM-DD)")
    doc_status: str = Field(default="amalda", description="Статус действия акта (amalda / kuchini yo'qotgan)")
    chunk_id: str = Field(..., description="ID чанка, обязательно присутствующий в contexts")
    quote: str = Field(..., description="Дословная цитата из текста нормы")


class ContextItem(BaseModel):
    """Найденный контекстный фрагмент."""

    rank: int = Field(..., ge=1, description="Ранг в выдаче (1..top_k)")
    chunk_id: str = Field(..., description="Уникальный идентификатор чанка")
    doc_id: str = Field(..., description="ID документа (lex:XXXXXXX)")
    article: str = Field(..., description="Статья/пункт")
    text: str = Field(..., description="Полный текст чанка")
    retrieval_score: float = Field(..., description="Оценка первого этапа (Dense+Sparse RRF)")
    rerank_score: float = Field(..., description="Оценка кросс-энкодер реранкера")
    used_in_answer: bool = Field(default=True, description="Использован ли данный фрагмент в итоговом ответе")


class ItemTiming(BaseModel):
    """Временные метрики обработки вопроса (мс)."""

    retrieval_ms: int = Field(default=0, ge=0)
    rerank_ms: int = Field(default=0, ge=0)
    generation_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)


class ItemUsage(BaseModel):
    """Количество токенов модели."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ItemDebug(BaseModel):
    """Отладочная информация RAG."""

    rewritten_query: Optional[str] = None
    retrieved_total: Optional[int] = None


class ItemError(BaseModel):
    """Описание ошибки обработки конкретного вопроса."""

    code: str = Field(
        ...,
        description="Код ошибки: RETRIEVAL_FAILED, INDEX_UNAVAILABLE, LLM_TIMEOUT, LLM_ERROR, CONTEXT_TOO_LONG, INVALID_LANGUAGE",
    )
    message: str = Field(..., description="Текстовое пояснение ошибки")
    retryable: bool = Field(default=False, description="Возможно ли повторить запрос")


class ItemResult(BaseModel):
    """Результат обработки единичного вопроса."""

    id: str = Field(..., description="ID вопроса (q001)")
    status: Literal["ok", "no_answer", "error", "timeout"] = Field(..., description="Статус обработки")
    answer: Optional[str] = Field(default=None, description="Сформированный ответ со сносками [1], [2]")
    answer_language: str = Field(default="uz-Latn", description="Язык ответа (uz-Latn, uz-Cyrl, ru)")
    no_answer: bool = Field(default=False, description="true, если вопрос не регулируется или ответ отсутствует в базе")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Уверенность ответа (0.0..1.0)")
    citations: List[Citation] = Field(default_factory=list, description="Список официальных источников")
    contexts: List[ContextItem] = Field(default_factory=list, description="Список найденных контекстов")
    debug: Optional[ItemDebug] = Field(default=None, description="Отладочные данные")
    timing: ItemTiming = Field(default_factory=ItemTiming, description="Тайминги обработки")
    usage: ItemUsage = Field(default_factory=ItemUsage, description="Расход токенов")
    error: Optional[ItemError] = Field(default=None, description="Объект ошибки при status != ok")


class SystemInfo(BaseModel):
    """Информация о версиях компонентов RAG пайплайна."""

    pipeline_version: str = "0.4.2"
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-m3"
    reranker: Optional[str] = "BAAI/bge-reranker-v2-m3"
    index_version: str = "lex-idx-2026-10-28"
    lex_snapshot_date: str = "2026-10-28"


class BatchTiming(BaseModel):
    """Общее время выполнения пакета."""

    received_at: str
    completed_at: str
    total_ms: int


class BatchSummary(BaseModel):
    """Сводка результатов пакета."""

    total: int = 0
    ok: int = 0
    no_answer: int = 0
    error: int = 0


class BatchResponse(BaseModel):
    """Полный ответ на пакетный запрос (Section 4)."""

    run_id: str
    batch_id: str
    status: Literal["completed", "processing", "failed"] = "completed"
    system: SystemInfo
    timing: BatchTiming
    summary: BatchSummary
    results: List[ItemResult]


class AsyncBatchStatusResponse(BaseModel):
    """Асинхронный статус пакета (HTTP 202)."""

    batch_id: str
    status: Literal["processing", "completed", "failed"] = "processing"


class QueryRequest(BaseModel):
    """Одиночный запрос для отладки (POST /v1/rag/query)."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = Field(default="q_debug_1", description="ID вопроса")
    question: str = Field(..., min_length=1, max_length=2000, description="Текст вопроса")
    language: Optional[str] = Field(default=None, description="Язык вопроса")
    options: Optional[OptionsRequest] = Field(default_factory=OptionsRequest, description="Опции")


class QueryResponse(BaseModel):
    """Ответ на одиночный запрос."""

    system: SystemInfo
    result: ItemResult


class HealthResponse(BaseModel):
    """Статус здоровья сервиса (GET /v1/health)."""

    status: str = "ok"
    version: str = "0.4.2"
    qdrant: Dict[str, Any]
    gemini_configured: bool
    uptime_seconds: float
    timestamp: str
