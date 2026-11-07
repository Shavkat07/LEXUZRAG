"""
lexuz.retrieval.reranker

Модуль точечного переранжирования (Cross-Encoder Reranker):
- Использует мультиязычный BAAI/bge-reranker-v2-m3
- Оценивает семантическую пару (query, candidate_document)
- Отбирает Top-N наиболее точных статей из Top-K кандидатов гибридного поиска
"""

import logging
import re
from typing import Any, List, Optional, Tuple

from lexuz.retrieval.models import RerankedResult, SearchResult

logger = logging.getLogger("lexuz.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def compute_legal_composite_score(
    query: str,
    raw_rerank_score: float,
    result: SearchResult,
) -> float:
    """
    Вычисляет композитный правовой скор кандидата:
    - Семантический скор кросс-энкодера (BGE Reranker)
    - Гибридный скор первичного поиска Qdrant (Dense + Sparse RRF)
    - Иерархия нормативно-правовых актов (Кодекс > Закон > Указ > Постановление > Подзаконный акт)
    - Тип структурной единицы (Статья кодекса/закона имеет приоритет над пунктом подзаконного акта)
    - Совпадение ключевых терминов вопроса в заголовке статьи / breadcrumb
    - Пенализация раздутых чанков, где норма тонет в таблицах приложений
    """
    q_lower = query.lower()
    q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

    title_lower = result.doc_title.lower()
    unit_title_lower = (result.unit_title or "").lower()
    breadcrumb_str = " ".join(result.breadcrumb).lower() if result.breadcrumb else ""
    content_lower = result.content.lower()

    # 1. Иерархия нормативно-правовых актов
    h_boost = 0.0
    if "kodeks" in title_lower or "кодекс" in title_lower:
        # Проверяем, что это сам кодекс, а не постановление/закон о введении кодекса
        if not any(term in title_lower for term in ["amalga oshirish", "tasdiqlash to'g'risida", "o'zgartirish"]):
            h_boost = 0.09
        else:
            h_boost = 0.03
    elif any(term in title_lower for term in ["qonun", "закон", "зру", "ўрқ"]):
        h_boost = 0.06
    elif any(term in title_lower for term in ["farmon", "указ", "пф"]):
        h_boost = 0.03
    elif any(term in title_lower for term in ["qaror", "постановление", "пкм", "пп"]):
        h_boost = 0.01

    # 2. Тип структурной единицы: статья закона/кодекса содержит общую правовую норму
    u_boost = 0.0
    if result.unit_type == "article" or (result.unit_number and "modda" in str(result.unit_number)):
        u_boost = 0.04
    elif result.unit_type == "clause":
        u_boost = 0.01

    # 3. Совпадение темы вопроса с заголовком статьи / рубрики (Breadcrumb)
    target_text = f"{unit_title_lower} {breadcrumb_str}"
    target_words = set(re.findall(r"\b\w{3,}\b", target_text))
    title_overlap = len(q_words & target_words)
    t_boost = min(0.08, title_overlap * 0.02)

    # 4. Пенализация раздутых чанков и чисто процедурных/санкционных статей
    penalty = 0.0
    if ("buzilishida aybdor" in content_lower or "javob beradilar" in content_lower) and not any(
        w in q_lower for w in ["javobgar", "jarima", "jazo", "ответствен"]
    ):
        if len(result.content) > 2000:
            penalty += 0.08

    # 5. Соответствие интенту вопроса (срок, продолжительность)
    intent_boost = 0.0
    if any(w in q_lower for w in ["muddat", "davomiylig", "qancha", "necha kun", "срок", "длительност"]):
        if any(w in content_lower for w in ["kalendar kun", "ish kuni", "davomiyligi", "tashkil etadi", "дней", "года", "лет"]):
            intent_boost += 0.03

    # Первичный retrieval score (от 0.0 до 1.0)
    ret_score = float(result.score) if result.score is not None else 0.0

    # Взвешенная сумма: кросс-энкодер 60%, первичный поиск 25%, правовые приоритеты 15%
    composite = (
        (raw_rerank_score * 0.60)
        + (ret_score * 0.25)
        + h_boost
        + u_boost
        + t_boost
        + intent_boost
        - penalty
    )
    return round(float(min(0.9999, max(0.01, composite))), 4)


class BGEReranker:
    """Кросс-энкодер реранкер на базе FlagEmbedding."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 2048,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._device = device
        self._model = None

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        """Ленивая загрузка весов кросс-энкодера."""
        if self._model is None:
            logger.info(
                "Loading FlagReranker '%s' (device=%s, fp16=%s)...",
                self.model_name,
                self.device,
                self.use_fp16,
            )
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=self.use_fp16 if self.device == "cuda" else False,
                    device=self.device,
                )
            except ImportError as e:
                logger.error("FlagEmbedding is required for BGEReranker.")
                raise ImportError("FlagEmbedding required") from e
        return self._model

    def clear_cache(self) -> None:
        """Очищает кэш CUDA памяти при наличии доступного GPU."""
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[RerankedResult]:
        """
        Переранжирует список кандидатов SearchResult и возвращает Top-K результатов.
        """
        if not results:
            return []

        # Формируем пары (запрос, текст документа)
        pairs: List[Tuple[str, str]] = []
        for r in results:
            # Включаем хлебные крошки и заголовок для глубокой оценки
            header = " > ".join(r.breadcrumb) if r.breadcrumb else r.doc_title
            doc_repr = f"{header}\n\n{r.content}".strip()
            pairs.append((query, doc_repr))

        cur_batch_size = batch_size
        try:
            scores = self.model.compute_score(
                pairs,
                batch_size=cur_batch_size,
                max_length=self.max_length,
                normalize=True,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg and cur_batch_size > 1:
                new_bs = max(1, cur_batch_size // 2)
                logger.warning(
                    "CUDA OOM encountered during rerank (batch_size=%d). "
                    "Clearing cache and retrying with batch_size=%d...",
                    cur_batch_size,
                    new_bs,
                )
                self.clear_cache()
                return self.rerank(query, results, top_k=top_k, batch_size=new_bs)
            raise
        finally:
            self.clear_cache()

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif isinstance(scores, (int, float)):
            scores = [float(scores)]

        scores = [float(s) for s in scores]

        # Применяем композитный правовой скоринг с учетом иерархии актов
        scored_results: List[Tuple[float, SearchResult]] = []
        for s, res in zip(scores, results):
            comp_score = compute_legal_composite_score(query, s, res)
            scored_results.append((comp_score, res))

        scored_results.sort(key=lambda x: x[0], reverse=True)

        reranked: List[RerankedResult] = []
        for rank, (score, res) in enumerate(scored_results[:top_k], start=1):
            reranked.append(
                RerankedResult(
                    search_result=res,
                    rerank_score=round(float(score), 4),
                    rank=rank,
                )
            )

        return reranked


class MockReranker:
    """Мок-реранкер для быстрых тестов."""

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[RerankedResult]:
        if not results:
            return []

        query_terms = set(query.lower().split())

        scored: List[Tuple[float, SearchResult]] = []
        for r in results:
            doc_terms = set(r.content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            base_score = 0.5 + 0.1 * overlap
            comp_score = compute_legal_composite_score(query, base_score, r)
            scored.append((comp_score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked: List[RerankedResult] = []
        for rank, (score, res) in enumerate(scored[:top_k], start=1):
            reranked.append(
                RerankedResult(
                    search_result=res,
                    rerank_score=round(float(score), 4),
                    rank=rank,
                )
            )
        return reranked
