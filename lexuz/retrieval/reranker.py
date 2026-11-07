"""
lexuz.retrieval.reranker

Модуль точечного переранжирования (Cross-Encoder Reranker):
- Использует мультиязычный BAAI/bge-reranker-v2-m3
- Оценивает семантическую пару (query, candidate_document)
- Отбирает Top-N наиболее точных статей из Top-K кандидатов гибридного поиска
"""

import logging
from typing import Any, List, Optional, Tuple

from lexuz.retrieval.models import RerankedResult, SearchResult

logger = logging.getLogger("lexuz.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


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

        scored_results: List[Tuple[float, SearchResult]] = list(zip(scores, results))
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
            score = 0.5 + 0.1 * overlap
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked: List[RerankedResult] = []
        for rank, (score, res) in enumerate(scored[:top_k], start=1):
            reranked.append(
                RerankedResult(
                    search_result=res,
                    rerank_score=round(score, 4),
                    rank=rank,
                )
            )
        return reranked
