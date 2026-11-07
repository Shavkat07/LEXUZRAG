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


def extract_doc_year(doc_title: str) -> int:
    """Извлекает год принятия документа из заголовка."""
    m = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", doc_title)
    if m:
        return int(m.group(3))
    m2 = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", doc_title)
    if m2:
        return int(m2.group(1))
    return 2026


def stem_uz_word(w: str) -> str:
    """Простой правовой стеммер для узбекского языка."""
    w = w.lower().replace("'", "").replace("ʻ", "").replace("ʼ", "")
    for suffix in ["ining", "larga", "larda", "ning", "dagi", "dida", "sida", "lik", "lar", "gi", "ga", "da", "ni", "si", "i"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def compute_legal_composite_score(
    query: str,
    raw_rerank_score: float,
    result: SearchResult,
    as_of_date: Optional[str] = None,
) -> float:
    """
    Вычисляет композитный правовой скор кандидата:
    - Семантический скор кросс-энкодера (BGE Reranker) — основной вес 80%
    - Первичный гибридный поиск Qdrant (Dense + Sparse RRF) — вес 5%
    - Иерархия нормативно-правовых актов и актуальность редакции (Кодекс > Закон > Указ > Постановление)
    - Статус действия нормы: новые кодексы (2022+) вытесняют утратившие силу (1995)
    - Тип структурной единицы (Статья кодекса/закона имеет приоритет над пунктом подзаконного акта)
    - Совпадение ключевых терминов вопроса в заголовке статьи / breadcrumb
    - Пенализация процедурных деклараций ('aybdor shaxslar javob beradilar', 'nazorat qilish yuklansin')
    - Приоритет общей нормы над узкими демографическими исключениями (для несовершеннолетних, инвалидов)
    """
    q_lower = query.lower()

    title_lower = result.doc_title.lower()
    unit_title_lower = (result.unit_title or "").lower()
    breadcrumb_str = " ".join(result.breadcrumb).lower() if result.breadcrumb else ""
    content_lower = result.content.lower()
    doc_year = extract_doc_year(result.doc_title)

    # 1. Иерархия нормативно-правовых актов и актуальность
    h_boost = 0.0
    is_code = ("kodeks" in title_lower or "кодекс" in title_lower) and not any(
        term in title_lower for term in ["amalga oshirish", "tasdiqlash to'g'risida", "o'zgartirish"]
    )

    as_of_year = int(as_of_date[:4]) if as_of_date and len(as_of_date) >= 4 else 2026

    if doc_year > as_of_year:
        # Документ принят позже контрольной даты as_of_date
        h_boost = -0.50
    elif is_code:
        # Если кодекс принят в 2020+ (действующий), даем высокий приоритет.
        # Если старый (до 2020, например Mehnat kodeksi 1995 года), он утратил силу при наличии нового.
        if doc_year >= 2020:
            h_boost = 0.10
        else:
            h_boost = -0.15
    elif any(term in title_lower for term in ["qonun", "закон", "зру", "ўрқ"]):
        h_boost = 0.05 if doc_year >= 2018 else 0.01
    elif any(term in title_lower for term in ["farmon", "указ", "пф"]):
        h_boost = 0.02
    elif any(term in title_lower for term in ["qaror", "постановление", "пкм", "пп"]):
        h_boost = 0.00

    # 2. Тип структурной единицы: статья закона/кодекса содержит общую правовую норму
    u_boost = 0.0
    if result.unit_type == "article" or (result.unit_number and "modda" in str(result.unit_number)):
        u_boost = 0.04
    elif result.unit_type == "clause":
        u_boost = 0.01

    # 3. Совпадение темы вопроса с заголовком статьи / рубрики (Breadcrumb) с учетом стемминга
    target_text = f"{unit_title_lower} {breadcrumb_str}"
    q_stems = {stem_uz_word(w) for w in re.findall(r"\b\w{3,}\b", q_lower)}
    target_stems = {stem_uz_word(w) for w in re.findall(r"\b\w{3,}\b", target_text)}
    title_overlap = len(q_stems & target_stems)
    t_boost = min(0.08, title_overlap * 0.02)

    # 4. Пенализация процедурных декларативных/санкционных статей
    penalty = 0.0
    is_liability_query = any(
        w in q_lower
        for w in [
            "javobgar", "jarima", "jazo", "nazorat", "kuchga kirish",
            "ответствен", "штраф", "наказани", "контрол", "вступлени",
        ]
    )
    if not is_liability_query:
        if any(
            phrase in content_lower
            for phrase in [
                "buzilishida aybdor", "javob beradilar", "виновные лица несут",
                "ijrosini nazorat qilish", "rasmiy e'lon qilingan kundan",
                "контроль за исполнением возложить", "со дня его официального опубликования",
            ]
        ):
            penalty += 0.15

    # 5. Соответствие интенту вопроса (срок, продолжительность, размер)
    intent_boost = 0.0
    if any(
        w in q_lower
        for w in [
            "muddat", "davomiylig", "qancha", "necha kun", "ta'til",
            "муддат", "қанча", "неча кун", "таътил", "срок", "длительност", "отпуск",
        ]
    ):
        if any(
            w in content_lower
            for w in [
                "asosiy eng kam", "kalendar kun", "ish kuni", "davomiyligi",
                "tashkil etadi", "календар", "дней", "составляет", "не менее",
            ]
        ):
            intent_boost += 0.03

    # 6. Пенализация специальных демографических исключений при общем вопросе
    is_general_leave = (
        any(w in q_lower for w in ["ta'til", "таътил", "отпуск"])
        and not any(
            w in q_lower
            for w in [
                "yosh", "bola", "voyaga yetmagan", "nogiron", "homilador",
                "ёш", "бола", "ногирон", "дети", "несовершеннолет", "инвалид",
            ]
        )
    )
    if is_general_leave:
        if any(
            w in content_lower[:120]
            for w in [
                "o'n sakkiz yosh", "ўн саккиз ёш", "nogironligi bo'lgan", "ногирон", "homilador",
            ]
        ):
            penalty += 0.08

        # Если не спрашивают о дополнительном отпуске, статья об основном имеет приоритет
        if not any(w in q_lower for w in ["qo'shimcha", "qoshimcha", "дополнительн"]):
            if any(w in unit_title_lower for w in ["qo'shimcha ta'til", "дополнительн"]):
                penalty += 0.06

    # Первичный retrieval score (от 0.0 до 1.0)
    ret_score = float(result.score) if result.score is not None else 0.0

    # Взвешенная сумма: кросс-энкодер 80%, первичный поиск 5%, правовые приоритеты 15%
    # Возвращаем несжатый скор для точной сортировки (без потери различий из-за min(0.9999))
    composite = (
        (raw_rerank_score * 0.80)
        + (ret_score * 0.05)
        + h_boost
        + u_boost
        + t_boost
        + intent_boost
        - penalty
    )
    return float(composite)


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
        as_of_date: Optional[str] = None,
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
                return self.rerank(query, results, top_k=top_k, batch_size=new_bs, as_of_date=as_of_date)
            raise
        finally:
            self.clear_cache()

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif isinstance(scores, (int, float)):
            scores = [float(scores)]

        scores = [float(s) for s in scores]

        # Применяем композитный правовой скоринг с учетом иерархии актов
        scored_results: List[Tuple[float, float, SearchResult]] = []
        for s, res in zip(scores, results):
            comp_score = compute_legal_composite_score(query, s, res, as_of_date=as_of_date)
            scored_results.append((comp_score, s, res))

        # Сортируем по некэпированному композитному скору с тай-брейком по сырому кросс-энкодеру
        scored_results.sort(key=lambda x: (x[0], x[1]), reverse=True)

        reranked: List[RerankedResult] = []
        for rank, (score, raw_s, res) in enumerate(scored_results[:top_k], start=1):
            reranked.append(
                RerankedResult(
                    search_result=res,
                    rerank_score=round(float(min(0.9999, max(0.01, score))), 4),
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
        as_of_date: Optional[str] = None,
    ) -> List[RerankedResult]:
        if not results:
            return []

        query_terms = set(query.lower().split())

        scored: List[Tuple[float, float, SearchResult]] = []
        for r in results:
            doc_terms = set(r.content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            base_score = 0.5 + 0.1 * overlap
            comp_score = compute_legal_composite_score(query, base_score, r, as_of_date=as_of_date)
            scored.append((comp_score, base_score, r))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        reranked: List[RerankedResult] = []
        for rank, (score, raw_s, res) in enumerate(scored[:top_k], start=1):
            reranked.append(
                RerankedResult(
                    search_result=res,
                    rerank_score=round(float(min(0.9999, max(0.01, score))), 4),
                    rank=rank,
                )
            )
        return reranked
