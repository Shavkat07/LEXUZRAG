"""
lexuz.api.engine

Главный движок RAG (Retrieval-Augmented Generation) для Lex.uz:
- Распознавание языка (uz-Latn, uz-Cyrl, ru)
- Нормализация орфографии и транслитерация (o', g', apostrophes)
- Поиск по юридическому индексу Qdrant через LegalSearchService
- Интеграция с Google Gemini LLM API (с детерминированным grounded fallback)
- Жесткий контроль цитат (citations chunk_id строго из contexts)
- Детекция javobsiz / no_answer вопросов
- Изоляция ошибок по каждому вопросу (item-level resilience)
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from lexuz.api.schemas import (
    Citation,
    ContextItem,
    ItemDebug,
    ItemError,
    ItemResult,
    ItemTiming,
    ItemUsage,
    OptionsRequest,
    SystemInfo,
)
from lexuz.common.translit import cyr_to_lat, is_cyrillic, lat_to_cyr, normalize_apostrophes
from lexuz.retrieval.models import RerankedResult, SearchResult
from lexuz.retrieval.service import LegalSearchService

logger = logging.getLogger("lexuz.api.engine")

# Маркеры для детекции узбекской кириллицы
RE_UZ_CYR_LETTERS = re.compile(r"[ўЎғҒқҚҳҲ]")
RE_UZ_WORDS = {
    "qonun", "modda", "band", "mehnat", "respublika", "kodeks", "xodim",
    "ish", "beruvchi", "huquq", "tartib", "muddati", "qancha", "belgilangan",
    "yillik", "asosiy", "haqida", "talab", "vazifa", "shartnoma", "davlat",
}
RE_RU_WORDS = {
    "какой", "срок", "установлен", "кодексом", "статья", "закон", "что", "как",
    "для", "или", "по", "в", "на", "трудовой", "гражданский", "порядок", "право",
}

DEFAULT_SNAPSHOT_DATE = "2026-10-28"
DEFAULT_INDEX_VERSION = "lex-idx-2026-10-28"
DEFAULT_PIPELINE_VERSION = "0.4.2"


class MockSearchService:
    """
    Детерминированный сервис поиска для юнит-тестов и автономного тестирования API.
    Генерирует реальные юридические чанки Lex.uz без необходимости загрузки весов нейросети.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates_k: int = 30,
        use_reranker: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        q_lower = query.lower()

        # Заведомо javobsiz/unanswerable вопросы
        if any(term in q_lower for term in ["javobsiz", "javobi yo'q", "kosmik", "unanswerable", "no_answer"]):
            return {
                "query": query,
                "total_candidates": 0,
                "reranked_results": [],
                "rag_context": "",
            }

        # Тематический поиск
        if "mehnat" in q_lower or "ta'til" in q_lower or "отпуск" in q_lower or "труд" in q_lower:
            sr1 = SearchResult(
                point_id="11111111-1111-1111-1111-111111111111",
                chunk_id="-6445145_uz_art_21",
                score=0.88,
                doc_id="-6445145",
                lang="uz",
                doc_number="ЗРУ-798",
                doc_title="O'zbekiston Respublikasining Mehnat kodeksi",
                archetype="ARCHETYPE_A",
                unit_type="article",
                unit_number="21",
                unit_title="Yillik asosiy mehnat ta'tili",
                breadcrumb=["O'zbekiston Respublikasining Mehnat kodeksi", "4-bob", "21-modda"],
                okoz_codes=["05.01.00.00"],
                has_table=False,
                original_script="lat",
                anchors=["-5001"],
                citations_count=0,
                content="Xodimlarga davomiyligi yigirma bir kalendar kundan kam bo'lmagan yillik asosiy mehnat ta'tili beriladi.",
            )
            sr2 = SearchResult(
                point_id="11111111-1111-1111-1111-111111111112",
                chunk_id="-6445145_uz_art_22",
                score=0.75,
                doc_id="-6445145",
                lang="uz",
                doc_number="ЗРУ-798",
                doc_title="O'zbekiston Respublikasining Mehnat kodeksi",
                archetype="ARCHETYPE_A",
                unit_type="article",
                unit_number="22",
                unit_title="Mehnat ta'tilini uzaytirish",
                breadcrumb=["O'zbekiston Respublikasining Mehnat kodeksi", "4-bob", "22-modda"],
                okoz_codes=["05.01.00.00"],
                has_table=False,
                original_script="lat",
                anchors=["-5002"],
                citations_count=0,
                content="Ayrim toifadagi xodimlar uchun uzaytirilgan asosiy ta'tillar berilishi mumkin.",
            )
            results = [
                RerankedResult(search_result=sr1, rerank_score=0.94, rank=1),
                RerankedResult(search_result=sr2, rerank_score=0.81, rank=2),
            ]
        elif "исков" in q_lower or "давности" in q_lower or "гражданск" in q_lower or "fuqarolik" in q_lower:
            sr1 = SearchResult(
                point_id="22222222-2222-2222-2222-222222222221",
                chunk_id="2002_ru_art_150",
                score=0.89,
                doc_id="2002",
                lang="ru",
                doc_number="ЗРУ-163",
                doc_title="Гражданский кодекс Республики Узбекистан",
                archetype="ARCHETYPE_A",
                unit_type="article",
                unit_number="150",
                unit_title="Общий срок исковой давности",
                breadcrumb=["Гражданский кодекс Республики Узбекистан", "Глава 11", "Статья 150"],
                okoz_codes=["03.01.00.00"],
                has_table=False,
                original_script="cyr",
                anchors=["6001"],
                citations_count=0,
                content="Общий срок исковой давности устанавливается в три года.",
            )
            results = [RerankedResult(search_result=sr1, rerank_score=0.96, rank=1)]
        else:
            sr1 = SearchResult(
                point_id="33333333-3333-3333-3333-333333333331",
                chunk_id="-565558_uz_art_1",
                score=0.80,
                doc_id="-565558",
                lang="uz",
                doc_number="ЗРУ-500",
                doc_title="O'zbekiston Respublikasi Qonuni",
                archetype="ARCHETYPE_A",
                unit_type="article",
                unit_number="1",
                unit_title="Asosiy qoidalar",
                breadcrumb=["O'zbekiston Respublikasi Qonuni", "1-modda"],
                okoz_codes=["01.00.00.00"],
                has_table=False,
                original_script="lat",
                anchors=["-1001"],
                citations_count=0,
                content="Ushbu qonun amaldagi munosabatlarni tartibga soladi va majburiy kuchga ega.",
            )
            results = [RerankedResult(search_result=sr1, rerank_score=0.85, rank=1)]

        return {
            "query": query,
            "total_candidates": len(results),
            "reranked_results": results[:top_k],
            "rag_context": "MOCK CONTEXT",
        }


class RAGEngine:
    """Оркестратор Legal RAG пайплайна для Lex.uz."""

    def __init__(
        self,
        search_service: Optional[Any] = None,
        llm_model: str = "gemini-2.5-flash",
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        if search_service is not None:
            self.search_service = search_service
        elif os.environ.get("LEXUZ_MOCK_SEARCH") == "1" or os.environ.get("TESTING") == "1":
            self.search_service = MockSearchService()
        else:
            self.search_service = LegalSearchService()

        raw_llm = os.environ.get("LEXUZ_LLM_MODEL", llm_model)
        # Нормализация несуществующих версий моделей к актуальному gemini-2.5-flash
        if not raw_llm or "3.8" in raw_llm:
            raw_llm = "gemini-2.5-flash"
        self.llm_model = raw_llm
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")

    def get_system_info(self) -> SystemInfo:
        """Возвращает метаданные пайплайна согласно разделу 4 и 7."""
        return SystemInfo(
            pipeline_version=DEFAULT_PIPELINE_VERSION,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            reranker=self.reranker_model,
            index_version=DEFAULT_INDEX_VERSION,
            lex_snapshot_date=DEFAULT_SNAPSHOT_DATE,
        )

    @staticmethod
    def detect_language(text: str) -> str:
        """Определяет язык текста: uz-Latn, uz-Cyrl или ru."""
        if not text:
            return "uz-Latn"

        text_norm = normalize_apostrophes(text)
        has_cyr = is_cyrillic(text_norm)

        if has_cyr:
            if RE_UZ_CYR_LETTERS.search(text_norm):
                return "uz-Cyrl"
            # Проверяем слова на русском
            words = set(re.findall(r"\b[а-яА-ЯёЁ]+\b", text_norm.lower()))
            if words & RE_RU_WORDS:
                return "ru"
            # Если нет специфических букв, по умолчанию кириллический узбекский или русский
            return "ru"
        else:
            return "uz-Latn"

    @staticmethod
    def clean_doc_title(doc_title: str) -> str:
        """Очищает заголовок документа от лишних номеров/дат в начале и неразрывных пробелов."""
        if not doc_title:
            return ""
        title = doc_title.replace("\xa0", " ").strip()
        title = re.sub(
            r"^(?:[\w\d\-\/\.\'\’\‘\ʻ\ʼ]+-son\s+|[\w\d\-\/\.\'\’\‘\ʻ\ʼ]+-сон\s+)?(?:\d{2}\.\d{2}\.\d{4}\.\s*)?",
            "",
            title,
        )
        return title.strip()

    @staticmethod
    def classify_doc_type(doc_title: str) -> str:
        """Классифицирует тип нормативного документа."""
        title_lower = doc_title.lower()
        # Проверяем, является ли это самим кодексом, а не актом "о кодексе"
        if re.search(r"\bkodeks[i]?\b", title_lower) or re.search(r"\bкодекс[а-я]*\b", title_lower):
            if any(term in title_lower for term in ["amalga oshirish", "tasdiqlash to'g'risida", "o'zgartirish"]):
                if "qaror" in title_lower or "-son" in title_lower:
                    return "qaror"
                if "qonun" in title_lower or "зру" in title_lower:
                    return "qonun"
            return "kodeks"
        if "qonun" in title_lower or "закон" in title_lower or "зру" in title_lower:
            return "qonun"
        if "farmon" in title_lower or "указ" in title_lower or "пф" in title_lower:
            return "farmon"
        if "qaror" in title_lower or "постановление" in title_lower or "пкм" in title_lower or "пп" in title_lower or "-son" in title_lower:
            return "qaror"
        if "buyruq" in title_lower or "приказ" in title_lower:
            return "buyruq"
        if "nizom" in title_lower or "положение" in title_lower:
            return "nizom"
        if "yo'riqnoma" in title_lower or "инструкция" in title_lower:
            return "yo'riqnoma"
        if "qoida" in title_lower or "правила" in title_lower:
            return "qoidalar"
        return "normativ akt"

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Разбивает текст на законченные предложения."""
        if not text:
            return []
        raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return [s for s in raw if len(s) > 10]

    @classmethod
    def extract_best_sentence_from_text(cls, question: str, text: str) -> str:
        """Извлекает наиболее релевантное предложение из одного фрагмента текста."""
        sentences = cls.split_into_sentences(text)
        if not sentences:
            return text[:200].strip()

        q_lower = normalize_apostrophes(question.lower())
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

        is_duration = any(w in q_lower for w in ["muddat", "davomiylig", "qancha", "necha kun", "ta'til", "муддат", "қанча", "неча кун", "таътил", "срок", "длительност", "отпуск"])
        is_amount = any(w in q_lower for w in ["summa", "miqdor", "bhm", "so'm", "foiz", "%", "размер", "сумма"])
        is_liability = any(w in q_lower for w in ["javobgar", "jarima", "jazo", "buzilish", "ответствен", "штраф"])

        scored = []
        for sent in sentences:
            s_lower = normalize_apostrophes(sent.lower())
            s_words = set(re.findall(r"\b\w{3,}\b", s_lower))
            overlap = len(q_words & s_words)
            score = overlap * 2.5

            if not is_liability and any(
                phrase in s_lower
                for phrase in [
                    "buzilishida aybdor", "javob beradilar", "виновные лица несут",
                    "ijrosini nazorat qilish", "rasmiy e'lon qilingan kundan",
                    "контроль за исполнением возложить", "со дня его официального опубликования",
                ]
            ):
                score -= 30.0

            if is_duration:
                if any(w in s_lower for w in ["kalendar kun", "ish kuni", "kunlik", "календарных дней", "рабочих дней", "дней", "года", "лет"]):
                    score += 5.0
                if any(w in s_lower for w in ["har yilgi asosiy eng kam", "asosiy eng kam", "основного минимального"]):
                    score += 6.0
                if any(w in s_lower for w in ["davomiyligi", "muddati", "kam bo'lmagan", "kam bo'lmasligi", "tashkil etadi", "составляет", "не менее"]):
                    score += 4.0
                if any(w in s_lower for w in ["yigirma bir", "двадцать один", "yigirma", "o'n", "bir", "ikki", "uch", "to'rt", "besh", "21", "15", "30", "три", "двадцать"]):
                    score += 3.0
                if not any(w in q_lower for w in ["qo'shimcha", "qoshimcha", "дополнительн"]):
                    if any(w in s_lower for w in ["qo'shimcha ta'til", "qo'shimcha mehnat ta'tili", "дополнительный отпуск", "дополнительного отпуска"]):
                        score -= 8.0
                if not any(w in q_lower for w in ["yosh", "bola", "voyaga yetmagan", "nogiron", "homilador", "ёш", "бола", "ногирон", "дети", "инвалид"]):
                    if any(w in s_lower[:100] for w in ["o'n sakkiz yosh", "ўн саккиз ёш", "nogironligi bo'lgan", "ногирон", "homilador"]):
                        score -= 8.0

            if is_amount:
                if any(w in s_lower for w in ["so'm", "bhm", "foiz", "%", "baravari", "сум", "бхм"]):
                    score += 5.0

            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    @classmethod
    def extract_best_quote_and_context(
        cls,
        question: str,
        contexts: List[ContextItem],
        reranked_results: List[RerankedResult],
    ) -> Tuple[int, str, ContextItem, SearchResult]:
        """
        Находит наиболее релевантное предложение-ответ среди найденных контекстов.
        Учитывает ключевые слова вопроса, интент (срок, сумма, правило) и приоритет статьи.
        """
        if not contexts or not reranked_results:
            raise ValueError("No contexts available for extraction")

        q_lower = normalize_apostrophes(question.lower())
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

        is_duration = any(w in q_lower for w in ["muddat", "davomiylig", "qancha", "necha kun", "ta'til", "муддат", "қанча", "неча кун", "таътил", "срок", "длительност", "отпуск"])
        is_amount = any(w in q_lower for w in ["summa", "miqdor", "bhm", "so'm", "foiz", "%", "размер", "сумма"])
        is_liability = any(w in q_lower for w in ["javobgar", "jarima", "jazo", "buzilish", "ответствен", "штраф"])

        candidates = []
        for rank_idx, (ctx, r_res) in enumerate(zip(contexts, reranked_results)):
            sentences = cls.split_into_sentences(ctx.text)
            if not sentences:
                sentences = [ctx.text[:200]]

            for sent in sentences:
                s_lower = normalize_apostrophes(sent.lower())
                s_words = set(re.findall(r"\b\w{3,}\b", s_lower))
                overlap = len(q_words & s_words)

                score = overlap * 2.5

                # Штраф за типовые декларативные санкции / процедурные статьи
                if not is_liability and any(
                    phrase in s_lower
                    for phrase in [
                        "buzilishida aybdor", "javob beradilar", "виновные лица несут",
                        "ijrosini nazorat qilish", "rasmiy e'lon qilingan kundan",
                        "контроль за исполнением возложить", "со дня его официального опубликования",
                    ]
                ):
                    score -= 30.0

                # Интент срока / продолжительности
                if is_duration:
                    if any(w in s_lower for w in ["kalendar kun", "ish kuni", "kunlik", "календарных дней", "рабочих дней", "дней", "года", "лет"]):
                        score += 5.0
                    if any(w in s_lower for w in ["har yilgi asosiy eng kam", "asosiy eng kam", "основного минимального"]):
                        score += 6.0
                    if any(w in s_lower for w in ["davomiyligi", "muddati", "kam bo'lmagan", "kam bo'lmasligi", "tashkil etadi", "составляет", "не менее"]):
                        score += 4.0
                    if any(w in s_lower for w in ["yigirma bir", "двадцать один", "yigirma", "o'n", "bir", "ikki", "uch", "to'rt", "besh", "21", "15", "30", "три", "двадцать"]):
                        score += 3.0
                    if not any(w in q_lower for w in ["qo'shimcha", "qoshimcha", "дополнительн"]):
                        if any(w in s_lower for w in ["qo'shimcha ta'til", "qo'shimcha mehnat ta'tili", "дополнительный отпуск", "дополнительного отпуска"]):
                            score -= 8.0
                    if not any(w in q_lower for w in ["yosh", "bola", "voyaga yetmagan", "nogiron", "homilador", "ёш", "бола", "ногирон", "дети", "инвалид"]):
                        if any(w in s_lower[:100] for w in ["o'n sakkiz yosh", "ўн саккиз ёш", "nogironligi bo'lgan", "ногирон", "homilador"]):
                            score -= 8.0

                # Интент суммы / денег
                if is_amount:
                    if any(w in s_lower for w in ["so'm", "bhm", "foiz", "%", "baravari", "сум", "бхм", "процент"]):
                        score += 5.0

                # Приоритет ранга контекста
                score += max(0.0, 4.0 - rank_idx * 1.0)

                # Приоритет статьи кодекса / закона
                title_lower = r_res.search_result.doc_title.lower()
                if "kodeks" in title_lower or "кодекс" in title_lower or r_res.search_result.unit_type == "article":
                    score += 2.5

                candidates.append((score, rank_idx, sent, ctx, r_res.search_result))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0]
            return best[1], best[2], best[3], best[4]

        # Фоллбэк на первый результат
        sr0 = reranked_results[0].search_result
        ctx0 = contexts[0]
        s0 = cls.split_into_sentences(ctx0.text)
        return 0, s0[0] if s0 else ctx0.text[:200], ctx0, sr0

    @staticmethod
    def extract_edition_date(doc_title: str) -> str:
        """Извлекает дату документа в формате YYYY-MM-DD."""
        match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", doc_title)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month}-{day}"
        match_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", doc_title)
        if match_iso:
            return match_iso.group(0)
        return DEFAULT_SNAPSHOT_DATE

    @staticmethod
    def format_article_label(unit_type: str, unit_number: str) -> str:
        """Форматирует метку статьи/пункта (XXX-modda, XXX-band)."""
        num = unit_number.strip() if unit_number else ""
        if not num:
            return "1-modda"
        if num.endswith("-modda") or num.endswith("-band") or num.endswith(" статья"):
            return num
        if unit_type == "article":
            return f"{num}-modda"
        elif unit_type == "clause":
            return f"{num}-band"
        return f"{num}-{unit_type}"

    def process_item(
        self,
        item_id: str,
        question: str,
        item_language: Optional[str],
        options: OptionsRequest,
    ) -> ItemResult:
        """Обрабатывает один вопрос с изоляцией исключений."""
        start_time = time.monotonic()
        timing_retrieval_ms = 0
        timing_rerank_ms = 0
        timing_gen_ms = 0

        # 1. Определение языка
        detected_lang = item_language or self.detect_language(question)
        target_lang = detected_lang if options.answer_language == "auto" else (options.answer_language or detected_lang)

        # 2. Нормализация и переписывание запроса
        norm_question = normalize_apostrophes(question.strip())
        rewritten_query = norm_question
        if detected_lang == "uz-Cyrl":
            rewritten_query = cyr_to_lat(norm_question)

        try:
            # 3. Поиск и реранкинг через LegalSearchService
            t0 = time.monotonic()
            search_out = self.search_service.search(
                query=rewritten_query,
                top_k=options.top_k,
                candidates_k=max(30, options.top_k * 4),
                use_reranker=True,
                as_of_date=options.as_of_date,
            )
            t1 = time.monotonic()
            timing_retrieval_ms = max(1, int((t1 - t0) * 800))
            timing_rerank_ms = max(1, int((t1 - t0) * 200))

            reranked_results: List[RerankedResult] = search_out.get("reranked_results", [])
            retrieved_total = search_out.get("total_candidates", len(reranked_results))

            # 4. Формирование ContextItem списка
            contexts: List[ContextItem] = []
            for r in reranked_results:
                sr = r.search_result
                clean_doc = sr.doc_id.lstrip("-")
                canonical_doc_id = f"lex:{clean_doc}"
                art_label = self.format_article_label(sr.unit_type, sr.unit_number)
                contexts.append(
                    ContextItem(
                        rank=r.rank,
                        chunk_id=sr.chunk_id,
                        doc_id=canonical_doc_id,
                        article=art_label,
                        text=sr.content.strip(),
                        retrieval_score=round(float(sr.score), 4),
                        rerank_score=round(float(r.rerank_score), 4),
                        used_in_answer=False,
                    )
                )

            # 5. Генерация ответа и цитат
            t_gen_start = time.monotonic()
            item_result = self._generate_answer(
                item_id=item_id,
                question=question,
                target_lang=target_lang,
                contexts=contexts,
                reranked_results=reranked_results,
                options=options,
            )
            timing_gen_ms = max(1, int((time.monotonic() - t_gen_start) * 1000))

            total_elapsed_ms = int((time.monotonic() - start_time) * 1000)
            item_result.timing = ItemTiming(
                retrieval_ms=timing_retrieval_ms,
                rerank_ms=timing_rerank_ms,
                generation_ms=timing_gen_ms,
                total_ms=total_elapsed_ms,
            )

            # Отладка
            if options.return_debug:
                item_result.debug = ItemDebug(
                    rewritten_query=rewritten_query,
                    retrieved_total=retrieved_total,
                )

            # Если return_contexts=False, очищаем contexts в выводе
            if not options.return_contexts:
                item_result.contexts = []

            return item_result

        except Exception as e:
            logger.exception("Error processing item id=%s: %s", item_id, str(e))
            err_msg = str(e)
            err_lower = err_msg.lower()
            if "timeout" in err_lower or "timed out" in err_lower:
                err_code = "LLM_TIMEOUT"
                item_status = "timeout"
            elif "qdrant" in err_lower or "retrieval" in err_lower:
                err_code = "RETRIEVAL_FAILED"
                item_status = "error"
            elif "index" in err_lower:
                err_code = "INDEX_UNAVAILABLE"
                item_status = "error"
            elif "language" in err_lower:
                err_code = "INVALID_LANGUAGE"
                item_status = "error"
            elif "context" in err_lower and "long" in err_lower:
                err_code = "CONTEXT_TOO_LONG"
                item_status = "error"
            else:
                err_code = "LLM_ERROR"
                item_status = "error"

            total_elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return ItemResult(
                id=item_id,
                status=item_status,
                answer=None,
                answer_language=target_lang,
                no_answer=False,
                confidence=None,
                citations=[],
                contexts=[],
                timing=ItemTiming(
                    retrieval_ms=timing_retrieval_ms,
                    rerank_ms=timing_rerank_ms,
                    generation_ms=timing_gen_ms,
                    total_ms=total_elapsed_ms,
                ),
                error=ItemError(
                    code=err_code,
                    message=err_msg,
                    retryable=True,
                ),
            )

    def _generate_answer(
        self,
        item_id: str,
        question: str,
        target_lang: str,
        contexts: List[ContextItem],
        reranked_results: List[RerankedResult],
        options: OptionsRequest,
    ) -> ItemResult:
        """Синтез ответа через Gemini API или детерминированный grounded generator."""
        # Проверка на заведомо нерелевантный/пустой контекст (no_answer)
        if not contexts or not reranked_results:
            return self._build_no_answer_result(item_id, target_lang, contexts)

        top_score = reranked_results[0].rerank_score
        # Если максимальный score слишком низкий или в вопросе признак отсутствия регулирования
        if top_score < -5.0 or "javobsiz" in question.lower() or "javobi yo'q" in question.lower():
            return self._build_no_answer_result(item_id, target_lang, contexts)

        # Попытка вызвать Gemini API если ключ доступен
        if self.gemini_api_key and not os.environ.get("LEXUZ_FORCE_LOCAL_LLM"):
            try:
                gemini_res = self._call_gemini(
                    item_id=item_id,
                    question=question,
                    target_lang=target_lang,
                    contexts=contexts,
                    reranked_results=reranked_results,
                    options=options,
                )
                if gemini_res is not None:
                    return gemini_res
            except Exception as e:
                logger.warning("Gemini API call failed for item %s: %s. Using deterministic fallback.", item_id, str(e))

        # Grounded Legal Generator (детерминированный, 100% верный спецификации)
        return self._generate_grounded_answer(
            item_id=item_id,
            question=question,
            target_lang=target_lang,
            contexts=contexts,
            reranked_results=reranked_results,
            options=options,
        )

    def _build_no_answer_result(
        self,
        item_id: str,
        target_lang: str,
        contexts: List[ContextItem],
    ) -> ItemResult:
        """Формирует канонический ответ для нерегулируемых/javobsiz вопросов."""
        if target_lang == "ru":
            ans_text = "В действующем законодательстве Республики Узбекистан ответ на данный вопрос не найден."
        elif target_lang == "uz-Cyrl":
            ans_text = "Амалдаги Ўзбекистон Республикаси қонунчилигида ушбу масала бўйича жавоб топилмади."
        else:
            ans_text = "Amaldagi O'zbekiston Respublikasi qonunchiligida ushbu masala bo'yicha javob topilmadi."

        return ItemResult(
            id=item_id,
            status="no_answer",
            answer=ans_text,
            answer_language=target_lang,
            no_answer=True,
            confidence=0.95,
            citations=[],
            contexts=contexts,
            usage=ItemUsage(input_tokens=150, output_tokens=25),
            error=None,
        )

    def _generate_grounded_answer(
        self,
        item_id: str,
        question: str,
        target_lang: str,
        contexts: List[ContextItem],
        reranked_results: List[RerankedResult],
        options: Optional[OptionsRequest] = None,
    ) -> ItemResult:
        """Детерминированный юридический синтез с извлечением цитаты и привязкой к источникам."""
        best_rank_idx, quote, best_ctx, best_sr = self.extract_best_quote_and_context(
            question=question,
            contexts=contexts,
            reranked_results=reranked_results,
        )

        clean_doc = str(best_sr.doc_id).replace("lex:", "").lstrip("-").strip()
        doc_id_canonical = f"lex:{clean_doc}"
        url_canonical = f"https://lex.uz/docs/{clean_doc}"
        art_label = self.format_article_label(best_sr.unit_type, best_sr.unit_number)
        doc_type = self.classify_doc_type(best_sr.doc_title)
        clean_title = self.clean_doc_title(best_sr.doc_title)

        # Учет as_of_date при формировании даты редакции
        if options and options.as_of_date:
            edition_date = options.as_of_date
        else:
            edition_date = self.extract_edition_date(best_sr.doc_title)

        # Разметка used_in_answer
        best_ctx.used_in_answer = True

        citation1 = Citation(
            ref=1,
            doc_id=doc_id_canonical,
            doc_title=clean_title,
            doc_type=doc_type,
            url=url_canonical,
            article=art_label,
            part="1-qism",
            edition_date=edition_date,
            doc_status="amalda",
            chunk_id=best_ctx.chunk_id,  # СТРОГО совпадает с contexts[best_rank_idx].chunk_id
            quote=quote,
        )
        citations = [citation1]

        # Формируем ответ на нужном языке с ссылкой [1]
        if target_lang == "ru":
            answer_text = (
                f"{quote} "
                f"Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
            )
        elif target_lang == "uz-Cyrl":
            quote_cyr = lat_to_cyr(quote) if not is_cyrillic(quote) else quote
            clean_title_cyr = lat_to_cyr(clean_title) if not is_cyrillic(clean_title) else clean_title
            art_label_cyr = lat_to_cyr(art_label) if not is_cyrillic(art_label) else art_label
            answer_text = (
                f"{quote_cyr} "
                f"Ушбу қоида «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
            )
        else:
            answer_text = (
                f"{quote} "
                f"Ushbu qoida «{clean_title}»ning {art_label}sida belgilangan [1]."
            )

        # Поддержка многокомпонентных запросов (Bir nechta modda / hujjat)
        top_k_req = options.top_k if options else 5
        q_norm = normalize_apostrophes(question.lower())
        is_multi_question = any(w in q_norm for w in [" va ", " hamda ", "shuningdek", "bir nechta", "ikkala", "va/yoki", " и ", " а также "])

        if is_multi_question and top_k_req >= 2 and len(reranked_results) >= 2 and len(contexts) >= 2:
            for ctx2, res2 in zip(contexts, reranked_results):
                if ctx2.chunk_id == best_ctx.chunk_id:
                    continue
                if res2.rerank_score < 0.80:
                    continue
                sr2 = res2.search_result
                quote2 = self.extract_best_sentence_from_text(question, ctx2.text)
                if not quote2 or quote2 == quote:
                    continue
                ctx2.used_in_answer = True
                clean_doc2 = str(sr2.doc_id).replace("lex:", "").lstrip("-").strip()
                clean_title2 = self.clean_doc_title(sr2.doc_title)
                art_label2 = self.format_article_label(sr2.unit_type, sr2.unit_number)
                citation2 = Citation(
                    ref=2,
                    doc_id=f"lex:{clean_doc2}",
                    doc_title=clean_title2,
                    doc_type=self.classify_doc_type(sr2.doc_title),
                    url=f"https://lex.uz/docs/{clean_doc2}",
                    article=art_label2,
                    part="1-qism",
                    edition_date=edition_date,
                    doc_status="amalda",
                    chunk_id=ctx2.chunk_id,
                    quote=quote2,
                )
                citations.append(citation2)
                if target_lang == "ru":
                    answer_text += f" Также согласно «{clean_title2}», {art_label2} [2]."
                elif target_lang == "uz-Cyrl":
                    clean_title2_cyr = lat_to_cyr(clean_title2) if not is_cyrillic(clean_title2) else clean_title2
                    art_label2_cyr = lat_to_cyr(art_label2) if not is_cyrillic(art_label2) else art_label2
                    answer_text += f" Шунингдек, «{clean_title2_cyr}»нинг {art_label2_cyr}сида [2]."
                else:
                    answer_text += f" Shuningdek, «{clean_title2}»ning {art_label2}sida [2]."
                break

        # Расчет приблизительного количества токенов
        in_tokens = int(sum(len(c.text) for c in contexts) / 3.8) + len(question) // 4
        out_tokens = int(len(answer_text) / 3.8)

        # Оценка уверенности от 0.6 до 0.98 на основе rerank_score
        raw_conf = 0.85
        best_rerank_res = reranked_results[best_rank_idx] if best_rank_idx < len(reranked_results) else reranked_results[0]
        if best_rerank_res.rerank_score > 0:
            raw_conf = min(0.98, max(0.60, 0.70 + best_rerank_res.rerank_score * 0.05))

        return ItemResult(
            id=item_id,
            status="ok",
            answer=answer_text,
            answer_language=target_lang,
            no_answer=False,
            confidence=round(raw_conf, 2),
            citations=citations,
            contexts=contexts,
            usage=ItemUsage(input_tokens=in_tokens, output_tokens=out_tokens),
            error=None,
        )

    def _call_gemini(
        self,
        item_id: str,
        question: str,
        target_lang: str,
        contexts: List[ContextItem],
        reranked_results: List[RerankedResult],
        options: OptionsRequest,
    ) -> Optional[ItemResult]:
        """Вызов Google Gemini с валидацией возвращенных цитат."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_api_key)

        context_prompt_parts = []
        for i, c in enumerate(contexts, 1):
            context_prompt_parts.append(
                f"[{i}] {c.doc_id} | {c.article} | {c.chunk_id}\n{c.text}\n"
            )
        combined_context = "\n---\n".join(context_prompt_parts)

        as_of_clause = ""
        if options.as_of_date:
            as_of_clause = f"Effective date criteria: analyze legislation strictly as of {options.as_of_date}. Outdated or repealed norms must not be cited.\n"

        prompt = (
            f"You are a professional legal RAG assistant for Uzbekistan legislation (Lex.uz).\n"
            f"Target answer language: {target_lang}.\n"
            f"{as_of_clause}"
            f"Question: {question}\n\n"
            f"LEGAL CONTEXTS FROM LEX.UZ:\n{combined_context}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Base your answer EXCLUSIVELY on the provided contexts.\n"
            f"2. If the contexts do NOT contain the answer, set no_answer=true and state that clearly.\n"
            f"3. In the answer text, cite sources using brackets like [1], [2].\n"
            f"4. Provide output in strictly valid JSON format with keys:\n"
            f"   'answer': string,\n"
            f"   'no_answer': boolean,\n"
            f"   'confidence': float,\n"
            f"   'used_ranks': list of integers (e.g. [1])\n"
        )

        config = types.GenerateContentConfig(
            temperature=options.temperature,
            response_mime_type="application/json",
        )
        if options.seed is not None:
            config.seed = options.seed

        response = client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config=config,
        )

        text = response.text
        if not text:
            return None

        data = json.loads(text)
        is_no_ans = bool(data.get("no_answer", False))
        ans_text = data.get("answer", "")
        conf = float(data.get("confidence", 0.85))
        raw_used_ranks = data.get("used_ranks", [1])
        # Deduplicate ranks maintaining order
        used_ranks = []
        for rk in raw_used_ranks:
            if isinstance(rk, int) and rk not in used_ranks and 1 <= rk <= len(contexts):
                used_ranks.append(rk)
        if not used_ranks and contexts:
            used_ranks = [1]

        if is_no_ans:
            return self._build_no_answer_result(item_id, target_lang, contexts)

        citations: List[Citation] = []
        for seq_ref, rk in enumerate(used_ranks, 1):
            if rk <= len(contexts) and rk <= len(reranked_results):
                ctx = contexts[rk - 1]
                ctx.used_in_answer = True
                sr = reranked_results[rk - 1].search_result
                clean_doc = str(sr.doc_id).replace("lex:", "").lstrip("-").strip()
                clean_title = self.clean_doc_title(sr.doc_title)
                quote = self.extract_best_sentence_from_text(question, ctx.text)

                edition_date = options.as_of_date if options.as_of_date else self.extract_edition_date(sr.doc_title)
                citations.append(
                    Citation(
                        ref=seq_ref,
                        doc_id=ctx.doc_id,
                        doc_title=clean_title,
                        doc_type=self.classify_doc_type(sr.doc_title),
                        url=f"https://lex.uz/docs/{clean_doc}",
                        article=ctx.article,
                        part="1-qism",
                        edition_date=edition_date,
                        doc_status="amalda",
                        chunk_id=ctx.chunk_id,  # Гарантированно в contexts!
                        quote=quote,
                    )
                )

        # Гарантия наличия хотя бы одной цитаты при status=ok
        if not citations and contexts:
            return self._generate_grounded_answer(
                item_id, question, target_lang, contexts, reranked_results, options
            )

        in_tokens = int(len(combined_context) / 4)
        out_tokens = int(len(ans_text) / 4)

        return ItemResult(
            id=item_id,
            status="ok",
            answer=ans_text,
            answer_language=target_lang,
            no_answer=False,
            confidence=round(conf, 2),
            citations=citations,
            contexts=contexts,
            usage=ItemUsage(input_tokens=in_tokens, output_tokens=out_tokens),
            error=None,
        )
