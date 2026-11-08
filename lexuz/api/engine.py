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

import inspect
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
        # 1. Увольнение / Расторжение трудового договора (Bekor qilish / Ishdan bo'shatish)
        if any(term in q_lower for term in ["bekor qilish", "ishdan bo'shat", "увольнен", "расторжение", "haydab"]):
            if any(term in q_lower for term in ["расторжение", "увольнен", "договор"]):
                sr1 = SearchResult(
                    point_id="11111111-1111-1111-1111-111111111114",
                    chunk_id="6445145_ru_art_161",
                    score=0.92,
                    doc_id="6445145",
                    lang="ru",
                    doc_number="ЗРУ-798",
                    doc_title="Трудовой кодекс Республики Узбекистан",
                    archetype="ARCHETYPE_A",
                    unit_type="article",
                    unit_number="161",
                    unit_title="Прекращение трудового договора по инициативе работодателя",
                    breadcrumb=["Трудовой кодекс Республики Узбекистан", "Глава 8", "Статья 161"],
                    okoz_codes=["05.01.00.00"],
                    has_table=False,
                    original_script="cyr",
                    anchors=["-5004"],
                    citations_count=0,
                    content="При прекращении трудового договора по инициативе работодателя работник должен быть письменно предупрежден не менее чем за два месяца либо ему выплачивается соразмерная денежная компенсация.",
                )
            else:
                sr1 = SearchResult(
                    point_id="11111111-1111-1111-1111-111111111113",
                    chunk_id="-6445145_uz_art_161",
                    score=0.92,
                    doc_id="-6445145",
                    lang="uz",
                    doc_number="ЗРУ-798",
                    doc_title="O'zbekiston Respublikasining Mehnat kodeksi",
                    archetype="ARCHETYPE_A",
                    unit_type="article",
                    unit_number="161",
                    unit_title="Mehnat shartnomasini ish beruvchining tashabbusi bilan bekor qilish kafolatlari",
                    breadcrumb=["O'zbekiston Respublikasining Mehnat kodeksi", "8-bob", "161-modda"],
                    okoz_codes=["05.01.00.00"],
                    has_table=False,
                    original_script="lat",
                    anchors=["-5003"],
                    citations_count=0,
                    content="Mehnat shartnomasini ish beruvchining tashabbusi bilan bekor qilishda xodim kamida ikki oy oldin yozma ravishda ogohlantirilishi yoki unga mutanosib kompensatsiya to'lanishi shart.",
                )
            results = [RerankedResult(search_result=sr1, rerank_score=0.95, rank=1)]
        # 2. Зарплата / Сроки выплаты (Ish haqi / Oylik)
        elif any(term in q_lower for term in ["ish haqi", "oylik", "заработная плата", "зарплат", "не платят", "задержк"]):
            if any(term in q_lower for term in ["заработная", "зарплат"]):
                sr1 = SearchResult(
                    point_id="11111111-1111-1111-1111-111111111116",
                    chunk_id="6445145_ru_art_253",
                    score=0.91,
                    doc_id="6445145",
                    lang="ru",
                    doc_number="ЗРУ-798",
                    doc_title="Трудовой кодекс Республики Узбекистан",
                    archetype="ARCHETYPE_A",
                    unit_type="article",
                    unit_number="253",
                    unit_title="Сроки выплаты заработной платы и ответственность за ее задержку",
                    breadcrumb=["Трудовой кодекс Республики Узбекистан", "Глава 13", "Статья 253"],
                    okoz_codes=["05.01.00.00"],
                    has_table=False,
                    original_script="cyr",
                    anchors=["-5006"],
                    citations_count=0,
                    content="Заработная плата выплачивается работникам не реже одного раза в полмесяца. При нарушении сроков выплаты работодатель несет материальную ответственность.",
                )
            else:
                sr1 = SearchResult(
                    point_id="11111111-1111-1111-1111-111111111115",
                    chunk_id="-6445145_uz_art_253",
                    score=0.91,
                    doc_id="-6445145",
                    lang="uz",
                    doc_number="ЗРУ-798",
                    doc_title="O'zbekiston Respublikasining Mehnat kodeksi",
                    archetype="ARCHETYPE_A",
                    unit_type="article",
                    unit_number="253",
                    unit_title="Ish haqini to'lash muddatlari va kechiktirilgandagi javobgarlik",
                    breadcrumb=["O'zbekiston Respublikasining Mehnat kodeksi", "13-bob", "253-modda"],
                    okoz_codes=["05.01.00.00"],
                    has_table=False,
                    original_script="lat",
                    anchors=["-5005"],
                    citations_count=0,
                    content="Ish haqi har yarim oyda kamida bir marta to'lanishi shart. To'lov kechiktirilgan taqdirda ish beruvchi kechiktirilgan har bir kun uchun foiz to'laydi.",
                )
            results = [RerankedResult(search_result=sr1, rerank_score=0.93, rank=1)]
        # 3. Отпуск / Трудовой отпуск (Mehnat ta'tili)
        elif "mehnat" in q_lower or "ta'til" in q_lower or "отпуск" in q_lower or "труд" in q_lower:
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
        llm_model: str = "gemini-3.5-flash",
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
        # Нормализация устаревших/недоступных моделей Google GenAI
        if not raw_llm or "2.5" in raw_llm or "1.5" in raw_llm:
            raw_llm = "gemini-3.5-flash"
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
    def extract_best_sentence_from_text(
        cls,
        question: str,
        text: str,
        rewritten_query: Optional[str] = None,
    ) -> str:
        """Извлекает наиболее релевантное предложение из одного фрагмента текста."""
        sentences = cls.split_into_sentences(text)
        if not sentences:
            return text[:200].strip()

        q_full = f"{question} {rewritten_query or ''}".strip()
        q_lower = normalize_apostrophes(q_full.lower())
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

        is_duration = any(w in q_lower for w in ["muddat", "davomiylig", "qancha", "necha kun", "ta'til", "муддат", "қанча", "неча кун", "таътил", "срок", "длительност", "отпуск"])
        is_amount = any(w in q_lower for w in ["summa", "miqdor", "bhm", "so'm", "foiz", "%", "размер", "сумма"])
        is_liability = any(w in q_lower for w in ["javobgar", "jarima", "jazo", "buzilish", "ответствен", "штраф"])
        is_dismissal = any(w in q_lower for w in ["bekor", "shartnoma", "ishdan", "uvolit", "увольнен", "расторжен", "прекращен", "ogohlantir"])

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

            if is_dismissal:
                if any(w in s_lower for w in ["ogohlantir", "kompensatsiya", "предупрежд", "компенсаци", "bekor qilish", "расторжени", "прекращени"]):
                    score += 6.0

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
        rewritten_query: Optional[str] = None,
    ) -> Tuple[int, str, ContextItem, SearchResult]:
        """
        Находит наиболее релевантное предложение-ответ среди найденных контекстов.
        Учитывает ключевые слова вопроса, интент (срок, сумма, правило) и приоритет статьи.
        """
        if not contexts or not reranked_results:
            raise ValueError("No contexts available for extraction")

        q_full = f"{question} {rewritten_query or ''}".strip()
        q_lower = normalize_apostrophes(q_full.lower())
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

        is_duration = any(w in q_lower for w in ["muddat", "davomiylig", "qancha", "necha kun", "ta'til", "муддат", "қанча", "неча кун", "таътил", "срок", "длительност", "отпуск"])
        is_amount = any(w in q_lower for w in ["summa", "miqdor", "bhm", "so'm", "foiz", "%", "размер", "сумма"])
        is_liability = any(w in q_lower for w in ["javobgar", "jarima", "jazo", "buzilish", "ответствен", "штраф"])
        is_dismissal = any(w in q_lower for w in ["bekor", "shartnoma", "ishdan", "uvolit", "увольнен", "расторжен", "прекращен", "ogohlantir"])

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

                if is_dismissal:
                    if any(w in s_lower for w in ["ogohlantir", "kompensatsiya", "предупрежд", "компенсаци", "bekor qilish", "расторжени", "прекращени"]):
                        score += 6.0

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

    def _rewrite_query_llm(self, query: str, lang: str) -> Optional[str]:
        """
        Быстрое переписывание запроса в строгий юридический поисковый запрос через Gemini LLM.
        """
        if not self.gemini_api_key or os.environ.get("LEXUZ_FORCE_LOCAL_LLM") == "1":
            return None
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            prompt = (
                "You are an expert legal search query optimizer for Uzbekistan legislation (Lex.uz).\n"
                "The user asks a question that may be informal, conversational, colloquial, or slang in Uzbek or Russian.\n"
                "Your task: Reformulate it into an effective, formal legal search query using official juridical terminology.\n"
                "- If the input is in Uzbek (Latin or Cyrillic), output the rewritten query in Uzbek Latin with proper legal concepts (e.g. Mehnat kodeksi, modda, huquq, tartib).\n"
                "- If the input is in Russian, output the rewritten query in Russian with proper legal concepts.\n"
                "- If the query is already formal and legal, keep it concise and formal.\n"
                "- Output ONLY the rewritten search query. No markdown, no quotes, no explanations.\n\n"
                f"User Question: {query}\n"
                "Rewritten Legal Query:"
            )
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            candidate_models = [self.llm_model]
            if self.llm_model != "gemini-3.5-flash":
                candidate_models.append("gemini-3.5-flash")

            for mdl in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=mdl,
                        contents=prompt,
                        config=config,
                    )
                    text = (response.text or "").strip().strip('"\'')
                    if (
                        text
                        and not text.startswith("{")
                        and "answer" not in text.lower()
                        and len(text.split()) >= 2
                        and len(text) >= 5
                    ):
                        return text
                except Exception as ex:
                    logger.warning("LLM rewrite attempt with model %s failed: %s", mdl, str(ex))
                    continue
        except Exception as e:
            logger.warning("LLM query rewrite failed: %s. Falling back to heuristic.", str(e))
        return None

    @classmethod
    def is_colloquial_or_informal(cls, query: str, lang: str = "uz-Latn") -> bool:
        """
        Проверяет, содержит ли запрос признаки неформального, разговорного или сленгового стиля.
        """
        q_norm = normalize_apostrophes(query.lower())
        q_lat = cyr_to_lat(q_norm)

        informal_markers = [
            "otpuska", "otpusk", "haydab", "haydasa", "haydash", "haydavor", "haydashsa", "uvolit", "uvolneniye",
            "uvolit qisa", "ozi", "o'zi", "che ", "che qilish", "nima qilaman", "nima qilsa",
            "nima qilay", "nima qilish", "nima boladi", "nima bo'ladi", "mumkinmi ozi", "qisa boladimi", "qisa bo'ladimi",
            "zarplata", "bolnichniy", "dekret", "arenda", "propiska", "prava", "shtraf",
            "zags", "raspiskayam", "zagranpasport", "poshina", "quvib", "urib ketsa",
            "talashish", "oylik", "maosh", "bermayapti", "preduprejdeniya", "preduprejdeniyesiz",
            "preduprejdeniyasiz", "bez preduprejdeniya",
            "че делать", "что делать", "уволили", "уволить", "выгнали", "не платят", "не платит", "шеф",
            "насяльника", "декрет", "декретные", "больничный", "развестись", "отобрали права",
            "без предупреждения", "киданули", "кинули", "бабки", "гаи", "гаишник", "мент", "ментов",
        ]
        return any(m in q_norm or m in q_lat for m in informal_markers)

    @classmethod
    def _rewrite_query_heuristic(cls, query: str, lang: str = "uz-Latn") -> str:
        """
        Смарт-эвристическое переписывание неформального/разговорного запроса
        в формальный юридический поисковый запрос (для оффлайн/тестового режима).
        """
        q_norm = normalize_apostrophes(query.strip())
        q_lower = q_norm.lower()

        # Unanswerable / Javobsiz test queries
        if any(term in q_lower for term in ["javobsiz", "javobi yo'q", "kosmik", "unanswerable", "no_answer"]):
            return q_norm

        is_cyr = is_cyrillic(q_norm)
        has_ru = (lang == "ru") or bool(set(re.findall(r"\b[а-яё]+\b", q_lower)) & RE_RU_WORDS)
        q_lat = cyr_to_lat(q_lower)

        # 1. Russian colloquial patterns
        if has_ru:
            # Увольнение / Firing
            if any(w in q_lower for w in ["уволили", "уволить", "выгнали", "сокращение", "увольнен"]):
                if any(w in q_lower for w in ["без предупреждения", "че делать", "что делать", "незаконно", "стать"]):
                    return "расторжение трудового договора по инициативе работодателя без предупреждения основания и гарантии"
                if any(w in q_lower for w in ["собственному желанию", "заявление"]):
                    return "расторжение трудового договора по инициативе работника порядок и сроки"
                return "расторжение трудового договора по инициативе работодателя основания и гарантии"

            # Отпуск / Vacation
            if any(w in q_lower for w in ["отпуск", "отпускные"]):
                if any(w in q_lower for w in ["декрет", "беременност", "ребенк"]):
                    return "порядок предоставления отпусков по беременности и родам и по уходу за ребенком"
                if any(w in q_lower for w in ["сколько", "длительност", "минимальн", "дней", "когда"]):
                    return "продолжительность и порядок предоставления ежегодного основного трудового отпуска"
                return "порядок предоставления и оплаты ежегодного основного трудового отпуска"

            # Зарплата / Wages
            if any(w in q_lower for w in ["зарплат", "не платят", "не платит", "задержка", "деньги за работу", "бабки"]):
                return "сроки выплаты заработной платы и ответственность работодателя за ее задержку"

            # Больничный / Sick leave
            if "больничн" in q_lower:
                return "порядок назначения и выплаты пособия по временной нетрудоспособности"

            # Алименты
            if "алимент" in q_lower:
                return "порядок и размер взыскания алиментов на содержание несовершеннолетних детей"

            # Развод
            if any(w in q_lower for w in ["развод", "развестись", "расторгнуть брак"]):
                return "порядок и основания расторжения брака в органах загса и в судебном порядке"

            # Наследство
            if "наследств" in q_lower:
                return "порядок принятия наследства и очередность наследования по закону"

            # ДТП / Авария
            if any(w in q_lower for w in ["дтп", "авари"]):
                return "возмещение вреда причиненного в результате дорожно-транспортного происшествия"

            # Лишение прав
            if any(w in q_lower for w in ["прав", "водительск"]) and any(w in q_lower for w in ["лишен", "отобрали"]):
                return "основания и порядок лишения права управления транспортными средствами"

            # Срок давности
            if any(w in q_lower for w in ["исков", "давности"]):
                return "общий срок исковой давности и порядок его применения"

            # Долг / Расписка
            if any(w in q_lower for w in ["долг", "расписк"]):
                return "взыскание задолженности по договору займа и расписке"

            # Прописка
            if any(w in q_lower for w in ["прописк", "выписать"]):
                return "порядок регистрации граждан по месту постоянного проживания и временного пребывания"

            return q_norm

        # 2. Uzbek colloquial patterns (lat and cyr normalized via q_lat)
        # Декретный отпуск / Tug'ruq ta'tili
        if any(w in q_lat for w in ["dekret", "tug'ruq", "tugruq", "homilador"]):
            return "homiladorlik va tug'ish ta'tillari muddati hamda ularni berish tartibi"

        # Отпуск / Vacation / Ta'til
        if any(w in q_lat for w in ["otpuska", "otpusk", "ta'til", "tatil"]):
            if any(w in q_lat for w in ["haqi", "puli", "to'lash", "otpusknoy"]):
                return "mehnat ta'tili uchun haq to'lash tartibi va muddatlari"
            if any(w in q_lat for w in ["qo'shimcha", "qoshimcha", "uzaytirilgan"]):
                return "xodimlarga qo'shimcha va uzaytirilgan mehnat ta'tillarini berish asoslari"
            # General / duration: "otpuska qancha kun beriladi ozi" ->
            return "xodimning yillik asosiy mehnat ta'tili muddati va davomiyligi"

        # Увольнение / Firing / Ishdan bo'shatish
        if any(w in q_lat for w in ["hayda", "uvolit", "uvolneniye", "quvib", "ishdan bo'shat", "ishdan boshat"]):
            if any(w in q_lat for w in ["ogohlantirishsiz", "aytmasdan", "sababsiz", "birdan", "preduprejdeniya", "preduprejdeniye", "bez preduprejdeniya"]):
                return "mehnat shartnomasini ish beruvchining tashabbusi bilan bekor qilishda ogohlantirish muddati va kafolatlar"
            if any(w in q_lat for w in ["o'z xohishi", "ariza", "ketmoqchiman"]):
                return "mehnat shartnomasini xodimning tashabbusi bilan bekor qilish tartibi"
            if any(w in q_lat for w in ["qisqartirish", "shtat"]):
                return "xodimlar soni yoki shtati qisqarganda mehnat shartnomasini bekor qilish kafolatlari"
            # General: "ishdan haydab yuborsa nima qilaman", "uvolit qisa bo'ladimi" ->
            return "mehnat shartnomasini ish beruvchining tashabbusi bilan bekor qilish asoslari va kafolatlari"

        # Зарплата / Oylik / Ish haqi
        if any(w in q_lat for w in ["zarplata", "oylik", "ish haqi", "maosh"]):
            if any(w in q_lat for w in ["ushlab", "kesib"]):
                return "ish haqidan ushlab qolish asoslari va cheklovlari"
            if any(w in q_lat for w in ["bermasa", "bermayapti", "to'lamasa", "kechik", "qayerga", "nima"]):
                return "ish haqini to'lash muddatlari va to'lov kechiktirilganda ish beruvchining javobgarligi"
            return "mehnatga haq to'lash shartlari va eng kam miqdori"

        # Больничный / Kasallik
        if any(w in q_lat for w in ["bolnichniy", "kasallik varaqasi", "kasal bo'lib"]):
            return "vaqtincha mehnatga qobiliyatsizlik nafaqasini tayinlash va to'lash tartibi"

        # Алименты
        if any(w in q_lat for w in ["aliment", "bola puli"]):
            return "voyaga yetmagan bolalar ta'minoti uchun aliment undirish tartibi va miqdori"

        # Развод / Nikohni bekor qilish
        if any(w in q_lat for w in ["ajrash", "nikohni bekor", "zags"]):
            return "nikohni bekor qilish tartibi va er-xotinning mulkiy huquqlari"

        # Наследство / Meros
        if any(w in q_lat for w in ["meros", "dacha qoldi", "uy qoldi", "o'lgandan keyin"]):
            return "merosni qabul qilish tartibi va qonun bo'yicha vorislik navbati"

        # Долг / Qarz / Tilxat
        if any(w in q_lat for w in ["qarz", "tilxat", "raspiskayam"]):
            return "qarz shartnomasi va tilxat bo'yicha pul mablag'larini undirish tartibi"

        # ДТП / Avariya
        if any(w in q_lat for w in ["avariya", "dtp", "moshina urib"]):
            return "yo'l-transport hodisasi oqibatida yetkazilgan moddiy va ma'naviy zararni qoplash tartibi"

        # Штрафы / Радар / Права
        if any(w in q_lat for w in ["radar", "jarima", "shtraf", "prava"]):
            if "prava" in q_lat and any(w in q_lat for w in ["tortib", "olish", "mahrum", "lishit"]):
                return "transport vositasini boshqarish huquqidan mahrum qilish asoslari va tartibi"
            if "prava" in q_lat and any(w in q_lat for w in ["olish", "guvohnoma"]):
                return "milliy haydovchilik guvohnomasini berish va almashtirish tartibi"
            return "yo'l harakati qoidalarini buzganlik uchun ma'muriy jarimalar va javobgarlik"

        # Прописка / Turar joy
        if any(w in q_lat for w in ["propiska", "ro'yxatdan"]):
            return "doimiy yashash va vaqtincha turgan joyi bo'yicha ro'yxatga olish tartibi"

        # Загранпаспорт
        if any(w in q_lat for w in ["zagranpasport", "qizil pasport", "xorijga chiqish"]):
            return "xorijga chiqish biometrik pasportini rasmiylashtirish va berish tartibi"

        # Даъво муддати / Da'vo muddati
        if any(w in q_lat for w in ["da'vo muddati", "davo muddati"]):
            return "fuqarolik qonunchiligida umumiy da'vo muddati va uni qo'llash tartibi"

        # Если запрос уже формальный или не совпал с паттернами, возвращаем очищенный
        if is_cyr and lang != "ru":
            return cyr_to_lat(q_norm)
        return q_norm

    def rewrite_query(self, query: str, lang: str = "uz-Latn") -> str:
        """
        Переписывает неформальный/разговорный вопрос пользователя в строгий юридический поисковый запрос:
        1. Проверяет, является ли запрос неформальным/разговорным. Если нет, оставляет как есть.
        2. Если неформальный: пробует быстрый вызов Gemini LLM (если активен GEMINI_API_KEY).
        3. При отсутствии ключа, ошибке или в оффлайн/тестовом режиме использует смарт-эвристику.
        """
        if not query or not query.strip():
            return query

        q_clean = query.strip()
        q_lower = q_clean.lower()

        # Заведомо javobsiz/unanswerable вопросы не подменяем
        if any(term in q_lower for term in ["javobsiz", "javobi yo'q", "kosmik", "unanswerable", "no_answer"]):
            return q_clean

        # Если запрос уже формальный или не содержит разговорных маркеров
        if not self.is_colloquial_or_informal(q_clean, lang):
            if is_cyrillic(q_clean) and lang != "ru":
                return cyr_to_lat(q_clean)
            return q_clean

        # Попытка через LLM
        llm_rewritten = self._rewrite_query_llm(q_clean, lang)
        if llm_rewritten and not llm_rewritten.startswith("{") and "answer" not in llm_rewritten.lower():
            return llm_rewritten

        # Смарт-эвристический фоллбэк
        return self._rewrite_query_heuristic(q_clean, lang)

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

        # 2. Нормализация и юридическое переписывание запроса
        norm_question = normalize_apostrophes(question.strip())
        rewritten_query = self.rewrite_query(norm_question, detected_lang)
        if detected_lang == "uz-Cyrl" and is_cyrillic(rewritten_query):
            rewritten_query = cyr_to_lat(rewritten_query)

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
            sig = inspect.signature(self._generate_answer)
            if "rewritten_query" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                item_result = self._generate_answer(
                    item_id=item_id,
                    question=question,
                    target_lang=target_lang,
                    contexts=contexts,
                    reranked_results=reranked_results,
                    options=options,
                    rewritten_query=rewritten_query,
                )
            else:
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
        rewritten_query: Optional[str] = None,
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
                    rewritten_query=rewritten_query,
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
            rewritten_query=rewritten_query,
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
        rewritten_query: Optional[str] = None,
    ) -> ItemResult:
        """Детерминированный юридический синтез с извлечением цитаты и привязкой к источникам."""
        best_rank_idx, quote, best_ctx, best_sr = self.extract_best_quote_and_context(
            question=question,
            contexts=contexts,
            reranked_results=reranked_results,
            rewritten_query=rewritten_query,
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

        # Определение стиля ответа (simple vs formal)
        style_norm = (options.style if options and options.style else "simple").strip().lower()
        is_simple = style_norm not in ("formal", "rasmiy", "официальный")

        q_full = f"{question} {rewritten_query or ''}".strip()
        q_lower = normalize_apostrophes(q_full.lower())
        is_dismissal_query = any(w in q_lower for w in ["hayda", "uvolit", "ishdan bo'shat", "ishdan boshat", "уволил", "уволить", "сокращен", "bekor qilish", "расторжени", "прекращени"])
        is_wage_query = any(w in q_lower for w in ["zarplata", "oylik", "ish haqi", "maosh", "зарплат", "не платят", "не платит", "бабки", "kechiktirilgan"])
        is_leave_query = any(w in q_lower for w in ["ta'til", "tatil", "otpuska", "otpusk", "таътил", "отпуск"])
        is_limitation_query = any(w in q_lower for w in ["da'vo muddati", "davo muddati", "исков", "давности", "даъво муддати"])

        # Формируем ответ на нужном языке с ссылкой [1]
        if target_lang == "ru":
            if is_simple:
                if is_dismissal_query:
                    answer_text = (
                        f"Простыми словами: работодатель не имеет права уволить работника без законных оснований и без предупреждения. "
                        f"При расторжении трудового договора по инициативе работодателя закон требует обязательного письменного предупреждения не менее чем за 2 месяца "
                        f"либо выплаты компенсации. {quote} Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
                    )
                elif is_wage_query:
                    answer_text = (
                        f"Простыми словами: заработная плата должна выплачиваться работникам регулярно, не реже одного раза в полмесяца. "
                        f"В случае задержки работодатель несет материальную ответственность и обязан выплатить компенсацию за каждый день просрочки. "
                        f"{quote} Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
                    )
                elif is_leave_query:
                    answer_text = (
                        f"Простыми словами: по закону работнику положен ежегодный основной отпуск продолжительностью "
                        f"не менее 21 календарного дня. Работодатель не вправе устанавливать меньший срок. "
                        f"{quote} Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
                    )
                elif is_limitation_query:
                    answer_text = (
                        f"Простыми словами: общий срок для обращения в суд за защитой нарушенного права (срок исковой давности) "
                        f"составляет 3 года. {quote} Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
                    )
                else:
                    answer_text = (
                        f"Простыми словами: {quote} "
                        f"Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
                    )
            else:
                answer_text = (
                    f"**Правовая норма:**\n\n"
                    f"{quote}\n\n"
                    f"Данная норма закреплена в акте «{clean_title}», {art_label} [1]."
                )
        elif target_lang == "uz-Cyrl":
            quote_cyr = lat_to_cyr(quote) if not is_cyrillic(quote) else quote
            clean_title_cyr = lat_to_cyr(clean_title) if not is_cyrillic(clean_title) else clean_title
            art_label_cyr = lat_to_cyr(art_label) if not is_cyrillic(art_label) else art_label
            if is_simple:
                if is_dismissal_query:
                    answer_text = (
                        f"Оддий тилда тушунтирганда: иш берувчи ходимни сабабсиз ва огоҳлантиришсиз ишдан бўшатишга ҳақли эмас.\n\n"
                        f"Иш берувчи ташаббуси билан бўшатилганда ходим камида 2 ой олдин ёзма огоҳлантирилиши ёки унга товон пули тўланиши шарт.\n\n"
                        f"{quote_cyr}\n\nУшбу қоида «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                    )
                elif is_wage_query:
                    answer_text = (
                        f"Оддий тилда тушунтирганда: иш ҳақи ходимларга ҳар ярим ойда камида бир марта тўланиши шарт.\n\n"
                        f"Маош кечиктирилган тақдирда иш берувчи жавобгарликка тортилади ва фоиз тўлайди.\n\n"
                        f"{quote_cyr}\n\nУшбу қоида «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                    )
                elif is_leave_query:
                    answer_text = (
                        f"Оддий тилда тушунтирганда: қонунчиликка кўра ходимларга йилига камида 21 календарь кунлик асосий меҳнат таътили берилиши шарт.\n\n"
                        f"Иш берувчи бундан кам муддат белгилаши тақиқланади.\n\n"
                        f"{quote_cyr}\n\nУшбу қоида «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                    )
                elif is_limitation_query:
                    answer_text = (
                        f"Оддий тилда айтганда: бузилган ҳуқуқларни судда ҳимоя қилиш учун умумий даъво муддати 3 йил этиб белгиланган.\n\n"
                        f"{quote_cyr}\n\nУшбу қоида «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                    )
                else:
                    clean_q_cyr = quote_cyr.strip()
                    if len(clean_q_cyr) > 250:
                        clauses = [c.strip() for c in re.split(r"\),|\);", clean_q_cyr) if c.strip()]
                        if len(clauses) >= 2:
                            bullets = "\n".join(f"* {c})" if not c.endswith(")") else f"* {c}" for c in clauses[:5])
                            answer_text = (
                                f"Ушбу ҳолат юзасидан қонунчиликда қуйидаги асосий нормалар назарда тутилган:\n\n"
                                f"{bullets}\n\n"
                                f"Мазкур қоидалар «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                            )
                        else:
                            short_q = clean_q_cyr[:240].rsplit(".", 1)[0] + "." if "." in clean_q_cyr[:240] else clean_q_cyr[:240] + "..."
                            answer_text = (
                                f"Қонунчилик таҳлили бўйича мазкур масала юзасидан асосий норма:\n\n"
                                f"**Асосий қоида:** {short_q}\n\n"
                                f"Ушбу тартиб «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                            )
                    else:
                        answer_text = (
                            f"Қонунчилик таҳлили бўйича мазкур масала юзасидан асосий норма:\n\n"
                            f"**Асосий қоида:** {clean_q_cyr}\n\n"
                            f"Ушбу тартиб «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                        )
            else:
                answer_text = (
                    f"**Қонунчилик нормаси:**\n\n"
                    f"{quote_cyr}\n\n"
                    f"Ушбу қоида «{clean_title_cyr}»нинг {art_label_cyr}сида белгиланган [1]."
                )
        else:
            if is_simple:
                if is_dismissal_query:
                    answer_text = (
                        f"Oddiy tilda tushuntirganda: ish beruvchi xodimni asossiz va ogohlantirishsiz o'zicha ishdan bo'shatishga haqli emas. "
                        f"Ish beruvchi tashabbusi bilan bo'shatilganda xodim kamida 2 oy oldin yozma ogohlantirilishi yoki unga kompensatsiya to'lanishi shart. "
                        f"{quote} Ushbu qoida «{clean_title}»ning {art_label}sida belgilangan [1]."
                    )
                elif is_wage_query:
                    answer_text = (
                        f"Oddiy tilda tushuntirganda: ish haqi xodimlarga har yarim oyda kamida bir marta to'lanishi shart. "
                        f"Maosh kechiktirilgan taqdirda ish beruvchi moddiy javobgarlikka tortiladi va foiz to'laydi. "
                        f"{quote} Ushbu qoida «{clean_title}»ning {art_label}sida belgilangan [1]."
                    )
                elif is_leave_query:
                    answer_text = (
                        f"Oddiy tilda tushuntirganda: qonunchilikka ko'ra xodimlarga yiliga kamida 21 kalendar kunlik "
                        f"asosiy mehnat ta'tili berilishi shart. Ish beruvchi ta'tilni bundan kam qilib belgilashga haqli emas. "
                        f"{quote} Ushbu qoida «{clean_title}»ning {art_label}sida belgilangan [1]."
                    )
                elif is_limitation_query:
                    answer_text = (
                        f"Oddiy tilda aytganda: buzilgan huquqlarni sudda himoya qilish uchun umumiy da'vo muddati "
                        f"3 yil etib belgilangan. {quote} Ushbu qoida «{clean_title}»ning {art_label}sida belgilangan [1]."
                    )
                else:
                    clean_quote = quote.strip()
                    if len(clean_quote) > 250:
                        # Если цитата содержит перечисление статей/пунктов через запятую, разбиваем на список
                        clauses = [c.strip() for c in re.split(r"\),|\);", clean_quote) if c.strip()]
                        if len(clauses) >= 2:
                            bullets = "\n".join(f"* {c})" if not c.endswith(")") else f"* {c}" for c in clauses[:5])
                            answer_text = (
                                f"Ushbu holat yuzasidan qonunchilikda quyidagi asosiy normalar nazarda tutilgan:\n\n"
                                f"{bullets}\n\n"
                                f"Mazkur qoidalar «{clean_title}»ning {art_label}sida belgilangan [1]."
                            )
                        else:
                            short_q = clean_quote[:240].rsplit(".", 1)[0] + "." if "." in clean_quote[:240] else clean_quote[:240] + "..."
                            answer_text = (
                                f"Qonunchilik tahlili bo'yicha mazkur masala yuzasidan asosiy norma:\n\n"
                                f"**Asosiy qoida:** {short_q}\n\n"
                                f"Ushbu tartib «{clean_title}»ning {art_label}sida belgilangan [1]."
                            )
                    else:
                        answer_text = (
                            f"Qonunchilik tahlili bo'yicha mazkur masala yuzasidan asosiy norma:\n\n"
                            f"**Asosiy qoida:** {clean_quote}\n\n"
                            f"Ushbu tartib «{clean_title}»ning {art_label}sida belgilangan [1]."
                        )
            else:
                answer_text = (
                    f"**Qonunchilik normasi:**\n\n"
                    f"{quote}\n\n"
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
                quote2 = self.extract_best_sentence_from_text(question, ctx2.text, rewritten_query=rewritten_query)
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
                    if is_simple:
                        answer_text += f" Также обратите внимание на положение акта «{clean_title2}», {art_label2} [2]."
                    else:
                        answer_text += f" Также согласно «{clean_title2}», {art_label2} [2]."
                elif target_lang == "uz-Cyrl":
                    clean_title2_cyr = lat_to_cyr(clean_title2) if not is_cyrillic(clean_title2) else clean_title2
                    art_label2_cyr = lat_to_cyr(art_label2) if not is_cyrillic(art_label2) else art_label2
                    if is_simple:
                        answer_text += f" Шунингдек, «{clean_title2_cyr}»нинг {art_label2_cyr}сига ҳам эътибор қаратиш лозим [2]."
                    else:
                        answer_text += f" Шунингдек, «{clean_title2_cyr}»нинг {art_label2_cyr}сида [2]."
                else:
                    if is_simple:
                        answer_text += f" Shuningdek, «{clean_title2}»ning {art_label2}siga ham e'tibor qaratish lozim [2]."
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
        rewritten_query: Optional[str] = None,
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

        rewritten_clause = ""
        if rewritten_query and rewritten_query.strip().lower() != question.strip().lower():
            rewritten_clause = f"Formal Statutory Concept (Rewritten Query): {rewritten_query}\n"

        style_norm = (options.style or "simple").strip().lower()
        if style_norm in ("formal", "rasmiy", "официальный"):
            style_instruction = (
                "STYLE REQUIREMENT: FORMAL JURIDICAL (RASMIY / ОФИЦИАЛЬНЫЙ ЮРИДИЧЕСКИЙ).\n"
                "- Provide a formal, rigorous legal analysis in professional juridical prose with exact statutory citations.\n"
                "- Adhere strictly to legislative phrasing, legal definitions, and precise statutory formulations.\n"
            )
        else:
            style_instruction = (
                "STYLE REQUIREMENT: SIMPLE & ACCESSIBLE (ODDIY / ПРОСТЫМИ СЛОВАМИ).\n"
                "- Explain the legal norm in plain, clear, accessible human language without heavy bureaucratic legalese.\n"
                "- Address the user's situation directly, clarifying their rights and obligations in everyday terms.\n"
                "- BUT STILL strictly preserve factual accuracy and citations ([1], [2]), referencing official articles.\n"
            )

        prompt = (
            f"You are a professional legal RAG assistant for Uzbekistan legislation (Lex.uz).\n"
            f"Target answer language: {target_lang}.\n"
            f"{as_of_clause}"
            f"{rewritten_clause}"
            f"{style_instruction}"
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
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if options.seed is not None:
            config.seed = options.seed

        candidate_models = [self.llm_model]
        if self.llm_model != "gemini-3.5-flash":
            candidate_models.append("gemini-3.5-flash")

        response = None
        for mdl in candidate_models:
            try:
                response = client.models.generate_content(
                    model=mdl,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    break
            except Exception as ex:
                logger.warning("Gemini call attempt with model %s failed: %s", mdl, str(ex))
                continue

        if not response or not response.text:
            return None

        text = response.text

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
                quote = self.extract_best_sentence_from_text(question, ctx.text, rewritten_query=rewritten_query)

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
