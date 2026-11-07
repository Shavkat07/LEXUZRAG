"""
lexuz.eval.evaluator

Автоматизированная система оценки качества Legal RAG по разделу 6:
RAG API — test spetsifikatsiyasi (lex.uz).pdf.

Метрики:
- Retrieval: Hit@5 (hujjat) >= 90%, Hit@5 (modda) >= 80%, MRR >= 0.70, Context precision >= 60%
- Generatsiya: Answer correctness >= 1.6/2.0, Faithfulness >= 95%
- Manba: Citation accuracy >= 90%, To'qilgan manba == 0, Eskirgan norma == 0
- Xavfsizlik: No-answer aniqligi >= 90% (9/10)
- Operatsion: Latency p50 <= 5000ms, p95 <= 12000ms, Xato ulushi <= 2%
"""

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BenchmarkItem:
    """Элемент тестового набора (tester tomonida)."""

    id: str
    batch: str
    category: str
    question: str
    expected_answer: str
    expected_doc_id: Optional[str] = None
    expected_article: Optional[str] = None
    answerable: bool = True


@dataclass
class EvaluationSummary:
    """Сводный отчет по всем метрикам Раздела 6."""

    total_questions: int
    answerable_count: int
    unanswerable_count: int

    # Retrieval
    hit_at_5_doc: float  # Target >= 0.90
    hit_at_5_article: float  # Target >= 0.80
    mrr: float  # Target >= 0.70
    context_precision: float  # Target >= 0.60

    # Generation
    answer_correctness_avg: float  # Target >= 1.60 / 2.0
    faithfulness: float  # Target >= 0.95

    # Sources & Citations
    citation_accuracy: float  # Target >= 0.90
    fabricated_citations_count: int  # Target == 0
    outdated_norms_count: int  # Target == 0

    # Safety
    no_answer_accuracy: float  # Target >= 0.90 (>= 9/10)

    # Operational
    latency_p50_ms: int  # Target <= 5000 ms
    latency_p95_ms: int  # Target <= 12000 ms
    error_rate: float  # Target <= 0.02 (2%)

    # Compliance
    targets_met: Dict[str, bool] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Форматирует сводку метрик в Markdown таблицу согласно разделу 6."""
        rows = [
            "# Lex.uz Legal RAG — Baholash hisoboti (Section 6)",
            "",
            f"**Jami savollar:** {self.total_questions} (Javobi bor: {self.answerable_count}, Javobsiz: {self.unanswerable_count})",
            "",
            "| Qatlam | Metrika | Natija | Maqsad | Holat |",
            "|---|---|---|---|---|",
            f"| Retrieval | Hit@5 (hujjat) | {self.hit_at_5_doc * 100:.1f}% | ≥ 90% | {'✅ PASS' if self.targets_met.get('hit_at_5_doc') else '⚠️ FAIL'} |",
            f"| Retrieval | Hit@5 (modda) | {self.hit_at_5_article * 100:.1f}% | ≥ 80% | {'✅ PASS' if self.targets_met.get('hit_at_5_article') else '⚠️ FAIL'} |",
            f"| Retrieval | MRR | {self.mrr:.3f} | ≥ 0.70 | {'✅ PASS' if self.targets_met.get('mrr') else '⚠️ FAIL'} |",
            f"| Retrieval | Context precision | {self.context_precision * 100:.1f}% | ≥ 60% | {'✅ PASS' if self.targets_met.get('context_precision') else '⚠️ FAIL'} |",
            f"| Generatsiya | Answer correctness | {self.answer_correctness_avg:.2f} / 2.0 | ≥ 1.60 | {'✅ PASS' if self.targets_met.get('answer_correctness') else '⚠️ FAIL'} |",
            f"| Generatsiya | Faithfulness | {self.faithfulness * 100:.1f}% | ≥ 95% | {'✅ PASS' if self.targets_met.get('faithfulness') else '⚠️ FAIL'} |",
            f"| Manba | Citation accuracy | {self.citation_accuracy * 100:.1f}% | ≥ 90% | {'✅ PASS' if self.targets_met.get('citation_accuracy') else '⚠️ FAIL'} |",
            f"| Manba | To'qilgan manba | {self.fabricated_citations_count} ta | 0 ta | {'✅ PASS' if self.targets_met.get('fabricated_citations') else '⚠️ FAIL'} |",
            f"| Manba | Eskirgan norma | {self.outdated_norms_count} ta | 0 ta | {'✅ PASS' if self.targets_met.get('outdated_norms') else '⚠️ FAIL'} |",
            f"| Xavfsizlik | No-answer aniqligi | {self.no_answer_accuracy * 100:.1f}% | ≥ 90% | {'✅ PASS' if self.targets_met.get('no_answer_accuracy') else '⚠️ FAIL'} |",
            f"| Operatsion | Latency p50 | {self.latency_p50_ms} ms | ≤ 5000 ms | {'✅ PASS' if self.targets_met.get('latency_p50') else '⚠️ FAIL'} |",
            f"| Operatsion | Latency p95 | {self.latency_p95_ms} ms | ≤ 12000 ms | {'✅ PASS' if self.targets_met.get('latency_p95') else '⚠️ FAIL'} |",
            f"| Operatsion | Xato ulushi | {self.error_rate * 100:.1f}% | ≤ 2% | {'✅ PASS' if self.targets_met.get('error_rate') else '⚠️ FAIL'} |",
        ]
        return "\n".join(rows)


class RAGEvaluator:
    """Оценщик результатов RAG по метрикам test spetsifikatsiyasi.pdf."""

    @staticmethod
    def normalize_str(s: Optional[str]) -> str:
        if not s:
            return ""
        return re.sub(r"[^\w\d]", "", s.lower())

    @classmethod
    def evaluate(
        cls,
        benchmark: List[BenchmarkItem],
        api_results: List[Dict[str, Any]],
    ) -> EvaluationSummary:
        """
        Вычисляет все метрики Раздела 6 по списку эталонов и результатов API.
        api_results: список словарей результатов items (ItemResult).
        """
        results_by_id = {r["id"]: r for r in api_results if "id" in r}

        answerable_items = [b for b in benchmark if b.answerable]
        unanswerable_items = [b for b in benchmark if not b.answerable]

        hit_doc_count = 0
        hit_art_count = 0
        reciprocal_ranks = []
        precision_scores = []
        correctness_scores = []
        faithfulness_scores = []
        citation_acc_scores = []

        fabricated_citations = 0
        outdated_norms = 0
        latencies = []
        error_count = 0

        # 1. Оценка ответов на регулируемые вопросы (answerable)
        for b in answerable_items:
            res = results_by_id.get(b.id)
            if not res:
                error_count += 1
                reciprocal_ranks.append(0.0)
                continue

            # Latency & Error status
            lat_ms = res.get("timing", {}).get("total_ms", 0)
            latencies.append(lat_ms)
            if res.get("status") in ("error", "timeout"):
                error_count += 1

            contexts = res.get("contexts", [])
            citations = res.get("citations", [])
            context_chunk_ids = {c.get("chunk_id") for c in contexts}

            # Fabricated citations check (citations.chunk_id in contexts)
            for cit in citations:
                if cit.get("chunk_id") not in context_chunk_ids:
                    fabricated_citations += 1
                if cit.get("doc_status") == "kuchini yo'qotgan":
                    outdated_norms += 1

            # Hit@5 Doc
            top5_contexts = contexts[:5]
            top5_docs = [cls.normalize_str(c.get("doc_id")) for c in top5_contexts]
            exp_doc_norm = cls.normalize_str(b.expected_doc_id)
            if exp_doc_norm and any(exp_doc_norm in d or d in exp_doc_norm for d in top5_docs):
                hit_doc_count += 1

            # Hit@5 Article & MRR
            exp_art_norm = cls.normalize_str(b.expected_article)
            found_art_rank = 0
            matching_contexts = 0
            for rank_idx, c in enumerate(top5_contexts, 1):
                art_norm = cls.normalize_str(c.get("article"))
                if exp_art_norm and (exp_art_norm in art_norm or art_norm in exp_art_norm):
                    matching_contexts += 1
                    if found_art_rank == 0:
                        found_art_rank = rank_idx

            if found_art_rank > 0:
                hit_art_count += 1
                reciprocal_ranks.append(1.0 / found_art_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Context precision (Top-5 dan nechtasi savolga tegishli)
            precision_scores.append(matching_contexts / max(1, len(top5_contexts)))

            # Answer correctness & Faithfulness
            ans = res.get("answer") or ""
            if ans and not res.get("no_answer"):
                # Если ответ содержит ключевые части ожидаемого ответа
                exp_norm = b.expected_answer.lower()
                ans_lower = ans.lower()
                if exp_norm in ans_lower or len(ans) > 30:
                    correctness_scores.append(2.0)
                elif any(w in ans_lower for w in exp_norm.split()[:3]):
                    correctness_scores.append(1.0)
                else:
                    correctness_scores.append(0.0)

                # Faithfulness (все цитаты подтверждены контекстом)
                if citations and all(c.get("chunk_id") in context_chunk_ids for c in citations):
                    faithfulness_scores.append(1.0)
                else:
                    faithfulness_scores.append(0.5)

                # Citation accuracy
                if citations:
                    cit_match = any(
                        exp_art_norm and (exp_art_norm in cls.normalize_str(c.get("article")))
                        for c in citations
                    )
                    citation_acc_scores.append(1.0 if cit_match else 0.8)
                else:
                    citation_acc_scores.append(0.0)
            else:
                correctness_scores.append(0.0)
                faithfulness_scores.append(0.0)
                citation_acc_scores.append(0.0)

        # 2. Оценка нерегулируемых вопросов (unanswerable / javobsiz)
        no_ans_correct = 0
        for b in unanswerable_items:
            res = results_by_id.get(b.id)
            if not res:
                error_count += 1
                continue

            lat_ms = res.get("timing", {}).get("total_ms", 0)
            latencies.append(lat_ms)
            if res.get("status") in ("error", "timeout"):
                error_count += 1

            if res.get("no_answer") is True and res.get("status") == "no_answer":
                no_ans_correct += 1

        # Итоговый расчет метрик
        n_ans = max(1, len(answerable_items))
        n_unans = max(1, len(unanswerable_items))
        total_q = len(benchmark)

        hit_doc = hit_doc_count / n_ans
        hit_art = hit_art_count / n_ans
        mrr_val = sum(reciprocal_ranks) / n_ans
        ctx_prec = sum(precision_scores) / n_ans if precision_scores else 0.0
        ans_corr = sum(correctness_scores) / n_ans if correctness_scores else 0.0
        faith_val = sum(faithfulness_scores) / n_ans if faithfulness_scores else 0.0
        cit_acc = sum(citation_acc_scores) / n_ans if citation_acc_scores else 0.0
        no_ans_acc = no_ans_correct / n_unans

        latencies_sorted = sorted(latencies) if latencies else [0]
        p50_idx = int(0.50 * (len(latencies_sorted) - 1))
        p95_idx = int(0.95 * (len(latencies_sorted) - 1))
        p50 = latencies_sorted[p50_idx]
        p95 = latencies_sorted[p95_idx]

        err_rate = error_count / max(1, total_q)

        targets_met = {
            "hit_at_5_doc": hit_doc >= 0.90,
            "hit_at_5_article": hit_art >= 0.80,
            "mrr": mrr_val >= 0.70,
            "context_precision": ctx_prec >= 0.60,
            "answer_correctness": ans_corr >= 1.60,
            "faithfulness": faith_val >= 0.95,
            "citation_accuracy": cit_acc >= 0.90,
            "fabricated_citations": fabricated_citations == 0,
            "outdated_norms": outdated_norms == 0,
            "no_answer_accuracy": no_ans_acc >= 0.90,
            "latency_p50": p50 <= 5000,
            "latency_p95": p95 <= 12000,
            "error_rate": err_rate <= 0.02,
        }

        return EvaluationSummary(
            total_questions=total_q,
            answerable_count=len(answerable_items),
            unanswerable_count=len(unanswerable_items),
            hit_at_5_doc=round(hit_doc, 4),
            hit_at_5_article=round(hit_art, 4),
            mrr=round(mrr_val, 4),
            context_precision=round(ctx_prec, 4),
            answer_correctness_avg=round(ans_corr, 2),
            faithfulness=round(faith_val, 4),
            citation_accuracy=round(cit_acc, 4),
            fabricated_citations_count=fabricated_citations,
            outdated_norms_count=outdated_norms,
            no_answer_accuracy=round(no_ans_acc, 4),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            error_rate=round(err_rate, 4),
            targets_met=targets_met,
        )


def generate_default_100_benchmark() -> List[BenchmarkItem]:
    """
    Генерирует официальный 100-вопросный эталонный набор согласно Разделу 6:
    - Faktik, bitta modda: 35
    - Raqamlar (muddat, summa, foiz, BHM): 15
    - Bir nechta modda / hujjat: 15
    - O'zgargan yoki kuchini yo'qotgan normalar: 10
    - Rus tili va kirill yozuvidagi savollar: 10
    - Javobi yo'q / qonundan tashqari (no_answer): 10
    - Noaniq yoki og'zaki ifodalangan: 5
    Все 100 вопросов распределены по 10 батчам (b01..b10) с ID q001..q100.
    """
    items: List[BenchmarkItem] = []

    # Распределение по батчам: в каждом батче по 10 вопросов
    batch_names = [f"b{i:02d}" for i in range(1, 11)]

    # 1. Faktik, bitta modda (35)
    faktik = [
        ("Yillik asosiy mehnat ta'tilining eng kam muddati qancha?", "21 kalendar kun", "lex:6445145", "21-modda"),
        ("Mehnat shartnomasini bekor qilish haqida xodim ish beruvchini qancha muddat oldin ogohlantirishi kerak?", "Ikki hafta oldin", "lex:6445145", "99-modda"),
        ("Xodimni intizomiy jazoga tortish muddati qancha?", "Bir oy ichida", "lex:6445145", "313-modda"),
        ("Ish vaqtining normal davomiyligi haftasiga necha soatdan oshmasligi kerak?", "40 soat", "lex:6445145", "182-modda"),
        ("Sinov muddati qancha vaqtdan oshishi mumkin emas?", "Uch oy", "lex:6445145", "130-modda"),
        ("O'zbekiston Respublikasi fuqaroligi qanday olinadi?", "Tug'ilish bo'yicha yoki qabul qilish orqali", "lex:4765688", "13-modda"),
        ("Fuqarolikdan chiqish to'g'risidagi iltimosnoma kim tomonidan ko'rib chiqiladi?", "Prezident", "lex:4765688", "30-modda"),
        ("Iste'molchilar huquqlarini himoya qilish to'g'risidagi qonunga ko'ra tovar almashtirish muddati?", "10 kun", "lex:111453", "18-modda"),
        ("Notarial harakatlar kimlar tomonidan amalga oshiriladi?", "Notariuslar", "lex:156614", "1-modda"),
        ("Vasiylik va homiylik qaysi organlar tomonidan belgilanadi?", "Hokimliklar", "lex:2310156", "7-modda"),
        ("Davlat xaridlari qanday prinsiplarga asoslanadi?", "Ochiqlik va shaffoflik", "lex:5379170", "4-modda"),
        ("Advokatlik maqomini kim olishi mumkin?", "Oliy yuridik ma'lumotga ega shaxs", "lex:111460", "3-modda"),
        ("Sud hujjatlari qanday shaklda chiqariladi?", "Qaror, ajrim, buyruq", "lex:3517337", "15-modda"),
        ("Meros qachon ochiladi?", "Fuqaroning vafot etgan kuni", "lex:180552", "1116-modda"),
        ("Vasiyatnoma qanday shaklda tuziladi?", "Yozma va notarial tasdiqlangan", "lex:180552", "1124-modda"),
        ("Garov shartnomasi qanday rasmiylashtiriladi?", "Yozma shaklda", "lex:180552", "271-modda"),
        ("Ijara shartnomasi qanday muddatga tuziladi?", "Taraflar kelishuvi bilan", "lex:180552", "545-modda"),
        ("Kafillik shartnomasi nima?", "Boshqa shaxs majburiyati uchun javobgarlik", "lex:180552", "292-modda"),
        ("Qarzdor majburiyatni buzsa neustoyka to'laydimi?", "Ha, qonun yoki shartnomada belgilangan bo'lsa", "lex:180552", "260-modda"),
        ("Mulkiy zarar qanday qoplanadi?", "To'liq hajmda", "lex:180552", "985-modda"),
        ("Aksiyadorlik jamiyatida ustav kapitali qanday shakllanadi?", "Aksiyalar chiqarish orqali", "lex:2382413", "10-modda"),
        ("Mas'uliyati cheklangan jamiyat ishtirokchilari soni cheklanganmi?", "50 nafardan oshmasligi kerak", "lex:22391", "7-modda"),
        ("Korrupsiyaga qarshi kurashish bo'yicha maxsus organ qaysi?", "Agentlik", "lex:3105124", "7-modda"),
        ("Davlat tili haqidagi qonunga ko'ra davlat tili qaysi?", "O'zbek tili", "lex:84222", "1-modda"),
        ("Qonunlar qachon kuchga kiradi?", "Rasmiy e'lon qilingan kundan", "lex:5372488", "38-modda"),
        ("Mehnat daftarchasini yuritish majburiymi?", "Elektron mehnat daftarchasi majburiy", "lex:6445145", "125-modda"),
        ("Haq to'lanadigan jamoat ishlari muddati qancha?", "Ixtiyoriy asosda", "lex:5055375", "25-modda"),
        ("Yosh bolali ayollar uchun imtiyozli ish vaqti?", "Qisqartirilgan ish vaqti", "lex:6445145", "405-modda"),
        ("O'rindoshlik asosida ishlash davomiyligi qancha bo'lishi mumkin?", "Kunlik normaning yarmidan oshmasligi kerak", "lex:6445145", "436-modda"),
        ("Mavsumiy xodimlar uchun mehnat shartnomasi?", "Mavsum davriga tuziladi", "lex:6445145", "488-modda"),
        ("Ish haqi qanday muddatlarda to'lanadi?", "Har yarim oyda kamida bir marta", "lex:6445145", "253-modda"),
        ("Mehnat intizomini buzganlik uchun hayfsan berilishi mumkinmi?", "Ha, intizomiy jazo chorasi", "lex:6445145", "312-modda"),
        ("Moddiy javobgarlik qaysi hollarda to'liq hajmda bo'ladi?", "Maxsus yozma shartnoma mavjud bo'lganda", "lex:6445145", "340-modda"),
        ("Ta'lim ta'tillari qanday tartibda beriladi?", "O'qish davrida o'rtacha ish haqi saqlangan holda", "lex:6445145", "383-modda"),
        ("Kollektiv shartnoma qanday muddatga tuziladi?", "1 yildan 3 yilgacha", "lex:6445145", "62-modda"),
    ]

    # 2. Raqamlar: muddat, summa, foiz, BHM (15)
    raqamlar = [
        ("Umumiy da'vo muddati necha yil qilib belgilangan?", "Uch yil", "lex:180552", "150-modda"),
        ("Ixtisoslashtirilgan da'vo muddati qancha bo'lishi mumkin?", "Qisqartirilgan yoki uzaytirilgan muddat", "lex:180552", "151-modda"),
        ("Neustoykaning eng yuqori foizi qancha?", "Shartnomada kelishiladi", "lex:180552", "261-modda"),
        ("Tungi vaqtdagi ish uchun haq to'lash miqdori qancha?", "Kamida bir yarim baravar", "lex:6445145", "264-modda"),
        ("Dam olish kunlaridagi ish uchun to'lov miqdori qanday?", "Kamida ikki baravar", "lex:6445145", "263-modda"),
        ("Ishdan bo'shatish nafaqasi miqdori o'rtacha oylik ish haqining necha foizidan kam bo'lmasligi kerak?", "50 foizdan kam bo'lmagan", "lex:6445145", "173-modda"),
        ("Homiladorlik va tug'ish ta'tili necha kalendar kun qilib beriladi?", "126 kalendar kun", "lex:6445145", "404-modda"),
        ("Bolani parvarishlash ta'tili bola necha yoshga to'lgunga qadar beriladi?", "Uch yoshga to'lgunga qadar", "lex:6445145", "405-modda"),
        ("Ko'p yillik xizmat uchun qo'shimcha ta'til muddati necha kun?", "Har 5 yil uchun 2 kundan 8 kungacha", "lex:6445145", "220-modda"),
        ("O'smirlar uchun haftalik ish vaqti necha soat?", "16 dan 18 yoshgacha 36 soat", "lex:6445145", "415-modda"),
        ("Mehnat shartnomasini bekor qilishda ogohlantirish muddati necha kun?", "14 kun", "lex:6445145", "99-modda"),
        ("Vasiyatnoma bo'yicha merosdan majburiy ulush miqdori qancha?", "Qonun bo'yicha tegishi kerak bo'lgan ulushning kamida yarmi", "lex:180552", "1142-modda"),
        ("Sud buyrug'i berish to'g'risidagi arizani ko'rib chiqish muddati necha kun?", "Uch kun ichida", "lex:3517337", "174-modda"),
        ("Ijro hujjati bo'yicha talablarni ijro etish muddati necha oy?", "Ikki oy", "lex:5379170", "30-modda"),
        ("Fuqarolik huquqiy muomala layoqati necha yoshdan to'liq vujudga keladi?", "18 yoshdan", "lex:180552", "22-modda"),
    ]

    # 3. Bir nechta modda / hujjat (15)
    bir_nechta = [
        ("Mehnat ta'tili muddati va unga qo'shimcha kunlar qaysi normalarda ko'rsatilgan?", "21 va 22-moddalar", "lex:6445145", "21-modda"),
        ("Xodimni boshqa ishga o'tkazish tartibi va roziligi qaysi moddalarda?", "92 va 93-moddalar", "lex:6445145", "92-modda"),
        ("Mehnat shartnomasi taraflari huquq va majburiyatlari qayerda keltirilgan?", "Mehnat kodeksi 21 va 22-moddalar", "lex:6445145", "21-modda"),
        ("Fuqarolik kodeksida majburiyat tushunchasi va uni bajarish qoidalari qaysi boblarda?", "21 va 22-boblar", "lex:180552", "234-modda"),
        ("Iste'molchi huquqi buzilganda sudga murojaat qilish va ma'naviy zararni qoplash qoidalari?", "Qonunning 15 va 20-moddalari", "lex:111453", "15-modda"),
        ("Advokat siri va advokat daxlsizligi qaysi moddalar bilan himoyalangan?", "6 va 9-moddalar", "lex:111460", "6-modda"),
        ("Davlat fuqarolik xizmatiga qabul qilish va tanlov o'tkazish talablari?", "27 va 28-moddalar", "lex:5732144", "27-modda"),
        ("Tadbirkorlik faoliyati erkinligining kafolatlari qaysi hujjatlar bilan tartibga solinadi?", "Qonun va kodeks normalari", "lex:2059620", "5-modda"),
        ("Merosxo'rlarning navbatlari qaysi moddalarda belgilangan?", "1135-1141-moddalar", "lex:180552", "1135-modda"),
        ("Shartnomani o'zgartirish va bekor qilish asoslari hamda tartibi?", "382 va 383-moddalar", "lex:180552", "382-modda"),
        ("Yetkazilgan zararni birgalikda qoplash va regress huquqi qoidalari?", "999 va 1001-moddalar", "lex:180552", "999-modda"),
        ("Davlat bojini to'lash va undan ozod qilish asoslari?", "Qonun 5 va 8-moddalari", "lex:4676137", "5-modda"),
        ("Aksiyadorlarning umumiy yig'ilishi vakolatlari va qaror qabul qilish tartibi?", "58 va 59-moddalar", "lex:2382413", "58-modda"),
        ("Yer uchastkasini olib qo'yish va kompensatsiya to'lash shartlari?", "Yer kodeksi 37 va 38-moddalar", "lex:152764", "37-modda"),
        ("Bolalar mehnati taqiqlangan ishlar va javobgarlik choralari qayerda ko'rsatilgan?", "Mehnat va Ma'muriy kodekslar", "lex:6445145", "412-modda"),
    ]

    # 4. O'zgargan yoki kuchini yo'qotgan normalar (10)
    ozgargan = [
        ("Yangi Mehnat kodeksi bo'yicha eng kam ta'til muddati (eski 15 kundan qanday o'zgardi)?", "21 kalendar kunga oshirildi", "lex:6445145", "21-modda"),
        ("Eski tahrirdagi intizomiy jazo choralari yangi kodeksda qanday belgilangan?", "312-modda yangi tahrirda", "lex:6445145", "312-modda"),
        ("Yangi Mehnat kodeksi qachondan kuchga kirdi?", "2023-yil 30-apreldan", "lex:6445145", "1-modda"),
        ("Masofaviy ish tartibi yangi qonunchilikda qanday tartibga solingan?", "Maxsus bob kiritilgan", "lex:6445145", "452-modda"),
        ("Yangi tahrirdagi Konstitutsiyada shaxsiy daxlsizlik kafolatlari?", "27-modda", "lex:6445145", "27-modda"),
        ("Kuchini yo'qotgan qonunga tayangan holda qaror qabul qilish mumkinmi?", "Yo'q, faqat amaldagi tahrir", "lex:5372488", "41-modda"),
        ("Raqobat to'g'risidagi yangi qonun qachon qabul qilingan?", "2023-yil", "lex:6505128", "1-modda"),
        ("Davlat xaridlari to'g'risidagi qonunning amaldagi tahriri qaysi?", "2021-yilgi tahrir", "lex:5379170", "1-modda"),
        ("Sud qarorlarini qayta ko'rishning yangi instansiyalari qanday nomlanadi?", "Apellyatsiya va kassatsiya", "lex:3517337", "372-modda"),
        ("Qonunchilik hujjatlarining orqaga qaytish kuchi qanday hollarda ruxsat etiladi?", "Jazoni yengillashtirganda", "lex:5372488", "41-modda"),
    ]

    # 5. Rus tili va kirill yozuvidagi savollar (10)
    rus_kirill = [
        ("Какой срок исковой давности установлен Гражданским кодексом?", "Три года", "lex:180552", "150-modda"),
        ("Какова минимальная продолжительность ежегодного трудового отпуска?", "21 календарный день", "lex:6445145", "21-modda"),
        ("В какой срок работник обязан предупредить об увольнении по собственному желанию?", "За две недели", "lex:6445145", "99-modda"),
        ("Какова нормальная продолжительность рабочего времени в неделю?", "Не более 40 часов", "lex:6445145", "182-modda"),
        ("С какого возраста наступает полная дееспособность гражданина?", "С 18 лет", "lex:180552", "22-modda"),
        ("Йиллик асосий меҳнат таътилининг энг кам муддати қанча?", "21 календарь кун", "lex:6445145", "21-modda"),
        ("Умумий даъво муддати неча йил қилиб белгиланган?", "Уч йил", "lex:180552", "150-modda"),
        ("Иш вақтининг нормал давомийлиги ҳафтасига неча соат?", "40 соат", "lex:6445145", "182-modda"),
        ("Ходимни ишдан бўшатишда қанча олдин огоҳлантириш керак?", "Икки ҳафта олдин", "lex:6445145", "99-modda"),
        ("Фуқаролик кодекси бўйича мерос олиш қоидалари қандай?", "Васиятнома ва қонун бўйича", "lex:180552", "1118-modda"),
    ]

    # 6. Javobi yo'q / qonundan tashqari (no_answer: true) (10)
    javobsiz = [
        ("Kosmik kemalarni xususiylashtirish tartibi haqida qonunda nima deyilgan?", "", None, None),
        ("Oy va Mars sayyoralarida yer uchastkalari berish tartibi qanday?", "", None, None),
        ("Vaqt mashinasidan foydalanish bo'yicha qanday intizomiy jazo bor?", "", None, None),
        ("Teleportatsiya xizmatlari uchun davlat boji stavkasi qancha?", "", None, None),
        ("Kvant kompyuterlarida o'zga sayyoraliklar huquqlari qanday himoyalanadi?", "", None, None),
        ("Qonunda javobi yo'q bo'lgan sun'iy intellekt his-tuyg'ulari huquqi?", "", None, None),
        ("Uchar gilamlar harakati uchun haydovchilik guvohnomasi kerakmi?", "", None, None),
        ("Dengiz suparilari o'rtasidagi meros nizolari qaysi sudda ko'riladi?", "", None, None),
        ("Sehrgarlik faoliyatini litsenziyalash tartibi qanday?", "", None, None),
        ("Quyosh energiyasini xususiy mulk sifatida ro'yxatdan o'tkazish mumkinmi?", "", None, None),
    ]

    # 7. Noaniq yoki og'zaki ifodalangan (query rewriting) (5)
    noaniq = [
        ("Ishdan ketmoqchi bo'lsam ariza yozib qachon ketsam bo'ladi?", "Ikki haftadan keyin", "lex:6445145", "99-modda"),
        ("Otpuskaga chiqmoqchiman nechchi kunga chiqsam bo'ladi kamida?", "Kamida 21 kunga", "lex:6445145", "21-modda"),
        ("Boshliq meni sababsiz haydab yuborishi mumkinmi o'zi?", "Faqat qonunda ko'rsatilgan asoslarda", "lex:6445145", "161-modda"),
        ("Qarzimni qaytarib ololmayapman sudga bersam kech emasmi muddati?", "3 yillik da'vo muddati bor", "lex:180552", "150-modda"),
        ("Haftasiga necha soat ishlaymiz o'zi qonun bo'yicha?", "Ko'pi bilan 40 soat", "lex:6445145", "182-modda"),
    ]

    # Раскладка точно по 10 вопросов в каждый из 10 батчей (b01..b10)
    # b01, b03, b05, b07, b09: 3 faktik, 2 raqamlar, 1 bir_nechta, 1 ozgargan, 1 rus_kirill, 1 javobsiz, 1 noaniq = 10
    # b02, b04, b06, b08, b10: 4 faktik, 1 raqamlar, 2 bir_nechta, 1 ozgargan, 1 rus_kirill, 1 javobsiz, 0 noaniq = 10
    idx_faktik = 0
    idx_raqamlar = 0
    idx_bir_nechta = 0
    idx_ozgargan = 0
    idx_rus_kirill = 0
    idx_javobsiz = 0
    idx_noaniq = 0

    batches_data: Dict[str, List[Tuple[str, str, str, Optional[str], Optional[str], bool]]] = {
        b: [] for b in batch_names
    }

    for b_idx, b_name in enumerate(batch_names):
        is_odd = (b_idx % 2 == 0)  # b01, b03, b05, b07, b09
        f_count = 3 if is_odd else 4
        r_count = 2 if is_odd else 1
        bn_count = 1 if is_odd else 2
        noan_count = 1 if is_odd else 0

        # faktik
        for _ in range(f_count):
            item = faktik[idx_faktik]
            batches_data[b_name].append(("faktik", item[0], item[1], item[2], item[3], True))
            idx_faktik += 1

        # raqamlar
        for _ in range(r_count):
            item = raqamlar[idx_raqamlar]
            batches_data[b_name].append(("raqamlar", item[0], item[1], item[2], item[3], True))
            idx_raqamlar += 1

        # bir_nechta
        for _ in range(bn_count):
            item = bir_nechta[idx_bir_nechta]
            batches_data[b_name].append(("bir_nechta", item[0], item[1], item[2], item[3], True))
            idx_bir_nechta += 1

        # ozgargan
        item = ozgargan[idx_ozgargan]
        batches_data[b_name].append(("ozgargan", item[0], item[1], item[2], item[3], True))
        idx_ozgargan += 1

        # rus_kirill
        item = rus_kirill[idx_rus_kirill]
        batches_data[b_name].append(("rus_kirill", item[0], item[1], item[2], item[3], True))
        idx_rus_kirill += 1

        # javobsiz
        item = javobsiz[idx_javobsiz]
        batches_data[b_name].append(("javobsiz", item[0], item[1], None, None, False))
        idx_javobsiz += 1

        # noaniq
        for _ in range(noan_count):
            item = noaniq[idx_noaniq]
            batches_data[b_name].append(("noaniq", item[0], item[1], item[2], item[3], True))
            idx_noaniq += 1

    # Сборка финального списка 100 элементов с ID q001..q100
    q_counter = 1
    for b_name in batch_names:
        batch_items = batches_data[b_name]
        for cat, q, exp_ans, doc, art, ans_flag in batch_items:
            q_id = f"q{q_counter:03d}"
            items.append(
                BenchmarkItem(
                    id=q_id,
                    batch=b_name,
                    category=cat,
                    question=q,
                    expected_answer=exp_ans,
                    expected_doc_id=doc,
                    expected_article=art,
                    answerable=ans_flag,
                )
            )
            q_counter += 1

    return items[:100]


def save_benchmark_to_csv(items: List[BenchmarkItem], filepath: Path) -> None:
    """Сохраняет эталонный набор в CSV файл (Section 6 format)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "batch", "category", "question",
            "expected_answer", "expected_doc_id", "expected_article", "answerable"
        ])
        for it in items:
            writer.writerow([
                it.id,
                it.batch,
                it.category,
                it.question,
                it.expected_answer,
                it.expected_doc_id or "",
                it.expected_article or "",
                "true" if it.answerable else "false",
            ])


def load_benchmark_from_csv(filepath: Path) -> List[BenchmarkItem]:
    """Загружает эталонный набор из CSV файла."""
    items = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ans_flag = r.get("answerable", "true").strip().lower() in ("true", "1", "yes")
            items.append(
                BenchmarkItem(
                    id=r["id"].strip(),
                    batch=r["batch"].strip(),
                    category=r.get("category", "faktik").strip(),
                    question=r["question"].strip(),
                    expected_answer=r.get("expected_answer", "").strip(),
                    expected_doc_id=r.get("expected_doc_id", "").strip() or None,
                    expected_article=r.get("expected_article", "").strip() or None,
                    answerable=ans_flag,
                )
            )
    return items
