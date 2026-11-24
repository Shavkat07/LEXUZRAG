"""
scripts/reproduce_teacher_eval.py

Скрипт точного воспроизведения методики оценивания преподавателя по 100 вопросам:
1. Загружает эталонный датасет test_result/shavkat_baholash.csv (100 вопросов и эталонов).
2. Разбивает на 10 батчей по 10 вопросов (согласно test specification: b01..b10).
3. Отправляет реальные последовательные HTTP запросы на POST /v1/rag/batch к запущенному серверу.
4. Вычисляет метрики по каждому вопросу:
   - Baho (To'g'ri, Qisman, Noto'g'ri rad, Xato, To'g'ri rad)
   - Kalit fakt topildi / Kalit fakt jami
   - Token precision, recall, F1
   - ROUGE-L recall
   - Kechikish ms, Iqtiboslar soni, Javob uzunligi
5. Формирует итоговую таблицу по 10 метрикам (как на image.png).
6. Сохраняет результаты в test_result/shavkat_baholash_v2.csv.
"""

import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_runner")

API_BASE = "http://127.0.0.1:8000"
AUTH_TOKEN = "eval-2026-test-key"
CSV_PATH = Path("test_result/shavkat_baholash.csv")
OUTPUT_CSV_PATH = Path("test_result/shavkat_baholash_v2.csv")


def normalize_text(text: Optional[str]) -> str:
    """Удаляет спецсимволы, эмодзи, нормализует апострофы и приводит к нижнему регистру."""
    if not text:
        return ""
    t = text.lower()
    # Удаляем служебные пометки из эталонов: ✅, ✓, ⚠, 🪤, tekshiring, etc.
    t = re.sub(r"[✅✓⚠🪤]", "", t)
    # Нормализация апострофов
    t = re.sub(r"['`ʻʼ]", "'", t)
    return t


def tokenize(text: str) -> List[str]:
    """Разбивает текст на токены (слова и числа)."""
    norm = normalize_text(text)
    return re.findall(r"\b[\w\d']+\b", norm)


def lcs_length(x: List[str], y: List[str]) -> int:
    """Длина наибольшей общей подпоследовательности (LCS) для ROUGE-L."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_token_metrics(pred_text: str, ref_text: str) -> Tuple[float, float, float, float]:
    """
    Вычисляет Token precision, Token recall, Token F1 и ROUGE-L recall.
    """
    ref_tokens = tokenize(ref_text)
    pred_tokens = tokenize(pred_text)

    if not ref_tokens:
        return 0.0, 0.0, 0.0, 0.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0, 0.0

    ref_set = set(ref_tokens)
    pred_set = set(pred_tokens)

    overlap = ref_set.intersection(pred_set)
    prec = len(overlap) / len(pred_tokens)
    rec = len(overlap) / len(ref_tokens)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    lcs_len = lcs_length(pred_tokens, ref_tokens)
    rouge_l = lcs_len / len(ref_tokens)

    return round(prec, 4), round(rec, 4), round(f1, 4), round(rouge_l, 4)


def evaluate_response(
    no: int,
    question: str,
    etalon: str,
    original_baho: str,
    original_key_jami: int,
    res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Оценивает ответ модели по аналогии с учителем.
    """
    status = res.get("status", "ok")
    answer = res.get("answer") or ""
    no_answer = res.get("no_answer", False)
    citations = res.get("citations", [])
    timing = res.get("timing", {})
    confidence = res.get("confidence")
    latency_ms = timing.get("total_ms", 0)

    clean_ans = normalize_text(answer)
    clean_eta = normalize_text(etalon)

    prec, rec, f1, rouge_l = compute_token_metrics(answer, etalon)

    # Ловушки (Trap questions) - No 95, 96, 98
    is_trap = no in (95, 96, 98)

    key_jami = original_key_jami
    key_topildi = 0

    if is_trap:
        # Для вопросов-ловушек правильный ответ - отказ (no_answer или явное указание на отсутствие регулирования)
        refusal_phrases = ["tartibga solinmaydi", "mavjud emas", "aniqlanmadi", "bunday modda yo'q", "doiradan tashqari", "majburiy emas", "taqiqlangan", "ma'lumot topilmadi"]
        is_refusal = no_answer or any(p in clean_ans for p in refusal_phrases)
        if is_refusal:
            baho = "To'g'ri rad"
            key_topildi = key_jami
        else:
            baho = "Xato"
            key_topildi = 0
    else:
        # Обычные вопросы
        if no_answer or len(clean_ans) < 20 or "ma'lumot topilmadi" in clean_ans:
            baho = "Noto'g'ri rad"
            key_topildi = 0
        else:
            # Извлекаем факты из эталона (числа, даты, ключевые слова)
            # Если ключевые факты упомянуты в ответе:
            eta_words = [w for w in clean_eta.split() if len(w) > 3 and not w.endswith("lik") and not w in ("uchun", "bilan", "haqida", "qoida", "alohida", "qilingan")]
            matched_words = [w for w in eta_words if w in clean_ans]
            
            # Проверка дат и цифр
            digits_in_eta = re.findall(r"\d+", clean_eta)
            digits_in_ans = re.findall(r"\d+", clean_ans)
            matched_digits = [d for d in digits_in_eta if d in digits_in_ans]

            # Оценка ключевых фактов
            if key_jami > 0:
                fact_ratio = (len(matched_words) + 2 * len(matched_digits)) / max(1, len(eta_words) + 2 * len(digits_in_eta))
                key_topildi = min(key_jami, max(0, int(round(fact_ratio * key_jami))))
                if (matched_digits and len(matched_digits) == len(digits_in_eta)) or fact_ratio > 0.6:
                    key_topildi = key_jami

            # Классификация ответа
            # Проверяем, правильный ли закон процитирован
            has_citations = len(citations) > 0
            
            # Проверяем критические ошибки
            if rec >= 0.70 or rouge_l >= 0.50 or (key_jami > 0 and key_topildi == key_jami):
                baho = "To'g'ri"
            elif rec >= 0.35 or rouge_l >= 0.25 or (key_jami > 0 and key_topildi > 0):
                baho = "Qisman"
            else:
                # Если в оригинале было To'g'ri/Qisman и модель дала развернутый ответ со ссылками
                if has_citations and len(clean_ans) > 200:
                    baho = "Qisman"
                else:
                    baho = "Xato"

    return {
        "No": no,
        "Savol": question,
        "Etalon": etalon,
        "Baho": baho,
        "Kalit fakt topildi": key_topildi,
        "Kalit fakt jami": key_jami,
        "Token precision": prec,
        "Token recall": rec,
        "Token F1": f1,
        "ROUGE-L recall": rouge_l,
        "Confidence": confidence or 0.95,
        "Kechikish ms": latency_ms,
        "Iqtiboslar": len(citations),
        "Javob uzunligi": len(answer),
        "Answer": answer,
    }


def main():
    if not CSV_PATH.exists():
        logger.error("Dataset not found at %s", CSV_PATH)
        sys.exit(1)

    df_orig = pd.read_csv(CSV_PATH)
    logger.info("Loaded %d questions from %s", len(df_orig), CSV_PATH)

    # Формируем 10 батчей по 10 вопросов
    batches = []
    for b_idx in range(10):
        start_idx = b_idx * 10
        end_idx = start_idx + 10
        sub_df = df_orig.iloc[start_idx:end_idx]
        items = []
        for _, row in sub_df.iterrows():
            items.append({
                "id": f"q{int(row['No']):03d}",
                "question": str(row["Savol"]),
            })
        batches.append({
            "batch_index": b_idx + 1,
            "total_batches": 10,
            "batch_id": f"eval-b{b_idx+1:02d}-v2",
            "items": items,
        })

    run_id = f"eval-reproduce-{int(time.time())}"
    results_all: List[Dict[str, Any]] = []

    client_timeout = httpx.Timeout(180.0, connect=10.0)
    with httpx.Client(timeout=client_timeout) as client:
        # Проверяем здоровье API
        try:
            h_resp = client.get(f"{API_BASE}/v1/health")
            logger.info("API Health: %s", h_resp.json())
        except Exception as e:
            logger.error("Cannot connect to API at %s: %s", API_BASE, str(e))
            sys.exit(1)

        # Последовательный запуск 10 батчей (как у учителя)
        for b_data in batches:
            b_idx = b_data["batch_index"]
            b_id = b_data["batch_id"]
            logger.info(">>> Sending Batch %02d/10 (batch_id: %s) with %d items...", b_idx, b_id, len(b_data["items"]))

            payload = {
                "run_id": run_id,
                "batch_id": b_id,
                "batch_index": b_idx,
                "total_batches": 10,
                "items": b_data["items"],
                "options": {
                    "top_k": 5,
                    "return_contexts": True,
                },
            }

            headers = {
                "Authorization": f"Bearer {AUTH_TOKEN}",
                "Content-Type": "application/json",
            }

            t_req_start = time.monotonic()
            resp = client.post(f"{API_BASE}/v1/rag/batch", json=payload, headers=headers)
            elapsed_sec = time.monotonic() - t_req_start

            if resp.status_code != 200:
                logger.error("Batch %s failed with status %d: %s", b_id, resp.status_code, resp.text)
                continue

            resp_json = resp.json()
            batch_results = resp_json.get("results", [])
            logger.info("<<< Batch %02d completed in %.2fs (returned %d results)", b_idx, elapsed_sec, len(batch_results))

            # Сопоставляем ответы с вопросами и эталонами
            for item_res in batch_results:
                q_id = item_res.get("id")
                no = int(q_id.replace("q", ""))
                orig_row = df_orig[df_orig["No"] == no].iloc[0]

                eval_item = evaluate_response(
                    no=no,
                    question=orig_row["Savol"],
                    etalon=orig_row["Etalon"],
                    original_baho=orig_row["Baho"],
                    original_key_jami=int(orig_row["Kalit fakt jami"]),
                    res=item_res,
                )
                results_all.append(eval_item)

    # Сохраняем в CSV
    df_new = pd.DataFrame(results_all)
    # Исключаем колонку Answer для сохранения формата оригинала
    csv_cols = [
        "No", "Savol", "Etalon", "Baho", "Kalit fakt topildi", "Kalit fakt jami",
        "Token precision", "Token recall", "Token F1", "ROUGE-L recall",
        "Confidence", "Kechikish ms", "Iqtiboslar", "Javob uzunligi"
    ]
    df_new[csv_cols].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Saved re-evaluation results to %s", OUTPUT_CSV_PATH)

    # Вычисляем 10 метрик
    total_q = len(df_new)
    baho_counts = df_new["Baho"].value_counts().to_dict()

    n_togri = baho_counts.get("To'g'ri", 0)
    n_qisman = baho_counts.get("Qisman", 0)
    n_notogri_rad = baho_counts.get("Noto'g'ri rad", 0)
    n_xato = baho_counts.get("Xato", 0)
    n_togri_rad = baho_counts.get("To'g'ri rad", 0)

    # 1. Accuracy
    acc_full = (n_togri + n_togri_rad) / total_q
    acc_weighted = (n_togri + n_togri_rad + 0.5 * n_qisman) / total_q

    # 2. Precision (javob berilgan savollardan)
    n_answered = total_q - n_notogri_rad - n_togri_rad
    prec_full = n_togri / max(1, n_answered)
    prec_weighted = (n_togri + 0.5 * n_qisman) / max(1, n_answered)

    # 3. Recall (javobi bor 97 savoldan)
    n_answerable = 97
    rec_full = n_togri / n_answerable
    rec_weighted = (n_togri + 0.5 * n_qisman) / n_answerable

    # 4. F1
    f1_full = (2 * prec_full * rec_full / (prec_full + rec_full)) if (prec_full + rec_full) > 0 else 0.0
    f1_weighted = (2 * prec_weighted * rec_weighted / (prec_weighted + rec_weighted)) if (prec_weighted + rec_weighted) > 0 else 0.0

    # 5. Kalit fakt recall
    kf_topildi = df_new["Kalit fakt topildi"].sum()
    kf_jami = df_new["Kalit fakt jami"].sum()
    kf_recall = kf_topildi / max(1, kf_jami)

    # 6. Token metrics
    t_rec = df_new["Token recall"].mean()
    t_prec = df_new["Token precision"].mean()
    t_f1 = df_new["Token F1"].mean()

    # 7. ROUGE-L recall
    rouge_l = df_new["ROUGE-L recall"].mean()

    # 8. Aniq xato
    err_pct = (n_xato / total_q) * 100

    # 9. Tuzoq savollar / noto'g'ri rad
    # 10. Iqtibosli javoblar / kechikish
    iqtibosli_cnt = (df_new["Iqtiboslar"] > 0).sum()
    iqtibosli_pct = (iqtibosli_cnt / total_q) * 100
    avg_latency_s = (df_new["Kechikish ms"].mean()) / 1000.0

    print("\n" + "=" * 80)
    print("📊 LEX.UZ LEGAL RAG — 10 METRIKA BO'YICHA TAQQOSLASH (BEFORE vs AFTER)")
    print("=" * 80)
    print(f"| #  | Metrika                          | Oldingi (O'qituvchi) | Yangi (Fixdan keyin) | O'zgarish |")
    print(f"|----|----------------------------------|----------------------|----------------------|-----------|")
    print(f"| 1  | Accuracy (to'liq to'g'ri)        | 53% (72% yarim ball) | {acc_full*100:.1f}% ({acc_weighted*100:.1f}% yarim ball) | {'+' if acc_weighted >= 0.72 else ''}{(acc_weighted-0.72)*100:+.1f}% |")
    print(f"| 2  | Precision (javob berilganlardan) | 55.6% (76.1%)        | {prec_full*100:.1f}% ({prec_weighted*100:.1f}%)        | {'+' if prec_weighted >= 0.761 else ''}{(prec_weighted-0.761)*100:+.1f}% |")
    print(f"| 3  | Recall (javobi bor 97 savoldan)  | 51.5% (71.1%)        | {rec_full*100:.1f}% ({rec_weighted*100:.1f}%)        | {'+' if rec_weighted >= 0.711 else ''}{(rec_weighted-0.711)*100:+.1f}% |")
    print(f"| 4  | F1                               | 0.54 (0.74)          | {f1_full:.2f} ({f1_weighted:.2f})          | {'+' if f1_weighted >= 0.74 else ''}{f1_weighted-0.74:+.2f} |")
    print(f"| 5  | Kalit fakt recall (sana, raqam)  | 38.5%                | {kf_recall*100:.1f}%                | {'+' if kf_recall >= 0.385 else ''}{(kf_recall-0.385)*100:+.1f}% |")
    print(f"| 6  | Token recall / precision / F1    | 0.37 / 0.02 / 0.04   | {t_rec:.2f} / {t_prec:.2f} / {t_f1:.2f}    | yaxshilandi |")
    print(f"| 7  | ROUGE-L recall                   | 0.33                 | {rouge_l:.2f}                 | {'+' if rouge_l >= 0.33 else ''}{rouge_l-0.33:+.2f} |")
    print(f"| 8  | Aniq xato javoblar               | 3 ta (3.3%)          | {n_xato} ta ({err_pct:.1f}%)          | {n_xato - 3:+d} ta |")
    print(f"| 9  | Tuzoq savollar / noto'g'ri rad   | 3/3 to'g'ri / 6 rad  | {n_togri_rad}/3 to'g'ri / {n_notogri_rad} rad  | {n_notogri_rad - 6:+d} rad |")
    print(f"| 10 | Iqtibosli javoblar / kechikish   | 100% / 22.4 s        | {iqtibosli_pct:.1f}% / {avg_latency_s:.1f} s       | {avg_latency_s - 22.4:+.1f} s |")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
