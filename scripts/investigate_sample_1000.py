"""
data/investigate_sample_1000.py

Скрипт глубокого исследования и верификации на 1000 реальных документах:
- Стратифицированная выборка 1000 документов (21 категория ОКОЗ, Архетипы A и B, UZ Latin, UZ Cyrillic, RU).
- Запуск парсера ArchetypeAParser.
- Точечная и сквозная проверка строчку за строчкой.
- Детальный сбор метрик и подготовка отчета для координатора.
"""

import gzip
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from lexuz.common.translit import (
    is_cyrillic,
    normalize_uz_text,
    normalize_apostrophes,
    RE_HAS_CYRILLIC,
)
from lexuz.parsers.archetype_a import ArchetypeAParser, parse_structural_header
from lexuz.parsers.table_parser import parse_table_grid
from selectolax.lexbor import LexborHTMLParser

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

RAW_DIR = Path("data/raw_html")
MASTER_REGISTRY_PATH = Path("data/master_registry.json")
OUTPUT_CHUNKS_PATH = Path("data/sample_1000_chunks.jsonl.gz")
OUTPUT_REPORT_PATH = Path("data/sample_1000_report.json")


def load_master_registry():
    print("Загрузка master_registry.json...")
    with open(MASTER_REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc_meta = {}
    for d in data.get("documents", []):
        did = str(d["doc_id"])
        top_okoz = "UNKNOWN"
        for bc in d.get("breadcrumbs", []):
            m = re.match(r"^(\d{2})", bc.strip())
            if m:
                top_okoz = m.group(1)
                break
        doc_meta[did] = {
            "doc_id": did,
            "top_okoz": top_okoz,
            "title": d.get("title", ""),
            "doc_number": d.get("doc_number", ""),
            "breadcrumbs": d.get("breadcrumbs", []),
        }
    print(f"Загружено метаданных для {len(doc_meta)} актов.")
    return doc_meta


def scan_and_select_sample(doc_meta: Dict[str, Dict], target_total: int = 1000):
    print("Сканирование корпуса raw_html для формирования стратифицированной выборки...")
    
    # 1. Индексируем доступные файлы на диске
    available_files = defaultdict(dict)  # doc_id -> {lang: file_path}
    for p in RAW_DIR.rglob("*.html.gz"):
        name_parts = p.name.replace(".html.gz", "").rsplit("_", 1)
        if len(name_parts) == 2:
            did, lang = name_parts
            available_files[did][lang] = p

    print(f"Всего доступно уникальных doc_id на диске: {len(available_files)}")

    # 2. Выделяем важные категории
    # Найдем известные кодексы
    codes_doc_ids = []
    for did, meta in doc_meta.items():
        title_lower = meta["title"].lower()
        if ("kodeks" in title_lower or "кодекс" in title_lower) and did in available_files:
            codes_doc_ids.append(did)

    print(f"Найдено кодексов в базе: {len(codes_doc_ids)}")

    # Группируем по (top_okoz, has_both_langs)
    okoz_groups = defaultdict(list)
    for did, langs in available_files.items():
        meta = doc_meta.get(did, {})
        top_okoz = meta.get("top_okoz", "UNKNOWN")
        okoz_groups[top_okoz].append(did)

    selected_files = []  # list of (file_path, doc_id, lang, is_archetype_b_hint)
    selected_doc_ids = set()

    # А. Гарантированно берем все найденные кодексы (в обоих языках, если есть)
    for did in codes_doc_ids[:25]:
        langs = available_files[did]
        for lang in ("uz", "ru"):
            if lang in langs:
                selected_files.append((langs[lang], did, lang, False, "CODE"))
        selected_doc_ids.add(did)

    print(f"После добавления кодексов: {len(selected_files)} файлов ({len(selected_doc_ids)} документов).")

    # Б. Гарантированно берем стратифицированную выборку по ОКОЗ (01-21) с двуязычными парами
    all_okoz = sorted([k for k in okoz_groups.keys() if k.isdigit()])
    print(f"Категории ОКОЗ в базе: {all_okoz}")

    target_per_okoz = 32  # 21 * 32 ~= 672 файлов
    for okoz in all_okoz:
        dids = okoz_groups[okoz]
        # Перемешаем
        shuffled = list(dids)
        random.seed(int(okoz) + 100)
        random.shuffle(shuffled)

        added_okoz = 0
        for did in shuffled:
            if did in selected_doc_ids:
                continue
            langs = available_files[did]
            # Предпочитаем пары UZ + RU
            if "uz" in langs and "ru" in langs:
                selected_files.append((langs["uz"], did, "uz", False, f"OKOZ_{okoz}"))
                selected_files.append((langs["ru"], did, "ru", False, f"OKOZ_{okoz}"))
                selected_doc_ids.add(did)
                added_okoz += 2
            elif "uz" in langs:
                selected_files.append((langs["uz"], did, "uz", False, f"OKOZ_{okoz}"))
                selected_doc_ids.add(did)
                added_okoz += 1
            elif "ru" in langs:
                selected_files.append((langs["ru"], did, "ru", False, f"OKOZ_{okoz}"))
                selected_doc_ids.add(did)
                added_okoz += 1

            if added_okoz >= target_per_okoz:
                break

    print(f"После ОКОЗ стратификации: {len(selected_files)} файлов ({len(selected_doc_ids)} документов).")

    # В. Добавляем Архетип B (файлы без классов .ACT_TEXT и .CLAUSE_DEFAULT)
    print("Поиск кандидатов Архетипа B...")
    arch_b_candidates = []
    for did, langs in available_files.items():
        if did in selected_doc_ids:
            continue
        for lang, fpath in langs.items():
            # Быстрая проверка
            try:
                with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as fp:
                    head = fp.read(15000)
                if len(head) > 800 and "divCont" in head:
                    if "CLAUSE_DEFAULT" not in head and "ACT_TEXT" not in head and "TEXT_HEADER_DEFAULT" not in head:
                        arch_b_candidates.append((fpath, did, lang, True, "ARCHETYPE_B"))
            except Exception:
                continue
            if len(arch_b_candidates) >= 200:
                break
        if len(arch_b_candidates) >= 200:
            break

    print(f"Найдено {len(arch_b_candidates)} кандидатов Архетипа B.")
    # Берем 150 из них
    for item in arch_b_candidates[:150]:
        selected_files.append(item)
        selected_doc_ids.add(item[1])

    print(f"После добавления Архетипа B: {len(selected_files)} файлов.")

    # Г. Добавляем UZ Cyrillic документы
    print("Поиск кириллических узбекских актов...")
    cyr_candidates = []
    for did, langs in available_files.items():
        if did in selected_doc_ids or "uz" not in langs:
            continue
        fpath = langs["uz"]
        try:
            with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as fp:
                txt = fp.read(10000)
            if "divCont" in txt and is_cyrillic(txt[:5000]):
                cyr_candidates.append((fpath, did, "uz", False, "UZ_CYR"))
                if "ru" in langs:
                    cyr_candidates.append((langs["ru"], did, "ru", False, "UZ_CYR_PAIR"))
        except Exception:
            continue
        if len(cyr_candidates) >= 150:
            break

    print(f"Найдено {len(cyr_candidates)} файлов с кириллическим узбекским.")
    for item in cyr_candidates:
        if len(selected_files) >= target_total:
            break
        selected_files.append(item)
        selected_doc_ids.add(item[1])

    # Д. Если еще не хватает до 1000, добираем из оставшихся документов
    if len(selected_files) < target_total:
        print(f"Добираем до {target_total} из общего пула...")
        for did, langs in available_files.items():
            if did in selected_doc_ids:
                continue
            for lang, fpath in langs.items():
                selected_files.append((fpath, did, lang, False, "GENERAL"))
                if len(selected_files) >= target_total:
                    break
            selected_doc_ids.add(did)
            if len(selected_files) >= target_total:
                break

    # Обрезаем ровно до target_total, если превысило
    selected_files = selected_files[:target_total]
    print(f"Итоговая выборка: {len(selected_files)} файлов из {len(selected_doc_ids)} уникальных актов.")
    return selected_files


def run_investigation():
    doc_meta = load_master_registry()
    sample = scan_and_select_sample(doc_meta, target_total=1000)

    print("=" * 70)
    print("ЗАПУСК ПАРСЕРА НА 1,000 ДОКУМЕНТАХ...")
    print("=" * 70)

    start_time = time.time()
    parsed_files = 0
    zero_chunk_files = 0
    error_files = 0
    total_chunks = 0

    results_by_doc = {}
    all_chunks = []
    errors = []

    # Для детального аудита
    archetype_counts = Counter()
    lang_script_counts = Counter()
    unit_type_counts = Counter()
    okoz_counts = Counter()

    # Хранилище для точечной проверки
    spot_check_candidates = defaultdict(list)

    for idx, (fpath, did, lang, is_arch_b, category_tag) in enumerate(sample, 1):
        try:
            with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as fp:
                html = fp.read()
        except Exception as e:
            error_files += 1
            errors.append({"file": str(fpath), "doc_id": did, "lang": lang, "error": str(e)})
            continue

        try:
            parser = ArchetypeAParser(html, doc_id=did, lang=lang)
            chunks = parser.parse_chunks()

            meta = doc_meta.get(did, {})
            top_okoz = meta.get("top_okoz", "UNKNOWN")
            okoz_counts[top_okoz] += 1

            script = "cyr" if parser.is_original_cyrillic else "lat"
            lang_script_counts[f"{lang}_{script}"] += 1

            if is_arch_b:
                archetype_counts["ARCHETYPE_B"] += 1
            else:
                archetype_counts["ARCHETYPE_A"] += 1

            if not chunks:
                zero_chunk_files += 1
                results_by_doc[(did, lang)] = {
                    "chunks_count": 0,
                    "category": category_tag,
                    "top_okoz": top_okoz,
                    "is_arch_b": is_arch_b,
                    "script": script,
                    "title": meta.get("title", ""),
                }
            else:
                parsed_files += 1
                total_chunks += len(chunks)
                results_by_doc[(did, lang)] = {
                    "chunks_count": len(chunks),
                    "category": category_tag,
                    "top_okoz": top_okoz,
                    "is_arch_b": is_arch_b,
                    "script": script,
                    "title": meta.get("title", ""),
                }
                for ch in chunks:
                    utype = ch["structural_unit"]["type"]
                    unit_type_counts[utype] += 1
                    all_chunks.append(ch)

                    # Отбираем кандидатов для точечной проверки
                    if len(spot_check_candidates[utype]) < 10:
                        spot_check_candidates[utype].append(ch)
                    if ch["metadata"].get("has_table") and len(spot_check_candidates["table"]) < 10:
                        spot_check_candidates["table"].append(ch)
                    if script == "cyr" and lang == "uz" and len(spot_check_candidates["cyr_converted"]) < 10:
                        spot_check_candidates["cyr_converted"].append(ch)
                    if ch["metadata"].get("citations") and len(spot_check_candidates["citations"]) < 10:
                        spot_check_candidates["citations"].append(ch)

        except Exception as e:
            error_files += 1
            errors.append({
                "file": str(fpath),
                "doc_id": did,
                "lang": lang,
                "category": category_tag,
                "error": f"{type(e).__name__}: {str(e)}"
            })

        if idx % 100 == 0 or idx == len(sample):
            elapsed = time.time() - start_time
            print(f"[{idx:4d}/1000] Обработано ({elapsed:.1f}s) | Чанков: {total_chunks:6d} | Успешно: {parsed_files} | 0 чанков: {zero_chunk_files} | Ошибок: {error_files}")

    total_time = time.time() - start_time
    print(f"\nВремя обработки 1,000 документов: {total_time:.2f} сек ({len(sample)/total_time:.1f} док/сек).")

    # Сохраняем полученные чанки в JSONL.GZ
    print(f"Сохранение {len(all_chunks)} чанков в {OUTPUT_CHUNKS_PATH}...")
    with gzip.open(OUTPUT_CHUNKS_PATH, "wt", encoding="utf-8") as out_f:
        for ch in all_chunks:
            out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # Проводим углубленный аудит
    print("\n" + "=" * 70)
    print("ГЛУБОКИЙ АУДИТ ПОЛУЧЕННЫХ ЧАНКОВ (LINE-BY-LINE & STATISTICAL AUDIT)")
    print("=" * 70)

    # 1. Распределение по типам единиц
    print("\n--- 1. Распределение типов структурных единиц ---")
    for utype, cnt in unit_type_counts.most_common():
        pct = cnt / total_chunks * 100 if total_chunks else 0
        print(f"  * {utype:15s}: {cnt:6d} ({pct:5.1f}%)")

    # 2. Распределение по языкам и скриптам
    print("\n--- 2. Распределение языков и исходных скриптов документов ---")
    for ls, cnt in lang_script_counts.most_common():
        print(f"  * {ls:15s}: {cnt:4d} документов")

    # 3. Распределение по ОКОЗ
    print("\n--- 3. Распределение обработанных документов по ОКОЗ ---")
    for okoz, cnt in sorted(okoz_counts.items()):
        print(f"  * ОКОЗ {okoz:8s}: {cnt:4d} документов")

    # 4. Анализ длин контента
    lengths = [len(ch["content"]) for ch in all_chunks]
    lengths.sort()
    min_len = lengths[0] if lengths else 0
    max_len = lengths[-1] if lengths else 0
    median_len = lengths[len(lengths) // 2] if lengths else 0
    p90_len = lengths[int(len(lengths) * 0.9)] if lengths else 0
    p99_len = lengths[int(len(lengths) * 0.99)] if lengths else 0

    print("\n--- 4. Метрики длин контента чанков ---")
    print(f"  * Мин длина:      {min_len} символов")
    print(f"  * Медиана длины:  {median_len} символов")
    print(f"  * 90-й перцентиль:{p90_len} символов")
    print(f"  * 99-й перцентиль:{p99_len} символов")
    print(f"  * Макс длина:     {max_len} символов")

    short_chunks = [ch for ch in all_chunks if len(ch["content"]) < 30]
    giant_chunks = [ch for ch in all_chunks if len(ch["content"]) > 10000]
    print(f"  * Микро-чанки (< 30 симв): {len(short_chunks)}")
    print(f"  * Сверхдлинные (> 10 000 симв): {len(giant_chunks)}")

    # 5. Аудит транслитерации и остаточной кириллицы
    print("\n--- 5. Аудит узбекской орфографии и транслитерации ---")
    residual_cyrillic_chunks = []
    apostrophe_deviations = []
    ks_rule_checks = []

    for ch in all_chunks:
        if ch["lang"] == "uz":
            text_to_check = ch["content"] + " " + " ".join(ch["breadcrumb"])
            found_cyr = RE_HAS_CYRILLIC.findall(text_to_check)
            if found_cyr:
                residual_cyrillic_chunks.append({
                    "chunk_id": ch["chunk_id"],
                    "symbols": list(set(found_cyr)),
                    "snippet": text_to_check[:120],
                })
            # Проверка апострофов (наличие неканонических апострофов)
            non_ascii_apos = re.findall(r"[\u2018\u2019\u02BB\u02BC\u0060\u00B4\u2032]", text_to_check)
            if non_ascii_apos:
                apostrophe_deviations.append({
                    "chunk_id": ch["chunk_id"],
                    "apos": list(set(non_ascii_apos)),
                })
            # Проверка правила ks (наличие aktsiy... вместо aksiy...)
            if re.search(r"\baktsiy", text_to_check, re.IGNORECASE):
                ks_rule_checks.append(ch["chunk_id"])

    print(f"  * Чанков с остаточной кириллицей в UZ: {len(residual_cyrillic_chunks)}")
    if residual_cyrillic_chunks:
        print(f"    Пример остаточной кириллицы: {residual_cyrillic_chunks[:3]}")
    print(f"  * Неканонических апострофов в UZ:      {len(apostrophe_deviations)}")
    print(f"  * Ошибок правила 'кц' -> 'ks' (aktsiy...): {len(ks_rule_checks)}")

    # 6. Аудит цитат и якорей
    print("\n--- 6. Аудит якорей и графа цитирования ---")
    total_anchors = 0
    negative_anchors = 0
    positive_anchors = 0
    total_cites = 0
    cites_with_anchor = 0
    cites_with_query_params_cleaned = 0

    for ch in all_chunks:
        meta = ch["metadata"]
        anchors = meta.get("anchors", [])
        total_anchors += len(anchors)
        for a in anchors:
            try:
                val = int(a)
                if val < 0:
                    negative_anchors += 1
                else:
                    positive_anchors += 1
            except ValueError:
                pass

        cites = meta.get("citations", [])
        total_cites += len(cites)
        for c in cites:
            if c.get("target_anchor"):
                cites_with_anchor += 1
            # Проверяем, не осталось ли '?' в target_doc_id
            if "?" in c.get("target_doc_id", ""):
                cites_with_query_params_cleaned += 1

    print(f"  * Всего извлечено якорей:        {total_anchors}")
    print(f"    - Отрицательных (UZ сегмент):  {negative_anchors}")
    print(f"    - Положительных (RU сегмент):  {positive_anchors}")
    print(f"  * Всего ссылок цитирования:      {total_cites}")
    print(f"    - Содержат целевой якорь (#):  {cites_with_anchor}")
    print(f"    - С неочищенными query (?):    {cites_with_query_params_cleaned} (должно быть 0)")

    # 7. Билингвальное выравнивание для пар UZ <-> RU
    print("\n--- 7. Билингвальный аудит паритета пар UZ <-> RU ---")
    doc_langs = defaultdict(dict)
    for ch in all_chunks:
        did = ch["doc_id"]
        lang = ch["lang"]
        if lang not in doc_langs[did]:
            doc_langs[did][lang] = []
        doc_langs[did][lang].append(ch)

    paired_docs = [did for did, langs in doc_langs.items() if "uz" in langs and "ru" in langs]
    print(f"  * Найдено полностью распарсенных двуязычных пар: {len(paired_docs)}")

    parity_exact_matches = 0
    parity_diffs = []

    for did in paired_docs:
        uz_chs = doc_langs[did]["uz"]
        ru_chs = doc_langs[did]["ru"]
        uz_len = len(uz_chs)
        ru_len = len(ru_chs)
        if uz_len == ru_len:
            parity_exact_matches += 1
        else:
            parity_diffs.append((did, uz_len, ru_len, abs(uz_len - ru_len)))

    print(f"  * Пар с точным совпадением 1-в-1 количества чанков: {parity_exact_matches} ({parity_exact_matches/len(paired_docs)*100:.1f}%)" if paired_docs else "0")
    if parity_diffs:
        parity_diffs.sort(key=lambda x: x[3], reverse=True)
        print(f"  * Топ расхождений по количеству чанков: {parity_diffs[:5]}")

    # 8. Аудит поведения на Архетипе B
    print("\n--- 8. Поведение парсера на Архетипе B ---")
    arch_b_results = [info for (did, lang), info in results_by_doc.items() if info.get("is_arch_b")]
    arch_b_total = len(arch_b_results)
    arch_b_zero = sum(1 for r in arch_b_results if r["chunks_count"] == 0)
    arch_b_parsed = sum(1 for r in arch_b_results if r["chunks_count"] > 0)
    print(f"  * Всего документов Архетипа B протестировано: {arch_b_total}")
    print(f"  * Из них дали 0 чанков:                     {arch_b_zero} ({arch_b_zero/arch_b_total*100:.1f}%)" if arch_b_total else "0")
    print(f"  * Из них дали >0 чанков:                    {arch_b_parsed} ({arch_b_parsed/arch_b_total*100:.1f}%)" if arch_b_total else "0")

    # Сохраняем отчет в JSON
    report_data = {
        "total_files_analyzed": len(sample),
        "total_chunks_extracted": total_chunks,
        "parsed_files": parsed_files,
        "zero_chunk_files": zero_chunk_files,
        "error_files": error_files,
        "total_time_seconds": total_time,
        "throughput_docs_per_sec": len(sample) / total_time,
        "unit_type_distribution": dict(unit_type_counts),
        "lang_script_distribution": dict(lang_script_counts),
        "okoz_distribution": dict(okoz_counts),
        "content_length_stats": {
            "min": min_len,
            "median": median_len,
            "p90": p90_len,
            "p99": p99_len,
            "max": max_len,
            "short_count": len(short_chunks),
            "giant_count": len(giant_chunks),
        },
        "transliteration_audit": {
            "residual_cyrillic_uz_chunks": len(residual_cyrillic_chunks),
            "non_ascii_apostrophes": len(apostrophe_deviations),
            "ks_errors": len(ks_rule_checks),
        },
        "anchors_and_citations": {
            "total_anchors": total_anchors,
            "negative_anchors": negative_anchors,
            "positive_anchors": positive_anchors,
            "total_citations": total_cites,
            "citations_with_anchor": cites_with_anchor,
            "uncleaned_query_params": cites_with_query_params_cleaned,
        },
        "bilingual_parity": {
            "paired_docs_count": len(paired_docs),
            "exact_chunk_count_matches": parity_exact_matches,
            "diffs_sample": parity_diffs[:10],
        },
        "archetype_b_behavior": {
            "total": arch_b_total,
            "zero_chunks": arch_b_zero,
            "parsed": arch_b_parsed,
        },
        "errors_sample": errors[:10],
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\nИтоговый JSON отчет сохранен в: {OUTPUT_REPORT_PATH}")
    return report_data, spot_check_candidates, short_chunks, giant_chunks, parity_diffs


if __name__ == "__main__":
    run_investigation()

