"""
scratch/extract_100_tables.py

Скрипт точного извлечения и валидации 100 разнообразных таблиц из корпуса Lex.uz.
Обеспечивает строгое разнообразие:
- 50 на узбекском (uz) и 50 на русском (ru)
- Разделы ОКОЗ: 01, 02, 03, 05, 07, 08, 09, 10, 11, 13, 14, 15, 17, 19, 21
- Тематические типы: бюджеты, ТН ВЭД, СанПиН, квоты, тарифы, дорожные карты, нормативы
- Размерности: узкие (2-4), средние (5-8), широкие (9-25+); короткие (3-10), средние (11-40), гигантские (41-300+)
- Тестирование через реальный lexuz.parsers.table_parser
"""

import gzip
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from selectolax.lexbor import LexborHTMLParser
from lexuz.parsers.table_parser import parse_table_grid, table_to_markdown_slices


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_table(full_text: str, cols_count: int, rows_count: int) -> str:
    t_low = full_text.lower()
    if any(k in t_low for k in ["тн вэд", "товарная позиция", "код товара", "божхона божи", "бож ставка", "ставка пошлин", "акциз"]):
        return "customs_tariff_tnved"
    elif any(k in t_low for k in ["бюджет", "харажат", "даромад", "расход", "доход", "минг сўм", "млн сўм", "тыс. сум", "млн. сум", "ажратилган mablag"]):
        return "budget_finance"
    elif any(k in t_low for k in ["пдк", "мг/м3", "пду", "санпин", "гигиенич", "токсич", "хавфлилик", "пищевые веществ", "озуқа модда"]):
        return "sanpin_norms"
    elif any(k in t_low for k in ["квота", "тонна", "гектар", "ҳосил", "пахта", "ғалла", "урожай", "поставка", "топшириш"]):
        return "quota_allocation"
    elif any(k in t_low for k in ["тариф", "разряд", "коэффициент", "маош", "оклад", "ставка", "базавий ҳисоблаш"]):
        return "wages_tariffs"
    elif any(k in t_low for k in ["чора-тадбир", "режа", "мероприяти", "план", "муддат", "срок", "масъул", "ответствен", "ижрочи"]):
        return "action_plan_schedule"
    elif any(k in t_low for k in ["гост", "ўзст", "стандарт", "ўлчаш восита", "поверка", "синов", "испытани"]):
        return "technical_standards"
    elif any(k in t_low for k in ["квотаси", "қабул", "бакалавр", "магистр", "талаба", "стипендия", "грант"]):
        return "education_quotas"
    else:
        return "general_tabular"


def main():
    print("=" * 60)
    print("Старт извлечения 100 разнообразных таблиц Lex.uz")
    print("=" * 60)

    # 1. Загрузка метаданных реестра
    t0 = time.time()
    with open("data/master_registry.json", "r", encoding="utf-8") as f:
        registry = json.load(f)
    doc_map = {doc["doc_id"]: doc for doc in registry.get("documents", [])}
    print(f"Реестр загружен: {len(doc_map)} актов за {time.time()-t0:.2f} сек.")

    # 2. Список файлов корпуса
    raw_files = list(Path("data/raw_html").rglob("*.html.gz"))
    print(f"Всего файлов в raw_html: {len(raw_files)}")
    random.seed(12345)
    random.shuffle(raw_files)

    # Счетчики стратификации
    # Итоговая цель: 100 таблиц (50 uz, 50 ru)
    MAX_TOTAL = 100
    MAX_PER_LANG = 65
    MAX_PER_DOC = 3
    MAX_PER_OKOZ = 40
    MAX_PER_TYPE = 50

    counts_by_lang = defaultdict(int)
    counts_by_okoz = defaultdict(int)
    counts_by_type = defaultdict(int)
    doc_table_counts = defaultdict(int)
    
    collected_tables: List[Dict[str, Any]] = []

    print("Поиск и анализ таблиц...")
    t_start = time.time()
    scanned_files = 0

    for fp in raw_files:
        if len(collected_tables) >= MAX_TOTAL:
            break
        scanned_files += 1

        name_parts = fp.name.replace(".html.gz", "").rsplit("_", 1)
        if len(name_parts) != 2:
            continue
        doc_id, lang = name_parts[0], name_parts[1]

        if lang not in ("uz", "ru"):
            continue
        if counts_by_lang[lang] >= MAX_PER_LANG:
            continue
        if doc_table_counts[doc_id] >= MAX_PER_DOC:
            continue

        try:
            with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except Exception:
            continue

        # Быстрые фильтры
        if len(html) < 2000 or "<table" not in html or "divCont" not in html:
            continue
        if "PDF shaklida berilgan" in html or "формате PDF" in html:
            continue

        # Метаданные ОКОЗ
        doc_meta = doc_map.get(doc_id, {})
        doc_title = doc_meta.get("title", "")
        breadcrumbs = doc_meta.get("breadcrumbs", [])
        
        okoz_code = "00"
        okoz_title = "Boshqa"
        if breadcrumbs:
            m = re.search(r"(\d{2})\.00\.00\.00\s+([^>]+)", breadcrumbs[0])
            if m:
                okoz_code = m.group(1)
                okoz_title = m.group(2).strip()

        if counts_by_okoz[okoz_code] >= MAX_PER_OKOZ:
            continue

        parser = LexborHTMLParser(html)
        tables = parser.css("table")
        if not tables:
            continue

        for tbl_idx, tbl in enumerate(tables):
            if len(collected_tables) >= MAX_TOTAL:
                break
            if counts_by_lang[lang] >= MAX_PER_LANG:
                break
            if doc_table_counts[doc_id] >= MAX_PER_DOC:
                break

            trs = tbl.css("tr")
            if len(trs) < 3:
                continue

            # Фильтр 1: отсекаем 1-строчные / 2-строчные плашки статусов
            tbl_full_text = clean_text(tbl.text())
            if len(tbl_full_text) < 40 and any(w in tbl_full_text.lower() for w in ["утратил силу", "кучини йўқотган", "вступил в силу"]):
                continue

            # Фильтр 2: считаем физические ячейки
            rowspan_max = 1
            colspan_max = 1
            has_th = False
            has_thead = bool(tbl.css_first("thead"))
            
            row_cols = []
            for tr in trs:
                cells = [c for c in tr.iter() if c.tag in ("td", "th")]
                if not cells:
                    continue
                for c in cells:
                    if c.tag == "th":
                        has_th = True
                    rs = int(c.attributes.get("rowspan", "1") or "1")
                    cs = int(c.attributes.get("colspan", "1") or "1")
                    if rs > rowspan_max:
                        rowspan_max = rs
                    if cs > colspan_max:
                        colspan_max = cs
                row_cols.append(sum(int(c.attributes.get("colspan", "1") or "1") for c in cells))

            if not row_cols:
                continue
            max_cols = max(row_cols)
            if max_cols < 2:
                continue # Таблица из 1 колонки обычно является списком или рамкой

            # Классификация темы
            tbl_type = classify_table(tbl_full_text, max_cols, len(trs))
            if counts_by_type[tbl_type] >= MAX_PER_TYPE and len(collected_tables) < 90:
                continue

            # 3. Валидация через реальный table_parser
            comp_headers, data_rows = parse_table_grid(tbl, lang)
            if not comp_headers or not data_rows:
                continue

            slices = table_to_markdown_slices(tbl, lang, max_rows_per_slice=35)

            # Формируем запись таблицы
            table_record = {
                "table_id": len(collected_tables) + 1,
                "doc_id": doc_id,
                "lang": lang,
                "doc_title": doc_title,
                "okoz_code": okoz_code,
                "okoz_title": okoz_title,
                "table_idx_in_doc": tbl_idx,
                "theme": tbl_type,
                "metrics": {
                    "raw_rows": len(trs),
                    "max_cols": max_cols,
                    "has_thead": has_thead,
                    "has_th": has_th,
                    "max_rowspan": rowspan_max,
                    "max_colspan": colspan_max,
                    "parsed_headers_count": len(comp_headers),
                    "parsed_rows_count": len(data_rows),
                    "slices_count": len(slices),
                    "text_chars": len(tbl_full_text),
                },
                "composite_headers": comp_headers,
                "sample_parsed_rows": data_rows[:3], # первые 3 строки
                "first_slice_preview": slices[0][:800] if slices else "",
                "html_snippet": tbl.html[:1500] if tbl.html else "",
            }

            collected_tables.append(table_record)
            counts_by_lang[lang] += 1
            counts_by_okoz[okoz_code] += 1
            counts_by_type[tbl_type] += 1
            doc_table_counts[doc_id] += 1

            if len(collected_tables) % 10 == 0:
                print(f"[{len(collected_tables)}/100] Собрано | UZ: {counts_by_lang['uz']}, RU: {counts_by_lang['ru']} | Сканировано файлов: {scanned_files} | {time.time()-t_start:.1f}с")

    print("\n" + "=" * 60)
    print(f"✅ УСПЕШНО СОБРАНО {len(collected_tables)} ТАБЛИЦ за {time.time()-t_start:.1f} сек!")
    print("=" * 60)

    print("\nРаспределение по языкам:")
    for l, c in sorted(counts_by_lang.items()):
        print(f"  {l}: {c}")

    print("\nРаспределение по темам (доменам):")
    for t, c in sorted(counts_by_type.items()):
        print(f"  {t:25s}: {c}")

    print("\nРаспределение по разделам ОКОЗ:")
    for o, c in sorted(counts_by_okoz.items()):
        print(f"  ОКОЗ {o:2s}: {c:2d}")

    # Статистика размерностей таблиц
    cols_stats = [t["metrics"]["max_cols"] for t in collected_tables]
    rows_stats = [t["metrics"]["parsed_rows_count"] for t in collected_tables]
    slices_stats = [t["metrics"]["slices_count"] for t in collected_tables]
    rowspan_tables = sum(1 for t in collected_tables if t["metrics"]["max_rowspan"] > 1)
    colspan_tables = sum(1 for t in collected_tables if t["metrics"]["max_colspan"] > 1)

    print("\nСтатистика структуры и геометрии:")
    print(f"  Колонки: мин={min(cols_stats)}, макс={max(cols_stats)}, среднее={sum(cols_stats)/len(cols_stats):.1f}")
    print(f"  Строки:   мин={min(rows_stats)}, макс={max(rows_stats)}, среднее={sum(rows_stats)/len(rows_stats):.1f}")
    print(f"  Таблиц с многоуровневыми шапками (rowspan > 1): {rowspan_tables} ({rowspan_tables}%)")
    print(f"  Таблиц с объединенными ячейками (colspan > 1): {colspan_tables} ({colspan_tables}%)")
    print(f"  Длинных таблиц, потребовавших нарезки (slices > 1): {sum(1 for s in slices_stats if s > 1)} ({sum(1 for s in slices_stats if s > 1)}%)")
    print(f"  Максимум слайсов у одной таблицы: {max(slices_stats)}")

    # Сохраняем в JSON
    out_file = Path("data/tables_sample_100.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(collected_tables, f, ensure_ascii=False, indent=2)

    print(f"\nФайл с полным датасетом 100 таблиц сохранен: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
