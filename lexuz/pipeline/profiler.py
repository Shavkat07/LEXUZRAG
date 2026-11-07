"""
lexuz.pipeline.profiler

Lex.uz DOM Profiler & Corpus Classifier.
Параллельно сканирует все скачанные HTML-файлы, классифицирует документы по типам верстки
(Архетипы A, B, C) и собирает глобальные метрики цитирования и якорной структуры.
"""

import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from selectolax.lexbor import LexborHTMLParser
from tqdm import tqdm

DEFAULT_RAW_HTML_DIR = Path("data/raw_html")
DEFAULT_REPORT_PATH = Path("data/dom_inventory_report.json")
STUB_TEXT_THRESHOLD = 650


def analyze_file(file_path_str: str) -> Optional[Dict]:
    """Анализирует один сжатый HTML-документ и определяет его архетип."""
    file_path = Path(file_path_str)

    raw_name = file_path.name
    if raw_name.endswith(".html.gz"):
        base_name = raw_name[:-8]
    else:
        return None

    try:
        doc_id, lang_code = base_name.rsplit("_", 1)
    except ValueError:
        return None

    try:
        with gzip.open(file_path, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {
            "doc_id": doc_id,
            "lang_code": lang_code,
            "archetype": "CORRUPTED",
            "error": str(e),
        }

    tree = LexborHTMLParser(content)
    content_node = tree.css_first("#divCont") or tree.css_first("#divBody") or tree.css_first("#mD")

    if not content_node:
        return {
            "doc_id": doc_id,
            "lang_code": lang_code,
            "archetype": "NO_CONTAINER",
            "text_len": 0,
            "act_text": 0,
            "tables": 0,
            "anchors": 0,
            "citations": 0,
        }

    act_text_count = len(content_node.css(".ACT_TEXT"))
    clause_default_count = len(content_node.css(".CLAUSE_DEFAULT"))
    header_default_count = len(content_node.css(".TEXT_HEADER_DEFAULT"))
    table_count = len(content_node.css("table"))
    p_count = len(content_node.css("p"))
    anchor_nodes_count = len(content_node.css("[onmousemove*='lx_mo']"))
    citation_links_count = len(content_node.css("a[href*='/docs/']"))

    text_content = content_node.text(deep=True, strip=True)
    clean_text_len = len(text_content)

    # Классификация по архетипам:
    if clean_text_len < STUB_TEXT_THRESHOLD and act_text_count == 0 and table_count == 0:
        archetype = "ARCHETYPE_C"
    elif act_text_count > 0 or clause_default_count > 0 or header_default_count > 0:
        archetype = "ARCHETYPE_A"
    elif table_count > 0 or p_count >= 5 or clean_text_len >= STUB_TEXT_THRESHOLD:
        archetype = "ARCHETYPE_B"
    else:
        archetype = "ARCHETYPE_C"

    return {
        "doc_id": doc_id,
        "lang_code": lang_code,
        "archetype": archetype,
        "text_len": clean_text_len,
        "act_text": act_text_count,
        "tables": table_count,
        "paragraphs": p_count,
        "anchors": anchor_nodes_count,
        "citations": citation_links_count,
    }


def run_profiler(raw_dir: Path = DEFAULT_RAW_HTML_DIR, report_path: Path = DEFAULT_REPORT_PATH):
    if not raw_dir.exists():
        print(f"Директория {raw_dir} не найдена!")
        return

    print("Поиск скачанных файлов...")
    all_files = [str(p) for p in raw_dir.glob("*/*.html.gz")]
    total_files = len(all_files)
    print(f"Найдено файлов для анализа: {total_files}")

    if total_files == 0:
        print("Файлы не найдены.")
        return

    cpu_count = os.cpu_count() or 4
    workers = min(cpu_count, 16)
    print(f"Запуск аудита на {workers} процессах (Lexbor Engine)...")

    results = []
    chunk_size = 250

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in tqdm(
            executor.map(analyze_file, all_files, chunksize=chunk_size),
            total=total_files,
            desc="Профилирование DOM",
            unit="doc",
        ):
            if res:
                results.append(res)

    stats = {
        "total_analyzed": len(results),
        "by_language": Counter(),
        "by_archetype": {
            "uz": Counter(),
            "ru": Counter(),
        },
        "totals": {
            "total_citations": 0,
            "total_anchors": 0,
            "total_tables": 0,
        },
        "stubs": {
            "uz_stubs_doc_ids": [],
            "ru_stubs_doc_ids": [],
        },
    }

    for r in results:
        lang = r["lang_code"]
        arch = r["archetype"]
        doc_id = r["doc_id"]

        stats["by_language"][lang] += 1
        if lang in stats["by_archetype"]:
            stats["by_archetype"][lang][arch] += 1

        stats["totals"]["total_citations"] += r.get("citations", 0)
        stats["totals"]["total_anchors"] += r.get("anchors", 0)
        stats["totals"]["total_tables"] += r.get("tables", 0)

        if arch == "ARCHETYPE_C":
            if lang == "uz":
                stats["stubs"]["uz_stubs_doc_ids"].append(doc_id)
            else:
                stats["stubs"]["ru_stubs_doc_ids"].append(doc_id)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Отчет сохранен в {report_path}")


if __name__ == "__main__":
    run_profiler()

