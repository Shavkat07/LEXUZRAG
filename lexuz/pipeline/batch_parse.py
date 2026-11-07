"""
lexuz.pipeline.batch_parse

Высокопроизводительный пакетный экстрактор чанков для корпусов Lex.uz.
Поддерживает Архетип A (законы, кодексы, постановления),
Архетип B (табличные акты, сметы, неструктурированные решения),
а также объединенный режим парсинга ('all').
"""

import gzip
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from lexuz.parsers.archetype_a import ArchetypeAParser
from lexuz.parsers.archetype_b import ArchetypeBParser

DEFAULT_RAW_DIR = Path("data/raw_html")
DEFAULT_OUTPUT_CHUNKS = Path("data/chunks_archetype_a.jsonl.gz")
DEFAULT_ERRORS_LOG = Path("data/archetype_a_errors.jsonl")

# Глобальная переменная для воркеров multiprocessing
_WORKER_MODE = "a"


def _worker_init(mode: str):
    global _WORKER_MODE
    _WORKER_MODE = mode


def process_single_file(file_path_str: str) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Воркер обработки одного файла с учетом выбранного режима архетипа (_WORKER_MODE).
    Возвращает: (status, chunks_list, error_info)
    """
    file_path = Path(file_path_str)

    # Парсим ID и язык из имени файла: {doc_id}_{lang}.html.gz
    name_parts = file_path.name.replace(".html.gz", "").rsplit("_", 1)
    if len(name_parts) != 2:
        return "ERROR", None, f"Некорректное имя файла: {file_path.name}"
    doc_id, lang = name_parts[0], name_parts[1]

    try:
        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        # Быстрый отсев поврежденных файлов и файлов без контейнера
        if len(html_content) < 650 or "divCont" not in html_content:
            return "SKIPPED", None, None

        if _WORKER_MODE == "a":
            if "CLAUSE_DEFAULT" not in html_content and "ACT_TEXT" not in html_content:
                return "SKIPPED", None, None
            parser_a = ArchetypeAParser(html_content, doc_id=doc_id, lang=lang)
            chunks = parser_a.parse_chunks()
            if not chunks:
                return "SKIPPED", None, None
            return "PARSED", chunks, None

        elif _WORKER_MODE == "b":
            if not ArchetypeBParser.can_parse(html_content):
                return "SKIPPED", None, None
            parser_b = ArchetypeBParser(html_content, doc_id=doc_id, lang=lang)
            chunks = parser_b.parse_chunks()
            if not chunks:
                return "SKIPPED", None, None
            return "PARSED", chunks, None

        elif _WORKER_MODE == "all":
            # Сначала проверяем Архетип A
            if "CLAUSE_DEFAULT" in html_content or "ACT_TEXT" in html_content:
                parser_a = ArchetypeAParser(html_content, doc_id=doc_id, lang=lang)
                chunks_a = parser_a.parse_chunks()
                if chunks_a:
                    return "PARSED_A", chunks_a, None

            # Если Архетип A не вернул чанки - проверяем Архетип B
            if ArchetypeBParser.can_parse(html_content):
                parser_b = ArchetypeBParser(html_content, doc_id=doc_id, lang=lang)
                chunks_b = parser_b.parse_chunks()
                if chunks_b:
                    return "PARSED_B", chunks_b, None

            return "SKIPPED", None, None

        return "SKIPPED", None, None

    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return "ERROR", None, f"DocID: {doc_id}_{lang} | {err_msg}"


def run_batch_parsing(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_CHUNKS,
    errors_path: Path = DEFAULT_ERRORS_LOG,
    archetype: Literal["a", "b", "all"] = "a",
    workers: Optional[int] = None,
    limit: Optional[int] = None,
):
    print("=" * 60)
    print(f"Старт пакетного парсинга Lex.uz (Режим: Архетип {archetype.upper()})")
    print("=" * 60)

    print(f"Сканирование директории {raw_dir}...")
    t0 = time.time()
    all_files = [str(p) for p in raw_dir.rglob("*.html.gz")]
    if limit:
        all_files = all_files[:limit]
    total_files = len(all_files)
    print(f"Найдено файлов для обработки: {total_files} за {time.time() - t0:.2f} сек.")

    if total_files == 0:
        print(f"❌ Файлы не найдены в {raw_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_workers = workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"Выделено воркеров: {num_workers} процессов")
    print(f"Выходной файл:    {output_path}")
    print(f"Лог ошибок:       {errors_path}")

    count_parsed_files = 0
    count_skipped_files = 0
    count_errors = 0
    total_chunks = 0
    chunksize = 50

    with gzip.open(output_path, "wt", encoding="utf-8") as out_f, \
            open(errors_path, "w", encoding="utf-8") as err_f:

        with mp.Pool(processes=num_workers, initializer=_worker_init, initargs=(archetype,)) as pool:
            results = pool.imap_unordered(process_single_file, all_files, chunksize=chunksize)

            last_report_time = time.time()
            for idx, (status, chunks, err_info) in enumerate(results, 1):
                if status.startswith("PARSED") and chunks:
                    count_parsed_files += 1
                    total_chunks += len(chunks)
                    for chk in chunks:
                        out_f.write(json.dumps(chk, ensure_ascii=False) + "\n")
                elif status == "ERROR":
                    count_errors += 1
                    err_f.write(json.dumps({"error": err_info}) + "\n")
                else:
                    count_skipped_files += 1

                now = time.time()
                if now - last_report_time >= 3.0 or idx == total_files:
                    elapsed = now - t0
                    speed = idx / elapsed if elapsed > 0 else 0
                    eta = (total_files - idx) / speed if speed > 0 else 0
                    pct = (idx / total_files) * 100
                    print(
                        f"[{idx:,}/{total_files:,}] ({pct:.1f}%) | "
                        f"Успешно: {count_parsed_files:,} | "
                        f"Чанков: {total_chunks:,} | "
                        f"Пропуск: {count_skipped_files:,} | "
                        f"Ошибок: {count_errors:,} | "
                        f"Скорость: {speed:.0f} ф/сек | "
                        f"ETA: {eta:.0f} сек."
                    )
                    last_report_time = now

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("Пакетный парсинг завершен!")
    print(f"Всего времени затрачено: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
    print(f"Обработано файлов:       {total_files:,}")
    print(f"Успешно распарсено:      {count_parsed_files:,} файлов")
    print(f"Сгенерировано чанков:    {total_chunks:,}")
    print(f"Пропущено файлов:        {count_skipped_files:,}")
    print(f"Ошибок парсинга:         {count_errors:,}")
    print(f"Результат сохранен в:    {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print("=" * 60)
