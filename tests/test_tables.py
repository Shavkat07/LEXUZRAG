"""
tests/test_tables.py
Тестирование парсинга таблиц, композитных заголовков и нарезки (Table Slicing).
"""

import gzip
from pathlib import Path
from selectolax.lexbor import LexborHTMLParser
from lexuz.parsers.archetype_a import ArchetypeAParser
from lexuz.parsers.table_parser import parse_table_grid, table_to_markdown

RAW_DIR = Path("data/raw_html")


def test_composite_headers_table():
    print("\n--- 1. Тест композитных шапок таблиц (-3689258) ---")
    p = list(RAW_DIR.rglob("-3689258_uz.html.gz"))[0]
    with gzip.open(p, "rt", encoding="utf-8") as f:
        chunks = ArchetypeAParser(f.read(), "-3689258", "uz").parse_chunks()

    tbl_chunks = [c for c in chunks if c["metadata"]["has_table"]]
    assert len(tbl_chunks) > 0, "Табличный чанк не найден!"
    content = tbl_chunks[0]["content"]

    assert "Chegara yoshi (1-daraja)" in content, "Композитный заголовок не сформирован!"
    assert "Chegara yoshi (2-daraja)" in content, "Второй уровень шапки не объединен!"
    print("✔ Композитные заголовки таблиц успешно проверены.")


def test_giant_table_slicing():
    print("\n--- 2. Тест нарезки таблиц-монстров (-1670542) ---")
    p = list(RAW_DIR.rglob("-1670542_uz.html.gz"))[0]
    with gzip.open(p, "rt", encoding="utf-8") as f:
        chunks = ArchetypeAParser(f.read(), "-1670542", "uz").parse_chunks()

    assert len(chunks) > 100, f"Ожидалось >100 чанков-слайсов, получено: {len(chunks)}"
    lengths = [len(c["content"]) for c in chunks]
    max_len = max(lengths)

    # Убеждаемся, что чанк больше не имеет аномальных 31 МБ
    assert max_len < 150_000, f"Чанк все еще слишком огромен: {max_len} символов!"
    print(f"✔ Таблица-монстр на 8 400 строк успешно нарезана на {len(chunks)} аккуратных слайсов.")
    print(f"  Максимальная длина слайса: {max_len:,} симв., медиана: {sorted(lengths)[len(lengths)//2]:,} симв.")


if __name__ == "__main__":
    test_composite_headers_table()
    test_giant_table_slicing()
    print("\nВсе тесты таблиц пройдены успешно!")

