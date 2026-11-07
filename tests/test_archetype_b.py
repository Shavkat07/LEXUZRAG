"""
tests/test_archetype_b.py

Тестирование парсера Архетипа B:
- Извлечение таблиц и приложений из актов без CLAUSE_DEFAULT (-3604663_ru)
- Связывание таблиц с точными заголовками приложений (annex titles)
- Пропуск пустых языковых заглушек (cross-language stubs)
- Корректная генерация структуры чанков (ChunkDict)
"""

import gzip
from pathlib import Path
from lexuz.parsers.archetype_b import ArchetypeBParser

RAW_DIR = Path("data/raw_html")


def test_archetype_b_presidential_decree():
    print("\n--- 1. Тест парсера Архетипа B на ПП-3630 (-3604663_ru) ---")
    p = list(RAW_DIR.rglob("*-3604663_ru.html.gz"))[0]
    with gzip.open(p, "rt", encoding="utf-8") as f:
        html = f.read()

    assert ArchetypeBParser.can_parse(html), "Документ -3604663_ru должен определяться как Архетип B!"
    parser = ArchetypeBParser(html, "-3604663", "ru")
    chunks = parser.parse_chunks()

    assert len(chunks) > 0, "Парсер не вернул чанки для -3604663_ru!"
    table_chunks = [c for c in chunks if c["metadata"]["has_table"]]
    assert len(table_chunks) >= 6, f"Ожидалось не менее 6 табличных чанков, получено: {len(table_chunks)}"

    # Проверяем, что заголовок приложения связался с таблицей
    annex_titles = [c["structural_unit"]["title"] for c in table_chunks]
    assert any("Обод кишлок" in t for t in annex_titles), "Заголовок программы 'Обод кишлок' не найден в названиях приложений!"
    print(f"✔ Документ -3604663_ru успешно распарсен: {len(chunks)} чанков (табличных: {len(table_chunks)}).")


def test_archetype_b_language_stub_filter():
    print("\n--- 2. Тест фильтрации языковых заглушек (8465244_ru, 8452399_ru) ---")
    for did in ["8465244", "8452399"]:
        files = list(RAW_DIR.rglob(f"*{did}_ru.html.gz"))
        if not files:
            continue
        with gzip.open(files[0], "rt", encoding="utf-8") as f:
            html = f.read()
        can_p = ArchetypeBParser.can_parse(html)
        assert not can_p, f"Заглушка {did}_ru не должна проходить валидацию can_parse()!"
    print("✔ Языковые заглушки успешно отфильтрованы и не создают ложных чанков.")


if __name__ == "__main__":
    test_archetype_b_presidential_decree()
    test_archetype_b_language_stub_filter()
    print("\nВсе тесты Архетипа B пройдены успешно!")

