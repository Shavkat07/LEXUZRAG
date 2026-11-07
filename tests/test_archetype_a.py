"""
tests/test_archetype_a.py
Верификация работы ArchetypeAParser на реальных законах, кодексах и постановлениях.
"""

import gzip
from pathlib import Path
from lexuz.parsers.archetype_a import ArchetypeAParser

RAW_DIR = Path("data/raw_html")


def load_raw_doc(doc_id: str, lang: str) -> str:
    """Загружает сжатый HTML документ из файловой структуры хранилища."""
    id_val = int(doc_id)
    folder = f"neg_{abs(id_val) // 100000}" if id_val < 0 else str(id_val // 100000)
    file_path = RAW_DIR / folder / f"{doc_id}_{lang}.html.gz"

    if not file_path.exists():
        found = list(RAW_DIR.rglob(f"{doc_id}_{lang}.html.gz"))
        if not found:
            raise FileNotFoundError(f"Файл {doc_id}_{lang}.html.gz не найден в {RAW_DIR}")
        file_path = found[0]

    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.read()


def test_laws_and_codes():
    print("\n--- 1. Тест структурированных законов (Admin Law -3492199) ---")
    for lang in ("uz", "ru"):
        html = load_raw_doc("-3492199", lang)
        parser = ArchetypeAParser(html, doc_id="-3492199", lang=lang)
        chunks = parser.parse_chunks()
        articles = [c for c in chunks if c["structural_unit"]["type"] == "article"]
        assert len(articles) == 88, f"Ожидалось 88 статей, получено: {len(articles)}"
        print(f"[{lang.upper()}] Успешно проверено: {len(articles)} статей, {len(chunks)} всего чанков.")

    print("\n--- 2. Тест кодексов (Трудовой кодекс -6257288) ---")
    for lang in ("uz", "ru"):
        html = load_raw_doc("-6257288", lang)
        parser = ArchetypeAParser(html, doc_id="-6257288", lang=lang)
        chunks = parser.parse_chunks()
        articles = [c for c in chunks if c["structural_unit"]["type"] == "article"]
        assert len(articles) == 593, f"Ожидалось 593 статьи ТК РУз, получено: {len(articles)}"
        assert len(chunks) == 608, f"Ожидалось 608 всего чанков ТК РУз, получено: {len(chunks)}"
        print(f"[{lang.upper()}] Успешно проверено: {len(articles)} статей, {len(chunks)} всего чанков ТК РУз.")


def test_government_resolutions():
    print("\n--- 3. Тест правительственных постановлений (8405084) ---")
    html = load_raw_doc("8405084", "uz")
    parser = ArchetypeAParser(html, doc_id="8405084", lang="uz")
    chunks = parser.parse_chunks()
    clauses = [c for c in chunks if c["structural_unit"]["type"] == "clause"]
    print(f"Постановление 8405084: всего чанков {len(chunks)}, из них пунктов: {len(clauses)}")
    assert len(clauses) > 10, f"Пункты постановления не распознаны! Найдено: {len(clauses)}"
    print("✔ Постановления успешно парсятся на нормативные пункты (clause).")


def test_false_preamble_suppression():
    print("\n--- 4. Тест фильтрации ложных преамбул (Бюджетный кодекс -2304138) ---")
    html = load_raw_doc("-2304138", "ru")
    parser = ArchetypeAParser(html, doc_id="-2304138", lang="ru")
    chunks = parser.parse_chunks()
    short_preambles = [
        c for c in chunks
        if c["structural_unit"]["type"] == "preamble" and len(c["content"]) < 60
    ]
    assert len(short_preambles) == 0, f"Найдено {len(short_preambles)} ложных микро-преамбул!"
    print("✔ Ложные микро-преамбулы ('Общая часть') успешно отфильтрованы.")


if __name__ == "__main__":
    test_laws_and_codes()
    test_government_resolutions()
    test_false_preamble_suppression()
    print("\nВсе тесты ArchetypeAParser успешно пройдены!")
