"""
tests/test_bilingual.py
Проверка билингвального паритета (UZ <-> RU) и нормализации кириллических актов.
"""

import gzip
import re
from pathlib import Path
from lexuz.parsers.archetype_a import ArchetypeAParser
from lexuz.common.translit import is_cyrillic

RAW_DIR = Path("data/raw_html")


def load_doc(doc_id: str, lang: str) -> str:
    found = list(RAW_DIR.rglob(f"{doc_id}_{lang}.html.gz"))
    if not found:
        raise FileNotFoundError(f"Файл {doc_id}_{lang}.html.gz не найден")
    with gzip.open(found[0], "rt", encoding="utf-8", errors="ignore") as f:
        return f.read()


def test_bilingual_admin_law():
    """Проверяет попарное соответствие для Закона об админ. процедурах -3492199."""
    _check_bilingual("-3492199")


def test_bilingual_labor_code():
    """Проверяет попарное соответствие для Трудового кодекса -6257288."""
    _check_bilingual("-6257288")


def _check_bilingual(doc_id: str):
    print(f"\n--- Билингвальное выравнивание (UZ <-> RU) для {doc_id} ---")
    uz_chunks = ArchetypeAParser(load_doc(doc_id, "uz"), doc_id, "uz").parse_chunks()
    ru_chunks = ArchetypeAParser(load_doc(doc_id, "ru"), doc_id, "ru").parse_chunks()

    print(f"Всего чанков: UZ={len(uz_chunks)}, RU={len(ru_chunks)}")
    min_len = min(len(uz_chunks), len(ru_chunks))
    mismatches = 0

    for i in range(min_len):
        u_u = uz_chunks[i]["structural_unit"]
        r_u = ru_chunks[i]["structural_unit"]

        if u_u["type"] != r_u["type"] or u_u["number"] != r_u["number"]:
            if mismatches < 5:
                print(f"  ❌ Рассинхрон на индексе {i}:")
                print(f"     UZ: type={u_u['type']}, num={u_u['number']} ({uz_chunks[i]['chunk_id']})")
                print(f"     RU: type={r_u['type']}, num={r_u['number']} ({ru_chunks[i]['chunk_id']})")
            mismatches += 1

    assert mismatches == 0, f"Обнаружено {mismatches} несовпадений структуры!"
    assert len(uz_chunks) == len(ru_chunks), f"Количество чанков не совпадает: {len(uz_chunks)} vs {len(ru_chunks)}"
    print(f"✔ Идеальный попарный паритет 1-в-1: все {min_len} чанков совпадают!")


def test_cyrillic_full_doc():
    """Находит в базе полноценный кириллический закон и проверяет качество нормализации."""
    print("\n--- Проверка полного кириллического закона ---")
    for gz_path in RAW_DIR.rglob("*_uz.html.gz"):
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        if "CLAUSE_DEFAULT" in html and is_cyrillic(html[:3000]):
            doc_id = gz_path.name.split("_")[0]
            chunks = ArchetypeAParser(html, doc_id, "uz").parse_chunks()
            if len(chunks) > 5:
                sample = chunks[1] if len(chunks) > 1 else chunks[0]
                residual = re.findall(r"[а-яА-ЯёЁўЎғҒҳҲқҚ]", " ".join(sample["breadcrumb"]))
                assert len(residual) == 0, f"Остаточная кириллица в breadcrumb: {residual}"
                print(f"✔ Кириллический закон ID {doc_id} ({len(chunks)} чанков) чисто нормализован в латиницу.")
                return

    raise AssertionError("Не удалось найти кириллический закон Архетипа A для теста.")


if __name__ == "__main__":
    test_bilingual_admin_law()
    test_bilingual_labor_code()
    test_cyrillic_full_doc()
    print("\nВсе проверки билингвального паритета пройдены успешно!")
