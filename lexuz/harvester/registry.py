"""
lexuz.harvester.registry

Формирование генерального реестра документов (master_registry.json)
на основе выкачанных таблиц .doc классификатора ОКОЗ.
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

DEFAULT_EXPORTS_DIR = Path("data/okoz_exports")
DEFAULT_TREE_FILE = Path("data/okoz_leaf_categories.json")
DEFAULT_OUTPUT_FILE = Path("data/master_registry.json")


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\xa0", " ").split()).strip()


def parse_col_meta(col_text: str):
    """Извлекает номер акта и дату принятия из колонки 1."""
    col_text = clean_text(col_text)
    parts = [p.strip() for p in col_text.split(",") if p.strip()]

    doc_number = parts[0] if parts else ""
    adoption_date = ""

    if len(parts) > 1:
        date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", parts[1])
        if date_match:
            adoption_date = date_match.group(1)

    return doc_number, adoption_date


def build_master_registry(
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    tree_file: Path = DEFAULT_TREE_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
):
    okoz_lookup = {}
    if tree_file.exists():
        with open(tree_file, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
            for node in tree_data.get("leaf_nodes", []):
                okoz_lookup[node["okoz_id"]] = {
                    "title": node.get("title", ""),
                    "breadcrumb": node.get("breadcrumb", "")
                }
    print(f"[1/3] Загружен справочник ОКОЗ: {len(okoz_lookup)} веток.")

    registry = {}
    doc_files = sorted(list(exports_dir.glob("*.doc")))
    print(f"[2/3] Обработка {len(doc_files)} файлов реестров...")

    total_rows_processed = 0

    for i, file_path in enumerate(doc_files, 1):
        okoz_id = file_path.stem.replace("okoz_", "")
        category_meta = okoz_lookup.get(okoz_id, {})
        breadcrumb = category_meta.get("breadcrumb", "")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tr in soup.find_all("tr"):
            link = tr.find("a", href=lambda h: h and "/docs/" in h)
            if not link:
                continue

            tds = tr.find_all("td")
            if len(tds) < 5:
                continue

            total_rows_processed += 1

            href = link.get("href", "").strip()
            id_match = re.search(r"/docs/(-?\d+)", href)
            if not id_match:
                continue
            doc_id = id_match.group(1)

            doc_number, adoption_date = parse_col_meta(tds[1].get_text())
            title = clean_text(tds[2].get_text())

            raw_effective_date = clean_text(tds[3].get_text())
            effective_date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", raw_effective_date)
            effective_date = effective_date_match.group(1) if effective_date_match else raw_effective_date.replace("-y.", "").strip()

            standard_url = f"https://lex.uz/docs/{doc_id}"

            if doc_id not in registry:
                registry[doc_id] = {
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "title": title,
                    "adoption_date": adoption_date,
                    "effective_date": effective_date,
                    "url": standard_url,
                    "okoz_ids": [okoz_id],
                    "breadcrumbs": [breadcrumb] if breadcrumb else []
                }
            else:
                if okoz_id not in registry[doc_id]["okoz_ids"]:
                    registry[doc_id]["okoz_ids"].append(okoz_id)
                if breadcrumb and breadcrumb not in registry[doc_id]["breadcrumbs"]:
                    registry[doc_id]["breadcrumbs"].append(breadcrumb)

        if i % 200 == 0 or i == len(doc_files):
            print(f"    Обработано файлов: {i}/{len(doc_files)} | Уникальных актов: {len(registry)}")

    print(f"\n[3/3] Сохранение в {output_file}...")
    output_payload = {
        "total_unique_documents": len(registry),
        "total_source_entries": total_rows_processed,
        "documents": list(registry.values())
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"Генеральный реестр создан: {output_file}")
    print(f"Уникальных актов: {len(registry)}")
    print("=" * 50)


if __name__ == "__main__":
    build_master_registry()

