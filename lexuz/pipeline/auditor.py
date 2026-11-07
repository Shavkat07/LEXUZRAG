"""
lexuz.pipeline.auditor

Глубокий аудит сформированного корпуса чанков (JSONL.GZ).
Проверяет целостность, распределение длин, остаточную кириллицу и цитатный граф.
"""

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

DEFAULT_CHUNKS_PATH = Path("data/chunks_archetype_a.jsonl.gz")
RE_CYRILLIC = re.compile(r"[а-яА-ЯёЁўЎғҒҳҲқҚ]")


def run_audit(chunks_path: Path = DEFAULT_CHUNKS_PATH) -> bool:
    print("=" * 60)
    print(f"Глубокий аудит корпуса: {chunks_path}")
    print("=" * 60)

    if not chunks_path.exists():
        print(f"❌ Файл {chunks_path} не найден!")
        return False

    total_chunks = 0
    by_lang = Counter()
    by_unit_type = Counter()
    by_script = Counter()

    total_anchors = 0
    total_citations = 0
    citations_with_anchor = 0
    table_chunks = 0

    lengths = []
    residual_cyrillic_uz_chunks = 0
    sample_residual = []

    too_short = []
    too_long = []

    with gzip.open(chunks_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue

            chunk = json.loads(line)
            total_chunks += 1

            lang = chunk["lang"]
            by_lang[lang] += 1

            unit_type = chunk["structural_unit"]["type"]
            by_unit_type[unit_type] += 1

            meta = chunk["metadata"]
            script = meta.get("original_script", "unknown")
            by_script[f"{lang}_{script}"] += 1

            # Якоря и цитаты
            anchors_cnt = len(meta.get("anchors", []))
            total_anchors += anchors_cnt

            cites = meta.get("citations", [])
            total_citations += len(cites)
            for c in cites:
                if c.get("target_anchor"):
                    citations_with_anchor += 1

            if meta.get("has_table"):
                table_chunks += 1

            # Анализ длины контента
            content = chunk.get("content", "")
            c_len = len(content)
            lengths.append(c_len)

            if c_len < 30:
                too_short.append((chunk["chunk_id"], content))
            elif c_len > 15000:
                too_long.append((chunk["chunk_id"], c_len))

            # Проверка остаточной кириллицы в узбекском корпусе
            if lang == "uz":
                full_text_to_check = content + " " + " ".join(chunk.get("breadcrumb", []))
                res = RE_CYRILLIC.findall(full_text_to_check)
                if res:
                    residual_cyrillic_uz_chunks += 1
                    if len(sample_residual) < 3:
                        sample_residual.append((chunk["chunk_id"], set(res), content[:100]))

    if total_chunks == 0:
        print("⚠ Корпус пуст.")
        return False

    lengths.sort()
    median_len = lengths[len(lengths) // 2] if lengths else 0
    p95_len = lengths[int(len(lengths) * 0.95)] if lengths else 0
    p99_len = lengths[int(len(lengths) * 0.99)] if lengths else 0

    print(f"\n1. ОБЩАЯ СТАТИСТИКА ЧАНКОВ:")
    print(f"   Всего валидных чанков:      {total_chunks:,}")
    print(f"   По языкам:                  UZ = {by_lang['uz']:,} | RU = {by_lang['ru']:,}")
    print(f"   Распределение скрипта:      {dict(by_script)}")

    print(f"\n2. СТРУКТУРНЫЕ ЕДИНИЦЫ (Unit Types):")
    for ut, cnt in by_unit_type.most_common():
        pct = cnt / total_chunks * 100
        print(f"   - {ut:16}: {cnt:,} ({pct:.1f}%)")

    print(f"\n3. СЕТЬ СВЯЗЕЙ И МЕТАДАННЫЕ (GraphRAG / Retrieval):")
    print(f"   Всего абзацных якорей (lx_mo):       {total_anchors:,}")
    print(f"   Всего ребер цитирования (/docs/...): {total_citations:,}")
    anchor_cite_pct = (citations_with_anchor / total_citations * 100) if total_citations else 0
    print(f"   Цитат с точным якорем перехода:      {citations_with_anchor:,} ({anchor_cite_pct:.1f}%)")
    print(f"   Чанков с Markdown-таблицами:         {table_chunks:,}")

    print(f"\n4. РАСПРЕДЕЛЕНИЕ ДЛИН КОНТЕНТА (символов):")
    print(f"   Минимальная длина:   {lengths[0] if lengths else 0} симв.")
    print(f"   Медианная длина:     {median_len} симв. (~{median_len // 4} токенов)")
    print(f"   95-й перцентиль:     {p95_len} симв.")
    print(f"   99-й перцентиль:     {p99_len} симв.")
    print(f"   Максимальная длина:  {lengths[-1] if lengths else 0} симв.")
    print(f"   Аномально коротких (<30 симв.): {len(too_short)}")
    print(f"   Аномально длинных (>15k симв.): {len(too_long)}")

    if too_short:
        print(f"\n   Примеры коротких чанков (<30 симв.):")
        for cid, txt in too_short[:3]:
            print(f"     - {cid}: {repr(txt)}")

    print(f"\n5. ЛИНГВИСТИЧЕСКИЙ АУДИТ УЗБЕКСКОГО КОРПУСА:")
    print(f"   Узбекских чанков с остаточной кириллицей: {residual_cyrillic_uz_chunks}")
    if residual_cyrillic_uz_chunks == 0:
        print("   ✔ 100% чистая латиница во всем узбекском корпусе!")
    else:
        print("   ⚠ Примеры чанков с кириллицей:")
        for cid, chars, snippet in sample_residual:
            print(f"     - {cid}: буквы {chars} | текст: {repr(snippet)}")

    print("=" * 60)
    return True


if __name__ == "__main__":
    run_audit()

