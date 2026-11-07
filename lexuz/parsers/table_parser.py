"""
lexuz.parsers.table_parser

Модуль сериализации HTML-таблиц Lex.uz в Markdown:
- Двумерная координатная сетка с корректной обработкой rowspan и colspan.
- Схлопывание многоуровневых шапок в составные заголовки (Composite Headers).
- Удаление служебных колонок-распорок.
- Извлечение прямых ячеек строк во избежание захвата вложенных таблиц.
- Нарезка сверхдлинных таблиц (Table Slicing) с дублированием Markdown-шапки
  для защиты контекстного окна от мегабайтных чанков.
"""

import re
from typing import List, Optional, Tuple
from selectolax.lexbor import LexborNode
from lexuz.common.translit import normalize_uz_text

DEFAULT_MAX_TABLE_ROWS = 35

HEADER_KEYWORDS = re.compile(
    r"(?i)\b("
    r"t/r|№|nomi|nomlanishi|mazmuni|mavzu|subyekt|bosqich|muddat|mas'ul|mablag'|summa|narx|kod|modda|harakat|tadbirlar|ijro|rejasi|haqida|hujjat|"
    r"naimenovaniye|naimenovanie|nazvaniye|nazvanie|pokazatel|soderzhaniye|soderzhanie|etap|srok|otvetstven|summa|kolichestvo|stoimost|stavka|razryad|tariff|tarifi|"
    r"edinitsa|ed\.\s*izm|izmereniya|primechaniye|primechanie|kod\s+tovara|tn\s*ved|tovar\s+pozitsiyasi|tarmoq|soha|hudud|tuman|viloyat|shahar|davr|yil|chorak|oy|"
    r"daraja|tarkib|kategoriya|guruh|sinf|gruppa|klass|stepen|chast|razdel|bob"
    r")\b"
)


def is_quote_row(cells: List[LexborNode]) -> bool:
    """Проверяет, состоит ли строка только из символов кавычек и пунктуации правок."""
    full = "".join(c.text() for c in cells)
    clean = re.sub(r"[\s“”«»\"\'\;,]+", "", full)
    return clean == ""


def is_header_cell(cell: LexborNode) -> bool:
    """Проверяет, содержит ли ячейка характерные признаки заголовка."""
    if cell.tag == "th":
        return True
    txt = cell.text().strip()
    if not txt:
        return False
    if re.match(r"^\d+(?:[\._]\d+)*[\.\)]?$", txt):
        return False
    if cell.css_first("b, strong") and len(txt) < 80:
        return True
    return bool(HEADER_KEYWORDS.search(txt))


def parse_table_grid(table_node: LexborNode, lang: str) -> Tuple[List[str], List[List[str]]]:
    """
    Разбирает HTML-таблицу в координатную сетку с композитными шапками и отсечением распорок.
    Возвращает (composite_headers, data_rows).
    """
    tr_nodes = table_node.css("tr")
    if not tr_nodes:
        return [], []

    # Вырезаем служебный UI Lex.uz из ячеек
    for junk in table_node.css(".lx_no_select, .lx_elem2, .lx_elem3, script, style"):
        junk.decompose()

    # Извлекаем строки и строго прямые дочерние ячейки (избегает раздувания от вложенных таблиц)
    row_cells_list = []
    for tr in tr_nodes:
        cells = [c for c in tr.iter() if c.tag in ("td", "th")]
        if cells:
            row_cells_list.append(cells)

    # 1. Отсекаем обрамляющие строки-кавычки правок (“ ... ”) в актах об изменениях
    while row_cells_list and is_quote_row(row_cells_list[0]):
        row_cells_list.pop(0)
    while row_cells_list and is_quote_row(row_cells_list[-1]):
        row_cells_list.pop(-1)

    if not row_cells_list:
        return [], []

    # 2. Определяем высоту шапки таблицы (header_row_count)
    has_th = any(any(c.tag == "th" for c in r) for r in row_cells_list)
    has_thead = bool(table_node.css_first("thead"))

    if has_thead:
        thead_trs = table_node.css("thead tr")
        header_row_count = min(len(thead_trs), len(row_cells_list))
    elif has_th:
        header_row_count = max(
            r + int(c.attributes.get("rowspan", 1) or 1)
            for r, row in enumerate(row_cells_list)
            for c in row
            if c.tag == "th"
        )
        header_row_count = min(header_row_count, len(row_cells_list) - 1 if len(row_cells_list) > 1 else 1)
    else:
        # Проверяем, похожа ли первая строка на заголовочную
        row0 = row_cells_list[0]
        first_txt = ""
        for c in row0:
            ct = c.text().strip()
            if ct:
                first_txt = ct
                break

        # Если первая непустая ячейка начинается с конкретного номера пункта (2.3., 14. и т.д.),
        # то это строка данных из фрагмента таблицы, а не шапка
        if first_txt and re.match(r"^\d+(?:[\._]\d+)*[\.\)]?$", first_txt):
            is_row0_header = False
        else:
            is_row0_header = any(is_header_cell(c) for c in row0)

        if not is_row0_header:
            # Вся таблица - строки данных (например, фрагмент поправки или таблица без шапки)
            header_row_count = 0
        else:
            max_rspan = 1
            for cell in row0:
                try:
                    rs = int(cell.attributes.get("rowspan", 1) or 1)
                    if rs > max_rspan:
                        max_rspan = rs
                except (ValueError, TypeError):
                    pass
            header_row_count = min(max_rspan, len(row_cells_list) - 1 if len(row_cells_list) > 1 else 1)

    # 3. Строим двумерную координатную сетку
    grid: List[List[Optional[str]]] = []

    for r_idx, cells in enumerate(row_cells_list):
        while len(grid) <= r_idx:
            grid.append([])

        c_idx = 0
        for cell in cells:
            while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                c_idx += 1

            raw_cell = " ".join(cell.text().split()).replace("|", "\\|")
            cell_text = normalize_uz_text(raw_cell) if lang == "uz" else raw_cell

            try:
                colspan = int(cell.attributes.get("colspan", 1))
            except (ValueError, TypeError):
                colspan = 1

            try:
                rowspan = int(cell.attributes.get("rowspan", 1))
            except (ValueError, TypeError):
                rowspan = 1

            while len(grid) < r_idx + rowspan:
                grid.append([])

            for r_offset in range(rowspan):
                curr_r = r_idx + r_offset
                while len(grid[curr_r]) < c_idx + colspan:
                    grid[curr_r].append(None)
                for c_offset in range(colspan):
                    if curr_r < header_row_count:
                        grid[curr_r][c_idx + c_offset] = cell_text
                    else:
                        grid[curr_r][c_idx + c_offset] = cell_text if (r_offset == 0 and c_offset == 0) else ""

            c_idx += colspan

    max_cols = max((len(r) for r in grid), default=0)
    if max_cols == 0:
        return [], []

    for r in grid:
        while len(r) < max_cols:
            r.append("")
        for idx in range(len(r)):
            if r[idx] is None:
                r[idx] = ""

    # 4. Отсекаем служебные колонки-распорки (без букв и цифр)
    valid_col_indices = []
    for col_j in range(max_cols):
        col_values = [grid[row_i][col_j] for row_i in range(len(grid))]
        if any(re.search(r"[\w\d]", val) for val in col_values):
            valid_col_indices.append(col_j)

    if not valid_col_indices:
        return [], []

    # 5. Формируем единые составные заголовки (Composite Headers)
    composite_headers = []
    if header_row_count > 0:
        for col_j in valid_col_indices:
            parts = []
            for r in range(header_row_count):
                val = grid[r][col_j].strip()
                if val and (not parts or val != parts[-1]):
                    parts.append(val)
            if len(parts) > 1:
                composite_headers.append(f"{parts[0]} ({' / '.join(parts[1:])})")
            elif parts:
                composite_headers.append(parts[0])
            else:
                composite_headers.append("")

    # Если шапка отсутствовала или все ячейки шапки пусты (например, оргсхемы):
    if not composite_headers or all(not h.strip() for h in composite_headers):
        prefix = "Ustun" if lang == "uz" else "Колонка"
        composite_headers = [f"{prefix} {i+1}" for i in range(len(valid_col_indices))]

    # 6. Собираем строки данных
    data_rows = []
    for r_idx in range(header_row_count, len(grid)):
        row_vals = [grid[r_idx][col_j].strip() for col_j in valid_col_indices]
        if any(row_vals):
            data_rows.append(row_vals)

    return composite_headers, data_rows


def format_table_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Форматирует заголовки и строки в Markdown таблицу."""
    if not headers or not rows:
        return ""
    col_count = len(headers)
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in rows:
        padded_row = row + [""] * max(0, col_count - len(row))
        md_lines.append("| " + " | ".join(padded_row[:col_count]) + " |")
    return "\n" + "\n".join(md_lines) + "\n"


def table_to_markdown_slices(
    table_node: LexborNode,
    lang: str,
    max_rows_per_slice: int = DEFAULT_MAX_TABLE_ROWS
) -> List[str]:
    """
    Конвертирует HTML-таблицу в один или несколько Markdown-сегментов (слайсов).
    Если строк данных больше max_rows_per_slice, нарезает на части с дублированием шапки.
    """
    headers, rows = parse_table_grid(table_node, lang)
    if not headers or not rows:
        return []

    if len(rows) <= max_rows_per_slice:
        return [format_table_markdown(headers, rows)]

    slices = []
    for i in range(0, len(rows), max_rows_per_slice):
        chunk_rows = rows[i:i + max_rows_per_slice]
        slices.append(format_table_markdown(headers, chunk_rows))
    return slices


def table_to_markdown(table_node: LexborNode, lang: str) -> str:
    """Конвертирует таблицу в единый Markdown блок (для небольших таблиц)."""
    headers, rows = parse_table_grid(table_node, lang)
    return format_table_markdown(headers, rows)
