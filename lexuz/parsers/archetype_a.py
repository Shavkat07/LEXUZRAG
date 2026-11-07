"""
lexuz.parsers.archetype_a

Экстрактор структурированных чанков для Архетипа A (Законы, кодексы, постановления).
Работает на базе selectolax.lexbor и детерминированного конечного автомата.

Ключевые возможности:
- Двойной режим детекции (Dual Mode):
    * Режим законов/кодексов: разбиение по статьям (.CLAUSE_DEFAULT / modda).
    * Режим постановлений/указов/приказов: разбиение по пунктам (band / ^\\d+\\.\\s+...),
      в том числе размеченным как .ACT_TEXT.
- Нарезка таблиц-гигантов (Table Slicing) с дублированием Markdown-шапки и сквозной нумерацией.
- Фильтрация ложных микро-преамбул ("Общая часть" / < 100 символов без норм).
- Полная поддержка отрицательных якорей uz (-3492528) и ссылок с query-параметрами.
- Нормализация индексов (35¹ -> 35-1) и тегов <br>.
- Дедупликация узлов DOM (защита от дубликатов Lexbor).
- Сохранение диспозитивных текстов однострочных пунктов.
- Корректная детекция скрипта (очистка интерфейсного мусора до вызова is_cyrillic).
"""

import re
from typing import Any, Dict, List, Optional
from selectolax.lexbor import LexborHTMLParser, LexborNode

from lexuz.common.translit import is_cyrillic, normalize_uz_text
from lexuz.parsers.table_parser import parse_table_grid, format_table_markdown, DEFAULT_MAX_TABLE_ROWS

# Регулярка для извлечения якорей (включая отрицательные ID в uz)
RE_LX_MO = re.compile(r"lx_mo\s*\(\s*event\s*,\s*['\"]?(-?\d+)['\"]?\s*\)")

# Регулярка для ссылок /docs/... с поддержкой query-параметров (?ONDATE=... и т.д.)
RE_DOC_LINK = re.compile(r"/docs/([-\d]+)(?:\?[^#]*)?(?:#(-?\d+))?")

# Регулярка классификатора ОКОЗ
RE_OKOZ = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\.\d{2}\b")

# Таблица трансляции Юникод-степеней
SUP_UNICODE_MAP = str.maketrans({
    "¹": "-1", "²": "-2", "³": "-3", "⁴": "-4", "⁵": "-5",
    "⁶": "-6", "⁷": "-7", "⁸": "-8", "⁹": "-9", "⁰": "-0",
})

# Римские цифры (для глав и разделов)
ROMAN_NUMERALS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
    "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
    "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30,
}

RE_ARTICLE_HEADER = re.compile(
    r"^(?:(\d+(?:[-_]\d+)?)\s*-\s*modda(?:si)?\b|стать[яеию]\s*(\d+(?:[-_]\d+)?)\b)\s*[\.:\-—]?\s*(.*)$",
    re.IGNORECASE
)
RE_CLAUSE_HEADER = re.compile(
    r"^(?:(\d+(?:[-_]\d+)?)\s*-\s*band(?:i)?\b|пункт[ау]?\s*(\d+(?:[-_]\d+)?)\b)\s*[\.:\-—]?\s*(.*)$",
    re.IGNORECASE
)
RE_NUM_DOT = re.compile(r"^(\d+(?:[\.-]\d+){0,3})\.?\s+([A-ZА-ЯЎҚҒ\"'«“„»”‘ʼ].*)$")

RE_APPENDIX = re.compile(
    r"^(?:(\d+)\s*-\s*)?(?:ilova|приложени[ея])(?:\s*(?:№\s*)?(\d+))?|"
    r"(?:qarori|buyrug['‘ʼ`]?i|farmoni)ga\s*(?:(\d+)\s*-\s*)?ilova\s*(?:№\s*)?(\d+)?\s*$|"
    r"(?:постановлению|приказу|указу|распоряжению).*?приложени[ея]\s*(?:№\s*)?(\d+)?\s*$",
    re.IGNORECASE
)


def roman_to_int(s: str) -> Optional[int]:
    """Преобразует римское число в арабское."""
    clean = s.strip().upper().rstrip(".")
    return ROMAN_NUMERALS.get(clean)


def parse_structural_header(
    text: str,
    is_clause_class: bool = False,
    allow_num_dot: bool = False
) -> Dict[str, str]:
    """
    Разбирает заголовки уровня Приложение / Раздел / Глава / Статья / Пункт.
    Возвращает dict: {'type': ..., 'number': ..., 'title': ...}
    """
    text = text.replace("\xa0", " ").strip()

    # 1. Приложения (Ilova / Приложение, включая юридические баннеры)
    m_app = RE_APPENDIX.search(text)
    if m_app:
        num = (
            m_app.group(1) or m_app.group(2) or m_app.group(3) or
            m_app.group(4) or m_app.group(5) or "1"
        )
        return {"type": "appendix", "number": num, "title": text}

    # 2. Разделы (Bo'lim / Раздел - поддержка арабских и римских цифр)
    m_part = re.match(
        r"^(?:([IVXLCDM\d]+)\s*-\s*bo['‘ʼ`]?lim|([IVXLCDM]+)\s+bo['‘ʼ`]?lim|раздел\s*([IVXLCDM\d]+))\s*[\.:\-—]?\s*(.*)$",
        text,
        re.IGNORECASE
    )
    if m_part:
        raw_num = m_part.group(1) or m_part.group(2) or m_part.group(3)
        num = str(roman_to_int(raw_num) if roman_to_int(raw_num) else raw_num)
        return {"type": "section", "number": num, "title": (m_part.group(4) or "").strip()}

    # 3. Главы (Bob / Глава - поддержка арабских и римских цифр)
    m_ch = re.match(
        r"^(?:([IVXLCDM\d]+)\s*-\s*bob|([IVXLCDM]+)\s+bob|глава\s*([IVXLCDM\d]+))\s*[\.:\-—]?\s*(.*)$",
        text,
        re.IGNORECASE
    )
    if m_ch:
        raw_num = m_ch.group(1) or m_ch.group(2) or m_ch.group(3)
        num = str(roman_to_int(raw_num) if roman_to_int(raw_num) else raw_num)
        return {"type": "chapter", "number": num, "title": (m_ch.group(4) or "").strip()}

    # 4. Статьи (Modda / Статья, включая прим: 35-1 и падежи)
    m_art = RE_ARTICLE_HEADER.search(text)
    if m_art:
        num = (m_art.group(1) or m_art.group(2)).replace("_", "-")
        return {"type": "article", "number": num, "title": m_art.group(3).strip()}

    # 5. Пункты (Band / Пункт)
    m_clause = RE_CLAUSE_HEADER.match(text)
    if m_clause:
        num = (m_clause.group(1) or m_clause.group(2)).replace("_", "-")
        return {"type": "clause", "number": num, "title": m_clause.group(3).strip()}

    # Если элемент имеет класс CLAUSE_DEFAULT или документ в режиме постановлений (allow_num_dot)
    if is_clause_class or allow_num_dot:
        m_num_dot = RE_NUM_DOT.match(text)
        if m_num_dot:
            return {
                "type": "clause",
                "number": m_num_dot.group(1).replace("_", "-"),
                "title": m_num_dot.group(2).strip()
            }

    return {"type": "unknown", "number": "", "title": text}


class ArchetypeAParser:
    """Парсер структурированных нормативно-правовых актов Архетипа A."""

    def __init__(self, html_content: str, doc_id: str, lang: str):
        self.doc_id = str(doc_id)
        self.lang = lang

        # 1. Замена <br> на пробелы (предотвращает склейку слов)
        clean_html = re.sub(r"<\s*br\s*/?>", " ", html_content, flags=re.IGNORECASE)

        # 2. Нормализация надстрочных индексов (35<sup>1</sup>-modda -> 35-1-modda)
        clean_html = re.sub(r"-\s*<sup\b[^>]*>(\d+)</sup>", r"-\1", clean_html, flags=re.IGNORECASE)
        clean_html = re.sub(r"<sup\b[^>]*>(\d+)</sup>", r"-\1", clean_html, flags=re.IGNORECASE)
        clean_html = clean_html.translate(SUP_UNICODE_MAP)
        clean_html = re.sub(r"-{2,}", "-", clean_html)

        self.tree = LexborHTMLParser(clean_html)

        # 3. Детекция контейнера
        self.container = self.tree.root.css_first("#divCont, #body, .ACT_BODY")
        if not self.container:
            self.container = self.tree.root

        # 4. Вырезаем служебный UI-мусор Lex.uz СТРОГО ДО проверки скрипта
        for junk in self.tree.root.css(".lx_no_select, .lx_elem2, .lx_elem3, script, style, noscript"):
            junk.decompose()

        # 5. Детекция исходного скрипта ДО транслитерации строго внутри очищенного контейнера
        raw_body_text = self.container.text()[:5000]
        self.is_original_cyrillic = is_cyrillic(raw_body_text)

        # 6. Определение режима документа (Закон со статьями vs Постановление с пунктами)
        self.has_articles = bool(self.container.css(".CLAUSE_DEFAULT")) or bool(
            RE_ARTICLE_HEADER.search(self.container.text()[:25000])
        )

    def extract_document_metadata(self) -> Dict[str, Any]:
        """Извлекает название акта, номер и классификаторы ОКОЗ."""
        title_el = self.tree.root.css_first(".ACT_TITLE, .doc_title, title")
        raw_title = title_el.text().strip() if title_el else ""
        if self.lang == "uz":
            raw_title = normalize_uz_text(raw_title)

        m_num = re.search(r"\b([A-ZО‘'ЎҚҒ]{1,4}[-—–]\s*\d+(?:[-_]\w+)?)\b", raw_title, re.IGNORECASE)
        doc_number = m_num.group(1).replace("—", "-").replace("–", "-").strip() if m_num else ""

        okoz_codes = set()
        for ior in self.tree.root.css(".INDEXES_ON_REF, span.iorVal, a[href*='okoz']"):
            found = RE_OKOZ.findall(ior.text())
            okoz_codes.update(found)

        return {
            "doc_title": raw_title,
            "doc_number": doc_number,
            "okoz_codes": sorted(list(okoz_codes)),
        }

    def parse_chunks(self) -> List[Dict[str, Any]]:
        """
        Основной цикл State Machine:
        формирует поток чанков (преамбула, статьи, пункты, приложения).
        """
        meta = self.extract_document_metadata()
        chunks: List[Dict[str, Any]] = []

        # Состояния автомата
        current_appendix: Optional[str] = None
        current_section: Optional[str] = None
        current_chapter: Optional[str] = None

        # Буфер активного чанка
        active_unit_type: Optional[str] = "preamble"
        active_unit_number = "0"
        active_unit_title = meta["doc_title"]
        active_content_parts: List[str] = []
        active_anchors: List[str] = []
        active_citations: List[Dict[str, str]] = []
        active_has_table = False
        table_counter = 0

        def commit_active_chunk():
            nonlocal active_content_parts, active_anchors, active_citations, active_has_table, active_unit_type, active_unit_number, active_unit_title
            full_content = "\n\n".join(p for p in active_content_parts if p.strip()).strip()
            if not full_content:
                return

            if not active_unit_type:
                if current_chapter:
                    active_unit_type = "chapter_header"
                    active_unit_title = current_chapter
                elif current_appendix:
                    active_unit_type = "appendix_header"
                    active_unit_title = current_appendix
                else:
                    active_unit_type = "preamble"
                    active_unit_title = "Muqaddima" if self.lang == "uz" else "Преамбула"

            # Фильтрация ложных преамбул (например, "Общая часть" / длина < 100 символов без реального текста)
            if active_unit_type == "preamble":
                clean_words = set(re.findall(r"[\w']+", full_content.lower()))
                structural_words = {
                    "umumiy", "qism", "maxsus", "bo'lim", "bob", "qoidalar", "asosiy",
                    "общая", "часть", "особенная", "раздел", "глава", "положения", "основные",
                    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"
                }
                if len(full_content) < 100 or clean_words.issubset(structural_words):
                    active_content_parts = []
                    active_anchors = []
                    active_citations = []
                    active_has_table = False
                    return

            breadcrumb = [meta["doc_title"]]
            if current_appendix:
                breadcrumb.append(current_appendix)
            if current_section:
                breadcrumb.append(current_section)
            if current_chapter:
                breadcrumb.append(current_chapter)

            if active_unit_type == "preamble":
                preamble_label = "Muqaddima" if self.lang == "uz" else "Преамбула"
                breadcrumb.append(preamble_label)
            elif active_unit_type == "appendix_header":
                pass
            else:
                if self.lang == "uz":
                    type_label = f"{active_unit_number}-modda" if active_unit_type == "article" else f"{active_unit_number}-band"
                else:
                    type_label = f"Статья {active_unit_number}" if active_unit_type == "article" else f"Пункт {active_unit_number}"
                
                # Компактный заголовок в хлебных крошках (защита от раздувания)
                title_snip = f": {active_unit_title[:80]}..." if len(active_unit_title) > 80 else (f". {active_unit_title}" if active_unit_title else "")
                breadcrumb.append(f"{type_label}{title_snip}".strip())

            prefix = f"{current_appendix}_" if current_appendix else ""
            unit_tag = "preamble" if active_unit_type == "preamble" else f"{active_unit_type[:3]}_{active_unit_number}"
            base_cid = f"{self.doc_id}_{self.lang}_{prefix}{unit_tag}"
            prior_count = len([c for c in chunks if c["chunk_id"] == base_cid or c["chunk_id"].startswith(f"{base_cid}_p")])
            chunk_id = f"{base_cid}_p{prior_count + 1}" if prior_count > 0 else base_cid

            chunk_obj = {
                "chunk_id": chunk_id,
                "doc_id": self.doc_id,
                "lang": self.lang,
                "doc_number": meta["doc_number"],
                "doc_title": meta["doc_title"],
                "archetype": "ARCHETYPE_A",
                "breadcrumb": breadcrumb,
                "structural_unit": {
                    "type": active_unit_type,
                    "number": active_unit_number if active_unit_type != "preamble" else "0",
                    "title": active_unit_title if active_unit_type != "preamble" else ("Muqaddima" if self.lang == "uz" else "Преамбула"),
                },
                "content": full_content,
                "metadata": {
                    "anchors": sorted(list(set(active_anchors))),
                    "okoz_codes": meta["okoz_codes"],
                    "citations": active_citations,
                    "has_table": active_has_table,
                    "original_script": "cyr" if self.is_original_cyrillic else "lat",
                },
            }
            chunks.append(chunk_obj)

            active_content_parts = []
            active_anchors = []
            active_citations = []
            active_has_table = False

        # Отбираем смысловые блоки документа с дедупликацией узлов и изоляцией таблиц
        raw_candidates = self.container.css(
            ".TEXT_HEADER_DEFAULT, .CLAUSE_DEFAULT, .ACT_TEXT, .PART_DEFAULT, .SUBPART_DEFAULT, "
            ".APPL_BANNER_LANDSCAPE_TITLE, .APPL_BANNER_PORTRAIT_TITLE, .ACT_TITLE_APPL, table, p, div"
        )
        seen_nodes = set()
        blocks = []
        blocks_set = set()
        for node in raw_candidates:
            if node in seen_nodes:
                continue
            seen_nodes.add(node)

            # Пропускаем любые элементы, находящиеся внутри таблиц (они обрабатываются табличным парсером)
            p_curr = node.parent
            in_tbl = False
            while p_curr:
                if p_curr.tag == "table":
                    in_tbl = True
                    break
                p_curr = p_curr.parent
            if in_tbl:
                continue

            # Пропускаем элементы, чей родитель уже добавлен в блоки (предотвращение дублирования вложенных div)
            anc_curr = node.parent
            in_added_anc = False
            while anc_curr:
                if anc_curr in blocks_set:
                    in_added_anc = True
                    break
                anc_curr = anc_curr.parent
            if in_added_anc:
                continue

            if node.tag == "table":
                blocks.append(node)
                blocks_set.add(node)
                continue
            cls = node.attributes.get("class", "")
            if any(c in cls for c in (
                "TEXT_HEADER_DEFAULT", "CLAUSE_DEFAULT", "ACT_TEXT", "PART_DEFAULT", "SUBPART_DEFAULT",
                "APPL_BANNER_LANDSCAPE_TITLE", "APPL_BANNER_PORTRAIT_TITLE", "ACT_TITLE_APPL"
            )):
                blocks.append(node)
                blocks_set.add(node)
                continue
            if node.tag in ("p", "div"):
                # Пропускаем родительские контейнеры, содержащие другие блочные элементы
                if any(child.tag in ("div", "table") for child in node.iter() if child != node):
                    continue
                txt = node.text().strip()
                if txt and not txt.startswith(("[OKOZ:", "[TSZ:")):
                    blocks.append(node)
                    blocks_set.add(node)

        for el in blocks:
            tag = el.tag
            class_name = el.attributes.get("class", "")

            # 1. Извлечение якорей абзацев (lx_mo)
            for target_node in [el] + el.css("[onmousemove*='lx_mo'], [id^='lx_mo_']"):
                on_mouse = target_node.attributes.get("onmousemove", "")
                if on_mouse:
                    m_mo = RE_LX_MO.search(on_mouse)
                    if m_mo:
                        active_anchors.append(m_mo.group(1))
                el_id = target_node.attributes.get("id", "")
                if el_id and el_id.startswith("lx_mo_"):
                    active_anchors.append(el_id.replace("lx_mo_", ""))

            # 2. Извлечение ссылок цитирования (/docs/...)
            for a_node in el.css("a[href*='/docs/']"):
                href = a_node.attributes.get("href", "")
                m_cite = RE_DOC_LINK.search(href)
                if m_cite:
                    ref_text = a_node.text().strip()
                    if self.lang == "uz":
                        ref_text = normalize_uz_text(ref_text)
                    active_citations.append({
                        "target_doc_id": m_cite.group(1),
                        "target_anchor": m_cite.group(2) or "",
                        "raw_ref_text": ref_text,
                    })

            # 3. Сериализация и нарезка таблиц (Table Slicing с уникальными ID)
            if tag == "table":
                headers, data_rows = parse_table_grid(el, self.lang)
                if headers and data_rows:
                    table_counter += 1
                    # Если таблица умеренная (до 35 строк) и есть активная статья или пункт
                    if len(data_rows) <= DEFAULT_MAX_TABLE_ROWS and active_unit_type in ("article", "clause"):
                        md_table = format_table_markdown(headers, data_rows)
                        curr_size = sum(len(p) for p in active_content_parts)
                        if active_has_table or curr_size + len(md_table) >= 3500:
                            commit_active_chunk()
                        active_has_table = True
                        active_content_parts.append(md_table)
                    else:
                        commit_active_chunk()
                        for s_idx, i in enumerate(range(0, len(data_rows), DEFAULT_MAX_TABLE_ROWS), 1):
                            slice_rows = data_rows[i:i + DEFAULT_MAX_TABLE_ROWS]
                            md_slice = format_table_markdown(headers, slice_rows)

                            breadcrumb = [meta["doc_title"]]
                            if current_appendix:
                                breadcrumb.append(current_appendix)
                            if current_section:
                                breadcrumb.append(current_section)
                            if current_chapter:
                                breadcrumb.append(current_chapter)

                            unit_label = "jadval" if self.lang == "uz" else "таблица"
                            breadcrumb.append(f"{unit_label} ({table_counter}_{s_idx})")

                            prefix = f"{current_appendix}_" if current_appendix else ""
                            chunk_id = f"{self.doc_id}_{self.lang}_{prefix}tbl_{table_counter}_s{s_idx}"

                            chunks.append({
                                "chunk_id": chunk_id,
                                "doc_id": self.doc_id,
                                "lang": self.lang,
                                "doc_number": meta["doc_number"],
                                "doc_title": meta["doc_title"],
                                "archetype": "ARCHETYPE_A",
                                "breadcrumb": breadcrumb,
                                "structural_unit": {
                                    "type": "table_slice",
                                    "number": f"{table_counter}_{s_idx}",
                                    "title": f"Jadval {table_counter}.{s_idx}" if self.lang == "uz" else f"Таблица {table_counter}.{s_idx}",
                                },
                                "content": md_slice,
                                "metadata": {
                                    "anchors": sorted(list(set(active_anchors))),
                                    "okoz_codes": meta["okoz_codes"],
                                    "citations": list(active_citations),
                                    "has_table": True,
                                    "original_script": "cyr" if self.is_original_cyrillic else "lat",
                                },
                            })
                            active_anchors = []
                            active_citations = []
                continue

            raw_text = el.text().strip()
            if not raw_text:
                continue

            norm_text = normalize_uz_text(raw_text) if self.lang == "uz" else raw_text

            if RE_OKOZ.search(norm_text) or norm_text.startswith(("[OKOZ:", "[TSZ:")) or norm_text == meta["doc_title"]:
                continue

            # 4. Анализ структурных заголовков
            is_clause_cls = "CLAUSE_DEFAULT" in class_name
            allow_num_dot = (not self.has_articles)

            parsed_struct = parse_structural_header(
                norm_text,
                is_clause_class=is_clause_cls,
                allow_num_dot=allow_num_dot
            )

            # Приложение
            if parsed_struct["type"] == "appendix":
                commit_active_chunk()
                current_appendix = f"ilova_{parsed_struct['number']}"
                current_section = None
                current_chapter = None
                active_unit_type = "appendix_header"
                active_unit_number = parsed_struct["number"]
                active_unit_title = norm_text
                continue

            # Раздел (сброс активной единицы)
            if parsed_struct["type"] == "section":
                commit_active_chunk()
                current_section = norm_text
                current_chapter = None
                active_unit_type = None
                active_unit_number = ""
                active_unit_title = ""
                continue

            # Глава (сброс активной единицы для предотвращения дубликатов)
            if parsed_struct["type"] == "chapter":
                commit_active_chunk()
                current_chapter = norm_text
                active_unit_type = None
                active_unit_number = ""
                active_unit_title = ""
                continue

            # Статья или Пункт
            if parsed_struct["type"] in ("article", "clause"):
                commit_active_chunk()
                active_unit_type = parsed_struct["type"]
                active_unit_number = parsed_struct["number"]
                active_unit_title = parsed_struct["title"]
                # Сохраняем текст пункта в контент (защита от потери однострочных пунктов)
                if parsed_struct["type"] == "clause":
                    active_content_parts.append(norm_text)
                continue

            # 5. Сбор текста нормы или преамбулы (с защитой от раздувания)
            if norm_text != active_unit_title:
                if len(norm_text) > 3500:
                    sub_units = []
                    for line in norm_text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if len(line) > 2500:
                            for sent in re.split(r"(?<=[.!?])\s+", line):
                                if sent.strip():
                                    sub_units.append(sent.strip())
                        else:
                            sub_units.append(line)

                    cur_buf = []
                    cur_len = 0
                    for su in sub_units:
                        if cur_len + len(su) >= 2500 and cur_buf:
                            if sum(len(p) for p in active_content_parts) + cur_len >= 3000:
                                commit_active_chunk()
                            active_content_parts.append("\n\n".join(cur_buf))
                            cur_buf = []
                            cur_len = 0
                        cur_buf.append(su)
                        cur_len += len(su)
                    if cur_buf:
                        if sum(len(p) for p in active_content_parts) + cur_len >= 3000:
                            commit_active_chunk()
                        active_content_parts.append("\n\n".join(cur_buf))
                else:
                    curr_size = sum(len(p) for p in active_content_parts)
                    split_threshold = 3500 if active_unit_type == "appendix_header" else (6500 if active_unit_type == "clause" else 12000)
                    if curr_size >= split_threshold:
                        commit_active_chunk()
                    active_content_parts.append(norm_text)

        commit_active_chunk()
        return chunks
