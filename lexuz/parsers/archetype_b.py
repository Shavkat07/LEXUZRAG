"""
lexuz.parsers.archetype_b

Экстрактор чанков для Архетипа B:
- Неструктурированные акты (без CLAUSE_DEFAULT и ACT_TEXT)
- Табличные акты, сметы, ведомости, перечни и приложения
- Постановления и решения с разметкой BY_DEFAULT
- Международные договоры, соглашения, типовые уставы

Ключевые возможности:
- Строгий фильтр языковых перенаправлений (cross-language stubs) и PDF-заглушек.
- Автономное извлечение и нарезка таблиц (table_slice) с сохранением контекста приложений.
- Детерминированное разбиение пунктов (при наличии нумерации 1., 2., 3.)
- Семантическое скользящее окно (sliding window) для нарративных текстов без нумерации.
- Полная нормализация текста (normalize_uz_text для uz), извлечение якорей (lx_mo) и цитат.
"""

import re
from typing import Any, Dict, List, Optional
from selectolax.lexbor import LexborHTMLParser, LexborNode

from lexuz.common.models import ChunkDict, StructuralUnitDict, ChunkMetadataDict, CitationDict
from lexuz.common.translit import is_cyrillic, normalize_uz_text
from lexuz.parsers.table_parser import parse_table_grid, table_to_markdown_slices, DEFAULT_MAX_TABLE_ROWS

RE_LX_MO = re.compile(r"lx_mo\s*\(\s*event\s*,\s*['\"]?(-?\d+)['\"]?\s*\)")
RE_DOC_LINK = re.compile(r"/docs/([-\d]+)(?:\?[^#]*)?(?:#(-?\d+))?")
RE_OKOZ = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\.\d{2}\b")

# Регулярка детекции языковых и PDF заглушек (с учетом тегов, опечаток и переносов)
STUB_REGEX = re.compile(
    r"(?is)(?:"
    r"(?:hujjat|qaror|farmon|buyruq|qonun|bitim|konvensiya|shartnoma|yo['‘`ʼ]?riqnoma|nizom|qoida|tartib)\s+matni\s+[^.\n]{0,80}(?:tilida|tillarida)\s+(?:berilgan|keltiriladi|keltirilgan|chop\s+etilgan|e['‘`ʼ]?lon\s+qilingan)|"
    r"matn\s+(?:rus|o['‘`ʼ]?zbek|ingliz)\s+tilida\s+(?:berilgan|keltiriladi|keltirilgan)|"
    r"текст\s+(?:акта|документа|постановления|указа|приказа|распоряжения|решения|закона|соглашения|договора|конвенции|инструкции|положения|правил)?\s*[^.\n]{0,80}(?:приводится|приовдится|опубликован|помещен)\s+(?:на|в)|"
    r"(?:ҳужжат|қарор|фармон|буйруқ|қонун|битим|конвенция|йўриқнома|низом|қоида)\s+матни\s+[^.\n]{0,80}(?:тилида|тилларида)\s+(?:берилган|келтирилади|келтирилган|чоп\s+этилган)|"
    r"матн\s+(?:рус|ўзбек|инглиз)\s+тилида\s+(?:берилган|келтирилади|келтирилган)|"
    r"онлайн-перевод.*?системном\s+меню|"
    r"online-tarjima|"
    r"pdf\s+shakl[i]?da\s+(?:berilgan|keltirilgan)|"
    r"формате\s+pdf"
    r")"
)

# Оперативные правовые глаголы (диспозиции норм)
OPERATIVE_VERBS = re.compile(
    r"(?i)\b("
    r"qaror\s+qiladi|buyuraman|belgilansin|tasdiqlansin|amalga\s+oshirilsin|topshirilsin|"
    r"kiritilsin|o'zgartirilsin|bekor\s+qilinsin|kuchini\s+yo'qotgan|tavsiya\s+etilsin|"
    r"постановляет|приказываю|утвердить|установить|возложить|внести|признать\s+утратившим|"
    r"рекомендовать|поручить"
    r")\b"
)

# Регулярки для нумерованных пунктов (строго с точкой после номера и заглавной буквы)
RE_CLAUSE_NUM = re.compile(r"^(\d+(?:[\.-]\d+){0,3})\.\s+([A-ZА-ЯЎҚҒ\"'«“„‘ʼ].*)$", re.DOTALL)
RE_CLAUSE_HEADER = re.compile(
    r"^(?:(\d+(?:[-_]\d+)?)\s*-\s*band(?:i)?\b|пункт[ау]?\s*(\d+(?:[-_]\d+)?)\b)\s*[\.:\-—]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)

RE_DATE_OR_SIGNATURE = re.compile(
    r"(?i)(?:г\.\s*Ташкент|Toshkent\s*sh\.|(?:январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр|yil)\b|Prezident|Vazir|Hokim|Rais|Президент|Министр|Хоким|Председатель)"
)

# Регулярка для заголовков приложений
RE_APPENDIX_TITLE = re.compile(
    r"(?:(\d+)\s*-\s*)?(?:ilova|приложени[ея])(?:\s*(?:№\s*)?(\d+))?|"
    r"(?:qarori|buyrug['‘ʼ`]?i|farmoni)ga\s*(?:(\d+)\s*-\s*)?ilova\s*(?:№\s*)?(\d+)?|"
    r"(?:постановлению|приказу|указу|распоряжению).*?приложени[ея]\s*(?:№\s*)?(\d+)?",
    re.IGNORECASE
)


class ArchetypeBParser:
    """Парсер для неструктурированных и табличных документов Архетипа B."""

    def __init__(self, html_content: str, doc_id: str, lang: str):
        self.raw_html = html_content
        self.doc_id = doc_id
        self.lang = lang
        self.parser = LexborHTMLParser(html_content)

        # Метаданные акта
        self.doc_title = ""
        self.doc_number = ""
        self.okoz_codes: List[str] = []
        self.original_script = "lat"

    @classmethod
    def can_parse(cls, html_content: str) -> bool:
        """
        Проверяет, подходит ли документ для Архетипа B:
        - Есть контейнер divCont и размер не менее 650 байт
        - Не содержит структурированных классов Архетипа A (CLAUSE_DEFAULT, ACT_TEXT)
        - Не является пустой языковой или PDF заглушкой
        - Не является пустой административной оболочкой (содержит таблицы, пункты или диспозитивные нормы)
        """
        if len(html_content) < 650 or "divCont" not in html_content:
            return False

        # Если есть явные классы Архетипа A - делегируем Архетипу A
        if "CLAUSE_DEFAULT" in html_content or "ACT_TEXT" in html_content:
            return False

        temp_p = LexborHTMLParser(html_content)
        div = temp_p.css_first("div#divCont")
        if not div:
            return False

        clean_text = " ".join(div.text().split())
        if len(clean_text) < 150:
            return False

        # 1. Проверяем наличие содержательных таблиц (tr >= 3)
        has_substantive_table = any(len(tbl.css("tr")) >= 3 for tbl in div.css("table"))

        # 2. Проверяем языковые или PDF заглушки
        is_stub = bool(STUB_REGEX.search(clean_text))
        if is_stub and not has_substantive_table:
            return False

        # 3. Фильтруем пустые административные оболочки (документы без таблиц, без пунктов и без норм)
        has_numbered_clauses = bool(RE_CLAUSE_NUM.search(clean_text) or RE_CLAUSE_HEADER.search(clean_text))
        has_operative_norms = bool(OPERATIVE_VERBS.search(clean_text))

        if not has_substantive_table and not has_numbered_clauses and not has_operative_norms:
            return False

        return True

    def _clean_dom(self, container: LexborNode):
        """Очистка служебного UI Lex.uz."""
        for junk in container.css(".lx_no_select, .lx_elem2, .lx_elem3, script, style"):
            junk.decompose()
        # Заменяем <br> на пробелы
        for br in container.css("br"):
            br.replace_with(" ")

    def _extract_doc_metadata(self, container: LexborNode):
        """Извлечение метаданных документа (название, номер, ОКОЗ)."""
        # 1. Заголовок акта
        title_node = container.css_first(".ACT_TITLE") or self.parser.css_first("title")
        if title_node:
            t = " ".join(title_node.text().split())
            self.doc_title = normalize_uz_text(t) if self.lang == "uz" else t

        # 2. Номер акта
        num_node = container.css_first(".ACT_ESSENTIAL_ELEMENTS_NUM")
        if num_node:
            self.doc_number = " ".join(num_node.text().split())
        else:
            m_num = re.search(r"№\s*([0-9\-_/]+)", self.doc_title)
            if m_num:
                self.doc_number = m_num.group(1)

        # 3. Коды ОКОЗ
        raw_text = container.text()
        self.okoz_codes = sorted(list(set(RE_OKOZ.findall(raw_text))))

        # 4. Определение скрипта
        self.original_script = "cyr" if is_cyrillic(raw_text[:2000]) else "lat"

    def _extract_citations(self, node: LexborNode) -> List[CitationDict]:
        """Извлечение исходящих ссылок на другие акты."""
        citations: List[CitationDict] = []
        for a_tag in node.css("a[href]"):
            href = a_tag.attributes.get("href", "")
            m_link = RE_DOC_LINK.search(href)
            if m_link:
                t_doc = m_link.group(1)
                t_anchor = m_link.group(2) or ""
                raw_txt = a_tag.text().strip()
                citations.append({
                    "target_doc_id": t_doc,
                    "target_anchor": t_anchor,
                    "raw_ref_text": normalize_uz_text(raw_txt) if self.lang == "uz" else raw_txt
                })
        return citations

    def _extract_anchors(self, node: LexborNode) -> List[str]:
        """Извлечение идентификаторов якорей lx_mo."""
        anchors = []
        raw_html = node.html or ""
        for m in RE_LX_MO.finditer(raw_html):
            anchors.append(m.group(1))
        # Также проверяем атрибуты id и name
        node_id = node.attributes.get("id")
        if node_id and re.match(r"^-?\d+$", node_id):
            anchors.append(node_id)
        node_name = node.attributes.get("name")
        if node_name and re.match(r"^-?\d+$", node_name):
            anchors.append(node_name)
        return sorted(list(set(anchors)))

    def parse_chunks(self) -> List[Dict[str, Any]]:
        """Главный метод генерации чанков Архетипа B."""
        container = self.parser.css_first("div#divCont") or self.parser.css_first("body")
        if not container:
            return []

        self._clean_dom(container)
        self._extract_doc_metadata(container)

        chunks: List[Dict[str, Any]] = []

        current_annex_title = ""
        current_annex_num = "1"
        table_counter = 0

        # Собираем прямые смысловые блоки внутри контейнера
        # 1. Извлекаем все содержательные таблицы в порядке следования DOM с привязкой к приложениям
        all_elements = container.css(".APPL_BANNER_LANDSCAPE_TITLE, .ACT_TITLE_APPL, table, .BY_DEFAULT, p")
        
        seen_tables = set()
        for node in all_elements:
            if node.tag == "table":
                if id(node) in seen_tables:
                    continue
                trs = node.css("tr")
                if len(trs) < 3:
                    continue

                tbl_text = node.text().strip()
                if len(tbl_text) < 40 and any(w in tbl_text.lower() for w in ["утратил силу", "кучини йўқотган", "вступил в силу"]):
                    continue

                table_counter += 1
                tbl_title = current_annex_title or self.doc_title

                slices = table_to_markdown_slices(node, self.lang, max_rows_per_slice=DEFAULT_MAX_TABLE_ROWS)
                if not slices:
                    continue

                anchors = self._extract_anchors(node)
                citations = self._extract_citations(node)

                for sl_idx, slice_md in enumerate(slices, 1):
                    chunk_id = f"{self.doc_id}_{self.lang}_tbl_{table_counter}_sl_{sl_idx}"
                    struct_unit: StructuralUnitDict = {
                        "type": "table_slice",
                        "number": f"T{table_counter}.S{sl_idx}",
                        "title": tbl_title
                    }
                    meta: ChunkMetadataDict = {
                        "anchors": anchors,
                        "okoz_codes": self.okoz_codes,
                        "citations": citations,
                        "has_table": True,
                        "original_script": self.original_script
                    }

                    content_parts = []
                    if tbl_title:
                        content_parts.append(f"### {tbl_title}\n")
                    content_parts.append(slice_md)

                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": self.doc_id,
                        "lang": self.lang,
                        "doc_number": self.doc_number,
                        "doc_title": self.doc_title,
                        "archetype": "archetype_b",
                        "breadcrumb": [self.doc_title, tbl_title] if tbl_title != self.doc_title else [self.doc_title],
                        "structural_unit": struct_unit,
                        "content": "\n".join(content_parts).strip(),
                        "metadata": meta
                    })

                seen_tables.add(id(node))
                # КРИТИЧНО: удаляем обработанную таблицу из DOM,
                # чтобы ее сырой текст не попал в нарративные текстовые чанки!
                node.decompose()
            else:
                cls_attr = node.attributes.get("class", "")
                if "APPL" in cls_attr or "TITLE" in cls_attr:
                    raw_txt = " ".join(node.text().split())
                    if raw_txt and len(raw_txt) < 300:
                        norm_txt = normalize_uz_text(raw_txt) if self.lang == "uz" else raw_txt
                        current_annex_title = norm_txt

        # 2. Извлекаем текстовые пункты и параграфы (строго вне таблиц)
        text_elements = []
        for el in container.css(".BY_DEFAULT, p, div"):
            # Пропускаем служебные блоки: подписи, издавший орган, форма акта, заголовки приложений
            cls_attr = el.attributes.get("class", "")
            if any(k in cls_attr for k in ["SIGNATURE", "ACCEPTING_BODY", "ACT_FORM", "ACT_ESSENTIAL_ELEMENTS", "APPL", "TITLE"]):
                continue

            # Пропускаем элементы внутри таблиц
            parent_table = el.parent
            is_inside_table = False
            while parent_table:
                if parent_table.tag == "table":
                    is_inside_table = True
                    break
                parent_table = parent_table.parent
            if is_inside_table:
                continue

            txt = " ".join(el.text().split())
            if not txt or len(txt) < 15:
                continue

            # Игнорируем дубликаты заголовка акта и перенаправлений
            if STUB_REGEX.search(txt) or txt == self.doc_title:
                continue

            text_elements.append(el)

        # Группируем параграфы в пункты или скользящие окна
        if text_elements:
            current_clause_num = ""
            current_clause_parts = []
            current_anchors = []
            current_citations = []
            clause_idx = 0

            for el in text_elements:
                raw_t = " ".join(el.text().split())
                norm_t = normalize_uz_text(raw_t) if self.lang == "uz" else raw_t

                m_num = RE_CLAUSE_NUM.match(norm_t)
                m_hdr = RE_CLAUSE_HEADER.match(norm_t)

                # Проверяем, не является ли это датой или подписью
                if m_num and (len(norm_t) < 40 or RE_DATE_OR_SIGNATURE.search(norm_t[:60])):
                    m_num = None

                is_new_clause = bool(m_num or m_hdr)

                if is_new_clause and current_clause_parts:
                    content_str = "\n\n".join(current_clause_parts).strip()
                    is_meaningful = (
                        (bool(current_clause_num) or bool(OPERATIVE_VERBS.search(content_str)))
                        and not bool(STUB_REGEX.search(content_str))
                    )
                    if is_meaningful and len(content_str) >= 60:
                        clause_idx += 1
                        c_num = current_clause_num or str(clause_idx)
                        chunk_id = f"{self.doc_id}_{self.lang}_clause_{c_num}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "doc_id": self.doc_id,
                            "lang": self.lang,
                            "doc_number": self.doc_number,
                            "doc_title": self.doc_title,
                            "archetype": "archetype_b",
                            "breadcrumb": [self.doc_title],
                            "structural_unit": {"type": "clause", "number": c_num, "title": f"Пункт {c_num}" if self.lang == "ru" else f"{c_num}-band"},
                            "content": content_str,
                            "metadata": {
                                "anchors": sorted(list(set(current_anchors))),
                                "okoz_codes": self.okoz_codes,
                                "citations": current_citations,
                                "has_table": False,
                                "original_script": self.original_script
                            }
                        })
                    current_clause_parts = []
                    current_anchors = []
                    current_citations = []

                if is_new_clause:
                    if m_num:
                        current_clause_num = m_num.group(1).rstrip(".")
                    elif m_hdr:
                        current_clause_num = m_hdr.group(1) or m_hdr.group(2) or str(clause_idx + 1)

                current_clause_parts.append(norm_t)
                current_anchors.extend(self._extract_anchors(el))
                current_citations.extend(self._extract_citations(el))

            # Сохраняем последний пункт
            if current_clause_parts:
                content_str = "\n\n".join(current_clause_parts).strip()
                is_meaningful = (
                    (bool(current_clause_num) or bool(OPERATIVE_VERBS.search(content_str)))
                    and not bool(STUB_REGEX.search(content_str))
                )
                if is_meaningful and len(content_str) >= 60:
                    clause_idx += 1
                    c_num = current_clause_num or str(clause_idx)
                    chunk_id = f"{self.doc_id}_{self.lang}_clause_{c_num}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": self.doc_id,
                        "lang": self.lang,
                        "doc_number": self.doc_number,
                        "doc_title": self.doc_title,
                        "archetype": "archetype_b",
                        "breadcrumb": [self.doc_title],
                        "structural_unit": {"type": "clause", "number": c_num, "title": f"Пункт {c_num}" if self.lang == "ru" else f"{c_num}-band"},
                        "content": content_str,
                        "metadata": {
                            "anchors": sorted(list(set(current_anchors))),
                            "okoz_codes": self.okoz_codes,
                            "citations": current_citations,
                            "has_table": False,
                            "original_script": self.original_script
                        }
                    })

        return chunks
