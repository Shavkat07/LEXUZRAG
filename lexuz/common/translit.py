"""
lexuz.common.translit

Детерминированный нормализатор орфографии и транслитератор узбекской кириллицы
в официальную латиницу для Legal RAG Lex.uz.

Особенности:
- Стандартизация всех 5 типов апострофов и кавычек к единому ASCII ' (\\u0027).
- Фонетическое правило юридической латиницы: 'кц' -> 'ks' (акция -> aksiya).
- Контекстное правило буквы 'Е': ye в начале слова и после гласных/знаков, e после согласных.
- Специфические буквы: 'Ў/ў' -> O'/o', 'Ғ/ғ' -> G'/g', 'Қ/қ' -> Q/q, 'Ҳ/ҳ' -> H/h.
- Сохранение регистра (ALL_CAPS vs TitleCase).
"""

import re
import unicodedata

# Регулярка для приведения всех вариантов апострофов к каноническому ASCII ' (\u0027)
RE_APOSTROPHES = re.compile(r"[\u2018\u2019\u02BB\u02BC\u0060\u00B4\u2032]")

# Гласные узбекской кириллицы (для правила буквы 'Е')
CYR_VOWELS = set("аоуиэеўяюёыАОУИЭЕЎЯЮЁЫ")

# Базовая таблица соответствий 1-в-1
SIMPLE_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "қ": "q", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ҳ": "h", "ы": "i", "э": "e",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Y", "К": "K",
    "Қ": "Q", "Л": "L", "М": "M", "Н": "N", "О": "O",
    "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "X", "Ҳ": "H", "Ы": "I", "Э": "E",
}

RE_HAS_CYRILLIC = re.compile(r"[а-яА-ЯёЁўЎғҒҳҲқҚ]")
RE_TOKENIZE = re.compile(r"(\w+|[^\w\s]+|\s+)")


def normalize_apostrophes(text: str) -> str:
    """Приводит все типы апострофов и кавычек к единому каноническому ASCII '."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return RE_APOSTROPHES.sub("'", text)


def is_all_caps_word(word: str) -> bool:
    """Проверяет, состоит ли слово целиком из заглавных букв."""
    letters = [ch for ch in word if ch.isalpha()]
    return len(letters) > 1 and all(ch.isupper() for ch in letters)


def _translit_token(token: str) -> str:
    """Транслитерирует отдельное слово/токен с учетом контекста и регистра."""
    if not token:
        return ""

    token = normalize_apostrophes(token)
    is_upper_word = is_all_caps_word(token)
    chars = list(token)
    length = len(chars)
    out = []

    i = 0
    while i < length:
        ch = chars[i]
        prev_ch = chars[i - 1] if i > 0 else None
        next_ch = chars[i + 1] if i + 1 < length else None

        # 0. Сочетание 'кц' -> 'ks' (акция -> aksiya, санкция -> sanksiya)
        if ch in ("к", "К") and next_ch in ("ц", "Ц"):
            if is_upper_word:
                out.append("KS")
            elif ch.isupper():
                out.append("Ks")
            else:
                out.append("ks")
            i += 2
            continue

        # 1. 'Ў' / 'ў' и 'Ғ' / 'ғ'
        if ch == "ў":
            out.append("o'")
            i += 1
            continue
        if ch == "Ў":
            out.append("O'")
            i += 1
            continue
        if ch == "ғ":
            out.append("g'")
            i += 1
            continue
        if ch == "Ғ":
            out.append("G'")
            i += 1
            continue

        # 2. 'Ш/ш', 'Ч/ч', 'Ц/ц', 'Щ/щ'
        if ch in ("ш", "ч", "ц", "щ"):
            mapping = {"ш": "sh", "ч": "ch", "ц": "ts", "щ": "shch"}
            out.append(mapping[ch])
            i += 1
            continue
        if ch in ("Ш", "Ч", "Ц", "Щ"):
            mapping = {
                "Ш": ("SH" if is_upper_word else "Sh"),
                "Ч": ("CH" if is_upper_word else "Ch"),
                "Ц": ("TS" if is_upper_word else "Ts"),
                "Щ": ("SHCH" if is_upper_word else "Shch"),
            }
            out.append(mapping[ch])
            i += 1
            continue

        # 3. Йотированные: 'Ё/ё', 'Ю/ю', 'Я/я'
        if ch in ("ё", "ю", "я"):
            mapping = {"ё": "yo", "ю": "yu", "я": "ya"}
            out.append(mapping[ch])
            i += 1
            continue
        if ch in ("Ё", "Ю", "Я"):
            mapping = {
                "Ё": ("YO" if is_upper_word else "Yo"),
                "Ю": ("YU" if is_upper_word else "Yu"),
                "Я": ("YA" if is_upper_word else "Ya"),
            }
            out.append(mapping[ch])
            i += 1
            continue

        # 4. Твердый знак 'Ъ / ъ'
        if ch in ("ъ", "Ъ"):
            if next_ch and next_ch.lower() in ("е", "ё", "ю", "я"):
                i += 1
                continue
            out.append("'")
            i += 1
            continue

        # 5. Мягкий знак 'Ь / ь'
        if ch in ("ь", "Ь"):
            if next_ch and next_ch.lower() == "о":
                out.append("Y" if ch.isupper() and is_upper_word else "y")
            elif next_ch and next_ch.lower() in ("е", "ё", "ю", "я"):
                pass
            i += 1
            continue

        # 6. Буква 'Е / е'
        if ch == "е":
            if (
                prev_ch is None
                or prev_ch in CYR_VOWELS
                or prev_ch in ("ъ", "Ъ", "ь", "Ь")
                or not prev_ch.isalpha()
            ):
                out.append("ye")
            else:
                out.append("e")
            i += 1
            continue
        if ch == "Е":
            if (
                prev_ch is None
                or prev_ch in CYR_VOWELS
                or prev_ch in ("ъ", "Ъ", "ь", "Ь")
                or not prev_ch.isalpha()
            ):
                out.append("YE" if is_upper_word else "Ye")
            else:
                out.append("E")
            i += 1
            continue

        # 7. Базовые буквы
        if ch in SIMPLE_MAP:
            out.append(SIMPLE_MAP[ch])
        else:
            out.append(ch)

        i += 1

    return "".join(out)


def is_cyrillic(text: str) -> bool:
    """Определяет, содержит ли текст символы узбекской кириллицы."""
    return bool(RE_HAS_CYRILLIC.search(text))


def cyr_to_lat(text: str) -> str:
    """Транслитерирует узбекский кириллический текст в каноническую латиницу."""
    if not text:
        return ""

    tokens = RE_TOKENIZE.findall(text)
    converted = []

    for token in tokens:
        if is_cyrillic(token):
            converted.append(_translit_token(token))
        else:
            converted.append(normalize_apostrophes(token))

    return normalize_apostrophes("".join(converted))


def normalize_uz_text(text: str) -> str:
    """
    Универсальная точка входа:
    1. Если кириллица -> транслитерирует в латиницу.
    2. Если латиница -> стандартизирует апострофы в ASCII '.
    """
    if not text:
        return ""
    if is_cyrillic(text):
        return cyr_to_lat(text)
    return normalize_apostrophes(text)


LAT_TO_CYR_MAP = {
    "a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ",
    "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "қ", "r": "р", "s": "с", "t": "т",
    "u": "у", "v": "в", "x": "х", "y": "й", "z": "з",
    "A": "А", "B": "Б", "D": "Д", "F": "Ф", "G": "Г", "H": "Ҳ",
    "I": "И", "J": "Ж", "K": "К", "L": "Л", "M": "М", "N": "Н",
    "O": "О", "P": "П", "Q": "Қ", "R": "Р", "S": "С", "T": "Т",
    "U": "У", "V": "В", "X": "Х", "Y": "Й", "Z": "З",
}
LAT_VOWELS = set("aeiouAEIOU")


def lat_to_cyr(text: str) -> str:
    """
    Транслитерирует узбекский текст из латиницы в каноническую кириллицу:
    - o' -> ў, g' -> ғ
    - sh -> ш, ch -> ч, ts -> ц
    - yo -> ё, yu -> ю, ya -> я, ye -> е
    - e -> э (в начале слова/после гласной), е (после согласной)
    - ' -> ъ (tutuq belgisi)
    """
    if not text:
        return ""
    text = normalize_apostrophes(text)

    tokens = re.findall(r"[a-zA-Z']+|[^a-zA-Z']+", text)
    res = []

    for token in tokens:
        if not re.match(r"[a-zA-Z']+", token):
            res.append(token)
            continue

        t_len = len(token)
        i = 0
        converted = []
        is_all_caps = token.isupper() and t_len > 1

        while i < t_len:
            sub2 = token[i : i + 2]
            sub2_low = sub2.lower()

            # 1. o' / g'
            if sub2_low == "o'":
                converted.append("Ў" if sub2[0].isupper() else "ў")
                i += 2
                continue
            if sub2_low == "g'":
                converted.append("Ғ" if sub2[0].isupper() else "ғ")
                i += 2
                continue

            # 2. sh / ch / ts
            if sub2_low == "sh":
                converted.append("Ш" if sub2[0].isupper() else "ш")
                i += 2
                continue
            if sub2_low == "ch":
                converted.append("Ч" if sub2[0].isupper() else "ч")
                i += 2
                continue
            if sub2_low == "ts":
                converted.append("Ц" if sub2[0].isupper() else "ц")
                i += 2
                continue

            # 3. yo / yu / ya / ye
            if sub2_low == "yo":
                converted.append("Ё" if sub2[0].isupper() else "ё")
                i += 2
                continue
            if sub2_low == "yu":
                converted.append("Ю" if sub2[0].isupper() else "ю")
                i += 2
                continue
            if sub2_low == "ya":
                converted.append("Я" if sub2[0].isupper() else "я")
                i += 2
                continue
            if sub2_low == "ye":
                converted.append("Е" if sub2[0].isupper() else "е")
                i += 2
                continue

            ch = token[i]

            # 4. Буква 'e / E'
            if ch in ("e", "E"):
                prev_ch = token[i - 1] if i > 0 else None
                if prev_ch is None or prev_ch in LAT_VOWELS or prev_ch == "'":
                    converted.append("Э" if ch.isupper() else "э")
                else:
                    converted.append("Е" if ch.isupper() else "е")
                i += 1
                continue

            # 5. Апостроф ' -> ъ (tutuq belgisi)
            if ch == "'":
                converted.append("ъ")
                i += 1
                continue

            # 6. Базовые буквы
            if ch in LAT_TO_CYR_MAP:
                converted.append(LAT_TO_CYR_MAP[ch])
            else:
                converted.append(ch)
            i += 1

        res.append("".join(converted))

    return "".join(res)


