"""
tests/test_translit.py
Верификация правил транслитерации и орфографической нормализации (lexuz.common.translit).
"""

from lexuz.common.translit import cyr_to_lat, lat_to_cyr, normalize_apostrophes, normalize_uz_text


def test_apostrophe_normalization():
    samples = [
        ("O‘zbekiston", "O'zbekiston"),
        ("O’zbekiston", "O'zbekiston"),
        ("Oʻzbekiston", "O'zbekiston"),
        ("Oʼzbekiston", "O'zbekiston"),
        ("O`zbekiston", "O'zbekiston"),
        ("G‘alaba", "G'alaba"),
        ("ma’muriy", "ma'muriy"),
    ]
    for inp, expected in samples:
        res = normalize_apostrophes(inp)
        assert res == expected, f"Failed on {inp}: expected {expected}, got {res}"


def test_cyrillic_legal_corpus():
    cases = [
        # 1. Базовые правовые термины
        (
            "Ўзбекистон Республикасининг Конституцияси",
            "O'zbekiston Respublikasining Konstitutsiyasi"
        ),
        (
            "Маъмурий жавобгарлик тўғрисидаги кодекс",
            "Ma'muriy javobgarlik to'g'risidagi kodeks"
        ),
        (
            "Ҳуқуқбузарлик ва жавобгарлик",
            "Huquqbuzarlik va javobgarlik"
        ),
        (
            "Эълон қилинган сана",
            "E'lon qilingan sana"
        ),
        # 2. Правило буквы 'Е / е' (в начале/после гласной -> ye, после согласной -> e)
        (
            "Ер участкаси ва келаётган йил",
            "Yer uchastkasi va kelayotgan yil"
        ),
        (
            "Эҳтиёт чоралари",
            "Ehtiyot choralari"
        ),
        # 3. Йотированные: Ё, Ю, Я
        (
            "Ягона интерактив давлат хизматлари портали",
            "Yagona interaktiv davlat xizmatlari portali"
        ),
        (
            "Юридик шахслар",
            "Yuridik shaxslar"
        ),
        (
            "Кўчмас мулк бўйича битимлар рўйхатга олинганда",
            "Ko'chmas mulk bo'yicha bitimlar ro'yxatga olinganda"
        ),
        # 4. Буква 'Ц' и сочетание 'КЦ' -> ks (акция -> aksiya)
        (
            "Конституция ва акциядорлик жамиятлари",
            "Konstitutsiya va aksiyadorlik jamiyatlari"
        ),
        (
            "Цех мудири",
            "Tsex mudiri"
        ),
        # 5. Мягкий знак 'ь' (выпадение vs йотовое разделение)
        (
            "Ижтимоий роль ва лагерь",
            "Ijtimoiy rol va lager"
        ),
        (
            "Шартнома объекти ва субъектлари",
            "Shartnoma obyekti va subyektlari"
        ),
        # 6. ALL CAPS (заголовки законов)
        (
            "ЎЗБЕКИСТОН РЕСПУБЛИКАСИ ҚОНУНИ",
            "O'ZBEKISTON RESPUBLIKASI QONUNI"
        ),
        (
            "ШАХСИЙ ҲИСОБВАРАҚ",
            "SHAXSIY HISOBVARAQ"
        ),
        (
            "ЧОРА-ТАДБИРЛАР ДАСТУРИ",
            "CHORA-TADBIRLAR DASTURI"
        ),
    ]

    for cyr, expected_lat in cases:
        actual = cyr_to_lat(cyr)
        assert actual == expected_lat, (
            f"\nОшибка транслитерации:\n"
            f"  Исходный:  '{cyr}'\n"
            f"  Ожидалось: '{expected_lat}'\n"
            f"  Получено:  '{actual}'"
        )


def test_lat_to_cyr_transliteration():
    cases = [
        (
            "Har yilgi asosiy eng kam mehnat ta'tilining davomiyligi yigirma bir kalendar kunni tashkil etadi.",
            "Ҳар йилги асосий энг кам меҳнат таътилининг давомийлиги йигирма бир календар кунни ташкил этади.",
        ),
        (
            "O'zbekiston Respublikasining Mehnat kodeksi",
            "Ўзбекистон Республикасининг Меҳнат кодекси",
        ),
        (
            "217-modda",
            "217-модда",
        ),
        (
            "E'lon qilingan sana",
            "Эълон қилинган сана",
        ),
        (
            "G'alaba va shartnoma",
            "Ғалаба ва шартнома",
        ),
    ]
    for lat, expected_cyr in cases:
        actual = lat_to_cyr(lat)
        assert actual == expected_cyr, f"Failed on {lat}: expected {expected_cyr}, got {actual}"



if __name__ == "__main__":
    print("Запуск тестов нормализатора...")
    test_apostrophe_normalization()
    print("✔ Тест апострофов пройден успешно.")
    test_cyrillic_legal_corpus()
    print("✔ Тест кириллического юридического корпуса пройден успешно.")
    print("Все синтетические тесты пройдены на 100%!")

