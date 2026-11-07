"""
tests/test_edge_cases.py
Проверка краевых случаев: query-параметры ссылок, якоря и трансляция степеней.
"""

import re
from lexuz.parsers.archetype_a import RE_DOC_LINK, SUP_UNICODE_MAP


def test_doc_links_with_query_params():
    sample_links = [
        ("/docs/-6445145#5129981", "-6445145", "5129981"),
        ("/docs/-6445145?ONDATE=25.09.2025#5129981", "-6445145", "5129981"),
        ("https://lex.uz/docs/3492203?action=compare#-3492528", "3492203", "-3492528"),
        ("/docs/12345", "12345", None),
    ]

    for url, expected_doc, expected_anchor in sample_links:
        m = RE_DOC_LINK.search(url)
        assert m is not None, f"Не удалось распарсить ссылку {url}"
        assert m.group(1) == expected_doc, f"DocID mismatch for {url}: expected {expected_doc}, got {m.group(1)}"
        assert m.group(2) == expected_anchor, f"Anchor mismatch for {url}: expected {expected_anchor}, got {m.group(2)}"

    print("✔ Ссылки с query-параметрами и отрицательными якорями парсятся без потерь.")


def test_superscript_indices():
    text = "35¹-modda va 42³-modda"
    translated = text.translate(SUP_UNICODE_MAP)
    assert translated == "35-1-modda va 42-3-modda"
    print("✔ Юникод-степени (¹²³) корректно транслируются в дефисную нотацию (-1, -3).")


if __name__ == "__main__":
    test_doc_links_with_query_params()
    test_superscript_indices()
    print("\nВсе краевые проверки успешно пройдены!")

