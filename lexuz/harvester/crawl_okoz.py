"""
lexuz.harvester.crawl_okoz

Обход дерева классификатора ОКОЗ на Lex.uz и сохранение списка конечных листьев.
"""

import json
import re
import time
from collections import deque
from pathlib import Path
import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://lex.uz"
TREE_AJAX_URL = "https://lex.uz/uz/ajax_methods_url?action=getLawClassifierTree&show_type=ind&parent_id={parent_id}"
DEFAULT_OUTPUT = Path("data/okoz_leaf_categories.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest"
}


def clean_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split()).strip()


def get_root_categories(client: httpx.Client):
    """Сбор корневых веток 01.00.00.00 - 21.00.00.00 со страницы классификатора"""
    url = f"{BASE_URL}/uz/klassifkator"
    response = client.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, follow_redirects=True)
    soup = BeautifulSoup(response.text, "html.parser")

    roots = []
    for a in soup.find_all("a", attrs={"onclick": lambda v: v and "loadLawClassifierChilds" in v}):
        onclick = a["onclick"]
        node_id_match = re.search(r"loadLawClassifierChilds\('(\d+)'\)", onclick)
        if node_id_match:
            node_id = node_id_match.group(1)
            title = clean_text(a.text)
            roots.append({
                "id": node_id,
                "title": title,
                "path": [title]
            })
    return roots


def crawl_tree(output_path: Path = DEFAULT_OUTPUT):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(headers=HEADERS, timeout=15.0)

    print("[1/3] Получение корневых разделов классификатора...")
    roots = get_root_categories(client)
    print(f"Найдено корневых разделов: {len(roots)}")

    queue = deque(roots)
    visited_folders = set()
    leaf_categories = []
    counter = 0

    print("[2/3] Обход дочерних узлов дерева...")
    while queue:
        current = queue.popleft()
        node_id = current["id"]

        if node_id in visited_folders:
            continue
        visited_folders.add(node_id)

        counter += 1
        if counter % 25 == 0:
            print(f"Обработано узлов: {counter} | В очереди: {len(queue)} | Листьев (ОКОЗ): {len(leaf_categories)}")

        ajax_url = TREE_AJAX_URL.format(parent_id=node_id)
        try:
            res = client.get(ajax_url)
            if res.status_code != 200 or not res.text.strip():
                continue
        except Exception as e:
            print(f"Ошибка при запросе узла {node_id}: {e}")
            continue

        soup = BeautifulSoup(res.text, "html.parser")

        # 1. Поиск промежуточных папок
        folder_links = soup.find_all("a", attrs={"onclick": lambda v: v and "loadLawClassifierChilds" in v})
        for f in folder_links:
            match = re.search(r"loadLawClassifierChilds\('(\d+)'\)", f.get("onclick", ""))
            if match:
                child_id = match.group(1)
                child_title = clean_text(f.text)
                queue.append({
                    "id": child_id,
                    "title": child_title,
                    "parent_id": node_id,
                    "path": current["path"] + [child_title]
                })

        # 2. Поиск конечных узлов
        leaf_links = soup.find_all("a", href=lambda h: h and "okoz=" in h)
        for link in leaf_links:
            href = link.get("href", "")
            okoz_match = re.search(r"okoz=(\d+)", href)
            if okoz_match:
                okoz_id = okoz_match.group(1)
                leaf_title = clean_text(link.text)
                leaf_categories.append({
                    "okoz_id": okoz_id,
                    "title": leaf_title,
                    "parent_node_id": node_id,
                    "breadcrumb": " > ".join(current["path"] + [leaf_title]),
                    "search_url": f"{BASE_URL}/uz/search/all?lang=4&okoz={okoz_id}"
                })

        time.sleep(0.05)

    client.close()

    print("\n[3/3] Сохранение результатов...")
    output_data = {
        "total_leaf_categories": len(leaf_categories),
        "total_folders_traversed": counter,
        "leaf_nodes": leaf_categories
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Готово! Собрано категорий: {len(leaf_categories)} -> {output_path}")


if __name__ == "__main__":
    crawl_tree()

