"""
lexuz.harvester.download_okoz

Асинхронная выгрузка списков документов (.doc) по категориям ОКОЗ через Playwright.
"""

import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

DEFAULT_OUTPUT_DIR = Path("data/okoz_exports")
DEFAULT_INPUT_JSON = Path("data/okoz_leaf_categories.json")


def load_and_filter_categories(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaf_nodes = data.get("leaf_nodes", [])
    filtered = []
    skipped_section_20 = 0
    skipped_repealed = 0

    for item in leaf_nodes:
        breadcrumb = item.get("breadcrumb", "")
        title = item.get("title", "")

        # 1. Отсекаем раздел 20 (индивидуальные кадровые указы, награждения)
        if "20.00.00.00" in breadcrumb:
            skipped_section_20 += 1
            continue

        # 2. Отсекаем явно утратившие силу ветки
        if "Кучини й" in title or "Кучини й" in breadcrumb:
            skipped_repealed += 1
            continue

        filtered.append(item)

    print(f"Всего категорий в файле: {len(leaf_nodes)}")
    print(f"Отсеяно кадровых актов (раздел 20): {skipped_section_20}")
    print(f"Отсеяно утративших силу веток: {skipped_repealed}")
    print(f"Осталось целевых категорий: {len(filtered)}")
    return filtered


async def download_category(page, okoz_id: str, output_dir: Path):
    file_path = output_dir / f"okoz_{okoz_id}.doc"
    if file_path.exists() and file_path.stat().st_size > 0:
        return "skipped"

    search_url = f"https://lex.uz/uz/search/all?lang=4&okoz={okoz_id}"

    try:
        await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)
        export_btn = page.locator("a:has-text('Ro‘yxatni Word ga eksport qilish')")

        if await export_btn.count() == 0:
            return "empty"

        async with page.expect_download(timeout=30000) as download_info:
            await export_btn.click()

        download = await download_info.value
        await download.save_as(file_path)
        return "downloaded"

    except Exception as e:
        return f"error: {str(e)}"


async def run_download_okoz(input_json: Path = DEFAULT_INPUT_JSON, output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = load_and_filter_categories(input_json)
    total = len(categories)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            accept_downloads=True
        )
        page = await context.new_page()

        start_time = time.time()
        success_count = 0
        empty_count = 0
        error_count = 0

        for i, item in enumerate(categories, 1):
            okoz_id = item["okoz_id"]

            if i % 50 == 0:
                await page.close()
                page = await context.new_page()

            result = await download_category(page, okoz_id, output_dir)

            if result == "downloaded":
                success_count += 1
                print(f"[{i}/{total}] [✓ Скачан] okoz_{okoz_id} ({item['title'][:35]})")
            elif result == "skipped":
                pass
            elif result == "empty":
                empty_count += 1
            else:
                error_count += 1
                print(f"[{i}/{total}] [! Ошибка] okoz_{okoz_id}: {result}")

            await asyncio.sleep(0.3)

        await browser.close()
        elapsed = time.time() - start_time
        print(f"Выгрузка завершена за {elapsed / 60:.1f} мин. Новых: {success_count}, Пусто: {empty_count}, Ошибок: {error_count}")


if __name__ == "__main__":
    asyncio.run(run_download_okoz())

