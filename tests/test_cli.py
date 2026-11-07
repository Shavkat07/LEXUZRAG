"""
tests/test_cli.py

Тесты консольного интерфейса main.py (CLI):
- index --mock (проверка аргументов, создания коллекции, сохранения чекпоинта)
- search --mock (проверка вывода форматированного Markdown и --raw JSON)
"""

import json
import subprocess
import sys
from pathlib import Path

from lexuz.storage.qdrant_manager import QdrantManager

TEST_CLI_COLLECTION = "test_cli_mock_collection"


def test_cli_index_and_search_mock(tmp_path: Path):
    ckpt_path = tmp_path / "cli_indexing_state.json"
    manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)

    try:
        # 1. Test CLI index with --mock and --limit 10
        cmd_index = [
            sys.executable,
            "main.py",
            "index",
            "--collection",
            TEST_CLI_COLLECTION,
            "--limit",
            "10",
            "--batch-size",
            "5",
            "--checkpoint",
            str(ckpt_path),
            "--recreate",
            "--mock",
        ]
        res_idx = subprocess.run(cmd_index, capture_output=True, text=True)
        assert res_idx.returncode == 0, f"Index failed:\n{res_idx.stderr}"
        assert ckpt_path.exists()

        assert manager.collection_exists(TEST_CLI_COLLECTION)
        assert manager.count(TEST_CLI_COLLECTION) == 10

        # 2. Test CLI search with --mock (Markdown output)
        cmd_search = [
            sys.executable,
            "main.py",
            "search",
            "qonun modda",
            "--collection",
            TEST_CLI_COLLECTION,
            "--top-k",
            "3",
            "--mock",
        ]
        res_search = subprocess.run(cmd_search, capture_output=True, text=True)
        assert res_search.returncode == 0, f"Search failed:\n{res_search.stderr}"
        assert "ПРАВОВОЙ КОНТЕКСТ ДЛЯ ОТВЕТА (LEX.UZ LEGAL RAG)" in res_search.stdout
        assert "Текст нормы:" in res_search.stdout

        # 3. Test CLI search with --raw (JSON output)
        cmd_search_raw = [
            sys.executable,
            "main.py",
            "search",
            "qonun",
            "--collection",
            TEST_CLI_COLLECTION,
            "--top-k",
            "2",
            "--raw",
            "--mock",
        ]
        res_raw = subprocess.run(cmd_search_raw, capture_output=True, text=True)
        assert res_raw.returncode == 0, f"Raw search failed:\n{res_raw.stderr}"
        data = json.loads(res_raw.stdout)
        assert "reranked_results" in data

        # 4. Test CLI search with --archetype
        cmd_search_arch = [
            sys.executable,
            "main.py",
            "search",
            "qonun",
            "--collection",
            TEST_CLI_COLLECTION,
            "--archetype",
            "ARCHETYPE_A",
            "--mock",
        ]
        res_arch = subprocess.run(cmd_search_arch, capture_output=True, text=True)
        assert res_arch.returncode == 0, f"Archetype search failed:\n{res_arch.stderr}"

        # 5. Test CLI search failure on nonexistent collection
        cmd_search_fail = [
            sys.executable,
            "main.py",
            "search",
            "qonun",
            "--collection",
            "nonexistent_collection_xyz",
            "--mock",
        ]
        res_fail = subprocess.run(cmd_search_fail, capture_output=True, text=True)
        assert res_fail.returncode == 1
        assert "Search failed:" in res_fail.stderr
        assert "NameError" not in res_fail.stderr
        assert "Traceback" not in res_fail.stderr

        # 6. Test CLI index failure on invalid checkpoint path
        fail_coll = "test_cli_fail_coll"
        cmd_index_fail = [
            sys.executable,
            "main.py",
            "index",
            "--collection",
            fail_coll,
            "--checkpoint",
            "/dev/null/impossible/state.json",
            "--limit",
            "1",
            "--mock",
        ]
        res_idx_fail = subprocess.run(cmd_index_fail, capture_output=True, text=True)
        assert res_idx_fail.returncode == 1
        assert "Indexing failed:" in res_idx_fail.stderr
        assert "NameError" not in res_idx_fail.stderr
        assert "Traceback" not in res_idx_fail.stderr
        if manager.collection_exists(fail_coll):
            manager.client.delete_collection(fail_coll)

    finally:
        if manager.collection_exists(TEST_CLI_COLLECTION):
            manager.client.delete_collection(TEST_CLI_COLLECTION)
