"""
lexuz.harvester.download_html

Lex.uz Raw HTML Harvester.
Выкачивает сырые HTML-страницы документов на узбекском (lang=4) и русском (lang=1) языках,
сохраняет их в gzip-архивы с шардированием и ведет учет в базе SQLite.
"""

import asyncio
import gzip
import json
import logging
import os
import random
import signal
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from tqdm.asyncio import tqdm

DEFAULT_REGISTRY = Path("data/master_registry.json")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_HTML_DIR = DEFAULT_DATA_DIR / "raw_html"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "harvest_state.db"

BASE_URL = "https://lex.uz/docs/{doc_id}"
LANGUAGES = [
    (4, "uz"),  # Базовый эталон: узбекский латиница
    (1, "ru"),  # Русский перевод
]

CONCURRENCY_LIMIT = 15
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 45.0
MAX_RETRIES = 3

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
]

logger = logging.getLogger("lexuz.harvester")


def init_db(db_path: Path) -> sqlite3.Connection:
    """Инициализирует SQLite базу для отслеживания статуса выкачки с включенным WAL-режимом."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harvest_log (
                doc_id TEXT NOT NULL,
                lang INTEGER NOT NULL,
                lang_code TEXT NOT NULL,
                http_status INTEGER,
                byte_size INTEGER,
                is_empty BOOLEAN DEFAULT 0,
                file_path TEXT,
                error_message TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (doc_id, lang)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_harvest_status
                ON harvest_log(http_status, is_empty);
            """
        )
    return conn


def get_shard_name(doc_id: str) -> str:
    """
    Формирует имя шардированной папки по doc_id (безопасно для отрицательных ID).
    Например: '8548754' -> '85', '-6257288' -> 'neg_62'
    """
    clean_id = doc_id.strip()
    if clean_id.startswith("-"):
        prefix = clean_id[1:3] if len(clean_id) >= 3 else clean_id[1:]
        return f"neg_{prefix}"
    return clean_id[:2] if len(clean_id) >= 2 else clean_id


def save_html_gz(doc_id: str, lang_code: str, content: bytes, raw_html_dir: Path = DEFAULT_RAW_HTML_DIR) -> Path:
    """Сохраняет бинарный HTML в сжатом виде .html.gz с шардированием."""
    shard = get_shard_name(doc_id)
    target_dir = raw_html_dir / shard
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{doc_id}_{lang_code}.html.gz"
    file_path = target_dir / file_name

    temp_path = file_path.with_suffix(".tmp")
    with gzip.open(temp_path, "wb", compresslevel=6) as f:
        f.write(content)
    temp_path.replace(file_path)

    return file_path


def load_registry(json_path: Path = DEFAULT_REGISTRY) -> List[str]:
    """Считывает doc_id из master_registry.json."""
    if not json_path.exists():
        raise FileNotFoundError(f"Реестр не найден по пути: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("documents", []) if isinstance(data, dict) else data

    doc_ids = []
    for item in items:
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id", "")).strip()
            if doc_id:
                doc_ids.append(doc_id)

    unique_ids = list(dict.fromkeys(doc_ids))
    logger.info(f"Загружено {len(unique_ids)} уникальных актов из реестра.")
    return unique_ids


def get_completed_tasks(conn: sqlite3.Connection) -> set:
    """Возвращает множество уже успешно обработанных пар (doc_id, lang)."""
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, lang FROM harvest_log WHERE http_status = 200")
    return {(row[0], row[1]) for row in cursor.fetchall()}

