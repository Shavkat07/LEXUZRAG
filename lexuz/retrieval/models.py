"""
lexuz.retrieval.models

Модели данных для результатов поиска и реранкинга.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Результат первого этапа поиска (Qdrant Hybrid RRF)."""

    point_id: str
    chunk_id: str
    score: float
    doc_id: str
    lang: str
    doc_number: str
    doc_title: str
    archetype: str
    unit_type: str
    unit_number: str
    unit_title: str
    breadcrumb: List[str]
    okoz_codes: List[str]
    has_table: bool
    original_script: str
    anchors: List[str]
    citations_count: int
    content: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankedResult:
    """Результат второго этапа поиска (Cross-Encoder Reranking)."""

    search_result: SearchResult
    rerank_score: float
    rank: int
