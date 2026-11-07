"""
lexuz.retrieval package

Модуль гибридного поиска, кросс-энкодер реранкинга и сборки контекста:
- HybridRetriever (Dense + Sparse + RRF + Metadata Filtering)
- BGEReranker (BAAI/bge-reranker-v2-m3)
- ParentDocumentContextBuilder (иерархическая контекстная сборка норм)
- LegalSearchService (фасад для CLI и API)
"""

from lexuz.retrieval.models import SearchResult, RerankedResult
from lexuz.retrieval.hybrid import HybridRetriever
from lexuz.retrieval.reranker import BGEReranker, MockReranker
from lexuz.retrieval.context_builder import ParentDocumentContextBuilder
from lexuz.retrieval.service import LegalSearchService

__all__ = [
    "SearchResult",
    "RerankedResult",
    "HybridRetriever",
    "BGEReranker",
    "MockReranker",
    "ParentDocumentContextBuilder",
    "LegalSearchService",
]
