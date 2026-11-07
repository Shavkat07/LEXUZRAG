"""
lexuz.retrieval.service

Единый фасад для поискового сервиса Legal RAG:
- Интегрирует HybridRetriever (Qdrant RRF)
- Интегрирует BGEReranker (Cross-Encoder)
- Интегрирует ParentDocumentContextBuilder
"""

import logging
from typing import Any, Dict, List, Optional, Union

from lexuz.retrieval.context_builder import ParentDocumentContextBuilder
from lexuz.retrieval.hybrid import HybridRetriever
from lexuz.retrieval.models import RerankedResult, SearchResult
from lexuz.retrieval.reranker import BGEReranker

logger = logging.getLogger("lexuz.retrieval.service")


class LegalSearchService:
    """Сервис высокоточного правового поиска с реранкингом и сборкой контекста."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Any] = None,
        context_builder: Optional[ParentDocumentContextBuilder] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or BGEReranker()
        self.context_builder = context_builder or ParentDocumentContextBuilder()

    def search(
        self,
        query: str,
        lang: Optional[str] = None,
        doc_id: Optional[Union[str, int]] = None,
        okoz_code: Optional[str] = None,
        unit_type: Optional[str] = None,
        archetype: Optional[str] = None,
        top_k: int = 5,
        candidates_k: int = 30,
        use_reranker: bool = True,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Полный цикл Legal RAG поиска:
        1. Гибридный поиск Qdrant (Dense + Sparse + RRF) -> Top candidates_k
        2. Кросс-энкодер реранкинг -> Top top_k
        3. Сборка контекста родительских документов
        """
        # 1. Первичный гибридный поиск
        candidates = self.retriever.search(
            query=query,
            limit=candidates_k,
            lang=lang,
            doc_id=doc_id,
            okoz_code=okoz_code,
            unit_type=unit_type,
            archetype=archetype,
        )

        if not candidates:
            return {
                "query": query,
                "total_candidates": 0,
                "reranked_results": [],
                "rag_context": self.context_builder.build_rag_context([], query=query),
            }

        # 2. Реранкинг
        if use_reranker and self.reranker is not None:
            reranked = self.reranker.rerank(
                query=query,
                results=candidates,
                top_k=top_k,
                as_of_date=as_of_date,
            )
        else:
            reranked = [
                RerankedResult(
                    search_result=cand,
                    rerank_score=cand.score,
                    rank=i + 1,
                )
                for i, cand in enumerate(candidates[:top_k])
            ]

        # 3. Сборка контекста
        rag_context = self.context_builder.build_rag_context(
            reranked_results=reranked,
            query=query,
        )

        return {
            "query": query,
            "total_candidates": len(candidates),
            "reranked_results": reranked,
            "rag_context": rag_context,
        }
