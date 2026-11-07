"""
lexuz.retrieval.hybrid

Модуль гибридного поиска по коллекции Qdrant:
- Совместный поиск по плотным векторам (Dense Cosine) и разреженным весам (Sparse IDF)
- Слияние рангов по алгоритму Reciprocal Rank Fusion (RRF)
- Фильтрация по языку (lang), нормативному акту (doc_id), классификатору (okoz_codes) и типу нормы (unit_type)
"""

import logging
from typing import Any, Dict, List, Optional, Union

from qdrant_client import models

from lexuz.indexer.embedder import BGEM3Embedder
from lexuz.retrieval.models import SearchResult
from lexuz.storage.qdrant_manager import QdrantManager, DEFAULT_COLLECTION_NAME

logger = logging.getLogger("lexuz.retrieval.hybrid")


class HybridRetriever:
    """Движок гибридного поиска по нормативно-правовым актам Lex.uz."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        embedder: Optional[Any] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.manager = qdrant_manager or QdrantManager()
        self.embedder = embedder or BGEM3Embedder()
        self.collection_name = collection_name

    def build_filter(
        self,
        lang: Optional[str] = None,
        doc_id: Optional[Union[str, int]] = None,
        okoz_code: Optional[str] = None,
        unit_type: Optional[str] = None,
        archetype: Optional[str] = None,
    ) -> Optional[models.Filter]:
        """Формирует фильтр Qdrant по метаданным полезной нагрузки (payload)."""
        must_conditions: List[models.Condition] = []

        if lang:
            must_conditions.append(
                models.FieldCondition(
                    key="lang",
                    match=models.MatchValue(value=lang),
                )
            )

        if doc_id:
            raw_id = str(doc_id).strip()
            clean_id = raw_id.lstrip("-")
            neg_id = f"-{clean_id}"
            match_variants = list(dict.fromkeys([raw_id, neg_id, clean_id]))
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=match_variants),
                )
            )

        if okoz_code:
            must_conditions.append(
                models.FieldCondition(
                    key="okoz_codes",
                    match=models.MatchValue(value=okoz_code),
                )
            )

        if unit_type:
            must_conditions.append(
                models.FieldCondition(
                    key="unit_type",
                    match=models.MatchValue(value=unit_type),
                )
            )

        if archetype:
            arch_norm = archetype.strip().upper()
            if not arch_norm.startswith("ARCHETYPE_"):
                arch_norm = f"ARCHETYPE_{arch_norm}"
            must_conditions.append(
                models.FieldCondition(
                    key="archetype",
                    match=models.MatchValue(value=arch_norm),
                )
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    def search(
        self,
        query: str,
        limit: int = 30,
        prefetch_limit: Optional[int] = None,
        lang: Optional[str] = None,
        doc_id: Optional[Union[str, int]] = None,
        okoz_code: Optional[str] = None,
        unit_type: Optional[str] = None,
        archetype: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Выполняет гибридный поиск по запросу пользователя с использованием RRF.
        
        Dense ветка находит семантический контекст и синонимы.
        Sparse ветка находит точные совпадения реквизитов (статьи, номера актов, коды).
        """
        if not query.strip():
            return []

        if not self.manager.collection_exists(self.collection_name):
            logger.error("Collection '%s' does not exist in Qdrant.", self.collection_name)
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in Qdrant. "
                f"Please index documents first with 'lexuz index'."
            )

        # 1. Векторизация запроса
        encoded = self.embedder.encode_queries([query])
        dense_vec = encoded["dense"][0]
        sparse_vec = encoded["sparse"][0]

        # 2. Построение фильтра
        query_filter = self.build_filter(
            lang=lang,
            doc_id=doc_id,
            okoz_code=okoz_code,
            unit_type=unit_type,
            archetype=archetype,
        )

        sub_limit = prefetch_limit or max(limit * 2, 50)

        # 3. Гибридный RRF запрос
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                filter=query_filter,
                limit=sub_limit,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                filter=query_filter,
                limit=sub_limit,
            ),
        ]

        response = self.manager.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        results: List[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            res = SearchResult(
                point_id=str(point.id),
                chunk_id=payload.get("chunk_id", ""),
                score=point.score,
                doc_id=str(payload.get("doc_id", "")),
                lang=payload.get("lang", ""),
                doc_number=payload.get("doc_number", ""),
                doc_title=payload.get("doc_title", ""),
                archetype=payload.get("archetype", ""),
                unit_type=payload.get("unit_type", ""),
                unit_number=str(payload.get("unit_number", "")),
                unit_title=payload.get("unit_title", ""),
                breadcrumb=payload.get("breadcrumb", []),
                okoz_codes=payload.get("okoz_codes", []),
                has_table=payload.get("has_table", False),
                original_script=payload.get("original_script", "lat"),
                anchors=payload.get("anchors", []),
                citations_count=payload.get("citations_count", 0),
                content=payload.get("content", ""),
                payload=payload,
            )
            results.append(res)

        return results
