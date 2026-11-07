"""
lexuz.common.models

Канонические структуры данных и схемы для Legal RAG Lex.uz.
Определяет контракт чанка для Dense Retrieval (BGE-M3), BM25 и GraphRAG.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


class CitationDict(TypedDict):
    target_doc_id: str
    target_anchor: str
    raw_ref_text: str


class StructuralUnitDict(TypedDict):
    type: str  # article, clause, preamble, appendix_header, section, chapter
    number: str
    title: str


class ChunkMetadataDict(TypedDict):
    anchors: List[str]
    okoz_codes: List[str]
    citations: List[CitationDict]
    has_table: bool
    original_script: Literal["lat", "cyr"]


class ChunkDict(TypedDict):
    chunk_id: str
    doc_id: str
    lang: str
    doc_number: str
    doc_title: str
    archetype: str
    breadcrumb: List[str]
    structural_unit: StructuralUnitDict
    content: str
    metadata: ChunkMetadataDict


@dataclass
class DocumentMetadata:
    doc_id: str
    doc_title: str
    doc_number: str
    okoz_codes: List[str] = field(default_factory=list)

