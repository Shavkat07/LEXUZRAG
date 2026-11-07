"""
lexuz.retrieval.context_builder

Модуль сборки контекста родительского документа (Parent-Document Context Assembly).
Собирает изолированные нормативные единицы в связанный правовой контекст:
- Полная иерархическая цепочка (Акт -> Приложение -> Глава -> Статья -> Пункт)
- Официальные реквизиты документа (Номер, Название, ОКОЗ)
- Ссылки на Lex.uz с точными абзацными якорями
- Форматирование готового промпта для LLM
"""

from typing import Any, Dict, List, Optional

from lexuz.retrieval.models import RerankedResult, SearchResult


class ParentDocumentContextBuilder:
    """Формирует канонический правовой контекст для LLM."""

    @staticmethod
    def build_lexuz_url(doc_id: str, anchors: List[str]) -> str:
        """Формирует прямую ссылку на документ и конкретный абзац на портале Lex.uz."""
        clean_doc_id = doc_id.lstrip("-")
        anchor_part = f"#{anchors[0]}" if (anchors and anchors[0]) else ""
        return f"https://lex.uz/docs/{clean_doc_id}{anchor_part}"

    def format_single_norm(
        self,
        res: SearchResult,
        rerank_score: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> str:
        """Форматирует одну найденную норму со всеми реквизитами и иерархией."""
        rank_str = f"#{rank} " if rank is not None else ""
        score_str = f" [Score: {rerank_score:.4f}]" if rerank_score is not None else ""
        
        doc_header = res.doc_title
        if res.doc_number:
            doc_header = f"{res.doc_title} (№ {res.doc_number})" if res.doc_title else f"№ {res.doc_number}"
        elif not doc_header:
            doc_header = f"Акт {res.doc_id}"

        hierarchy_str = " > ".join(res.breadcrumb) if res.breadcrumb else "Общая часть"
        url = self.build_lexuz_url(res.doc_id, res.anchors)

        unit_label = f"{res.unit_type} {res.unit_number}".strip()
        if res.unit_title:
            unit_label = f"{unit_label} ({res.unit_title})" if unit_label else res.unit_title

        lines = [
            f"### {rank_str}{doc_header}{score_str}",
            f"- **Иерархия (Breadcrumb):** {hierarchy_str}",
            f"- **Структурная единица:** {unit_label or 'Не указана'}",
            f"- **Отрасль права (ОКОЗ):** {', '.join(res.okoz_codes) if res.okoz_codes else 'Не указан'}",
            f"- **Официальный источник:** {url}",
            "",
            "#### Текст нормы:",
            res.content.strip(),
            "",
            "---",
        ]
        return "\n".join(lines)

    def build_rag_context(
        self,
        reranked_results: List[RerankedResult],
        query: Optional[str] = None,
    ) -> str:
        """
        Собирает полный правовой контекст из списка переранжированных результатов.
        Готов к подаче в System / Context prompt больших языковых моделей.
        """
        if not reranked_results:
            return "ПО ДАННОМУ ЗАПРОСУ НЕ НАЙДЕНО РЕЛЕВАНТНЫХ НОРМАТИВНЫХ АКТОВ."

        header_lines = [
            "# ПРАВОВОЙ КОНТЕКСТ ДЛЯ ОТВЕТА (LEX.UZ LEGAL RAG)",
        ]
        if query:
            header_lines.append(f"**Запрос:** {query}")
        header_lines.append(f"**Количество подтверждающих норм:** {len(reranked_results)}\n")

        norm_blocks = []
        for r in reranked_results:
            norm_str = self.format_single_norm(
                res=r.search_result,
                rerank_score=r.rerank_score,
                rank=r.rank,
            )
            norm_blocks.append(norm_str)

        return "\n".join(header_lines) + "\n" + "\n".join(norm_blocks)
