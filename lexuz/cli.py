"""
lexuz.cli

Единая консольная точка входа для Legal RAG Lex.uz.
Команды:
  parse    - запуск пакетного парсинга сырых HTML-документов в структурированные чанки
  audit    - аудит качества, метрик и распределения длин сформированного индекса
  profile  - профилирование DOM и классификация актов (Архетипы A, B, C)
  registry - построение генерального реестра документов из выгрузок ОКОЗ
  index    - пакетная векторизация корпуса (BGE-M3 Dense + Sparse) и заливка в Qdrant
  search   - гибридный поиск (Dense + Sparse + RRF) с реранкингом и сборкой контекста
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Enable expandable segments for PyTorch CUDA allocator to prevent VRAM fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lexuz.pipeline.auditor import run_audit
from lexuz.pipeline.batch_parse import run_batch_parsing
from lexuz.pipeline.profiler import run_profiler
from lexuz.harvester.registry import build_master_registry

logger = logging.getLogger("lexuz.cli")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="lexuz",
        description="Legal RAG Lex.uz CLI Tool - парсинг, векторизация, индексация и поиск по законодательству Узбекистана"
    )
    subparsers = parser.add_subparsers(dest="command", help="Команда для выполнения")

    # parse
    parse_p = subparsers.add_parser("parse", help="Пакетный парсинг сырых HTML-документов в структурированные чанки")
    parse_p.add_argument("--archetype", choices=["a", "b", "all"], default="a", help="Целевой архетип для парсинга (a, b или all)")
    parse_p.add_argument("--raw-dir", type=Path, default=Path("data/raw_html"), help="Путь к raw_html")
    parse_p.add_argument("--output", type=Path, default=None, help="Выходной файл JSONL.GZ")
    parse_p.add_argument("--workers", type=int, default=None, help="Количество параллельных процессов")
    parse_p.add_argument("--limit", type=int, default=None, help="Ограничить количество обрабатываемых файлов (для тестов)")

    # audit
    audit_p = subparsers.add_parser("audit", help="Глубокий аудит сформированного корпуса чанков")
    audit_p.add_argument("--chunks", type=Path, default=Path("data/chunks_archetype_a.jsonl.gz"), help="Путь к чанкам")

    # profile
    prof_p = subparsers.add_parser("profile", help="Профилирование DOM структуры корпуса")
    prof_p.add_argument("--raw-dir", type=Path, default=Path("data/raw_html"), help="Путь к raw_html")
    prof_p.add_argument("--report", type=Path, default=Path("data/dom_inventory_report.json"), help="Выходной JSON отчет")

    # registry
    reg_p = subparsers.add_parser("registry", help="Сборка master_registry.json из выгрузок ОКОЗ")
    reg_p.add_argument("--exports", type=Path, default=Path("data/okoz_exports"), help="Папка с .doc файлами")
    reg_p.add_argument("--tree", type=Path, default=Path("data/okoz_leaf_categories.json"), help="Файл категорий ОКОЗ")
    reg_p.add_argument("--output", type=Path, default=Path("data/master_registry.json"), help="Выходной файл реестра")

    # index
    idx_p = subparsers.add_parser("index", help="Пакетная векторизация (BGE-M3) и индексация корпуса в Qdrant")
    idx_p.add_argument("--input", type=Path, nargs="+", default=None, help="Пути к входным .jsonl.gz файлам")
    idx_p.add_argument("--batch-size", type=int, default=64, help="Размер батча для заливки в Qdrant (по умолчанию 64)")
    idx_p.add_argument("--encode-batch-size", type=int, default=32, help="Размер подбатча для векторизации на GPU (по умолчанию 32)")
    idx_p.add_argument("--max-length", type=int, default=1024, help="Максимальная длина последовательности токенов (по умолчанию 1024)")
    idx_p.add_argument("--device", type=str, default=None, help="Устройство: cuda или cpu (по умолчанию автоопределение)")
    idx_p.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию и сбросить чекпоинт")
    idx_p.add_argument("--checkpoint", type=Path, default=Path("data/indexing_state.json"), help="Путь к файлу чекпоинта")
    idx_p.add_argument("--save-interval", type=int, default=1000, help="Шаг сохранения контрольной точки (в чанках, по умолчанию 1000)")
    idx_p.add_argument("--collection", type=str, default="lexuz_collection", help="Имя целевой коллекции в Qdrant")
    idx_p.add_argument("--limit", type=int, default=None, help="Ограничить количество индексируемых чанков (для тестирования)")
    idx_p.add_argument("--mock", action="store_true", help="Использовать MockEmbedder вместо BGE-M3 (для тестов)")

    # search
    search_p = subparsers.add_parser("search", help="Высокоточный гибридный поиск по правовой базе")
    search_p.add_argument("query", type=str, help="Поисковый запрос на узбекском или русском языке")
    search_p.add_argument("--mock", action="store_true", help="Использовать MockEmbedder и MockReranker (для тестов)")
    search_p.add_argument("--lang", choices=["uz", "ru"], default=None, help="Фильтрация по языку (uz или ru)")
    search_p.add_argument("--doc-id", type=str, default=None, help="Фильтрация по конкретному документу (doc_id)")
    search_p.add_argument("--okoz", type=str, default=None, help="Фильтрация по коду классификатора ОКОЗ")
    search_p.add_argument("--unit-type", type=str, default=None, help="Фильтрация по типу нормы (article, clause, preamble)")
    search_p.add_argument("--archetype", choices=["a", "b", "ARCHETYPE_A", "ARCHETYPE_B"], default=None, help="Фильтрация по архетипу")
    search_p.add_argument("--top-k", type=int, default=5, help="Количество финальных статей после реранкинга")
    search_p.add_argument("--candidates", type=int, default=30, help="Количество кандидатов первого этапа гибридного поиска")
    search_p.add_argument("--collection", type=str, default="lexuz_collection", help="Имя коллекции в Qdrant")
    search_p.add_argument("--no-rerank", action="store_true", help="Отключить кросс-энкодер реранкер")
    search_p.add_argument("--raw", action="store_true", help="Вывести сырой JSON вместо форматированного Markdown")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "parse":
        output_file = args.output
        if not output_file:
            if args.archetype == "a":
                output_file = Path("data/chunks_archetype_a.jsonl.gz")
            elif args.archetype == "b":
                output_file = Path("data/chunks_archetype_b.jsonl.gz")
            else:
                output_file = Path("data/chunks_all.jsonl.gz")

        errors_file = Path(f"data/errors_archetype_{args.archetype}.jsonl")

        run_batch_parsing(
            raw_dir=args.raw_dir,
            output_path=output_file,
            errors_path=errors_file,
            archetype=args.archetype,
            workers=args.workers,
            limit=args.limit,
        )
    elif args.command == "audit":
        run_audit(chunks_path=args.chunks)
    elif args.command == "profile":
        run_profiler(raw_dir=args.raw_dir, report_path=args.report)
    elif args.command == "registry":
        build_master_registry(exports_dir=args.exports, tree_file=args.tree, output_file=args.output)
    elif args.command == "index":
        from lexuz.indexer.pipeline import run_indexing
        embedder = None
        if args.mock:
            from lexuz.indexer.embedder import MockEmbedder
            embedder = MockEmbedder()
        else:
            from lexuz.indexer.embedder import BGEM3Embedder
            embedder = BGEM3Embedder(
                device=args.device,
                use_fp16=True,
                max_length=args.max_length,
                batch_size=args.encode_batch_size,
            )

        try:
            run_indexing(
                input_files=args.input,
                collection_name=args.collection,
                batch_size=args.batch_size,
                device=args.device,
                recreate=args.recreate,
                checkpoint_path=args.checkpoint,
                save_interval=args.save_interval,
                limit=args.limit,
                embedder=embedder,
            )
        except Exception as e:
            logger.error("Indexing failed: %s", e)
            sys.exit(1)
    elif args.command == "search":
        from lexuz.retrieval.hybrid import HybridRetriever
        from lexuz.retrieval.service import LegalSearchService
        from lexuz.storage.qdrant_manager import QdrantManager

        embedder = None
        reranker = None
        if args.mock:
            from lexuz.indexer.embedder import MockEmbedder
            from lexuz.retrieval.reranker import MockReranker
            embedder = MockEmbedder()
            reranker = MockReranker()

        qdrant_manager = QdrantManager(host="localhost", port=6333, grpc_port=6334)
        retriever = HybridRetriever(
            qdrant_manager=qdrant_manager,
            embedder=embedder,
            collection_name=args.collection,
        )
        service = LegalSearchService(retriever=retriever, reranker=reranker)

        try:
            res = service.search(
                query=args.query,
                lang=args.lang,
                doc_id=args.doc_id,
                okoz_code=args.okoz,
                unit_type=args.unit_type,
                archetype=args.archetype,
                top_k=args.top_k,
                candidates_k=args.candidates,
                use_reranker=not args.no_rerank,
            )
        except Exception as e:
            logger.error("Search failed: %s", e)
            sys.exit(1)

        if args.raw:
            serializable = {
                "query": res["query"],
                "total_candidates": res["total_candidates"],
                "reranked_results": [
                    {
                        "rank": r.rank,
                        "score": r.rerank_score,
                        "chunk_id": r.search_result.chunk_id,
                        "doc_id": r.search_result.doc_id,
                        "doc_number": r.search_result.doc_number,
                        "doc_title": r.search_result.doc_title,
                        "breadcrumb": r.search_result.breadcrumb,
                        "content": r.search_result.content,
                    }
                    for r in res["reranked_results"]
                ],
            }
            print(json.dumps(serializable, indent=2, ensure_ascii=False))
        else:
            print(res["rag_context"])


if __name__ == "__main__":
    main()
