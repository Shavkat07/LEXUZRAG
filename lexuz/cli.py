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

    # ask
    ask_p = subparsers.add_parser("ask", help="Интерактивный вопрос к юридическому AI-агенту через LLM (Gemini)")
    ask_p.add_argument("question", type=str, help="Юридический вопрос на узбекском или русском языке")
    ask_p.add_argument("--top-k", type=int, default=5, help="Количество подтверждающих норм (по умолчанию 5)")
    ask_p.add_argument("--as-of-date", type=str, default=None, help="Дата среза законодательства (ГГГГ-ММ-ДД)")
    ask_p.add_argument("--lang", choices=["auto", "uz-Latn", "uz-Cyrl", "ru"], default="auto", help="Целевой язык ответа")
    ask_p.add_argument("--style", choices=["simple", "formal"], default="simple", help="Стиль ответа: 'simple' (простой) или 'formal' (строгий юридический)")
    ask_p.add_argument("--mock", action="store_true", help="Использовать MockSearchService для тестирования")
    ask_p.add_argument("--raw", action="store_true", help="Вывести сырой JSON ответа вместо форматированного Markdown")

    # serve
    serve_p = subparsers.add_parser("serve", help="Запуск FastAPI веб-сервера и REST API")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Хост для привязки (по умолчанию 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8000, help="Порт для привязки (по умолчанию 8000)")
    serve_p.add_argument("--reload", action="store_true", help="Включить автоперезагрузку при изменении кода")
    serve_p.add_argument("--workers", type=int, default=1, help="Количество worker процессов")

    # eval
    eval_p = subparsers.add_parser("eval", help="Оценка качества RAG пайплайна по Разделу 6 (100 вопросов)")
    eval_p.add_argument("--dataset", type=Path, default=Path("data/eval_testset_100.csv"), help="Путь к CSV файлу эталонов")
    eval_p.add_argument("--batch", type=str, default=None, help="Выполнить только указанный батч (например b01)")
    eval_p.add_argument("--output", type=Path, default=None, help="Путь для сохранения отчета (Markdown/JSON)")
    eval_p.add_argument("--mock", action="store_true", help="Использовать MockSearchService для тестирования пайплайна")

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
    elif args.command == "ask":
        from lexuz.api.engine import RAGEngine, MockSearchService
        from lexuz.api.schemas import OptionsRequest

        engine = RAGEngine(search_service=MockSearchService() if args.mock else None)
        options = OptionsRequest(
            top_k=args.top_k,
            return_contexts=True,
            return_debug=True,
            as_of_date=args.as_of_date,
            answer_language=args.lang,
            style=args.style,
        )

        res = engine.process_item(
            item_id="cli_q1",
            question=args.question,
            item_language=None if args.lang == "auto" else args.lang,
            options=options,
        )

        if args.raw:
            print(res.model_dump_json(indent=2))
        else:
            print("\n" + "=" * 70)
            print("⚖️  LEX.UZ LEGAL RAG AI JAVOBI (ЮРИДИЧЕСКАЯ КОНСУЛЬТАЦИЯ)")
            print("=" * 70)
            print(f"Savol: {args.question}")
            if res.debug and res.debug.rewritten_query and res.debug.rewritten_query != args.question:
                print(f"Yuridik so'rov (Rewritten): {res.debug.rewritten_query}")
            conf_val = int((res.confidence or 0) * 100)
            print(f"Holati: {res.status.upper()} | Uslub: {args.style.upper()} | Ishonchlilik: {conf_val}% | Til: {res.answer_language}\n")
            print(f"Javob:\n{res.answer}\n")
            if res.citations:
                print("-" * 70)
                print("📚 Havola qilingan normalar (Официальные источники):")
                for cit in res.citations:
                    print(f"[{cit.ref}] {cit.doc_title} ({cit.article or cit.chunk_id})")
                    if cit.url:
                        print(f"    URL: {cit.url}")
            print("=" * 70 + "\n")
    elif args.command == "serve":
        import uvicorn
        logger.info(f"Starting Lex.uz Legal RAG FastAPI server on {args.host}:{args.port}...")
        uvicorn.run(
            "lexuz.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
        )
    elif args.command == "eval":
        from lexuz.eval.evaluator import RAGEvaluator, load_benchmark_from_csv, generate_default_100_benchmark, save_benchmark_to_csv
        from lexuz.api.engine import RAGEngine, MockSearchService
        from lexuz.api.schemas import OptionsRequest

        csv_file = args.dataset
        if not csv_file.exists():
            logger.info("Generating default 100 benchmark testset at %s", csv_file)
            items = generate_default_100_benchmark()
            save_benchmark_to_csv(items, csv_file)
        else:
            items = load_benchmark_from_csv(csv_file)

        if args.batch:
            items = [it for it in items if it.batch == args.batch]
            logger.info("Evaluating batch %s (%d questions)", args.batch, len(items))
        else:
            logger.info("Evaluating full testset (%d questions)", len(items))

        engine = RAGEngine(search_service=MockSearchService() if args.mock else None)
        options = OptionsRequest(top_k=5, return_contexts=True, return_debug=True)

        results = []
        for it in items:
            res = engine.process_item(
                item_id=it.id,
                question=it.question,
                item_language=None,
                options=options,
            )
            results.append(res.model_dump())

        summary = RAGEvaluator.evaluate(items, results)
        md_report = summary.to_markdown()
        print("\n" + md_report + "\n")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(md_report, encoding="utf-8")
            logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
