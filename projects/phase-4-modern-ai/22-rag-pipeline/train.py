"""
CLI entry point for the Daedalus RAG pipeline.

Examples
--------
    # Ingest + chunk only, no models loaded (fast sanity check)
    python train.py --dry-run

    # Build the vector index from the corpus
    python train.py

    # Build the index and answer one question end to end
    python train.py --question "What is a text embedding?"

    # Query an index that was already built
    python train.py --no-build --question "Why is brute-force QnA not scalable?"
"""

import argparse

from solution import (
    DaedalusRAG,
    NotebookChunker,
)

DEFAULT_CORPUS = "corpus/Retrival_Ext_QnA_Complete.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query the Daedalus RAG pipeline."
    )

    # --- corpus and chunking ---
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=2)

    # --- vector store ---
    parser.add_argument("--qdrant-path", type=str, default="./daedalus_qdrant")
    parser.add_argument("--collection", type=str, default="daedalus_chunks")
    parser.add_argument("--batch-size", type=int, default=32)

    # --- models ---
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--generation-model", type=str, default="qwen3:8b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / mps / cpu. Auto-detected when omitted.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load embedding weights from the local HF cache only.")

    # --- retrieval ---
    parser.add_argument("--top-k", type=int, default=10,
                        help="Candidates pulled from the vector store.")
    parser.add_argument("--final-top-k", type=int, default=5,
                        help="Chunks kept after reranking.")
    parser.add_argument("--temperature", type=float, default=0.1)

    # --- what to run ---
    parser.add_argument("--question", type=str, default=None,
                        help="Ask one question after the index is ready.")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip indexing and query the existing collection.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ingest and chunk only. Loads no models.")

    return parser.parse_args()


def print_chunk_stats(chunks_data) -> None:
    stats = NotebookChunker.describe(chunks_data)

    print("\nChunk statistics")
    print("-" * 40)
    print(f"Total chunks:        {stats['total_chunks']}")
    print(f"Words   min/max:     {stats['min_words']} / {stats['max_words']}")
    print(f"Words   mean/median: {stats['mean_words']} / {stats['median_words']}")
    print(f"Units   min/max:     {stats['min_units']} / {stats['max_units']}")
    print(f"Chars   mean:        {stats['mean_characters']}")


def print_answer(result: dict) -> None:
    print("\n" + "=" * 100)
    print("QUESTION")
    print("=" * 100)
    print(result["question"])

    print("\n" + "=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(result["answer"])

    print("\n" + "=" * 100)
    print("EVIDENCE USED")
    print("=" * 100)

    for chunk in result["reranked_chunks"]:
        print(
            f"\nFinal rank {chunk['reranker_rank']} | "
            f"reranker {chunk['reranker_score']:.4f} | "
            f"vector rank {chunk['rank']} ({chunk['retrieval_score']:.4f}) | "
            f"{chunk['source']} cell {chunk['cell_number']} ({chunk['cell_type']})"
        )
        print(chunk["preview"].replace("\n", " ")[:300])

    print(f"\nGeneration time: {round(result['generation_time'], 2)} seconds")


def main() -> None:
    args = parse_args()

    rag = DaedalusRAG(
        corpus_path=args.corpus,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        max_words=args.max_words,
        sentence_overlap=args.overlap,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        generation_model=args.generation_model,
        ollama_url=args.ollama_url,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    try:
        # ---------- ingest + chunk only ----------
        if args.dry_run:
            chunks_data = rag.ingest()
            print_chunk_stats(chunks_data)
            return

        # ---------- build the vector index ----------
        if not args.no_build:
            rag.build_index(batch_size=args.batch_size)
            print_chunk_stats(rag.chunks_data)

        # ---------- answer one question ----------
        if args.question:
            rag.generator.check_connection()

            result = rag.answer(
                question=args.question,
                retrieve_top_k=args.top_k,
                final_top_k=args.final_top_k,
                temperature=args.temperature,
            )
            print_answer(result)

        elif args.no_build:
            print(
                "Nothing to do: --no-build was passed without --question.\n"
                "Pass --question \"...\" to query the existing collection."
            )

    finally:
        # Always release the storage lock, otherwise the next run fails with
        # "Storage folder is already accessed by another instance".
        rag.close()


if __name__ == "__main__":
    main()
