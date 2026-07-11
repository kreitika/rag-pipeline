import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

QUESTIONS_PATH   = Path("evals/golden_qa/questions.json")
COMPARISON_PATH  = Path("evals/strategy_comparison.json")


def run_strategy_eval(strategy: str, max_questions: int = 20) -> dict:
    from backend.ingestion.loader import load_document
    from backend.ingestion.chunker import chunk_fixed, chunk_by_section
    from backend.ingestion.deduplicator import Deduplicator
    from backend.generation.pipeline import full_pipeline
    from backend.evaluation.evaluator import score_correctness, score_idk_correctness

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    print(f"\nBuilding index for strategy: {strategy}")

    import chromadb
    import pickle
    from rank_bm25 import BM25Okapi

    DOCUMENT_FILES = [
        "data/raw/sample.md",
        "data/raw/sample.txt",
        "data/raw/sample.html",
        "data/raw/sample.pdf",
    ]

    chunker_map = {
        "fixed":   chunk_fixed,
        "section": chunk_by_section,
    }
    chunker_fn = chunker_map[strategy]

    documents   = [load_document(f) for f in DOCUMENT_FILES]
    all_chunks  = []
    for doc in documents:
        chunks = chunker_fn(doc)
        all_chunks.extend(chunks)

    deduplicator  = Deduplicator(threshold=0.95)
    unique_chunks = [c for c in all_chunks if deduplicator.add(c)]

    print(f"  Chunks created: {len(unique_chunks)}")

    response = client.embeddings.create(
        input=[c.text for c in unique_chunks],
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]

    chroma_path = f"indexes/chroma_{strategy}"
    bm25_path   = f"indexes/bm25/index_{strategy}.pkl"

    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    try:
        chroma_client.delete_collection("rag_chunks")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="rag_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        embeddings=embeddings,
        documents=[c.text for c in unique_chunks],
        metadatas=[{
            "source":      c.source,
            "chunk_index": c.chunk_index,
            "strategy":    c.strategy,
            "char_count":  c.char_count,
        } for c in unique_chunks],
        ids=[f"{c.source}_{c.chunk_index}_{c.strategy}"
             for c in unique_chunks]
    )

    tokenized = [c.text.lower().split() for c in unique_chunks]
    bm25      = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": unique_chunks}, f)

    print(f"  Index built. Running {max_questions} questions...")

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)[:max_questions]

    scores     = []
    idk_scores = []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['id']}...", end="", flush=True)

        # Pass index paths directly — no module variable hacking
        result = full_pipeline(
            q["question"],
            chroma_dir=chroma_path,
            bm25_path=bm25_path,
        )
        answer = result["answer"]

        if q["answer_exists"]:
            cr = score_correctness(
                q["question"], q["expected_answer"], answer
            )
            scores.append(cr["score"] / cr["max_score"])
            print(f" score={cr['score']}/2")
        else:
            idk = score_idk_correctness(answer, q["answer_exists"])
            idk_scores.append(idk["score"] / idk["max_score"])
            print(f" idk={'✓' if idk['score']==2 else '✗'}")

        time.sleep(1)

    avg_correctness = sum(scores) / len(scores) if scores else 0
    avg_idk         = sum(idk_scores) / len(idk_scores) if idk_scores else 0

    return {
        "strategy":           strategy,
        "chunks_created":     len(unique_chunks),
        "questions_tested":   len(questions),
        "answer_correctness": round(avg_correctness, 3),
        "idk_accuracy":       round(avg_idk, 3),
    }




if __name__ == "__main__":
    print("Chunking Strategy Comparison")
    print("Testing all 50 questions per strategy")
    print("=" * 60)

    results = {}

    for strategy in ["fixed", "section"]:
        result = run_strategy_eval(strategy, max_questions=50)
        results[strategy] = result
        results[strategy] = result

    with open(COMPARISON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Strategy':<12} {'Chunks':<10} {'Correctness':<15} {'IDK Acc'}")
    print("-" * 50)
    for strategy, r in results.items():
        print(f"{r['strategy']:<12} {r['chunks_created']:<10} {r['answer_correctness']:.1%}{'':>8} {r['idk_accuracy']:.1%}")

    print(f"\nResults saved to: {COMPARISON_PATH}")