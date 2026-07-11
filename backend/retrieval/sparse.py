import pickle
from pathlib import Path

BM25_PATH = Path("indexes/bm25/index.pkl")

def sparse_retrieve(query: str, n_results: int = 10, bm25_path: str = None) -> list[dict]:
    path = bm25_path or str(BM25_PATH)
    with open(path, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = []
    for i, score in enumerate(scores):
        scored_chunks.append({
            "text":        chunks[i].text,
            "source":      chunks[i].source,
            "chunk_index": chunks[i].chunk_index,
            "strategy":    chunks[i].strategy,
            "bm25_score":  round(float(score), 4),
            "index":       i
        })

    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)

    for rank, chunk in enumerate(scored_chunks[:n_results]):
        chunk["rank"] = rank + 1

    return scored_chunks[:n_results]


if __name__ == "__main__":
    query = "What is the rollback process if a deployment fails?"

    print(f"Query: {query}")
    print(f"\nBM25 sparse retrieval results:")
    print("=" * 60)

    results = sparse_retrieve(query, n_results=5)

    for result in results:
        print(f"\nRank {result['rank']} — BM25 score: {result['bm25_score']}")
        print(f"Source: {result['source']}")
        print(f"Preview: {result['text'][:150].replace(chr(10), ' ')}")

    print("\n" + "=" * 60)
    print("Comparing with dense retrieval:")
    print("Dense finds: meaning and intent")
    print("BM25 finds:  exact keywords and rare terms")