from sentence_transformers import CrossEncoder
from backend.retrieval.fusion import rrf_fuse

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query: str, top_k: int = 5) -> list[dict]:
    print("  Loading cross-encoder model...")
    model = CrossEncoder(MODEL_NAME)

    candidates = rrf_fuse(query, n_results=20)

    if not candidates:
        return []

    pairs = [(query, candidate["text"]) for candidate in candidates]

    print(f"  Scoring {len(pairs)} candidate chunks...")
    scores = model.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["cross_encoder_score"] = round(float(scores[i]), 4)
        candidate["rrf_rank"] = candidate["rank"]

    reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

    for rank, chunk in enumerate(reranked[:top_k]):
        chunk["rank"] = rank + 1

    return reranked[:top_k]



if __name__ == "__main__":
    query = "What is the rollback process if a deployment fails?"

    print(f"Query: {query}")
    print()

    results = rerank(query, top_k=5)

    print(f"\nFinal reranked results (top 5):")
    print("=" * 60)

    for result in results:
        print(f"\nRank {result['rank']} — cross-encoder: {result['cross_encoder_score']}")
        print(f"RRF rank was: {result['rrf_rank']}")
        print(f"Source:       {result['source']}")
        print(f"Preview:      {result['text'][:120].replace(chr(10), ' ')}")