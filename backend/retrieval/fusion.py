from backend.retrieval.dense import dense_retrieve
from backend.retrieval.sparse import sparse_retrieve

def rrf_fuse(query: str, n_results: int = 10, k: int = 60) -> list[dict]:
    dense_results = dense_retrieve(query, n_results=n_results)
    sparse_results = sparse_retrieve(query, n_results=n_results)

    rrf_scores = {}

    for result in dense_results:
        doc_id = f"{result['source']}_{result['chunk_index']}"
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "text":        result["text"],
                "source":      result["source"],
                "chunk_index": result["chunk_index"],
                "rrf_score":   0.0,
                "dense_rank":  None,
                "sparse_rank": None,
            }
        rrf_scores[doc_id]["rrf_score"] += 1 / (k + result["rank"])
        rrf_scores[doc_id]["dense_rank"] = result["rank"]

    for result in sparse_results:
        doc_id = f"{result['source']}_{result['chunk_index']}"
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "text":        result["text"],
                "source":      result["source"],
                "chunk_index": result["chunk_index"],
                "rrf_score":   0.0,
                "dense_rank":  None,
                "sparse_rank": None,
            }
        rrf_scores[doc_id]["rrf_score"] += 1 / (k + result["rank"])
        rrf_scores[doc_id]["sparse_rank"] = result["rank"]

    fused = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    for rank, chunk in enumerate(fused[:n_results]):
        chunk["rank"] = rank + 1
        chunk["rrf_score"] = round(chunk["rrf_score"], 6)

    return fused[:n_results]


if __name__ == "__main__":
    query = "What is the rollback process if a deployment fails?"

    print(f"Query: {query}")
    print(f"\nRRF Fused results (k=60):")
    print("=" * 60)

    results = rrf_fuse(query, n_results=5)

    for result in results:
        dense_rank  = result['dense_rank']  or 'not in list'
        sparse_rank = result['sparse_rank'] or 'not in list'
        print(f"\nRank {result['rank']} — RRF score: {result['rrf_score']}")
        print(f"Source:      {result['source']}")
        print(f"Dense rank:  {dense_rank}")
        print(f"Sparse rank: {sparse_rank}")
        print(f"Preview:     {result['text'][:120].replace(chr(10), ' ')}")