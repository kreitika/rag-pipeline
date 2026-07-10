import os
from dotenv import load_dotenv
from backend.retrieval.reranker import rerank
from backend.generation.prompt_builder import build_prompt
from backend.generation.generator import generate
from backend.generation.citation_parser import parse_citations, extract_claims
from backend.generation.citation_verifier import verify_all_citations
from backend.generation.confidence_scorer import compute_confidence

load_dotenv()

CONFIDENCE_THRESHOLD = 0.5

def idk_response(query: str, chunks: list[dict], confidence: dict) -> dict:
    if chunks:
        found_in = list(set(c["source"] for c in chunks))
        sources_str = ", ".join(found_in)
        message = (
            f"I don't have enough information in the provided documentation "
            f"to fully answer this question.\n\n"
            f"I found potentially related content in: {sources_str}\n"
            f"However, the confidence score ({confidence['composite_score']}) "
            f"was below the threshold ({CONFIDENCE_THRESHOLD}), suggesting "
            f"the retrieved content may not directly answer your question.\n\n"
            f"You may want to check these documents manually or rephrase your question."
        )
    else:
        message = (
            f"I don't have enough information in the provided documentation "
            f"to answer this question. No relevant content was found."
        )

    return {
        "query":            query,
        "answer":           message,
        "confident":        False,
        "composite_score":  confidence["composite_score"],
        "citations":        [],
        "chunks":           chunks,
    }



def full_pipeline(query: str, top_k: int = 3) -> dict:
    print(f"\nRunning RAG pipeline for: '{query}'")

    # Stage 1: Retrieve ONCE
    print("  [1/4] Retrieving chunks...")
    chunks = rerank(query, top_k=top_k)

    if not chunks:
        return idk_response(query, [], {"composite_score": 0.0})

    # Stage 2: Generate — pass chunks directly, no second rerank
    print("  [2/4] Generating answer...")
    result = generate(query, chunks=chunks)
    answer = result["answer"]

    # Stage 3: Verify citations
    print("  [3/4] Verifying citations...")
    verification = verify_all_citations(answer, chunks)
    citation_score = verification["citation_score"]

    # Stage 4: Score confidence
    print("  [4/4] Scoring confidence...")
    confidence = compute_confidence(
        chunks=chunks,
        citation_score=citation_score,
        query=query,
        answer=answer,
    )

    if not confidence["confident"]:
        print(f"  Low confidence ({confidence['composite_score']}) — returning IDK response")
        return idk_response(query, chunks, confidence)

    citations = parse_citations(answer, chunks)

    return {
        "query":           query,
        "answer":          answer,
        "confident":       True,
        "composite_score": confidence["composite_score"],
        "citations":       citations,
        "chunks":          chunks,
        "verification":    verification,
        "confidence":      confidence,
        "tokens_used":     result["tokens_used"],
    }

    

if __name__ == "__main__":
    # Test 1: Question we can answer
    print("TEST 1: Question with answer in documents")
    print("=" * 60)
    result1 = full_pipeline("What is the rollback process if a deployment fails?")
    print(f"\nAnswer: {result1['answer'][:300]}")
    print(f"Confident: {result1['confident']}")
    print(f"Score: {result1['composite_score']}")

    print()

    # Test 2: Question we cannot answer
    print("TEST 2: Question without answer in documents")
    print("=" * 60)
    result2 = full_pipeline("What is the company's parental leave policy?")
    print(f"\nAnswer: {result2['answer'][:300]}")
    print(f"Confident: {result2['confident']}")
    print(f"Score: {result2['composite_score']}")