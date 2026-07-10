import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def score_retrieval_confidence(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    similarities = []
    for chunk in chunks:
        if "similarity" in chunk:
            similarities.append(chunk["similarity"])
    if not similarities:
        return 0.5
    avg = sum(similarities) / len(similarities)
    return round(min(avg * 1.5, 1.0), 3)


def score_answer_completeness(query: str, answer: str, client: OpenAI) -> float:
    prompt = f"""Does the following answer completely address the question?

Question: {query}

Answer: {answer}

Rate completeness from 0.0 to 1.0:
1.0 = fully addresses all parts of the question
0.5 = partially addresses the question
0.0 = does not address the question at all

Respond with only a number between 0.0 and 1.0."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return round(min(max(score, 0.0), 1.0), 3)
    except ValueError:
        return 0.5


def compute_confidence(
    chunks: list[dict],
    citation_score: float,
    query: str,
    answer: str,
) -> dict:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    retrieval_score    = score_retrieval_confidence(chunks)
    completeness_score = score_answer_completeness(query, answer, client)

    weights = {
        "retrieval":    0.35,
        "citation":     0.40,
        "completeness": 0.25,
    }

    composite = (
        weights["retrieval"]    * retrieval_score +
        weights["citation"]     * citation_score +
        weights["completeness"] * completeness_score
    )

    return {
        "retrieval_score":    retrieval_score,
        "citation_score":     citation_score,
        "completeness_score": completeness_score,
        "composite_score":    round(composite, 3),
        "confident":          composite >= 0.5,
    }



if __name__ == "__main__":
    from backend.generation.generator import generate
    from backend.generation.citation_verifier import verify_all_citations

    query = "What is the rollback process if a deployment fails?"
    print(f"Query: {query}\n")

    result   = generate(query)
    answer   = result["answer"]
    chunks   = result["chunks"]

    verification = verify_all_citations(answer, chunks)
    citation_score = verification["citation_score"]

    confidence = compute_confidence(
        chunks=chunks,
        citation_score=citation_score,
        query=query,
        answer=answer,
    )

    print(f"{'='*60}")
    print("CONFIDENCE SCORES:")
    print(f"{'='*60}")
    print(f"  Retrieval confidence:  {confidence['retrieval_score']}")
    print(f"  Citation score:        {confidence['citation_score']}")
    print(f"  Completeness score:    {confidence['completeness_score']}")
    print(f"  ─────────────────────────────")
    print(f"  Composite score:       {confidence['composite_score']}")
    print(f"  Confident:             {confidence['confident']}")