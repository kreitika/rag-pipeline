import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.generation.citation_parser import parse_citations, extract_claims

load_dotenv()


def verify_citation(claim: str, chunk_text: str, client: OpenAI) -> dict:
    prompt = f"""Does the following source text support the claim?

Source text:
{chunk_text}

Claim: {claim}

Answer with exactly one of:
SUPPORTED - the source text directly supports this claim
PARTIAL - the source text partially supports this claim
UNSUPPORTED - the source text does not support this claim

Then on a new line, explain in one sentence why."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result_text = response.choices[0].message.content.strip()
    first_line = result_text.split('\n')[0].strip().upper()

    if "SUPPORTED" in first_line and "PARTIAL" not in first_line:
        verdict = "SUPPORTED"
    elif "PARTIAL" in first_line:
        verdict = "PARTIAL"
    else:
        verdict = "UNSUPPORTED"

    return {
        "claim":    claim,
        "verdict":  verdict,
        "explanation": result_text,
    }



def verify_all_citations(answer: str, chunks: list[dict]) -> dict:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        citations = parse_citations(answer, chunks)
        claims = extract_claims(answer)

        citation_map = {c["citation_number"]: c["chunk_text"] for c in citations}

        results = []
        for claim in claims:
            for cited_num in claim["cited_numbers"]:
                if cited_num in citation_map:
                    chunk_text = citation_map[cited_num]
                    verification = verify_citation(
                    claim["claim"],
                    chunk_text,
                    client
                    )
                    verification["citation_number"] = cited_num
                    results.append(verification)

        supported   = sum(1 for r in results if r["verdict"] == "SUPPORTED")
        partial     = sum(1 for r in results if r["verdict"] == "PARTIAL")
        unsupported = sum(1 for r in results if r["verdict"] == "UNSUPPORTED")
        total       = len(results)

        citation_score = (supported + 0.5 * partial) / total if total > 0 else 0.0

        return {
        "results":          results,
        "supported":        supported,
        "partial":          partial,
        "unsupported":      unsupported,
        "total_claims":     total,
        "citation_score":   round(citation_score, 3),
        }





if __name__ == "__main__":
        from backend.generation.generator import generate

        query = "What is the rollback process if a deployment fails?"
        print(f"Query: {query}\n")

        result = generate(query)
        answer = result["answer"]
        chunks = result["chunks"]

        print("Verifying citations...")
        print("(this makes one LLM call per claim)\n")

        verification = verify_all_citations(answer, chunks)

        print(f"{'='*60}")
        print("CITATION VERIFICATION RESULTS:")
        print(f"{'='*60}")

        for r in verification["results"]:
            print(f"\nClaim:   {r['claim'][:100]}")
            print(f"Cites:   [{r['citation_number']}]")
            print(f"Verdict: {r['verdict']}")
            print(f"Reason:  {r['explanation'].split(chr(10))[1] if chr(10) in r['explanation'] else ''}")

        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Supported:    {verification['supported']}/{verification['total_claims']}")
        print(f"  Partial:      {verification['partial']}/{verification['total_claims']}")
        print(f"  Unsupported:  {verification['unsupported']}/{verification['total_claims']}")
        print(f"  Citation score: {verification['citation_score']}")





