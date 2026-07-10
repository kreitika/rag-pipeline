import re

def parse_citations(answer: str, chunks: list[dict]) -> list[dict]:
    citation_pattern = re.compile(r'\[(\d+)\]')
    matches = citation_pattern.findall(answer)

    cited_numbers = sorted(set(int(m) for m in matches))

    citations = []
    for num in cited_numbers:
        chunk_index = num - 1
        if chunk_index < len(chunks):
            citations.append({
                "citation_number": num,
                "source":          chunks[chunk_index]["source"],
                "chunk_text":      chunks[chunk_index]["text"],
                "times_cited":     matches.count(str(num)),
            })

    return citations

    
    
def extract_claims(answer: str) -> list[dict]:
    citation_pattern = re.compile(r'\[(\d+)\]')

    sentences = re.split(r'(?<=[.!?])\s+', answer)

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        cited_nums = citation_pattern.findall(sentence)
        if cited_nums:
            clean_claim = citation_pattern.sub('', sentence).strip()
            claims.append({
                "claim":            clean_claim,
                "cited_numbers":    [int(n) for n in cited_nums],
                "original_sentence": sentence,
            })

    return claims



if __name__ == "__main__":
    from backend.generation.generator import generate

    query = "What is the rollback process if a deployment fails?"
    print(f"Query: {query}\n")

    result = generate(query)
    answer = result["answer"]
    chunks = result["chunks"]

    print("ANSWER:")
    print(answer)

    print(f"\n{'='*60}")
    print("PARSED CITATIONS:")
    citations = parse_citations(answer, chunks)
    for c in citations:
        print(f"\n  [{c['citation_number']}] cited {c['times_cited']} time(s)")
        print(f"  Source: {c['source']}")
        print(f"  Chunk preview: {c['chunk_text'][:100].replace(chr(10), ' ')}")

    print(f"\n{'='*60}")
    print("EXTRACTED CLAIMS:")
    claims = extract_claims(answer)
    for i, claim in enumerate(claims):
        print(f"\n  Claim {i+1}: {claim['claim'][:100]}")
        print(f"  Cites:  {claim['cited_numbers']}")


