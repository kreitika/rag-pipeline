from backend.retrieval.reranker import rerank

SYSTEM_PROMPT = """You are a precise question-answering assistant for internal company documentation.

Your rules:
1. Answer ONLY using the numbered context blocks provided below.
2. Cite every factual claim using bracketed numbers like [1] or [2].
3. If a claim is supported by multiple chunks, cite all of them like [1][3].
4. If the context does not contain enough information to answer, respond with exactly:
   "I don't have enough information in the provided documentation to answer this question."
5. Never use knowledge from outside the provided context.
6. Be concise and precise. Do not pad the answer with unnecessary text.
7. If the question has multiple parts, address each part separately."""


def build_context(chunks: list[dict]) -> str:
    context_blocks = []

    for i, chunk in enumerate(chunks):
        block = f"[{i + 1}] Source: {chunk['source']}\n{chunk['text']}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

def build_prompt(query: str, chunks: list[dict]) -> dict:
    context = build_context(chunks)

    user_message = f"""Context:
{context}

Question: {query}

Remember: cite every claim with [1], [2], etc. based on the context numbers above."""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_message,
        "chunks": chunks,
        "query": query,
    }


if __name__ == "__main__":
    query = "What is the rollback process if a deployment fails?"

    print(f"Query: {query}")
    print(f"\nRetrieving relevant chunks...")
    chunks = rerank(query, top_k=3)

    print(f"\nBuilding prompt...")
    prompt = build_prompt(query, chunks)

    print(f"\n{'='*60}")
    print("SYSTEM PROMPT:")
    print(f"{'='*60}")
    print(prompt["system"])

    print(f"\n{'='*60}")
    print("USER MESSAGE:")
    print(f"{'='*60}")
    print(prompt["user"])

    print(f"\n{'='*60}")
    print(f"Chunks included: {len(chunks)}")
    print(f"Total prompt length: {len(prompt['system']) + len(prompt['user'])} chars")