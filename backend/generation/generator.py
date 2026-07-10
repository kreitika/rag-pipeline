import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.generation.prompt_builder import build_prompt
from backend.retrieval.reranker import rerank

load_dotenv()


def generate(query: str, top_k: int = 3) -> dict:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    chunks = rerank(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user",   "content": prompt["user"]},
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return {
        "query":       query,
        "answer":      answer,
        "chunks":      chunks,
        "tokens_used": tokens_used,
        "model":       "gpt-4o-mini",
    }

if __name__ == "__main__":
    query = "What is the rollback process if a deployment fails?"

    print(f"Query: {query}")
    print(f"\nRunning full RAG pipeline...")
    print("(retrieval + reranking + generation)")
    print()

    result = generate(query)

    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Tokens used:  {result['tokens_used']}")
    print(f"Model:        {result['model']}")
    print(f"Chunks used:  {len(result['chunks'])}")