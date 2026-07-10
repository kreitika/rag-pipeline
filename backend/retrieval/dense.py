import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from pathlib import Path

load_dotenv()

CHROMA_DIR = Path("indexes/chroma")
EMBEDDING_MODEL = "text-embedding-3-small"


def dense_retrieve(query: str, n_results: int = 10) -> list[dict]:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(
        name="rag_chunks",
        embedding_function=None
    )

    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text":       results['documents'][0][i],
            "source":     results['metadatas'][0][i]['source'],
            "chunk_index": results['metadatas'][0][i]['chunk_index'],
            "strategy":   results['metadatas'][0][i]['strategy'],
            "similarity": round(1 - results['distances'][0][i], 4),
            "rank":       i + 1
        })

    return chunks


if __name__ == "__main__":
    query = "What is the rollback process if a deployment fails?"

    print(f"Query: {query}")
    print(f"\nDense retrieval results:")
    print("=" * 60)

    results = dense_retrieve(query, n_results=5)

    for result in results:
        print(f"\nRank {result['rank']} — similarity: {result['similarity']}")
        print(f"Source: {result['source']}")
        print(f"Preview: {result['text'][:150].replace(chr(10), ' ')}")