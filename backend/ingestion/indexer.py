import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_by_section, Chunk
from backend.ingestion.deduplicator import Deduplicator

load_dotenv()


DOCUMENTS_DIR = Path("data/raw")
CHROMA_DIR = Path("indexes/chroma")
BM25_PATH = Path("indexes/bm25/index.pkl")
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"

DOCUMENT_FILES = [
    "data/raw/sample.md",
    "data/raw/sample.txt",
    "data/raw/sample.html",
    "data/raw/sample.pdf",
]


def embed_in_batches(chunks: list[Chunk], client: OpenAI) -> list[list[float]]:
    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end="")

        response = client.embeddings.create(
            input=[chunk.text for chunk in batch],
            model=EMBEDDING_MODEL
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(" done ✓")

    return all_embeddings


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def save_bm25_index(bm25: BM25Okapi, chunks: list[Chunk]):
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "bm25": bm25,
        "chunks": chunks
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"  BM25 index saved to {BM25_PATH} ✓")


def run_ingestion():
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Step 1: Load all documents
    print("Loading documents...")
    documents = []
    for file_path in DOCUMENT_FILES:
        doc = load_document(file_path)
        documents.append(doc)
        print(f"  Loaded: {doc.source} ({doc.char_count} chars)")

    # Step 2: Chunk all documents
    print("\nChunking with section-aware strategy...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_by_section(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.source}: {len(chunks)} chunks")
    print(f"  Total: {len(all_chunks)} chunks")

    # Step 3: Deduplicate
    print("\nDeduplicating...")
    deduplicator = Deduplicator(threshold=0.95)
    unique_chunks = []
    for chunk in all_chunks:
        if deduplicator.add(chunk):
            unique_chunks.append(chunk)
    print(f"  Unique chunks: {len(unique_chunks)}")
    print(f"  Duplicates skipped: {deduplicator.skipped_count}")

    # Step 4: Embed in batches
    print("\nEmbedding in batches...")
    embeddings = embed_in_batches(unique_chunks, client)

    # Step 5: Store in ChromaDB
    print("\nStoring in ChromaDB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection_name = "rag_chunks"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        embeddings=embeddings,
        documents=[chunk.text for chunk in unique_chunks],
        metadatas=[{
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "strategy": chunk.strategy,
            "char_count": chunk.char_count
        } for chunk in unique_chunks],
        ids=[f"{chunk.source}_{chunk.chunk_index}_{chunk.strategy}"
             for chunk in unique_chunks]
    )
    print(f"  Stored {collection.count()} chunks in ChromaDB ✓")

    # Step 6: Build and save BM25 index
    print("\nBuilding BM25 index...")
    bm25 = build_bm25_index(unique_chunks)
    save_bm25_index(bm25, unique_chunks)
    print(f"  BM25 corpus size: {len(unique_chunks)} chunks ✓")

    print("\nIngestion complete!")
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  BM25:     {BM25_PATH}")



if __name__ == "__main__":
    run_ingestion()