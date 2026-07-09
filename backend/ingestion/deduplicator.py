import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from backend.ingestion.chunker import Chunk

load_dotenv()

class Deduplicator:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name="dedup_index"
        )
        self.stored_count = 0
        self.skipped_count = 0
    

    def is_duplicate(self, chunk: Chunk) -> bool:
        if self.collection.count() == 0:
            return False

        response = self.client.embeddings.create(
            input=chunk.text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )

        if results['distances'][0]:
            distance = results['distances'][0][0]
            similarity = 1 - distance
            if similarity >= self.threshold:
                return True

        return False
    

    def add(self, chunk: Chunk) -> bool:
        if self.is_duplicate(chunk):
            self.skipped_count += 1
            return False

        response = self.client.embeddings.create(
            input=chunk.text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        self.collection.add(
            embeddings=[embedding],
            documents=[chunk.text],
            ids=[f"{chunk.source}_{chunk.chunk_index}_{chunk.strategy}"]
        )

        self.stored_count += 1
        return True
    

    def report(self):
        total = self.stored_count + self.skipped_count
        print(f"\nDeduplication Report:")
        print(f"  Total chunks processed: {total}")
        print(f"  Unique chunks stored:   {self.stored_count}")
        print(f"  Duplicates skipped:     {self.skipped_count}")
        if total > 0:
            pct = (self.skipped_count / total) * 100
            print(f"  Duplicate rate:         {pct:.1f}%")



if __name__ == "__main__":
    from backend.ingestion.loader import load_document
    from backend.ingestion.chunker import chunk_fixed

    # Load the same document twice to simulate duplicates
    doc1 = load_document("data/raw/sample.md")
    doc2 = load_document("data/raw/sample.md")

    chunks1 = chunk_fixed(doc1)
    chunks2 = chunk_fixed(doc2)
    all_chunks = chunks1 + chunks2

    print(f"Total chunks to process: {len(all_chunks)}")
    print(f"(same document loaded twice to simulate duplicates)")
    print()

    deduplicator = Deduplicator(threshold=0.95)

    for chunk in all_chunks:
        stored = deduplicator.add(chunk)
        status = "stored  ✓" if stored else "DUPLICATE — skipped"
        preview = chunk.text[:50].replace("\n", " ")
        print(f"  Chunk {chunk.chunk_index} ({chunk.source}): {status}")

    deduplicator.report()