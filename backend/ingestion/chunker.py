import numpy as np
from openai import OpenAI
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.ingestion.loader import Document, load_document



load_dotenv()

def _cosine_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Chunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    strategy: str
    char_count: int


def chunk_fixed(document: Document, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    pieces = splitter.split_text(document.text)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk = Chunk(
            text=piece,
            source=document.source,
            chunk_index=i,
            strategy="fixed",
            char_count=len(piece),
        )
        chunks.append(chunk)

    return chunks


def chunk_by_section(document: Document, chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )

    pieces = splitter.split_text(document.text)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk = Chunk(
            text=piece,
            source=document.source,
            chunk_index=i,
            strategy="section",
            char_count=len(piece),
        )
        chunks.append(chunk)

    return chunks


def chunk_semantic(document: Document, threshold: float = 0.7) -> list[Chunk]:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Step 1: split document into individual sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', document.text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if len(sentences) < 2:
        return [Chunk(
            text=document.text,
            source=document.source,
            chunk_index=0,
            strategy="semantic",
            char_count=len(document.text),
        )]

    # Step 2: embed all sentences in one API call
    print(f"  Embedding {len(sentences)} sentences...")
    response = client.embeddings.create(
        input=sentences,
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]

    # Step 3: find split points where similarity drops
    split_points = []
    for i in range(len(embeddings) - 1):
        similarity = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if similarity < threshold:
            split_points.append(i + 1)

    # Step 4: group sentences into chunks
    chunks = []
    chunk_index = 0
    start = 0

    for split_point in split_points:
        chunk_text = '. '.join(sentences[start:split_point]) + '.'
        chunks.append(Chunk(
            text=chunk_text,
            source=document.source,
            chunk_index=chunk_index,
            strategy="semantic",
            char_count=len(chunk_text),
        ))
        chunk_index += 1
        start = split_point

    # Add the final chunk
    final_text = '. '.join(sentences[start:]) + '.'
    chunks.append(Chunk(
        text=final_text,
        source=document.source,
        chunk_index=chunk_index,
        strategy="semantic",
        char_count=len(final_text),
    ))

    return chunks


if __name__ == "__main__":
    doc = load_document("data/raw/sample.md")

    print(f"Document: {doc.source} ({doc.char_count} chars)")
    print()

    # Strategy A: Fixed-size
    print("=" * 50)
    print("STRATEGY A: Fixed-size (512 chars, 50 overlap)")
    print("=" * 50)
    fixed_chunks = chunk_fixed(doc)
    print(f"Chunks created: {len(fixed_chunks)}")
    for chunk in fixed_chunks:
        preview = chunk.text[:70].replace("\n", " ")
        print(f"  Chunk {chunk.chunk_index}: {chunk.char_count} chars — \"{preview}...\"")

    print()

    # Strategy B: Section-aware
    print("=" * 50)
    print("STRATEGY B: Section-aware (splits at headings)")
    print("=" * 50)
    section_chunks = chunk_by_section(doc)
    print(f"Chunks created: {len(section_chunks)}")
    for chunk in section_chunks:
        preview = chunk.text[:70].replace("\n", " ")
        print(f"  Chunk {chunk.chunk_index}: {chunk.char_count} chars — \"{preview}...\"")

    print()

    # Strategy C: Semantic
    print("=" * 50)
    print("STRATEGY C: Semantic (splits at topic boundaries)")
    print("=" * 50)
    print("Calling OpenAI API...")
    semantic_chunks = chunk_semantic(doc)
    print(f"Chunks created: {len(semantic_chunks)}")
    for chunk in semantic_chunks:
        preview = chunk.text[:70].replace("\n", " ")
        print(f"  Chunk {chunk.chunk_index}: {chunk.char_count} chars — \"{preview}...\"")