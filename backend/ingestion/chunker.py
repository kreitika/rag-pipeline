from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.ingestion.loader import Document, load_document

load_dotenv()


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