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


if __name__ == "__main__":
    doc = load_document("data/raw/sample.md")

    print(f"Document: {doc.source} ({doc.char_count} chars)")
    print(f"Strategy: fixed-size (512 chars, 50 overlap)")

    chunks = chunk_fixed(doc)

    print(f"Chunks created: {len(chunks)}")
    print()

    for chunk in chunks:
        preview = chunk.text[:80].replace("\n", " ")
        print(f"Chunk {chunk.chunk_index}: {chunk.char_count} chars — \"{preview}...\"")