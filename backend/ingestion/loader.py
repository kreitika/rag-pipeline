import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from pypdf import PdfReader
from bs4 import BeautifulSoup

load_dotenv()


class Document(BaseModel):
    text: str
    source: str
    format: str
    char_count: int

def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_html(path: Path) -> str:
    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_html(path: Path) -> str:
    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


LOADERS = {
    "md":   _load_markdown,
    "txt":  _load_text,
    "html": _load_html,
    "pdf":  _load_pdf,
}


def load_document(file_path: str) -> Document:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No file found at: {file_path}")

    fmt = path.suffix.lstrip(".")

    if fmt not in LOADERS:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {list(LOADERS.keys())}")

    loader_fn = LOADERS[fmt]
    text = loader_fn(path)

    return Document(
        text=text,
        source=path.name,
        format=fmt,
        char_count=len(text),
    )


if __name__ == "__main__":
    test_files = [
        "data/raw/sample.md",
        "data/raw/sample.txt",
        "data/raw/sample.html",
        "data/raw/sample.pdf",
    ]

    for file_path in test_files:
        print(f"\n{'='*50}")
        document = load_document(file_path)
        print(f"Source:     {document.source}")
        print(f"Format:     {document.format}")
        print(f"Characters: {document.char_count}")
        print(f"Preview:    {document.text[:150]}")