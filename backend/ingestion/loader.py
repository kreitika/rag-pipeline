import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_document(file_path: str) -> dict:
    """
    Load a single document from disk and return its content and metadata.
    
    Args:
        file_path: path to the document file
    
    Returns:
        dict with keys: text, source, format, char_count
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"No file found at: {file_path}")
    
    text = path.read_text(encoding="utf-8")
    
    return {
        "text": text,
        "source": path.name,
        "format": path.suffix.lstrip("."),
        "char_count": len(text),
    }

if __name__ == "__main__":
    file_path = "data/raw/sample.md"
    
    print(f"Loading document: {file_path}")
    document = load_document(file_path)
    
    print(f"\nDocument loaded successfully!")
    print(f"  Source:     {document['source']}")
    print(f"  Format:     {document['format']}")
    print(f"  Characters: {document['char_count']}")
    print(f"\nPreview (first 200 characters):")
    print(f"{document['text'][:200]}")    