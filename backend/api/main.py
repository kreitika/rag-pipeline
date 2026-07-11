import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG Pipeline API",
    description="Production RAG system with hybrid search",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class CitationResponse(BaseModel):
    citation_number: int
    source: str
    chunk_text: str
    times_cited: int


class AskResponse(BaseModel):
    question: str
    answer: str
    confident: bool
    composite_score: float
    citations: list[CitationResponse]
    tokens_used: int = 0


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    source: str



@app.post("/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        from backend.generation.pipeline import full_pipeline
        result = full_pipeline(request.question, top_k=request.top_k)

        citations = []
        for c in result.get("citations", []):
            citations.append(CitationResponse(
                citation_number=c["citation_number"],
                source=c["source"],
                chunk_text=c["chunk_text"],
                times_cited=c["times_cited"],
            ))

        return AskResponse(
            question=result["query"],
            answer=result["answer"],
            confident=result["confident"],
            composite_score=result["composite_score"],
            citations=citations,
            tokens_used=result.get("tokens_used", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ask/dense-only")
async def ask_dense_only(request: AskRequest):
    try:
        from backend.retrieval.dense import dense_retrieve
        from backend.generation.prompt_builder import build_prompt
        from backend.generation.generator import generate
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        chunks = dense_retrieve(request.question, n_results=request.top_k)

        result = generate(request.question, chunks=chunks)

        return {
            "question": request.question,
            "answer":   result["answer"],
            "method":   "dense-only",
            "chunks":   len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents")
async def list_documents():
    try:
        data_dir = Path("data/raw")
        documents = []
        for f in data_dir.iterdir():
            if f.is_file():
                documents.append({
                    "filename": f.name,
                    "format":   f.suffix.lstrip("."),
                    "size_kb":  round(f.stat().st_size / 1024, 2),
                })
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        save_path = Path("data/raw") / file.filename
        with open(save_path, "wb") as f:
            f.write(content)

        from backend.ingestion.indexer import run_ingestion
        run_ingestion()

        return IngestResponse(
            status="success",
            chunks_created=0,
            source=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}