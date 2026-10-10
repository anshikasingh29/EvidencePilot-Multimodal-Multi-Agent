import base64
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import get_settings
from .schemas import AskRequest, AskResponse
from .ingestion import SUPPORTED, extract_text, chunk_text
from .vector_store import VectorStore
from .agents import MultiAgentPipeline

settings = get_settings()
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
app = FastAPI(title="EvidencePilot API", description="Multimodal multi-agent RAG with verifier auditing", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/")
def root():
    return {"name": "EvidencePilot", "status": "ok", "docs": "/docs", "features": ["RAG", "vector search", "multimodal image input", "two-agent verification"]}

@app.get("/health")
def health():
    try:
        store = VectorStore()
        return {"status": "ok", "indexed_chunks": store.count(), "api_key_configured": bool(settings.openai_api_key and settings.openai_api_key != "replace_me")}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, "Unsupported type. Upload PDF, DOCX, TXT, or MD.")
    content = await file.read()
    if len(content) > settings.max_file_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_file_mb} MB limit.")
    dest = Path(settings.upload_dir) / f"{uuid4().hex}_{name}"
    dest.write_bytes(content)
    try:
        text = extract_text(dest)
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(422, "No extractable text found. Scanned PDFs need OCR before ingestion.")
        count = VectorStore().add_chunks(name, chunks)
        return {"filename": name, "chunks_indexed": count, "characters_extracted": len(text), "status": "indexed"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        top_k = request.top_k or settings.top_k
        sources = VectorStore().search(request.question, top_k=top_k)
        result = MultiAgentPipeline().answer(request.question, sources)
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@app.post("/ask-image", response_model=AskResponse)
async def ask_image(question: str, file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    raw = await file.read()
    if len(raw) > settings.max_file_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_file_mb} MB limit.")
    data_url = f"data:{content_type};base64,{base64.b64encode(raw).decode()}"
    try:
        sources = VectorStore().search(question, top_k=settings.top_k)
        result = MultiAgentPipeline().answer(question, sources, image_data_url=data_url)
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
