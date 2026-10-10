from pathlib import Path
from pypdf import PdfReader
from docx import Document

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

def extract_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip(): pages.append(f"[Page {page_no}]\n{text}")
        return "\n\n".join(pages)
    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF, DOCX, TXT, MD.")

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    """Simple boundary-aware chunker; overlap preserves context across boundaries."""
    text = " ".join(text.split())
    if not text: return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start + chunk_size // 2, end),
                           text.rfind("; ", start + chunk_size // 2, end),
                           text.rfind(" ", start + chunk_size // 2, end))
            if boundary > start: end = boundary + 1
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks
