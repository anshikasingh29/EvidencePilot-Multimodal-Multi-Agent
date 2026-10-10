from pydantic import BaseModel, Field
from typing import Any

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=12)

class Source(BaseModel):
    source: str
    chunk_id: str
    score: float | None = None
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    verdict: str
    verifier_notes: str
    revised: bool
    sources: list[Source]
    trace: list[dict[str, Any]]
