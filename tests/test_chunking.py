from app.ingestion import chunk_text

def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []

def test_chunking_covers_content_with_overlap():
    text = ("Sentence about retrieval. " * 250).strip()
    chunks = chunk_text(text, chunk_size=300, overlap=40)
    assert len(chunks) > 1
    assert all(chunks)
    assert "Sentence about retrieval" in chunks[0]

def test_small_text_one_chunk():
    assert chunk_text("small example") == ["small example"]
