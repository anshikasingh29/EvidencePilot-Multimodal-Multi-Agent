# EvidencePilot — Multimodal RAG with a Verifier Agent

**A portfolio-grade AI engineering project** demonstrating retrieval-augmented generation (RAG), vector similarity search, multimodal image understanding, and a two-model verification loop.

## Why this project is interesting

Many demo chatbots simply pass a prompt to one model. EvidencePilot has an explicit evidence pipeline and an auditable agent workflow:

1. **Ingest** PDF, DOCX, TXT, and Markdown documents; extract text and preserve PDF page markers.
2. **Chunk** extracted text with overlap so context survives chunk boundaries.
3. **Embed and index** chunks with OpenAI embeddings in a persistent ChromaDB collection.
4. **Retrieve** semantically similar chunks for a user question and show source excerpts and similarity scores.
5. **Researcher agent** drafts an answer with `[S1]`, `[S2]` citations grounded in retrieved context.
6. **Verifier agent** independently audits support, citation discipline, relevance, and uncertainty. It can pass the answer or request a revision.
7. **Multimodal mode** accepts an image plus question, while still retrieving relevant text evidence.
8. **Inspect the trace** to see which agents ran, whether revision occurred, and why.

This is a solid foundation, not a claim that a second model guarantees truth. The verifier can also be wrong; evaluate it with a benchmark and retain human review for high-impact use.

## Architecture

```text
             PDF / DOCX / TXT / MD
                       |
             Extract -> Chunk + overlap
                       |
       OpenAI embedding model (text embeddings)
                       |
           ChromaDB persistent vector store
                       |
Question ---------------------> Semantic top-k retrieval
   |                                   |
Optional image -> vision input          +-> source chunks + metadata
   |                                   |
   +-----------------------> Researcher model
                                      |
                                 Draft answer
                                      |
                               Verifier model
                              /              \
                           PASS            REVISE
                            |                 |
                       Final answer <- corrected/revised answer
                            |
                citations + sources + agent trace
```

## Stack

- Python 3.11 recommended
- FastAPI REST backend
- Streamlit UI
- OpenAI Responses API for text and image inputs
- OpenAI embeddings API
- ChromaDB persistent vector store
- PyPDF and python-docx for document extraction
- pytest for starter tests

## Requirements

- Python 3.11 or 3.12
- An API key for an OpenAI API project with access to the configured text/vision model and embeddings model. API usage may be billable; check your account's pricing and limits.
- Internet access for API calls and initial package/model requests

## Run locally

### 1. Create a virtual environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace `OPENAI_API_KEY=replace_me` with your own key. Never commit `.env`, paste keys into frontend code, or upload secrets to GitHub. The default model names are configurable; use model IDs enabled for your account.

### 4. Start the API

From the project root:

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs  
Health: http://localhost:8000/health

### 5. Start the UI in another terminal

Activate the same virtual environment, then run:

```bash
streamlit run app/ui.py
```

Open the local URL Streamlit prints (normally http://localhost:8501). Upload a document in the sidebar, index it, and ask questions. Upload an image as well to enable multimodal mode.

### Optional: API examples

Index a document:

```bash
curl -X POST http://localhost:8000/ingest -F 'file=@sample_docs/demo_knowledge.md'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"What does the verifier agent check?","top_k":4}'
```

Ask with an image:

```bash
curl -X POST 'http://localhost:8000/ask-image?question=Describe%20the%20image%20and%20relate%20it%20to%20the%20knowledge%20base' \\
  -F 'file=@your-image.jpg'
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Project metadata |
| GET | `/health` | Configuration and indexed-chunk status |
| POST | `/ingest` | Extract, chunk, embed, and index a document |
| POST | `/ask` | RAG answer + verifier audit + optional revision |
| POST | `/ask-image?question=...` | Image question-answering with retrieval and verification |

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Server-side API key | Required |
| `OPENAI_MODEL` | Researcher/vision model | `gpt-4.1-mini` |
| `OPENAI_VERIFIER_MODEL` | Independent verifier model | `gpt-4.1-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHROMA_PATH` | Persistent vector database directory | `./data/chroma` |
| `UPLOAD_DIR` | Local uploaded files directory | `./data/uploads` |
| `TOP_K` | Default retrieval count | `5` |
| `MAX_FILE_MB` | Upload limit | `15` |
| `API_URL` | UI backend URL | `http://localhost:8000` |

For a meaningful comparison, configure different researcher and verifier models when your account supports them. The code intentionally defaults to the same model family to keep setup simple; the two agents still have different roles and prompts.

## Evaluation plan — turn it into a research project

Do not stop at a working demo. Create `eval/questions.jsonl` with 30–100 question/answer/evidence examples based on documents you have permission to use. Include answerable, ambiguous, and unanswerable questions.

Measure:

- **Retrieval Recall@k**: fraction of expected evidence chunks present in the top-k results.
- **Faithfulness**: fraction of factual answer claims supported by retrieved evidence, judged with a documented rubric.
- **Citation precision**: fraction of citations that actually support the associated claim.
- **Answer relevance**: whether the response addresses the question.
- **Verifier catch rate**: fraction of deliberately seeded unsupported claims correctly flagged by the verifier.
- **Revision gain**: difference in groundedness before and after verifier revision.
- **Latency and cost**: median response time and token/API cost per question.

Compare at least these conditions: (A) one model without RAG, (B) RAG with one model, (C) RAG + verifier, (D) RAG + verifier with a different verifier model. Report sample size, model IDs, prompt versions, and limitations. Avoid claiming improvement until the experiment supports it.

## Resume bullet (customize after you measure it)

**EvidencePilot — Multimodal Multi-Agent RAG System** | Python, FastAPI, OpenAI API, ChromaDB, Streamlit

- Built a multimodal RAG application that ingests PDF/DOCX/text documents, embeds overlapping chunks, and retrieves semantically relevant evidence using a persistent vector database.
- Implemented a two-agent answer pipeline in which a researcher model produces source-grounded responses and an independent verifier audits evidence support and triggers corrective revision.
- Added image-question answering, source-level evidence inspection, REST endpoints, and an agent execution trace; designed evaluation metrics for retrieval recall, citation precision, faithfulness, verifier catch rate, latency, and cost.

**Important:** Add real quantitative results only after running an evaluation. For example, report measured Recall@5 or verifier catch rate only if you have collected the results.

## Suggested GitHub presentation

1. Add a short screen recording showing document ingestion, an answer with citations, and a verifier-triggered revision.
2. Add an architecture diagram and a small evaluation-results table.
3. Add 5–10 test questions and document expected source passages.
4. Pin dependency versions after testing in your environment.
5. Add a short section on limitations: scanned PDFs need OCR; the verifier is not infallible; API use may incur cost; confidential documents require careful handling.

## Tests

```bash
pytest -q
```

## Current limitations / next improvements

- Scanned PDFs require OCR; current PDF extraction handles text-based PDFs.
- Ingestion currently uses local development storage. Add authentication, rate limits, and per-user document isolation before public deployment.
- Similarity score is a transformed Chroma cosine distance, not a calibrated probability.
- Image mode uses the configured OpenAI model; ensure that model supports image input.
- The starter tests cover chunking only. Add integration tests with mocked API responses and a labelled retrieval/verification evaluation set.

## Documentation

- OpenAI Responses API: https://developers.openai.com/api/docs/guides/migrate-to-responses
- OpenAI API reference: https://developers.openai.com/api/reference/resources/responses/methods/create
- ChromaDB persistent client: https://docs.trychroma.com/reference/python
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
