# Demo knowledge: evidence-based AI systems

A retrieval-augmented generation (RAG) system retrieves relevant passages from a knowledge base and supplies them to a language model when answering a question. Retrieval can reduce unsupported claims, but it does not eliminate hallucinations. Source attribution and evaluation are important.

Vector search represents text as numerical embeddings and retrieves semantically similar items. Chunk size, overlap, embedding model, top-k, and distance metric can affect retrieval quality. A persistent vector database allows indexed data to be reused between application runs.

A multi-agent verification workflow can separate drafting from auditing. The first agent creates an answer grounded in retrieved evidence. A second agent checks whether claims are supported, citations are valid, and uncertainty is communicated. If the audit finds a problem, the answer can be revised. The verifier is itself a model and can make mistakes, so human review and benchmark evaluation remain necessary.

Multimodal question answering accepts more than text, for example an image alongside a question. The model should distinguish directly visible image details from facts that require external evidence. Images may contain sensitive information, and users should avoid uploading confidential material to third-party APIs without permission.
