# Evaluation starter

Create an `eval/` directory and add a JSONL file with one record per test case, for example:

```json
{"id":"q001","question":"What is vector search?","expected_source":"demo_knowledge.md","expected_answer_points":["embeddings","semantic similarity"],"answerable":true}
{"id":"q002","question":"What was the company's 2025 revenue?","expected_source":null,"expected_answer_points":["insufficient evidence"],"answerable":false}
```

For each test case, save the retrieved chunks, final answer, verifier verdict, latency, model IDs, and estimated cost. Score with a written rubric; have a human review a subset. Do not treat model-graded evaluation as ground truth.
