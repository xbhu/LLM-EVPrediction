# Use Case 5: LLM as a Post-Hoc Anomaly Explainer

## Purpose
Given a case where a model's forecast deviated significantly from actual demand, have the LLM generate a plausible *explanation* for the deviation — not a new prediction. The goal was to test post-hoc explanation quality (faithfulness/coherence) rather than predictive accuracy.

## Data
68 labeled anomaly samples, each with a `detection_prompt` (describing the anomaly) and a ground-truth `llm_explanation` label. Note: within each anomaly type, all rows share an identical `llm_explanation` label — an important detail for interpreting Route B below.

## Experiment Design
Three experimental routes:

- **Route A — Zero-shot prompting**: Flan-T5-base and Qwen3-4B, using self-constructed prompts that deliberately avoided leaking the ground-truth answer embedded in the dataset's `detection_prompt` field.
- **Route B — Fine-tuned Seq2Seq**: Flan-T5-large fine-tuned directly on the 68 samples.
- **Route C — RAG**: Qwen3-4B + FAISS (`IndexFlatIP`) + sentence-transformers embeddings, using leave-one-out retrieval from the 68-sample historical anomaly knowledge base (i.e., each test sample retrieves from the other 67).

## Results
| Route | Model | ROUGE-L | BERTScore |
|---|---|---|---|
| A (zero-shot) | Flan-T5-base | 0.0085 (degenerate/looping output) | — |
| A (zero-shot) | Qwen3-4B | 0.0449 | 0.8604 |
| B (fine-tuned) | Flan-T5-large | 1.0 | 1.0 |
| C (RAG) | Qwen3-4B | 0.3175 | 0.8951 |

- Route B's near-perfect score was identified as **pure memorization**, not genuine learning — a direct consequence of only 68 samples where all rows of a given anomaly type share one identical label.
- Route C (RAG) represents the most credible result: a genuine ~7× ROUGE-L improvement over zero-shot, driven by actual retrieval of relevant historical context rather than label leakage.

## Key Conceptual Points Discussed
- Post-hoc explanation vs. chain-of-thought reasoning: the distinction between *faithfulness* (does the explanation reflect the true cause?) and *operational coupling* (was the explanation actually used to produce the answer?).
- ROUGE-L (lexical overlap) vs. BERTScore (semantic similarity) — and why BERTScore can be misleadingly high when a model simply echoes prompt content back.
- RAG architecture (embedding normalization, FAISS inner-product search as cosine similarity, leave-one-out evaluation) is structurally identical to personal knowledge-management tools like Obsidian — same underlying retrieval mechanism, different UI.
