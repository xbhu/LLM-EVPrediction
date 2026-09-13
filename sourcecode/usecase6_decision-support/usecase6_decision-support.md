# Use Case 6: LLM as a Decision Support System

## Purpose
Translate a daily operational snapshot — forecasted demand combined with pricing, grid state, and queue length — into a one-line actionable operator recommendation. A core goal of this thread was to probe whether the LLM's apparent "reasoning" on this task is genuine causal decision-making or superficial pattern matching.

## Data
365 daily operational snapshots, split chronologically 292 train / 73 test.

## Experiment Design
Single model (Qwen3-4B) evaluated across three stages:

- **Stage 1 — Zero-shot inference.**
- **Stage 2 — SFT with LoRA fine-tuning.**
- **Stage 3 — RAG**, using historical decisions as the retrieval knowledge base.

(Notable debugging: TRL library API churn caused several failures during Stage 2 — `DataCollatorForCompletionOnlyLM` was removed in TRL 1.4.0, and `max_seq_length`/`dataset_text_field` were removed from `SFTTrainer`. Resolved by rewriting the script using `SFTConfig`, the correct pattern for TRL 1.4.0.)

## Results
| Stage | Method | ROUGE-L | Notes |
|---|---|---|---|
| 1 | Zero-shot | 0.09 | Model produced verbose "consulting report" style output instead of a one-line instruction |
| 2 | SFT (LoRA) | 0.93 | Dramatic jump, but judged as template memorization + numerical interpolation, not causal reasoning |
| 3 | RAG | 0.17 | Better than zero-shot, but degraded — embedding similarity scores clustered at 0.997–0.999 across nearly all prompts because prompts share a near-identical template structure, so retrieval couldn't meaningfully discriminate between snapshots |

## Key Conceptual Conclusion
The Stage 2 score does not reflect genuine decision intelligence — and scaling up data or model size would not fundamentally fix this, since the underlying issue is that the task's surface structure is highly templated. The most valuable research contribution from this use case is articulating the **boundary conditions** where LLMs add real decision-support value versus where a simpler rule engine would suffice. A possible Stage 4 (combining SFT for output-format learning with RAG restricted to structured numerical features instead of text embeddings) was scoped in discussion but not implemented — the thread moved on to Use Case 7 instead.
