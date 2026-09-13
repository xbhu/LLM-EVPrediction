# Use Case 3: End-to-End Prediction + Natural-Language Explanation (Model Comparison)

## Purpose
Combine prediction and explanation in a single LLM output — given historical demand plus context, the model outputs both a predicted kWh value and a human-readable explanation — and systematically compare instruction-tuned LLM architectures on the identical task.

## Data
357 QA pairs (prompt = historical demand + context; target = predicted value + explanation text), split 80/20 train/val.

## Experiment Design
Six models fine-tuned and evaluated on identical data/task:
- Flan-T5-large (encoder-decoder, full fine-tune, BF16 + gradient checkpointing)
- Phi-4-mini-instruct (decoder-only, LoRA)
- Llama-3.2-3B-Instruct (decoder-only, LoRA)
- Gemma-3-4B-it (decoder-only, LoRA)
- Qwen3-4B (decoder-only, LoRA, thinking mode disabled)
- DeepSeek-R1-Distill-Qwen-7B (decoder-only, QLoRA 4-bit)

Baseline: 7-day moving average.

## Results
| Model | MAE (kWh) | MAPE |
|---|---|---|
| **Qwen3-4B (best)** | **50.51** | **3.04%** |
| DeepSeek-R1-Distill-Qwen-7B | 50.69 | — |
| Phi-4-mini-instruct | 51.50 | — |
| Flan-T5-large (worst) | 68.51 | — |
| 7-day moving average (baseline) | 177.12 | — |

All LLM approaches substantially beat the naive moving-average baseline. Gemma's initial results were invalidated: checkpoint saving had been accidentally disabled mid-run, so evaluation used an overfit epoch-8 model instead of the intended best epoch-4 checkpoint; this was identified and corrected.

## Key Takeaway
This thread established the recurring code conventions used across the rest of the series: BF16 everywhere (FP16 causes NaN loss), `processing_class=tokenizer` (replacing the deprecated `tokenizer=` argument), `load_best_model_at_end=True` + `save_total_limit=1` for small datasets (overfitting typically begins around epoch 4–5), QLoRA via `BitsAndBytesConfig` + `prepare_model_for_kbit_training` + `device_map="auto"`, and model-specific quirks (Gemma's chat template doesn't support a system role; Qwen3 needs `enable_thinking=False`).
