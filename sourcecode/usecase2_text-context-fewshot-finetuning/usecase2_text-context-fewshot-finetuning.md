# Use Case 2: Text-Context-Based Next-Day Demand Prediction (Few-Shot & Fine-Tuning)

## Purpose
Shift from pure numeric time series (UC1) to natural-language context: can an LLM read a day's text description (weather, event, weekday/weekend) and predict next-day total demand? Compare API-based few-shot prompting against fine-tuning several open-source LLMs on the same task.

## Data
`dataset2` — 364 rows with columns: `date, total_demand, peak_demand, avg_temp, is_weekend, month, event_name, event_type, context_text, next_day_demand`. `context_text` is a pre-written natural-language description of the day (e.g., "Winter weekday. Average temperature: 32.5°F (cold). Regular weekday..."). Split 80/10/10 train/val/test (291 training rows — a fairly small dataset).

## Experiment Design
Two phases:

1. **Few-shot prompting** via the Anthropic API (Claude Haiku): k prior days' `context_text` + true `next_day_demand` used as in-context examples, then the model predicts the current day's demand; numeric answer parsed from the text output. Evaluated against a mean-predictor baseline and a persistence (yesterday's value) baseline.
2. **Fine-tuning open models** on the same input→output mapping: Flan-T5-base (regression-style, encoder-decoder), GPT-2 Medium (generative fine-tune), Llama-3.1-8B (QLoRA), Mistral-7B (QLoRA), Gemma-2-9B (QLoRA), Qwen3-8B (QLoRA). Gemma-4-E4B, DeepSeek-R1-0528-Qwen3-8B, and DeepSeek-V2-Lite-Chat were attempted but failed due to library incompatibilities (peft incompatibility, thinking-mode conflicts, and a transformers version mismatch requiring `trust_remote_code=True` for DeepSeek-V2's custom MoE architecture).

## Results
- Best fine-tuned model: **Gemma-2-9B — MAE = 36 kWh (MAPE 2.9%)**.
- Mistral-7B: MAE = 42 kWh. Qwen3-8B: MAE = 45 kWh.
- All fine-tuned models showed overfitting given the small dataset (291 training rows).

## Key Debugging Insight (kept — general/recurring lesson)
Generated-text decoding must slice on **token position** (`output[0][prompt_len:]`), not string length. Naive string-based slicing silently produced garbage predictions that collapsed to the dataset mean across several models before this was caught and fixed — this became a standing convention for all later generative fine-tuning use cases.
