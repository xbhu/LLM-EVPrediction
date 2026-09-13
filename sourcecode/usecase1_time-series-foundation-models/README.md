# Use Case 1: Time-Series Foundation Models for Short-Term EV Demand Forecasting

## Purpose
Establish the baseline use case of the LLM-EVPrediction series: test whether time-series foundation models (not text LLMs) can forecast EV charging demand 6 hours ahead, both zero-shot and fine-tuned. This thread also set up the shared research infrastructure (servers, conda environment, dataset generation) used across all later use cases.

## Data
`dataset1` — synthetic hourly EV charging demand series:
- 24-hour historical demand window (`demand_t-24` … `demand_t-1`)
- 24-hour historical temperature window, plus 6-hour future temperature (extrapolated by holding the last known value constant)
- Target: 6-hour-ahead demand (`target_t+1` … `target_t+6`)
- Chronological train/val/test split

## Experiment Design
Three time-series foundation models were tested:

- **Chronos** (`amazon/chronos-t5-small`): zero-shot inference vs. full fine-tuning (all parameters, AdamW + CosineAnnealingLR + early stopping, patience=3).
- **TimesFM** (`google/timesfm-1.0-200m-pytorch`): zero-shot only — Google did not release training code, only inference weights, so fine-tuning was not possible.
- **MOIRAI** (`Salesforce/moirai-1.0-R-small`): univariate zero-shot; multivariate zero-shot (temperature added as a dynamic covariate); and a residual-correction network (frozen MOIRAI backbone + a small trainable MLP, ~1,600 parameters), since `MoiraiForecast`'s sampling-based inference breaks gradient flow and cannot be fine-tuned directly like a normal model.

## Results
- **Chronos**: 21.3% MAPE zero-shot → 19.5% MAPE fine-tuned. Largest improvement at t+4h (−34% error). Notably, t+1h slightly *worsened* after fine-tuning, attributed to a mismatch between custom tokenization used for fine-tuning and Chronos's native tokenizer.
- **TimesFM**: 22.6% MAPE zero-shot.
- **MOIRAI**: 42.8% MAPE univariate zero-shot → 44.1% when temperature was naively added as a multivariate zero-shot covariate (confirms that multivariate inputs need training/fine-tuning to actually help — adding a covariate zero-shot can hurt). The residual-correction network improved this to 33.1% MAPE.

## Key Takeaway
"Fine-tuning" is not always literal gradient-based parameter updates — for models whose inference pipeline involves non-differentiable sampling (like MOIRAI), a frozen-backbone + small trainable correction module is the practical substitute. Also, native model-specific tokenization schemes can be sensitive to being overridden by custom fine-tuning tokenization.

## Infrastructure Established (for reference)
- Shared lab server (two RTX 6000 Ada GPUs, 49GB each) and personal office server (RTX A2000, 12GB), both Ubuntu, accessed via SSH/PSU VPN.
- Conda environment `ev_llm` (Python 3.11) at `/home/xzh5180/Research/llm-evprediction/`, organized into `datasets/`, `sourcecode/`, `outputs/`.
- Eight synthetic datasets generated for the full 8-use-case series, hosted on GitHub and copied to the research servers.
