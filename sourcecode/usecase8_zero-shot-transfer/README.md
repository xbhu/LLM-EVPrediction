# Use Case 8: Zero-Shot Cross-City Demand Transfer

## Purpose
Test whether an LLM can use its pretrained world knowledge to transfer EV charging demand patterns from a data-rich source site (State College, PA) to cities with **no historical charging data at all**, using only qualitative city descriptions and analogical reasoning. This mirrors a real deployment scenario: forecasting demand for a new site before it exists.

## Data
`dataset8_zero_shot_transfer.csv` — 60 rows (5 target cities × 12 months): Altoona, Harrisburg, Erie, Bethlehem, Philadelphia. Each row contains city features, a transfer-reasoning prompt, and an LLM-estimated monthly demand ratio relative to State College.

Note: this dataset is itself LLM-generated, for learning purposes only — real Kansas City and Pennsylvania statewide data are reserved separately for actual research-grade evaluation.

## Experiment Design
Iterative prompt-format engineering was required first: two earlier regex-based output-parsing versions failed on roughly half of all inferences (v1 accidentally extracted peak hourly values instead of daily demand; v2 had ~50% extraction failure). This was fixed by switching to a `>>DEMAND:` / `>>RATIO:` prefixed output format that could be reliably parsed. After that:

- **Full zero-shot inference** across all 5 cities × 12 months.
- **Ablation study**: 5 prompt conditions — full prompt / no city description / no transfer notes / no source data / numeric-only — tested on Altoona and Harrisburg (120 total inferences).

## Results
- **Directionality was correct** for all 5 cities: smaller/lower-EV-adoption cities were predicted below State College, larger cities above — the LLM's qualitative sense of city scale/EV adoption was sound.
- **Quantitative estimates were unstable**: the model frequently ignored the numeric features it was given and hallucinated its own city facts instead — e.g., inventing Altoona's population as 25,000–30,000 versus the actual ~43,000.
- **Ablation findings**:
  - Removing `transfer_notes` (explicit transfer-reasoning guidance) was the single most damaging change — Altoona's predicted ratio inverted from 0.44 to 1.20 (wrong direction), including one extreme hallucination (ratio = 6.05).
  - Removing city descriptions dropped Harrisburg's ratio below 1.0, showing its above-baseline prediction depended entirely on the qualitative text, not the numeric features.
  - The `numeric_only` condition had the lowest extraction-failure rate (2/24) but lost city-specific nuance in the predictions.

## Key Conceptual Takeaway
This is the "purest LLM" use case in the series — unlike UC1–UC7, it depends almost entirely on the model's pretrained world knowledge rather than on any data provided at inference time. Its main failure mode is not prediction-format breakdown but **confident hallucination of factual details** (population, demographics) even when correct numeric data was available in the prompt. A follow-up discussion also covered a more rigorous evaluation framework for future work: baseline comparisons (linear regression, XGBoost, KNN), ablation design, forecasting granularity (monthly/daily/hourly), and uncertainty quantification — noting that hourly granularity likely exceeds what LLM world-knowledge alone can support reliably.
