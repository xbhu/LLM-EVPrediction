# LLM-EVPrediction

A learning project exploring eight distinct roles that Large Language Models (LLMs) can play in EV charging demand forecasting. All datasets are built around the same virtual charging station in State College, PA, so results across use cases are directly comparable.

**Core prediction task:** Given the past 24 hours of data, predict charging demand for the next 6 hours.

---

## Eight LLM Use Cases

### Use Case 1 — LLM as a Time Series Forecaster

**Role:** Direct numeric prediction

**Core idea:** A Transformer-based LLM processes number sequences the same way it processes word sequences. Just as a language model predicts the next word, a time series LLM predicts the next value.

```
Language model:  "The  cat  sat  on  the  mat" → predicts next word
Time series LLM:  45.2  67.8  89.3  76.1  54.2  43.0 → predicts next number
```

Representative models: Google **TimesFM**, Amazon **Chronos**, Salesforce **MOIRAI**. Key advantage: leverages pre-trained representations, especially effective under data-scarce conditions.

**Research angle:** *Time Series LLM for EV Charging Demand Forecasting* — benchmark against LSTM and Transformer baselines; analyze performance under limited training data.

**Dataset:** `datasets/dataset1_timeseries.csv`
- 8,730 samples
- Each row: 24-hour rolling window of hourly demand + temperature → 6-hour future demand
- Pure numeric format, ready to feed directly into a time series model

---

### Use Case 2 — LLM as a Context Encoder

**Role:** Text → vector feature extraction for hybrid models

**Core idea:** Pure numeric models are blind to unstructured real-world context. A BERT-style encoder converts natural language descriptions into vectors that augment a traditional forecasting model.

```
[Numeric sequence] → LSTM ──────────────────────→ final prediction
                                                         ↑
[Text context]     → BERT → context vector ──────────────┘
"Penn State home game tonight, expect high downtown traffic"
```

**Research angle:** *Context-Aware EV Demand Prediction* — incorporate unstructured contextual signals (event calendars, weather reports, policy text) to improve forecast accuracy.

**Dataset:** `datasets/dataset2_text_context.csv`
- 364 daily samples (one full year)
- Each row: daily summary statistics + natural-language description → next-day total demand
- Events include Penn State game days, holidays, and graduation ceremonies

---

### Use Case 3 — LLM as a Reasoning Engine

**Role:** End-to-end prediction with natural language explanation

**Core idea:** Feed the LLM a complete prompt with historical data and context. It returns a forecast plus a human-readable explanation — directly useful for agency reporting.

```
Input:  "Past 24h data: [45.2, 67.8, 89.3...]. Tomorrow: Friday,
         Penn State game, forecast 82°F. Predict next 6 hours."

Output: "Predicted demand: [92.3, 105.6, 118.2, 97.4, 76.3, 61.2]
         Key drivers: event traffic (+23%), temperature effect (+8%)..."
```

**Research angle:** *Explainable EV Demand Forecasting with LLMs* — evaluate LLaMA-style models on structured prediction + explanation tasks; explore domain-specific fine-tuning.

**Dataset:** `datasets/dataset3_qa_pairs.csv`
- 357 QA pairs
- Each row: a complete LLM prompt (history + context) paired with a structured reference answer
- Ready for zero-shot evaluation or fine-tuning

---

### Use Case 4 — LLM as a Data Augmenter

**Role:** Synthetic data generation for rare and extreme scenarios

**Core idea:** Real datasets rarely capture extreme events — blizzards, grid failures, unexpected mass gatherings. LLMs can generate realistic synthetic samples for these edge cases, expanding the training set in a controllable way.

```
90 days real data (limited) → LLM generation prompts → synthetic edge-case scenarios
                                                         (cold snap, grid fault,
                                                          fleet charging event,
                                                          price spike)
         ↓
Augmented dataset → better-generalized prediction model
```

**Research angle:** *Data Augmentation for EV Charging Demand Forecasting Using LLMs* — particularly valuable in data-scarce settings or when targeting rare but high-impact events.

**Dataset:** `datasets/dataset4_augmentation.csv`
- 90-day "real" baseline + 4 synthetic scenario types
- Each augmented row includes the LLM generation prompt used to create it
- Scenario types: cold snap, grid outage, fleet charging event, electricity price spike

---

### Use Case 5 — LLM as an Anomaly Explainer

**Role:** Post-hoc interpretation of prediction deviations

**Core idea:** When a model's prediction is far off, the LLM investigates why — cross-referencing context, flagging likely causes, and finding analogous historical events.

```
Actual demand:    150 kWh
Model prediction:  62 kWh  ← large gap
                      ↓
                  [LLaMA]
                      ↓
"This deviation likely corresponds to an unrecorded campus event.
 Historical data shows similar spikes on Oct 28 and Nov 11."
```

**Research angle:** *LLM-Assisted Anomaly Detection and Explanation in EV Charging Systems* — high practical value for agency dashboards and operational reporting.

**Dataset:** `datasets/dataset5_anomaly.csv`
- Full year of hourly data with 7 injected anomaly events
- Anomaly types: equipment failure, unrecorded activity, heat wave, sensor error, Halloween event
- Each anomalous row includes an LLM explanation template

---

### Use Case 6 — LLM as a Decision Support System

**Role:** Translating forecasts into operational recommendations

**Core idea:** Beyond predicting demand, the LLM synthesizes forecast + pricing + grid state into actionable operator guidance — moving from "what will happen" to "what should we do."

```
Input:  forecast + current electricity price + grid load + queue length
           ↓
       [LLaMA]
           ↓
Output: "Raise charging price 15% from 2–4 PM.
         Expected peak load reduction: 23%.
         Projected impact on user satisfaction: minimal."
```

**Research angle:** *LLM-Assisted Dynamic Pricing and Demand Management for EV Charging* — directly applicable to DOT and utility operator needs.

**Dataset:** `datasets/dataset6_decision_support.csv`
- 365 daily operational snapshots
- Each row: demand forecast + electricity price + grid stress + queue length + LLM decision prompt + reference recommendation

---

### Use Case 7 — LLM-Based Multi-Agent Simulation

**Role:** Role-playing agents that negotiate charging schedules

**Core idea:** Multiple LLM agents each represent a stakeholder — the EV driver, the charging station operator, and the grid manager. They negotiate in natural language to reach an optimal charging schedule.

```
Agent A (EV driver):       "I need 80% charge before 5 PM."
Agent B (charging station): "3 spots available, $0.18/kWh."
Agent C (grid operator):   "Peak hours 4–6 PM — requesting load reduction."
              ↓
         Three-way negotiation
              ↓
         Optimal charging schedule + V2G dispatch
```

**Research angle:** *LLM-Based Multi-Agent Simulation for EV Charging Coordination* — connects naturally to CAV/V2G research; strong potential for high-impact publication.

**Dataset:** `datasets/dataset7_multiagent.csv`
- 120 negotiation sessions across 3 scenario types: peak-hour conflict, overnight fleet charging, V2G emergency discharge
- Each session: full three-agent dialogue, outcome, and satisfaction scores for all parties

---

### Use Case 8 — LLM-Enabled Zero-Shot Transfer

**Role:** Predicting demand at new sites with no historical data

**Core idea:** Traditional models need local historical data. An LLM can leverage its world knowledge about a city's demographics, economy, and geography to transfer patterns from a data-rich source site to a brand-new station.

```
State College (10 years of data) ──→ LLM reasoning
                                          ↓
"Altoona: mid-size city, ~40k population,
 manufacturing-based, no university, 45 miles from State College..."
                                          ↓
                             Zero-shot demand forecast for Altoona
```

**Research angle:** *Zero-Shot EV Demand Forecasting for New Charging Sites Using LLMs* — directly applicable with real data from State College as the source site.

**Dataset:** `datasets/dataset8_zero_shot_transfer.csv`
- State College as source site; 5 Pennsylvania cities as zero-shot targets: Altoona, Harrisburg, Erie, Bethlehem, Philadelphia
- Each row: city characteristics + transfer reasoning prompt + monthly demand estimate
- 5 cities × 12 months = 60 transfer scenarios

---

## Dataset Overview

| # | Dataset | Samples | Format | LLM Role |
|---|---|---|---|---|
| 1 | `dataset1_timeseries.csv` | 8,730 | Numeric sequences | Forecaster |
| 2 | `dataset2_text_context.csv` | 364 | Numeric + text | Context encoder |
| 3 | `dataset3_qa_pairs.csv` | 357 | Prompt + answer | Reasoning engine |
| 4 | `dataset4_augmentation.csv` | 90 + scenarios | Real + synthetic | Data augmenter |
| 5 | `dataset5_anomaly.csv` | ~8,760 (hourly) | Timeseries + labels | Anomaly explainer |
| 6 | `dataset6_decision_support.csv` | 365 | Operational snapshot | Decision support |
| 7 | `dataset7_multiagent.csv` | 120 sessions | Multi-turn dialogue | Agent negotiation |
| 8 | `dataset8_zero_shot_transfer.csv` | 60 | City profile + prompt | Zero-shot transfer |

All datasets are grounded in the same virtual charging station. Demand patterns — seasonal cycles, Penn State event spikes, weekday/weekend variation — are consistent across all eight datasets.

---

## Learning Roadmap

```
Phase 1 — Core LLM mechanics
  Use Case 1: Numeric sequences as tokens         ← dataset1  (Chronos / TimesFM)
  Use Case 2: Text embeddings meet numeric models ← dataset2  (BERT + LSTM)
  Use Case 3: End-to-end LLM prediction + explain ← dataset3  (LLaMA)

Phase 2 — Applied extensions
  Use Case 4: Synthetic data for rare events      ← dataset4
  Use Case 5: Anomaly detection + explanation     ← dataset5

Phase 3 — System-level research
  Use Case 6: Operational decision support        ← dataset6
  Use Case 8: Zero-shot transfer to new sites     ← dataset8  (real data ready)

Phase 4 — High-impact frontier
  Use Case 7: Multi-agent V2G simulation          ← dataset7  (connects to CAV work)
```

---

## Next Session — Starting Point

Pick up from **Use Case 1**, end-to-end from scratch:

1. **Connect to remote server** — SSH into the compute server, verify GPU availability
2. **Set up Python environment** — create a conda/venv environment, install dependencies (PyTorch, Chronos or TimesFM, pandas, etc.)
3. **Understand the mechanics** — walk through how numeric sequences are tokenized and embedded in a time series LLM
4. **Run the code** — train or run inference on `dataset1_timeseries.csv`
5. **See the predictions** — visualize predicted vs. actual demand for the next 6 hours

Once Use Case 1 is working end-to-end, move on to Use Case 2, then Use Case 3.

---

## Background

This project explores LLM techniques through EV charging demand forecasting — a domain with 10 years of real charging session data from Kansas City, Missouri. The eight synthetic datasets here are designed for learning and prototyping; Use Case 8 in particular is ready to be replicated with real data once the modeling pipeline is established.
