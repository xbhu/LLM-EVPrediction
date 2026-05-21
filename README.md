# LLM-EVPrediction

A hands-on learning project that explores **eight distinct roles** Large Language Models (LLMs) can play in EV charging demand forecasting. Each use case comes with a purpose-built dataset, working Python code, and output examples — making this a self-contained reference for researchers and practitioners who want to understand how LLMs fit into a real transportation energy problem.

**Core prediction task:** Given the past 24 hours of data, predict charging demand for the next 6 hours.

**Virtual site:** A charging station in State College, PA. All eight datasets share the same demand patterns — seasonal cycles, Penn State event spikes, weekday/weekend variation — so results across use cases are directly comparable.

---

## Repository Structure

```
llm-evprediction/
├── datasets/          # CSV datasets for all 8 use cases
├── sourcecode/        # Python scripts — one per model per use case
│   ├── usecase1_chronos.py
│   ├── usecase2_llama.py
│   └── ...
└── outputs/           # Generated plots, logs, and evaluation results
    ├── usecase1_chronos/
    ├── usecase2_llama/
    └── ...
```

Script naming convention: `usecase{N}_{model}.py` — each script is self-contained and maps directly to a subfolder under `outputs/`.

---

## Setup

**Python:** 3.10+

**Core dependencies:**

```bash
pip install torch transformers datasets peft accelerate
pip install pandas numpy matplotlib scikit-learn
```

**Use-case-specific dependencies:**

| Use Case | Additional packages |
|---|---|
| UC1 — Time Series LLM | `chronos-forecasting`, `timesfm`, `uni2ts` (MOIRAI) |
| UC5 — Anomaly RAG | `faiss-cpu`, `sentence-transformers` |
| UC6 — Decision RAG | `faiss-cpu`, `sentence-transformers` |

**GPU:** A single GPU with 16–24 GB VRAM is sufficient for all scripts using QLoRA (4-bit quantization). Zero-shot inference scripts run on smaller GPUs or CPU.

---

## Eight Use Cases

### Use Case 1 — LLM as a Time Series Forecaster

**Role:** Direct numeric prediction

**Core idea:** A Transformer-based LLM processes number sequences the same way it processes word sequences. Just as a language model predicts the next word, a time series LLM predicts the next value.

```
Language model:  "The  cat  sat  on  the  mat" → predicts next word
Time series LLM:  45.2  67.8  89.3  76.1  54.2  43.0 → predicts next number
```

**Research angle:** *Time Series LLM for EV Charging Demand Forecasting* — benchmark foundation models against LSTM and Transformer baselines; analyze performance under limited training data.

**Dataset:** `datasets/dataset1_timeseries.csv`
- 8,730 samples
- Each row: 24-hour rolling window of hourly demand + temperature → 6-hour future demand
- Pure numeric format, ready to feed directly into a time series model

**Models explored:**

| Model | Developer | Parameters | Architecture | Fine-tuning | Key Characteristics |
|---|---|---|---|---|---|
| Chronos | Amazon | 8M – 710M | T5 Encoder-Decoder | Supported | Tokenizes real values into discrete bins; trained on a large corpus of public time series; multiple model sizes (Tiny → Large) |
| TimesFM | Google | 200M | Patch Transformer (Decoder-only) | Not supported | Processes time series as patches rather than individual tokens; strong zero-shot performance |
| MOIRAI | Salesforce | 14M – 311M | Universal Time Series Transformer | Supported | Designed for multivariate forecasting; residual correction applied as a workaround for the univariate task |

> **Note:** All three models are zero-shot forecasters by default — no labeled training data is required. Chronos additionally supports fine-tuning on domain-specific data when available.

---

### Use Case 2 — LLM as a Context Encoder

**Role:** Text → vector feature extraction for hybrid models

**Core idea:** Pure numeric models are blind to unstructured real-world context. An LLM encoder converts natural language descriptions into vectors that augment a traditional forecasting model.

```
[Numeric sequence] → LSTM ──────────────────────→ final prediction
                                                         ↑
[Text context]     → LLM → context vector ───────────────┘
"Penn State home game tonight, expect high downtown traffic"
```

**Research angle:** *Context-Aware EV Demand Prediction* — incorporate unstructured contextual signals (event calendars, weather reports, policy text) to improve forecast accuracy.

**Dataset:** `datasets/dataset2_text_context.csv`
- 364 daily samples (one full year)
- Each row: daily summary statistics + natural-language description → next-day total demand
- Events include game days, holidays, and graduation ceremonies

**Models explored:**

| Model | Developer | Parameters | Architecture | Fine-tuning Method | Key Characteristics |
|---|---|---|---|---|---|
| Flan-T5-base | Google | 250M | Encoder-Decoder | Full fine-tune | Instruction-tuned T5; adapted for regression via a linear head; lightweight and fast to train |
| GPT-2 Medium | OpenAI | 345M | Decoder-only | Full fine-tune | Causal language model repurposed for regression; text and numeric context concatenated as a single sequence |
| Llama 3.1 8B | Meta | 8B | Decoder-only | QLoRA (4-bit) | Strong general-purpose instruction follower; quantized to 4-bit with LoRA adapters to fit on a single GPU |
| Mistral 7B | Mistral AI | 7B | Decoder-only | QLoRA (4-bit) | Apache 2.0 license; grouped-query attention and sliding window attention for efficiency |
| Gemma 2 9B | Google | 9B | Decoder-only | QLoRA (4-bit) | Trained with knowledge distillation from a larger teacher model; strong reasoning relative to its size |
| Gemma 4 E4B | Google | ~4B (active) | Decoder-only (MoE) | QLoRA (attempted) | Mixture-of-Experts variant; PEFT incompatibility encountered — run in zero-shot mode only |
| Qwen3-8B | Alibaba | 8B | Decoder-only | QLoRA (4-bit) | Thinking Mode for chain-of-thought reasoning; trained on 36T tokens |
| DeepSeek-R1-0528-Qwen3-8B | DeepSeek | 8B | Decoder-only | QLoRA (attempted) | Reinforcement-learning-based reasoning model; fine-tuning unstable — used in zero-shot mode only |
| DeepSeek-V2-Lite-Chat | DeepSeek | 15.7B total / 2.4B active | MoE Decoder-only | Not attempted | MoE architecture; incompatible with `transformers` 5.x at the time of writing |

> **Note:** This use case covers the full spectrum from lightweight Encoder-Decoder models to large Decoder-only models with parameter-efficient fine-tuning. MoE models are included to illustrate how active-parameter counts differ from total parameter counts.

---

### Use Case 3 — LLM as a Reasoning Engine

**Role:** End-to-end prediction with natural language explanation

**Core idea:** Feed the LLM a complete prompt with historical data and context. It returns a forecast plus a human-readable explanation — directly useful for operational reporting.

```
Input:  "Past 24h data: [45.2, 67.8, 89.3...]. Tomorrow: Friday,
         game day, forecast 82°F. Predict next 6 hours."

Output: "Predicted demand: [92.3, 105.6, 118.2, 97.4, 76.3, 61.2]
         Key drivers: event traffic (+23%), temperature effect (+8%)..."
```

**Research angle:** *Explainable EV Demand Forecasting with LLMs* — evaluate instruction-tuned models on structured prediction + explanation tasks; explore domain-specific fine-tuning.

**Dataset:** `datasets/dataset3_qa_pairs.csv`
- 357 QA pairs
- Each row: a complete LLM prompt (history + context) paired with a structured reference answer
- Ready for zero-shot evaluation or fine-tuning

**Models explored:**

| Model | Developer | Parameters | Architecture | Fine-tuning Method | Key Characteristics |
|---|---|---|---|---|---|
| Flan-T5-large | Google | 780M | Encoder-Decoder | Full fine-tune | Adapted for structured text generation (forecast values + explanation); all weights updated |
| Phi-4-mini-instruct | Microsoft | 3.8B | Decoder-only | LoRA | Trained on high-quality synthetic data; strong reasoning relative to its small size |
| Llama-3.2-3B-Instruct | Meta | 3.21B | Decoder-only | LoRA | Compact instruction-tuned model distilled from Llama 3.1 8B |
| Gemma-3-4B-it | Google | 4B | Decoder-only | LoRA | Sliding window attention for long-context efficiency; instruction-tuned variant |
| Qwen3-4B | Alibaba | 4B | Decoder-only | LoRA | Thinking Mode enables step-by-step reasoning before the final answer |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek | 7.6B | Decoder-only | QLoRA (4-bit) | Reasoning model distilled from DeepSeek-R1-671B; structured chain-of-thought behavior |

> **Note:** Encoder-Decoder models (Flan-T5) suit seq2seq tasks; Decoder-only instruction-tuned models suit open-ended generation. LoRA vs. QLoRA reflects a memory–quality trade-off: QLoRA adds 4-bit quantization to further reduce GPU memory.

---

### Use Case 4 — LLM as a Data Augmenter

**Role:** Synthetic data generation for rare and extreme scenarios

**Core idea:** Real datasets rarely capture extreme events — blizzards, grid failures, unexpected mass gatherings. LLMs can generate realistic synthetic samples for edge cases, expanding the training set in a controllable way.

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
- 90-day baseline + 4 synthetic scenario types
- Each augmented row includes the generation prompt used to create it
- Scenario types: cold snap, grid outage, fleet charging event, electricity price spike

**Models explored:**

This use case has two stages: (1) **synthetic data generation** and (2) **demand prediction** on the augmented dataset.

*Stage 1 — Synthetic data generation:*

| Model | Prompting Strategy | Key Characteristics |
|---|---|---|
| Claude (Anthropic) | Structured prompt with scenario description | Used to pre-generate the baseline dataset |
| Qwen3-4B | Zero-shot + few-shot prompting | Run locally to generate additional scenario-conditioned samples; demonstrates how any instruction-tuned LLM can act as a data generator |

*Stage 2 — Demand prediction on augmented data:*

| Model | Type | Key Characteristics |
|---|---|---|
| XGBoost | Traditional ML (gradient boosting) | Strong tabular baseline; trained on real-only vs. real+augmented data to isolate the augmentation effect |
| Flan-T5-base | LLM (Encoder-Decoder) | Fine-tuned on the augmented dataset to evaluate whether LLM-generated data helps an LLM-based predictor |

> **Note:** The two-stage design makes the augmentation contribution measurable — by training the same downstream model with and without synthetic data, you can directly quantify the effect.

---

### Use Case 5 — LLM as an Anomaly Explainer

**Role:** Post-hoc interpretation of prediction deviations

**Core idea:** When a model's prediction is far off, the LLM investigates why — cross-referencing context, flagging likely causes, and finding analogous historical events.

```
Actual demand:    150 kWh
Model prediction:  62 kWh  ← large gap
                      ↓
                  [LLM]
                      ↓
"This deviation likely corresponds to an unrecorded campus event.
 Historical data shows similar spikes on Oct 28 and Nov 11."
```

**Research angle:** *LLM-Assisted Anomaly Detection and Explanation in EV Charging Systems* — high practical value for operational dashboards and reporting.

**Dataset:** `datasets/dataset5_anomaly.csv`
- Full year of hourly data with 7 injected anomaly events
- Anomaly types: equipment failure, unrecorded activity, heat wave, sensor error, unexpected event
- Each anomalous row includes an explanation template

**Models explored:**

Three routes were implemented, ranging from zero-shot prompting to fine-tuning to retrieval-augmented generation (RAG):

| Route | Model | Approach | Fine-tuning | Key Characteristics |
|---|---|---|---|---|
| A — Zero-shot | Flan-T5-base | Prompt with anomaly context; request explanation | None | Lightweight baseline; relies on pre-trained knowledge only |
| A — Zero-shot | Qwen3-4B | Same zero-shot prompting with Thinking Mode | None | Structured chain-of-thought before the explanation |
| B — Fine-tune | Flan-T5-large | Seq2seq fine-tuning on labeled explanation pairs | Full fine-tune | Learns to map anomaly signatures to templated explanations |
| C — RAG | Qwen3-4B + FAISS | Retrieve similar historical anomalies; inject into prompt | None | Uses `sentence-transformers` for embedding and FAISS for retrieval; grounds explanations in historical precedent |

> **Note:** The three routes illustrate a key design choice: zero-shot (no data, fast), fine-tuning (labeled data, best domain fit), and RAG (unlabeled history, retrieval-grounded). RAG is especially natural for anomaly explanation because the most useful context is historical precedent.

---

### Use Case 6 — LLM as a Decision Support System

**Role:** Translating forecasts into operational recommendations

**Core idea:** Beyond predicting demand, the LLM synthesizes forecast + pricing + grid state into actionable operator guidance — moving from "what will happen" to "what should we do."

```
Input:  forecast + current electricity price + grid load + queue length
           ↓
       [LLM]
           ↓
Output: "Raise charging price 15% from 2–4 PM.
         Expected peak load reduction: 23%.
         Projected impact on user satisfaction: minimal."
```

**Research angle:** *LLM-Assisted Dynamic Pricing and Demand Management for EV Charging* — directly applicable to grid operator and transportation agency needs.

**Dataset:** `datasets/dataset6_decision_support.csv`
- 365 daily operational snapshots
- Each row: demand forecast + electricity price + grid stress + queue length + decision prompt + reference recommendation

**Models explored:**

A single model was progressively enhanced across three stages to isolate the contribution of each technique:

| Stage | Approach | Model | Key Characteristics |
|---|---|---|---|
| Stage 1 — Zero-shot | Prompt with operational snapshot; request recommendation | Qwen3-4B | No training; relies on pre-trained knowledge of grid operations and pricing |
| Stage 2 — SFT | Supervised fine-tuning on reference recommendations | Qwen3-4B (LoRA) | Learns the recommendation format and operational priorities from the dataset |
| Stage 3 — RAG | Retrieve similar past operational days; inject as context | Qwen3-4B + FAISS (`all-MiniLM-L6-v2`) | Grounds recommendations in historical precedent; no additional training required |

> **Note:** This staged design mirrors a realistic deployment workflow: zero-shot to validate feasibility, fine-tuning for domain alignment, RAG for situational grounding. Each stage is independently evaluable.

---

### Use Case 7 — LLM-Based Multi-Agent Simulation

**Role:** Role-playing agents that negotiate charging schedules

**Core idea:** Multiple LLM agents each represent a stakeholder — the EV driver, the charging station operator, and the grid manager. They negotiate in natural language to reach an optimal charging schedule.

```
Agent A (EV driver):        "I need 80% charge before 5 PM."
Agent B (charging station): "3 spots available, $0.18/kWh."
Agent C (grid operator):    "Peak hours 4–6 PM — requesting load reduction."
              ↓
         Three-way negotiation
              ↓
         Optimal charging schedule + V2G dispatch
```

**Research angle:** *LLM-Based Multi-Agent Simulation for EV Charging Coordination* — connects naturally to connected and automated vehicles (CAV) and vehicle-to-grid (V2G) research.

**Dataset:** `datasets/dataset7_multiagent.csv`
- 120 negotiation sessions across 3 scenario types: peak-hour conflict, overnight fleet charging, V2G emergency discharge
- Each session: full three-agent dialogue, outcome, and satisfaction scores for all parties

**Models explored:**

| Step | Setup | Model(s) | Key Characteristics |
|---|---|---|---|
| Step 1a — Minimal prompt | All three agents with minimal role instructions | Qwen3-4B | Baseline for coherent multi-turn negotiation |
| Step 1b — Rich prompt | Detailed role descriptions, constraints, and objectives per agent | Qwen3-4B | Tests whether richer prompting improves negotiation quality without changing the model |
| Step 1c — Heterogeneous personas | Agents assigned distinct communication styles (assertive, cooperative, regulatory) | Qwen3-4B | Explores whether persona diversity produces more realistic dynamics |
| Step 2 — LLM-as-Judge | Automated evaluation of negotiation transcripts | Qwen3-4B | Scores sessions on outcome quality, argument coherence, and fairness |
| Step 3 — SFT Grid Agent | Fine-tune only the Grid Agent on high-quality examples | Qwen3-4B (LoRA) | Demonstrates selective agent specialization; 240 training samples |

> **Note:** Multi-agent simulation does not require multiple model instances — a single model plays all roles sequentially. Key design choices are prompt isolation (preventing agents from seeing each other's internal state) and turn management.

---

### Use Case 8 — LLM-Enabled Zero-Shot Transfer

**Role:** Predicting demand at new sites with no historical data

**Core idea:** Traditional models require local historical data. An LLM can leverage world knowledge about a city's demographics, economy, and geography to transfer patterns from a data-rich source site to a brand-new station.

```
State College (data-rich source) ──→ LLM reasoning
                                          ↓
"Altoona: mid-size city, ~40k population,
 manufacturing-based, no university, 45 miles from State College..."
                                          ↓
                             Zero-shot demand forecast for Altoona
```

**Research angle:** *Zero-Shot EV Demand Forecasting for New Charging Sites Using LLMs* — directly applicable when deploying new stations without sufficient historical data.

**Dataset:** `datasets/dataset8_zero_shot_transfer.csv`
- State College as source site; 5 Pennsylvania cities as zero-shot targets: Altoona, Harrisburg, Erie, Bethlehem, Philadelphia
- Each row: city characteristics + transfer reasoning prompt + monthly demand estimate
- 5 cities × 12 months = 60 transfer scenarios

**Models explored:**

| Model | Developer | Parameters | Approach | Key Characteristics |
|---|---|---|---|---|
| Qwen3-4B | Alibaba | 4B | Zero-shot transfer with structured city-profile prompt | Receives source-site statistics, target-city profile, and seasonal context; reasons about demand scaling factors |

**Prompt ablation study** — five prompt variants were tested to identify which components of the context matter most:

| Variant | What is removed | Purpose |
|---|---|---|
| Full prompt | Nothing — all context included | Baseline |
| No city description | City demographic/economic narrative removed | Tests whether qualitative context adds value beyond raw statistics |
| No analyst notes | Domain-expert hints about transfer factors removed | Tests whether curated guidance improves reasoning |
| No source data | Source-site reference statistics removed | Tests whether the model can reason without a source anchor |
| Numeric-only | All natural language removed; only numeric values | Lower bound — pure numeric reasoning |

> **Note:** The ablation design answers a practical question: *which parts of the prompt is the model actually using?* If removing city descriptions causes no degradation, the model relies on pre-trained world knowledge alone. If removing source data causes degradation, the source anchor is essential for calibration.

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

All datasets are grounded in the same virtual charging station. Demand patterns — seasonal cycles, event-day spikes, weekday/weekend variation — are consistent across all eight datasets, making cross-use-case comparisons meaningful.

---

## How to Navigate This Project

The eight use cases are organized into four phases of increasing complexity:

```
Phase 1 — Core LLM mechanics
  Use Case 1: Numeric sequences as tokens         (Chronos / TimesFM / MOIRAI)
  Use Case 2: Text embeddings meet numeric models  (Flan-T5, GPT-2, Llama, Mistral, Qwen3...)
  Use Case 3: End-to-end LLM prediction + explain  (Phi, Llama, Gemma, Qwen3, DeepSeek...)

Phase 2 — Applied extensions
  Use Case 4: Synthetic data for rare events
  Use Case 5: Anomaly detection + explanation      (zero-shot / fine-tune / RAG)

Phase 3 — System-level applications
  Use Case 6: Operational decision support         (zero-shot → SFT → RAG)
  Use Case 8: Zero-shot transfer to new sites      (prompt ablation study)

Phase 4 — Multi-agent frontier
  Use Case 7: Multi-agent negotiation simulation   (V2G / CAV connection)
```

Each phase builds on the concepts of the previous one, but all use cases are self-contained — you can start anywhere depending on your research interest.

---

## Background

This project uses EV charging demand forecasting as a concrete application domain to explore a wide range of LLM techniques. The eight datasets are synthetically generated around a virtual charging station, designed to exhibit realistic demand patterns (seasonal trends, event-driven spikes, weather sensitivity) while remaining fully open for research use.

The framework is intentionally modular: each use case addresses a different research question, uses a different LLM paradigm, and can be extended independently. The shared dataset design makes it straightforward to compare approaches across use cases or to swap in new models as the field evolves.
