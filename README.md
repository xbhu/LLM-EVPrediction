# LLM-EVPrediction

A learning project exploring three distinct ways Large Language Models (LLMs) can be applied to EV charging demand forecasting. All three use cases share the same synthetic dataset from a single virtual charging station, making it easy to compare approaches side by side.

**Prediction task:** Given the past 24 hours of data, predict charging demand for the next 6 hours.

---

## Three LLM Use Cases

### Use Case 1 — LLM as a Time Series Forecaster (Direct Numeric Prediction)

**Core idea:** A Transformer-based LLM processes number sequences the same way it processes word sequences. Just as a language model predicts the next word, a time series LLM predicts the next value.

```
Language model:  "The  cat  sat  on  the  mat" → predicts next word
Time series LLM:  45.2  67.8  89.3  76.1  54.2  43.0 → predicts next number
```

Representative models in this space include Google's **TimesFM** and Amazon's **Chronos**. The key advantage is leveraging pre-trained representations, especially under data-scarce conditions.

**Research angle:** *Time Series LLM for EV Charging Demand Forecasting* — benchmark against LSTM and Transformer baselines; analyze performance under limited training data.

**Dataset:** `datasets/dataset1_timeseries.csv`
- 8,730 samples
- Each row: 24-hour rolling window of hourly demand + temperature → 6-hour future demand
- Pure numeric format, ready to feed directly into a time series model

---

### Use Case 2 — LLM as a Context Encoder (Hybrid Text + Numeric Prediction)

**Core idea:** Pure numeric models are blind to unstructured real-world context — they cannot know that a Penn State home game tonight will spike downtown charging demand. A BERT-style encoder converts natural language descriptions into vectors that augment a traditional forecasting model.

```
[Numeric sequence] → LSTM → initial forecast
                                  ↓
[Text context]     → BERT → context vector       → final prediction
"Penn State home game tonight,
 expect high downtown traffic"
```

**Research angle:** *Context-Aware EV Demand Prediction* — incorporate unstructured contextual signals (events, weather reports, policy text) to improve forecast accuracy.

**Dataset:** `datasets/dataset2_text_context.csv`
- 364 daily samples (one full year)
- Each row: daily summary statistics + one natural-language description (weather, season, local events) → next-day total demand
- Events encoded include Penn State game days, holidays, and graduation ceremonies

---

### Use Case 3 — LLM as a Reasoning & Explanation Engine (QA Format)

**Core idea:** Feed the LLM a complete prompt containing historical data, contextual factors, and the prediction question. The LLM returns both a forecast and a human-readable explanation of key drivers — directly useful for reporting to agencies or stakeholders.

```
Input:
"Past 24 hours data: [45.2, 67.8, 89.3...]
 Tomorrow is Friday, Penn State game, forecast 82°F.
 Predict charging demand for the next 6 hours."

Output:
"Predicted demand: [92.3, 105.6, 118.2, 97.4, 76.3, 61.2]
 Key drivers: event traffic (+23%), temperature effect (+8%)..."
```

**Research angle:** *Explainable EV Demand Forecasting with LLMs* — evaluate LLaMA-style models on structured prediction + explanation tasks; explore fine-tuning on domain-specific QA pairs.

**Dataset:** `datasets/dataset3_qa_pairs.csv`
- 357 QA pairs
- Each row: a complete LLM prompt (history + context) paired with a structured answer
- Ready to use for zero-shot evaluation or fine-tuning

---

## Dataset Overview

| Dataset | Samples | Format | Target Use |
|---|---|---|---|
| `dataset1_timeseries.csv` | 8,730 | Numeric sequences | TimesFM / Chronos |
| `dataset2_text_context.csv` | 364 | Numeric + text | BERT + LSTM fusion |
| `dataset3_qa_pairs.csv` | 357 | Prompt + answer | LLaMA evaluation / fine-tuning |

All three datasets are derived from the same virtual charging station over the same one-year period. The underlying demand patterns — seasonal variation, Penn State event spikes, weekday/weekend cycles — are consistent across all three, so results are directly comparable across use cases.

---

## Learning Roadmap

```
Embedding fundamentals
        ↓
Use Case 1: How numeric sequences become vectors   ← dataset1
        ↓
Contrastive learning
        ↓
Use Case 2: Train a BERT that understands EV context ← dataset2
        ↓
RAG architecture
        ↓
Use Case 3: Full prediction + explanation system     ← dataset3
```

---

## Context

This project was set up to learn LLM techniques through a domain-familiar problem (EV charging demand forecasting). The three synthetic datasets were generated to support hands-on experimentation across the three use cases above.
