"""
Use Case 8: Zero-Shot EV Demand Transfer — v3 (Robust Extraction)
=================================================================
Improvements over v2:
  - Output format changed to >>DEMAND / >>RATIO prefix to avoid confusion with reasoning text
  - Regex specifically matches >> prefix lines, significantly improving extraction success rate
"""

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset8_zero_shot_transfer.csv"
OUTPUT_DIR  = "/home/xzh5180/Research/llm-evprediction/outputs/usecase8_zeroshot"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "usecase8_zeroshot_v3_results.csv")
LOG_FILE    = os.path.join(OUTPUT_DIR, "usecase8_zeroshot_v3_log.txt")

MODEL_NAME     = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 600
TEMPERATURE    = 0.1
DO_SAMPLE      = True

# Format constraint appended to the end of every prompt
FORMAT_INSTRUCTION = """
At the very end of your response, write exactly these two lines and nothing after them:
>>DEMAND: [number]
>>RATIO: [number]

Where:
- DEMAND is your predicted average daily EV charging demand in kWh for the target site
- RATIO = your predicted demand / State College demand for the same month
  (ratio < 1 means lower than State College, ratio > 1 means higher)

Example of correct ending:
>>DEMAND: 850.5
>>RATIO: 0.59
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

# ─────────────────────────────────────────────
# Helper: run one inference
# ─────────────────────────────────────────────
def run_inference(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# ─────────────────────────────────────────────
# Helper: extract >>DEMAND and >>RATIO lines
# ─────────────────────────────────────────────
def extract_structured(response: str, source_kwh: float):
    """
    Match lines like:
      >>DEMAND: 850.5
      >>RATIO: 0.59
    Falls back to computing ratio from demand if >>RATIO line is missing.
    """
    demand_match = re.search(
        r">>DEMAND[:\s]+(\d+(?:\.\d+)?)", response, re.IGNORECASE
    )
    ratio_match = re.search(
        r">>RATIO[:\s]+(\d+(?:\.\d+)?)", response, re.IGNORECASE
    )

    predicted_kwh = float(demand_match.group(1)) if demand_match else None

    if ratio_match:
        transfer_ratio = float(ratio_match.group(1))
    elif predicted_kwh is not None and source_kwh > 0:
        transfer_ratio = round(predicted_kwh / source_kwh, 3)
    else:
        transfer_ratio = None

    return predicted_kwh, transfer_ratio

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows\n")

results   = []
log_lines = []

for idx, row in df.iterrows():
    site       = row["site_name"]
    month      = row["month_name"]
    source_kwh = row["source_avg_daily_demand_kwh"]

    prompt = row["zero_shot_prompt"] + FORMAT_INSTRUCTION

    print(f"[{idx+1:02d}/60] {site} — {month} ...", end=" ", flush=True)

    response = run_inference(prompt)
    predicted_kwh, transfer_ratio = extract_structured(response, source_kwh)

    if predicted_kwh is not None:
        print(f"predicted={predicted_kwh:.1f} kWh  ratio={transfer_ratio}")
    else:
        print("EXTRACT_FAILED")

    results.append({
        "site_id"                    : row["site_id"],
        "site_name"                  : site,
        "month"                      : row["month"],
        "month_name"                 : month,
        "population"                 : row["population"],
        "ev_penetration_pct"         : row["ev_penetration_pct"],
        "has_university"             : row["has_university"],
        "avg_income_k"               : row["avg_income_k"],
        "climate_zone"               : row["climate_zone"],
        "source_avg_daily_demand_kwh": source_kwh,
        "predicted_kwh"              : predicted_kwh,
        "transfer_ratio"             : transfer_ratio,
        "full_response"              : response,
    })

    log_lines.append("=" * 70)
    log_lines.append(f"[{idx+1:02d}] {site} | {month}")
    log_lines.append(f"SOURCE: {source_kwh} kWh  |  PREDICTED: {predicted_kwh}  |  RATIO: {transfer_ratio}")
    log_lines.append("--- RESPONSE ---")
    log_lines.append(response)
    log_lines.append("")

# ─────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nResults saved → {OUTPUT_FILE}")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"Full responses saved → {LOG_FILE}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY BY CITY")
print("=" * 60)
summary = results_df.groupby("site_name").agg(
    avg_source_kwh     = ("source_avg_daily_demand_kwh", "mean"),
    avg_predicted_kwh  = ("predicted_kwh", "mean"),
    avg_ratio          = ("transfer_ratio", "mean"),
    n_extracted        = ("predicted_kwh", "count"),
).round(3)
print(summary.to_string())

failed = results_df["predicted_kwh"].isna().sum()
print(f"\nExtraction failures: {failed}/60")
