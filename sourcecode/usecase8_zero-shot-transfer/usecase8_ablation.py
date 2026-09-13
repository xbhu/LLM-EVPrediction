"""
Use Case 8: Prompt Ablation Study
===================================
Goal: Test which types of information the LLM's predictions depend on
Cities: Altoona (ratio < 1) and Harrisburg (ratio > 1)
Conditions: 5 prompt versions, each × 12 months = 120 inferences

5 Ablation conditions:
  full          - full prompt (baseline)
  no_desc       - remove site_description (qualitative city description)
  no_notes      - remove transfer_notes (transfer reasoning guidance)
  no_source     - remove the specific State College numeric values
  numeric_only  - keep only numeric features, remove all natural language descriptions
"""

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset8_zero_shot_transfer.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase8_zeroshot"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "usecase8_ablation_results.csv")
LOG_FILE    = os.path.join(OUTPUT_DIR, "usecase8_ablation_log.txt")

MODEL_NAME     = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 600
TEMPERATURE    = 0.1
DO_SAMPLE      = True

# Run only these two cities
TARGET_CITIES = ["Altoona - Chestnut Ave Station", "Harrisburg - Market St Station"]

# Format constraint (same as v3)
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
# Prompt builder — 5 ablation conditions
# ─────────────────────────────────────────────
def build_prompt(row: pd.Series, condition: str) -> str:
    """
    Reconstruct prompt from dataset fields under each ablation condition.

    Dataset fields used:
      row["site_name"]                    — target city name
      row["month_name"]                   — month
      row["source_avg_daily_demand_kwh"]  — SC daily demand
      row["source_avg_peak_kwh"]          — SC peak demand
      row["source_avg_temp_f"]            — SC avg temperature
      row["site_description"]             — qualitative city description
      row["transfer_notes"]               — transfer reasoning guidance
      row["population"]                   — numeric features
      row["ev_penetration_pct"]
      row["avg_income_k"]
      row["has_university"]
    """

    site        = row["site_name"]
    month       = row["month_name"]
    src_demand  = row["source_avg_daily_demand_kwh"]
    src_peak    = row["source_avg_peak_kwh"]
    src_temp    = row["source_avg_temp_f"]
    description = row["site_description"]
    notes       = row["transfer_notes"]
    population  = row["population"]
    ev_pct      = row["ev_penetration_pct"]
    income      = row["avg_income_k"]
    has_univ    = "Yes" if row["has_university"] == 1 else "No"

    header = "You are an EV charging demand forecasting expert.\n\n"
    question = (
        f"\nQuestion: Estimate the average daily EV charging demand (kWh) "
        f"for {site} in {month}. Show your reasoning step by step, then give a final number."
    )

    # ── Source data block ──────────────────────────────────────────
    if condition == "no_source":
        source_block = (
            "You have data from a reference charging station:\n"
            "  Site: State College, PA (university town, pop. 42,000, EV penetration 8.2%)\n"
            f"  Month: {month}\n"
            "  [Specific demand figures withheld]\n"
        )
    else:
        source_block = (
            "You have detailed data from a reference charging station:\n"
            "  Site: State College, PA (university town, pop. 42,000, EV penetration 8.2%)\n"
            f"  Month: {month}\n"
            f"  Average daily demand: {src_demand:.1f} kWh\n"
            f"  Average peak hourly demand: {src_peak:.1f} kWh\n"
            f"  Average temperature: {src_temp:.1f}°F\n"
        )

    # ── Target site block ──────────────────────────────────────────
    if condition == "numeric_only":
        # Only numeric features, no qualitative language
        target_block = (
            f"\nTarget site with NO historical data:\n"
            f"  Site: {site}\n"
            f"  Month: {month}\n"
            f"  Population: {population:,}\n"
            f"  EV penetration: {ev_pct}%\n"
            f"  Average household income: ${income}k\n"
            f"  Has university: {has_univ}\n"
        )
    elif condition == "no_desc":
        # Remove site_description, keep numeric features only
        target_block = (
            f"\nTarget site with NO historical data:\n"
            f"  Site: {site}\n"
            f"  Population: {population:,}\n"
            f"  EV penetration: {ev_pct}%\n"
            f"  Average household income: ${income}k\n"
            f"  Has university: {has_univ}\n"
        )
    else:
        # full / no_notes / no_source — keep description
        target_block = (
            f"\nTarget site with NO historical data:\n"
            "  " + description + "\n"
        )

    # ── Transfer notes block ───────────────────────────────────────
    if condition in ("full", "no_desc", "no_source"):
        notes_block = f"\nTransfer guidance: {notes}\n"
    else:
        # no_notes and numeric_only: omit guidance
        notes_block = ""

    prompt = header + source_block + target_block + notes_block + question + FORMAT_INSTRUCTION
    return prompt


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
# Helper: inference
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
# Helper: extract >>DEMAND and >>RATIO
# ─────────────────────────────────────────────
def extract_structured(response: str, source_kwh: float):
    demand_match = re.search(r">>DEMAND[:\s]+(\d+(?:\.\d+)?)", response, re.IGNORECASE)
    ratio_match  = re.search(r">>RATIO[:\s]+(\d+(?:\.\d+)?)",  response, re.IGNORECASE)

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
df_subset = df[df["site_name"].isin(TARGET_CITIES)].reset_index(drop=True)
print(f"Subset: {len(df_subset)} rows × 5 conditions = {len(df_subset)*5} total inferences\n")

CONDITIONS = ["full", "no_desc", "no_notes", "no_source", "numeric_only"]

results   = []
log_lines = []
total     = len(df_subset) * len(CONDITIONS)
counter   = 0

for condition in CONDITIONS:
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition}")
    print(f"{'='*60}")

    for _, row in df_subset.iterrows():
        counter += 1
        site       = row["site_name"]
        month      = row["month_name"]
        source_kwh = row["source_avg_daily_demand_kwh"]

        prompt = build_prompt(row, condition)

        print(f"  [{counter:03d}/{total}] {site[:20]} — {month:>9} ...", end=" ", flush=True)

        response = run_inference(prompt)
        predicted_kwh, transfer_ratio = extract_structured(response, source_kwh)

        if predicted_kwh is not None:
            print(f"predicted={predicted_kwh:.1f}  ratio={transfer_ratio}")
        else:
            print("EXTRACT_FAILED")

        results.append({
            "condition"                  : condition,
            "site_name"                  : site,
            "month"                      : row["month"],
            "month_name"                 : month,
            "source_avg_daily_demand_kwh": source_kwh,
            "predicted_kwh"              : predicted_kwh,
            "transfer_ratio"             : transfer_ratio,
            "full_response"              : response,
        })

        log_lines.append("=" * 70)
        log_lines.append(f"[{counter:03d}] condition={condition} | {site} | {month}")
        log_lines.append(f"SOURCE: {source_kwh} | PREDICTED: {predicted_kwh} | RATIO: {transfer_ratio}")
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
# Summary: avg ratio by condition × city
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ABLATION SUMMARY: avg transfer_ratio by condition × city")
print("=" * 60)
pivot = results_df.pivot_table(
    values="transfer_ratio",
    index="condition",
    columns="site_name",
    aggfunc="mean"
).round(3)
# Reorder rows
pivot = pivot.reindex(CONDITIONS)
print(pivot.to_string())

print("\nExtraction failures by condition:")
fail_summary = results_df[results_df["predicted_kwh"].isna()].groupby("condition").size()
print(fail_summary.to_string() if len(fail_summary) > 0 else "  None")
