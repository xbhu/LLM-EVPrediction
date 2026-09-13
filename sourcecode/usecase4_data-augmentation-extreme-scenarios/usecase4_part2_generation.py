# =============================================================================
# Use Case 4 – Part 2: LLM Generation Pipeline
# Use Qwen3-4B to generate new synthetic rows from scenario prompts
# Then compare generated data with pre-existing synthetic data
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset4_augmentation.csv"
OUTPUT_DIR  = "/home/xzh5180/Research/llm-evprediction/outputs/usecase4_generation"
MODEL_NAME  = "Qwen/Qwen3-4B"
N_SAMPLES   = 24   # generate 24 rows per scenario (one full day)
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = ["cold_snap", "ev_fleet_event", "grid_outage", "price_spike"]
COLORS = {
    "real":           "#2c7bb6",
    "cold_snap":      "#d7191c",
    "ev_fleet_event": "#fdae61",
    "grid_outage":    "#1a9641",
    "price_spike":    "#9e0142",
}

# Scenario prompts — one per scenario
SCENARIO_PROMPTS = {
    "cold_snap": (
        "Simulate an unexpected cold snap: temperature drops 20°F below seasonal "
        "average, EV charging demand increases due to battery heating needs."
    ),
    "ev_fleet_event": (
        "Simulate a large EV fleet charging event: a commercial fleet of 50 electric "
        "vehicles arrives at the charging station, causing a major demand surge."
    ),
    "grid_outage": (
        "Simulate a partial grid outage: some charging stations are unavailable, "
        "causing demand to drop sharply during the outage window."
    ),
    "price_spike": (
        "Simulate an electricity price spike: prices double during peak hours, "
        "causing users to shift charging to off-peak hours."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# Load model
# =============================================================================
print("Loading Qwen3-4B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("✓ Model loaded\n")

# =============================================================================
# Generation function
# =============================================================================
def build_prompt(scenario_description: str, hour: int, month: int,
                 day_of_week: int, is_weekend: int) -> str:
    """Build a prompt asking the LLM to generate one row of EV charging data."""
    system_msg = (
        "You are a data generator for EV charging demand simulation. "
        "Given a scenario description and time context, output a single JSON object "
        "with exactly these fields: temperature_f (float), demand_kwh (float). "
        "Output ONLY the JSON object, nothing else. No explanation, no markdown."
    )
    user_msg = (
        f"Scenario: {scenario_description}\n\n"
        f"Time context:\n"
        f"  hour: {hour} (0-23)\n"
        f"  month: {month} (1=Jan, 12=Dec)\n"
        f"  day_of_week: {day_of_week} (0=Mon, 6=Sun)\n"
        f"  is_weekend: {is_weekend}\n\n"
        f"Generate realistic values for temperature_f and demand_kwh "
        f"that match the scenario and time context."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )


def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output, return None if parsing fails."""
    # Try to find a {...} block
    match = re.search(r'\{[^}]+\}', text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        # Validate required fields
        if "temperature_f" in obj and "demand_kwh" in obj:
            return {
                "temperature_f": float(obj["temperature_f"]),
                "demand_kwh":    float(obj["demand_kwh"]),
            }
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def generate_row(scenario_description: str, hour: int, month: int,
                 day_of_week: int, is_weekend: int) -> dict | None:
    prompt = build_prompt(scenario_description, hour, month, day_of_week, is_weekend)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_json_output(text)

# =============================================================================
# Generate data for all scenarios
# =============================================================================
# Use a fixed "day" for consistent hour coverage: Jan weekday
FIXED_MONTH      = 1
FIXED_DOW        = 2   # Wednesday
FIXED_IS_WEEKEND = 0

all_generated = []

for scenario in SCENARIOS:
    print(f"\n{'='*50}")
    print(f"Generating {N_SAMPLES} rows for: {scenario}")
    print(f"Prompt: {SCENARIO_PROMPTS[scenario][:80]}...")
    print('='*50)

    rows = []
    failed = 0
    for hour in range(N_SAMPLES):   # 0..23, one row per hour
        result = generate_row(
            SCENARIO_PROMPTS[scenario],
            hour, FIXED_MONTH, FIXED_DOW, FIXED_IS_WEEKEND
        )
        if result:
            result.update({
                "hour": hour,
                "month": FIXED_MONTH,
                "day_of_week": FIXED_DOW,
                "is_weekend": FIXED_IS_WEEKEND,
                "scenario": scenario,
                "source": "llm_generated",
            })
            rows.append(result)
            print(f"  hour={hour:02d}  temp={result['temperature_f']:6.1f}°F  "
                  f"demand={result['demand_kwh']:7.2f} kWh")
        else:
            failed += 1
            print(f"  hour={hour:02d}  [parse failed]")

    print(f"  → {len(rows)}/{N_SAMPLES} rows generated, {failed} failed")
    all_generated.extend(rows)

generated_df = pd.DataFrame(all_generated)
generated_df.to_csv(os.path.join(OUTPUT_DIR, "generated_samples.csv"), index=False)
print(f"\n✓ Saved {len(generated_df)} generated rows")

# =============================================================================
# Comparison: LLM-generated vs pre-existing synthetic
# =============================================================================
print("\n" + "="*50)
print("COMPARISON: Generated vs Pre-existing Synthetic")
print("="*50)

original_df = pd.read_csv(DATA_PATH)
original_df["scenario"] = original_df.apply(
    lambda r: r["augmentation_scenario"] if r["source"] == "synthetic" else "real",
    axis=1
)

print("\n--- Demand Stats Comparison ---")
print(f"{'Scenario':<20} {'Source':<16} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 72)
for sc in SCENARIOS:
    # pre-existing
    orig = original_df[original_df["scenario"] == sc]["demand_kwh"]
    print(f"{sc:<20} {'pre-existing':<16} "
          f"{orig.mean():8.2f} {orig.std():8.2f} "
          f"{orig.min():8.2f} {orig.max():8.2f}")
    # generated
    gen = generated_df[generated_df["scenario"] == sc]["demand_kwh"]
    if len(gen) > 0:
        print(f"{'':<20} {'llm_generated':<16} "
              f"{gen.mean():8.2f} {gen.std():8.2f} "
              f"{gen.min():8.2f} {gen.max():8.2f}")
    print()

# =============================================================================
# Plot: Hourly demand profile — generated vs pre-existing vs real
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

real_hourly = original_df[original_df["scenario"] == "real"].groupby("hour")["demand_kwh"].mean()

for ax, sc in zip(axes, SCENARIOS):
    # pre-existing hourly mean
    orig_hourly = original_df[original_df["scenario"] == sc].groupby("hour")["demand_kwh"].mean()
    ax.plot(orig_hourly.index, orig_hourly.values,
            color=COLORS[sc], linewidth=2, label="pre-existing synthetic")

    # LLM-generated hourly
    gen_sc = generated_df[generated_df["scenario"] == sc]
    if len(gen_sc) > 0:
        ax.plot(gen_sc["hour"], gen_sc["demand_kwh"],
                color="black", linewidth=1.5, linestyle="--",
                marker="o", markersize=4, label="llm_generated (new)")

    # real as reference
    ax.plot(real_hourly.index, real_hourly.values,
            color=COLORS["real"], linewidth=1.2, linestyle=":",
            alpha=0.7, label="real (reference)")

    ax.set_title(sc, fontsize=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("demand_kwh")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Hourly Demand: LLM-Generated vs Pre-existing Synthetic vs Real",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_generated_vs_preexisting.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Comparison plot saved")

# =============================================================================
# Show a few raw outputs for inspection
# =============================================================================
print("\n--- Sample Generated Rows (first 5 per scenario) ---")
for sc in SCENARIOS:
    sub = generated_df[generated_df["scenario"] == sc].head(5)
    print(f"\n[{sc}]")
    print(sub[["hour", "temperature_f", "demand_kwh"]].to_string(index=False))

print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
