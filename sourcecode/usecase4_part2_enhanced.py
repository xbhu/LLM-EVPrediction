# =============================================================================
# Use Case 4 – Part 2 (Revised): LLM Generation with Better Prompting
# Run 1: explicit scale constraints only
# Run 2: scale constraints + few-shot examples from real data
# Compare both against pre-existing synthetic data
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
DATA_PATH      = "/home/xzh5180/Research/llm-evprediction/datasets/dataset4_augmentation.csv"
OUTPUT_DIR     = "/home/xzh5180/Research/llm-evprediction/outputs/usecase4_generation_v2"
MODEL_NAME     = "Qwen/Qwen3-4B"
N_SAMPLES      = 24        # one full day, one row per hour
TEMPERATURE    = 0.7
MAX_NEW_TOKENS = 128
N_FEWSHOT      = 4         # how many real examples to show in Run 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = ["cold_snap", "ev_fleet_event", "grid_outage", "price_spike"]
COLORS = {
    "pre-existing":  {"cold_snap": "#d7191c", "ev_fleet_event": "#fdae61",
                      "grid_outage": "#1a9641", "price_spike": "#9e0142"},
    "constraints":   "#1f78b4",
    "fewshot":       "#33a02c",
    "real":          "#aaaaaa",
}

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

FIXED_MONTH      = 1
FIXED_DOW        = 2
FIXED_IS_WEEKEND = 0
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# Load model & real data
# =============================================================================
print("Loading Qwen3-4B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print("✓ Model loaded\n")

df_orig = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
real_df = df_orig[df_orig["source"] == "real"].reset_index(drop=True)
df_orig["scenario"] = df_orig.apply(
    lambda r: r["augmentation_scenario"] if r["source"] == "synthetic" else "real",
    axis=1
)

# Pre-compute real data stats for the constraints string
r_mean = real_df["demand_kwh"].mean()
r_std  = real_df["demand_kwh"].std()
r_min  = real_df["demand_kwh"].min()
r_max  = real_df["demand_kwh"].max()
t_min  = real_df["temperature_f"].min()
t_max  = real_df["temperature_f"].max()

CONSTRAINTS = (
    f"Constraints (based on real station data):\n"
    f"  - demand_kwh is typically between {r_min:.0f} and {r_max:.0f} kWh "
    f"(mean {r_mean:.0f}, std {r_std:.0f})\n"
    f"  - temperature_f is typically between {t_min:.0f} and {t_max:.0f}°F "
    f"for this location\n"
    f"  - This is a single EV charging station, not a grid or fleet total\n"
    f"  - Adjust values to reflect the scenario, but stay within physically "
    f"plausible ranges for a single station"
)

# =============================================================================
# Helper: sample few-shot examples from real data for a given hour
# =============================================================================
def get_fewshot_examples(hour: int, n: int = N_FEWSHOT) -> str:
    """Return n real rows near the target hour as formatted examples."""
    # grab rows within ±2 hours
    nearby = real_df[abs(real_df["hour"] - hour) <= 2].sample(
        min(n, len(real_df)), random_state=hour
    ).head(n)
    lines = ["Examples from real normal-condition data (for scale reference):"]
    for _, row in nearby.iterrows():
        lines.append(
            f"  hour={int(row['hour']):02d}, month={int(row['month'])}, "
            f"temp={row['temperature_f']:.1f}°F → demand={row['demand_kwh']:.1f} kWh"
        )
    return "\n".join(lines)

# =============================================================================
# Prompt builders
# =============================================================================
SYSTEM_MSG = (
    "You are a data generator for EV charging demand simulation. "
    "Given a scenario and time context, output a single JSON object "
    "with exactly these fields: temperature_f (float), demand_kwh (float). "
    "Output ONLY the JSON object. No explanation, no markdown, no extra text."
)

def build_prompt_v1(scenario_desc, hour, month, dow, is_weekend):
    """Run 1: constraints only."""
    user = (
        f"Scenario: {scenario_desc}\n\n"
        f"{CONSTRAINTS}\n\n"
        f"Time context:\n"
        f"  hour: {hour} (0=midnight, 8=morning, 20=evening)\n"
        f"  month: {month}\n"
        f"  day_of_week: {dow} (0=Mon, 6=Sun)\n"
        f"  is_weekend: {is_weekend}\n\n"
        f"Generate realistic temperature_f and demand_kwh for this scenario and time."
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_MSG},
         {"role": "user",   "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def build_prompt_v2(scenario_desc, hour, month, dow, is_weekend):
    """Run 2: constraints + few-shot examples."""
    fewshot = get_fewshot_examples(hour)
    user = (
        f"Scenario: {scenario_desc}\n\n"
        f"{CONSTRAINTS}\n\n"
        f"{fewshot}\n\n"
        f"Time context:\n"
        f"  hour: {hour} (0=midnight, 8=morning, 20=evening)\n"
        f"  month: {month}\n"
        f"  day_of_week: {dow} (0=Mon, 6=Sun)\n"
        f"  is_weekend: {is_weekend}\n\n"
        f"Now generate realistic temperature_f and demand_kwh for the "
        f"SCENARIO above (not normal conditions). The scenario should shift "
        f"values from the normal examples shown."
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_MSG},
         {"role": "user",   "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

# =============================================================================
# Generation & parsing
# =============================================================================
def parse_output(text: str):
    match = re.search(r'\{[^}]+\}', text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if "temperature_f" in obj and "demand_kwh" in obj:
            return {"temperature_f": float(obj["temperature_f"]),
                    "demand_kwh":    float(obj["demand_kwh"])}
    except (json.JSONDecodeError, ValueError):
        pass
    return None

def generate_row(prompt_fn, scenario_desc, hour, month, dow, is_weekend):
    prompt = prompt_fn(scenario_desc, hour, month, dow, is_weekend)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_output(text)

def run_generation(run_name, prompt_fn):
    print(f"\n{'='*60}")
    print(f"RUN: {run_name}")
    print('='*60)
    all_rows = []
    for sc in SCENARIOS:
        print(f"\n  Scenario: {sc}")
        rows = []
        for hour in range(N_SAMPLES):
            result = generate_row(
                prompt_fn, SCENARIO_PROMPTS[sc],
                hour, FIXED_MONTH, FIXED_DOW, FIXED_IS_WEEKEND
            )
            if result:
                result.update({"hour": hour, "scenario": sc, "run": run_name})
                rows.append(result)
                print(f"    hour={hour:02d}  temp={result['temperature_f']:6.1f}°F  "
                      f"demand={result['demand_kwh']:8.2f} kWh")
            else:
                print(f"    hour={hour:02d}  [parse failed]")
        all_rows.extend(rows)
        ok = len(rows)
        print(f"  → {ok}/{N_SAMPLES} rows generated")
    return pd.DataFrame(all_rows)

# =============================================================================
# Run 1 and Run 2
# =============================================================================
df_v1 = run_generation("constraints_only", build_prompt_v1)
df_v2 = run_generation("constraints+fewshot", build_prompt_v2)

df_v1.to_csv(os.path.join(OUTPUT_DIR, "generated_v1_constraints.csv"), index=False)
df_v2.to_csv(os.path.join(OUTPUT_DIR, "generated_v2_fewshot.csv"), index=False)
print(f"\n✓ Saved {len(df_v1)} rows (v1) and {len(df_v2)} rows (v2)")

# =============================================================================
# Stats comparison
# =============================================================================
print("\n" + "="*60)
print("STATS COMPARISON: pre-existing vs v1 vs v2")
print("="*60)
print(f"\n{'Scenario':<20} {'Source':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 76)
for sc in SCENARIOS:
    orig = df_orig[df_orig["scenario"] == sc]["demand_kwh"]
    print(f"{sc:<20} {'pre-existing':<22} "
          f"{orig.mean():8.2f} {orig.std():8.2f} "
          f"{orig.min():8.2f} {orig.max():8.2f}")
    for label, df_gen in [("v1 constraints", df_v1), ("v2 +fewshot", df_v2)]:
        sub = df_gen[df_gen["scenario"] == sc]["demand_kwh"]
        if len(sub):
            print(f"{'':<20} {label:<22} "
                  f"{sub.mean():8.2f} {sub.std():8.2f} "
                  f"{sub.min():8.2f} {sub.max():8.2f}")
    print()

# =============================================================================
# Plot: hourly profiles for all three sources, one subplot per scenario
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
real_hourly = real_df.groupby("hour")["demand_kwh"].mean()

for ax, sc in zip(axes, SCENARIOS):
    # pre-existing
    orig_h = df_orig[df_orig["scenario"] == sc].groupby("hour")["demand_kwh"].mean()
    ax.plot(orig_h.index, orig_h.values,
            color=COLORS["pre-existing"][sc], linewidth=2.5,
            label="pre-existing synthetic")

    # v1 constraints only
    v1_sc = df_v1[df_v1["scenario"] == sc]
    if len(v1_sc):
        ax.plot(v1_sc["hour"], v1_sc["demand_kwh"],
                color=COLORS["constraints"], linewidth=1.8,
                linestyle="--", marker="o", markersize=4,
                label="v1: constraints only")

    # v2 constraints + few-shot
    v2_sc = df_v2[df_v2["scenario"] == sc]
    if len(v2_sc):
        ax.plot(v2_sc["hour"], v2_sc["demand_kwh"],
                color=COLORS["fewshot"], linewidth=1.8,
                linestyle=":", marker="s", markersize=4,
                label="v2: +few-shot")

    # real reference
    ax.plot(real_hourly.index, real_hourly.values,
            color=COLORS["real"], linewidth=1.2,
            linestyle="-", alpha=0.5, label="real (reference)")

    ax.set_title(sc, fontsize=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("demand_kwh")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(
    "Hourly Demand: Pre-existing vs v1 (constraints) vs v2 (constraints+fewshot)",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_comparison_all.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Comparison plot saved")
print(f"✓ All outputs in: {OUTPUT_DIR}")
