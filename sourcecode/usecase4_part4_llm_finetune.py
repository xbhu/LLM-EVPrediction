# =============================================================================
# Use Case 4 – Part 4: LLM Fine-tuning Augmentation Experiment
# Model: Flan-T5-base (seq2seq)
# Compare:
#   Model A: fine-tuned on real data only
#   Model B: fine-tuned on real + all synthetic scenarios
# Evaluate on: real test set + each scenario test set
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset4_augmentation.csv"
OUTPUT_DIR  = "/home/xzh5180/Research/llm-evprediction/outputs/usecase4_llm"
MODEL_NAME  = "google/flan-t5-base"
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 16

TRAIN_ARGS = dict(
    num_train_epochs        = 5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    learning_rate           = 3e-4,
    bf16                    = True,
    gradient_checkpointing  = True,
    save_strategy           = "epoch",
    eval_strategy           = "epoch",
    load_best_model_at_end  = True,
    save_total_limit        = 1,
    logging_steps           = 50,
    predict_with_generate   = True,
)

SCENARIOS = ["cold_snap", "ev_fleet_event", "grid_outage", "price_spike"]
SCENARIO_COLORS = {
    "cold_snap":      "#d7191c",
    "ev_fleet_event": "#fdae61",
    "grid_outage":    "#1a9641",
    "price_spike":    "#9e0142",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# Load & split data (same splits as Part 3)
# =============================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["scenario"] = df.apply(
    lambda r: r["augmentation_scenario"] if r["source"] == "synthetic" else "real",
    axis=1
)

real_df    = df[df["source"] == "real"].sort_values("timestamp").reset_index(drop=True)
real_train = real_df.iloc[:1800]
real_test  = real_df.iloc[1800:]

syn_train, syn_test = {}, {}
for sc in SCENARIOS:
    sc_df = df[df["scenario"] == sc].reset_index(drop=True)
    syn_train[sc] = sc_df.iloc[:1800]
    syn_test[sc]  = sc_df.iloc[1800:]

print(f"Real train: {len(real_train)}  |  Real test: {len(real_test)}")
for sc in SCENARIOS:
    print(f"{sc}: {len(syn_train[sc])} train  |  {len(syn_test[sc])} test")

# =============================================================================
# Text formatting
# =============================================================================
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

SCENARIO_DESC = {
    "cold_snap":      "cold snap (temperature well below average, battery heating demand)",
    "ev_fleet_event": "EV fleet charging event (large fleet arriving simultaneously)",
    "grid_outage":    "partial grid outage (some charging stations unavailable)",
    "price_spike":    "electricity price spike (users shift charging to off-peak hours)",
    "real":           "normal operating conditions",
}

def row_to_input(row) -> str:
    sc     = row["scenario"]
    hour   = int(row["hour"])
    month  = int(row["month"])
    dow    = int(row["day_of_week"])
    is_we  = int(row["is_weekend"])
    temp   = float(row["temperature_f"])

    time_of_day = ("midnight" if hour < 4 else "early morning" if hour < 7
                   else "morning" if hour < 12 else "afternoon" if hour < 17
                   else "evening" if hour < 21 else "night")

    return (
        f"predict EV charging demand: "
        f"scenario={SCENARIO_DESC[sc]}, "
        f"hour={hour} ({time_of_day}), "
        f"month={MONTH_NAMES[month-1]}, "
        f"day={DAY_NAMES[dow]}, "
        f"weekend={'yes' if is_we else 'no'}, "
        f"temperature={temp:.1f}F"
    )

def row_to_target(row) -> str:
    return f"{row['demand_kwh']:.2f}"

def df_to_hf_dataset(data: pd.DataFrame) -> Dataset:
    records = [{"input_text":  row_to_input(row),
                "target_text": row_to_target(row)}
               for _, row in data.iterrows()]
    return Dataset.from_list(records)

# =============================================================================
# Tokenizer
# =============================================================================
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        batch["target_text"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# =============================================================================
# Evaluation helper
# =============================================================================
def decode_predictions(pred_ids, label_ids):
    pred_ids  = np.where(pred_ids  != -100, pred_ids,  tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    preds  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return preds, labels

def parse_float_safe(s):
    try:
        return float(s.strip())
    except ValueError:
        return float("nan")

def evaluate_on_set(model, dataset_raw: pd.DataFrame, label: str, batch_size=64):
    ds = df_to_hf_dataset(dataset_raw).map(tokenize, batched=True,
                                            remove_columns=["input_text","target_text"])
    ds.set_format("torch")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    loader   = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                            collate_fn=collator)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            gen = model.generate(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                max_new_tokens = MAX_TARGET_LEN,
            )
            preds, trues = decode_predictions(
                gen.cpu().numpy(),
                batch["labels"].cpu().numpy()
            )
            all_preds.extend([parse_float_safe(p) for p in preds])
            all_true.extend( [parse_float_safe(t) for t in trues])

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    valid  = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred, y_true = y_pred[valid], y_true[valid]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    parse_fail = (~valid).sum()
    print(f"  {label:<30}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}"
          + (f"  [parse_fail={parse_fail}]" if parse_fail else ""))
    return {"mae": mae, "rmse": rmse, "r2": r2,
            "y_pred": y_pred, "y_true": y_true}

# =============================================================================
# Train helper
# =============================================================================
def train_model(run_name: str, train_data: pd.DataFrame, eval_data: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"Training: {run_name}  ({len(train_data)} train rows)")
    print('='*60)

    train_ds = df_to_hf_dataset(train_data).map(
        tokenize, batched=True, remove_columns=["input_text","target_text"])
    eval_ds  = df_to_hf_dataset(eval_data).map(
        tokenize, batched=True, remove_columns=["input_text","target_text"])

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, f"ckpt_{run_name}")

    args = Seq2SeqTrainingArguments(output_dir=ckpt_dir, **TRAIN_ARGS)
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        processing_class= tokenizer,
        data_collator   = collator,
    )
    trainer.train()
    return model

# =============================================================================
# Model A: real only
# =============================================================================
model_A = train_model("model_A_real_only", real_train, real_test)

print(f"\n--- Model A Evaluation ---")
results_A = {}
results_A["real_test"] = evaluate_on_set(model_A, real_test,  "real test")
for sc in SCENARIOS:
    results_A[sc] = evaluate_on_set(model_A, syn_test[sc], sc)

# =============================================================================
# Model B: real + all synthetic
# =============================================================================
augmented_B = pd.concat([real_train] + [syn_train[sc] for sc in SCENARIOS],
                         ignore_index=True).sample(frac=1, random_state=42)

model_B = train_model("model_B_augmented", augmented_B, real_test)

print(f"\n--- Model B Evaluation ---")
results_B = {}
results_B["real_test"] = evaluate_on_set(model_B, real_test,  "real test")
for sc in SCENARIOS:
    results_B[sc] = evaluate_on_set(model_B, syn_test[sc], sc)

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: MAE — Model A vs Model B")
print("="*60)
test_sets = ["real_test"] + SCENARIOS
print(f"{'Test Set':<22} {'Model A':>12} {'Model B':>12} {'Improvement':>14}")
print("-" * 62)
for ts in test_sets:
    mae_a = results_A[ts]["mae"]
    mae_b = results_B[ts]["mae"]
    diff  = mae_a - mae_b
    arrow = "↑ better" if diff > 0 else "↓ worse"
    print(f"{ts:<22} {mae_a:12.2f} {mae_b:12.2f}  {diff:+.2f} {arrow}")

# =============================================================================
# Plot 1: MAE comparison A vs B
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(test_sets))
w = 0.35
mae_A = [results_A[ts]["mae"] for ts in test_sets]
mae_B = [results_B[ts]["mae"] for ts in test_sets]
b1 = ax.bar(x - w/2, mae_A, w, label="Model A (real only)",   color="#2c7bb6", alpha=0.85)
b2 = ax.bar(x + w/2, mae_B, w, label="Model B (real + all)", color="#d7191c", alpha=0.85)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(test_sets, rotation=15, ha="right")
ax.set_ylabel("MAE (kWh)")
ax.set_title("Flan-T5 Fine-tuning: Model A vs B\n"
             "Does LLM training also benefit from synthetic augmentation?", fontsize=11)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot1_mae_A_vs_B.png"), dpi=150)
plt.close()
print("\n✓ Plot 1 saved")

# =============================================================================
# Plot 2: MAE improvement B vs A per extreme scenario
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
improvements = [results_A[sc]["mae"] - results_B[sc]["mae"] for sc in SCENARIOS]
colors = ["#2ca25f" if v > 0 else "#de2d26" for v in improvements]
bars = ax.bar(SCENARIOS, improvements, color=colors, alpha=0.85)
for bar, val in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.2 if val >= 0 else -1.0),
            f"{val:+.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("MAE improvement (kWh)\npositive = Model B better")
ax.set_title("LLM Augmentation Effect per Scenario\n(Flan-T5: real-only vs augmented)", fontsize=11)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot2_improvement.png"), dpi=150)
plt.close()
print("✓ Plot 2 saved")

# =============================================================================
# Plot 3: Pred vs actual — real test set, both models
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, label, res, color in zip(
    axes,
    ["Model A (real only)", "Model B (real + all)"],
    [results_A["real_test"], results_B["real_test"]],
    ["#2c7bb6", "#d7191c"],
):
    ax.scatter(res["y_true"], res["y_pred"], alpha=0.3, s=8, color=color)
    lims = [min(res["y_true"].min(), res["y_pred"].min()),
            max(res["y_true"].max(), res["y_pred"].max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_title(f"{label}\nMAE={res['mae']:.2f}  R²={res['r2']:.3f}", fontsize=10)
    ax.set_xlabel("Actual demand_kwh")
    ax.set_ylabel("Predicted demand_kwh")
    ax.grid(alpha=0.3)
fig.suptitle("Flan-T5: Prediction vs Actual on Real Test Set", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot3_pred_vs_actual.png"), dpi=150)
plt.close()
print("✓ Plot 3 saved")

print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
