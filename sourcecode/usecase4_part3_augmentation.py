# =============================================================================
# Use Case 4 – Part 3: Augmentation Experiment (Traditional ML)
# Compare XGBoost trained on:
#   Model A: real data only
#   Model B: real + all synthetic scenarios
#   Model C: real + single scenario (ablation, one model per scenario)
# Evaluate on: real test set + each scenario's test set
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset4_augmentation.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase4_augmentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES  = ["temperature_f", "hour", "day_of_week", "month", "is_weekend"]
TARGET    = "demand_kwh"
SCENARIOS = ["cold_snap", "ev_fleet_event", "grid_outage", "price_spike"]
COLORS = {
    "Model A (real only)": "#2c7bb6",
    "Model B (real + all)": "#d7191c",
}
SCENARIO_COLORS = {
    "cold_snap":      "#d7191c",
    "ev_fleet_event": "#fdae61",
    "grid_outage":    "#1a9641",
    "price_spike":    "#9e0142",
}

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# Load & split data
# =============================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["scenario"] = df.apply(
    lambda r: r["augmentation_scenario"] if r["source"] == "synthetic" else "real",
    axis=1
)

# Real data: sorted by timestamp, first 75 days train / last 15 days test
real_df = df[df["source"] == "real"].sort_values("timestamp").reset_index(drop=True)
real_train = real_df.iloc[:1800]   # 75 days × 24h
real_test  = real_df.iloc[1800:]   # 15 days × 24h

print(f"Real train: {len(real_train)} rows  |  Real test: {len(real_test)} rows")

# Synthetic data: last 360 rows per scenario as test, rest as train pool
syn_train = {}
syn_test  = {}
for sc in SCENARIOS:
    sc_df = df[df["scenario"] == sc].reset_index(drop=True)
    syn_train[sc] = sc_df.iloc[:1800]
    syn_test[sc]  = sc_df.iloc[1800:]
    print(f"{sc}: {len(syn_train[sc])} train  |  {len(syn_test[sc])} test")


def get_XY(data: pd.DataFrame):
    return data[FEATURES].values, data[TARGET].values


def evaluate(model, X, y, label=""):
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)
    if label:
        print(f"  {label:<30}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred, "y_true": y}

# =============================================================================
# Model A: real data only
# =============================================================================
print("\n" + "="*60)
print("MODEL A: Real data only")
print("="*60)

X_train_A, y_train_A = get_XY(real_train)
model_A = XGBRegressor(**XGB_PARAMS)
model_A.fit(X_train_A, y_train_A)

results_A = {}
results_A["real_test"] = evaluate(model_A, *get_XY(real_test), "real test")
for sc in SCENARIOS:
    results_A[sc] = evaluate(model_A, *get_XY(syn_test[sc]), sc)

# =============================================================================
# Model B: real + all synthetic
# =============================================================================
print("\n" + "="*60)
print("MODEL B: Real + all synthetic scenarios")
print("="*60)

augmented_B = pd.concat([real_train] + [syn_train[sc] for sc in SCENARIOS],
                         ignore_index=True)
print(f"Training size: {len(augmented_B)} rows")

X_train_B, y_train_B = get_XY(augmented_B)
model_B = XGBRegressor(**XGB_PARAMS)
model_B.fit(X_train_B, y_train_B)

results_B = {}
results_B["real_test"] = evaluate(model_B, *get_XY(real_test), "real test")
for sc in SCENARIOS:
    results_B[sc] = evaluate(model_B, *get_XY(syn_test[sc]), sc)

# =============================================================================
# Model C: real + single scenario (ablation)
# =============================================================================
print("\n" + "="*60)
print("MODEL C: Ablation — real + one scenario at a time")
print("="*60)

models_C  = {}
results_C = {}
for sc in SCENARIOS:
    augmented_C = pd.concat([real_train, syn_train[sc]], ignore_index=True)
    model_C = XGBRegressor(**XGB_PARAMS)
    model_C.fit(*get_XY(augmented_C))
    models_C[sc] = model_C
    print(f"\n  Training with real + {sc} ({len(augmented_C)} rows):")
    results_C[sc] = {}
    results_C[sc]["real_test"] = evaluate(model_C, *get_XY(real_test), "real test")
    results_C[sc][sc] = evaluate(model_C, *get_XY(syn_test[sc]), sc)

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: MAE by model and test set")
print("="*60)

test_sets = ["real_test"] + SCENARIOS
header = f"{'Test Set':<22}" + f"{'Model A':>12}" + f"{'Model B':>12}"
for sc in SCENARIOS:
    header += f"{'C+'+sc[:8]:>14}"
print(header)
print("-" * (22 + 12 + 12 + 14 * len(SCENARIOS)))

for ts in test_sets:
    row = f"{ts:<22}"
    row += f"{results_A[ts]['mae']:12.2f}"
    row += f"{results_B[ts]['mae']:12.2f}"
    for sc in SCENARIOS:
        val = results_C[sc].get(ts, {}).get("mae", float("nan"))
        row += f"{val:14.2f}"
    print(row)

# =============================================================================
# Plot 1: MAE comparison — A vs B, per test set
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(test_sets))
width = 0.35
mae_A = [results_A[ts]["mae"] for ts in test_sets]
mae_B = [results_B[ts]["mae"] for ts in test_sets]

bars_A = ax.bar(x - width/2, mae_A, width, label="Model A (real only)",
                color=COLORS["Model A (real only)"], alpha=0.85)
bars_B = ax.bar(x + width/2, mae_B, width, label="Model B (real + all)",
                color=COLORS["Model B (real + all)"], alpha=0.85)

for bar in bars_A:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in bars_B:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(test_sets, rotation=15, ha="right")
ax.set_ylabel("MAE (kWh)")
ax.set_title("Model A vs Model B: MAE by Test Set\n"
             "(lower = better; key question: does B improve on A for extreme scenarios?)",
             fontsize=11)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot1_A_vs_B_mae.png"), dpi=150)
plt.close()
print("\n✓ Plot 1: A vs B MAE")

# =============================================================================
# Plot 2: Ablation — Model C per scenario vs Model A baseline
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(SCENARIOS))
width = 0.35

mae_A_sc  = [results_A[sc]["mae"] for sc in SCENARIOS]
mae_C_sc  = [results_C[sc][sc]["mae"] for sc in SCENARIOS]

bars_base = ax.bar(x - width/2, mae_A_sc, width,
                   label="Model A (no augmentation)", color="#2c7bb6", alpha=0.85)
bars_aug  = ax.bar(x + width/2, mae_C_sc, width,
                   label="Model C (real + matching scenario)", alpha=0.85,
                   color=[SCENARIO_COLORS[sc] for sc in SCENARIOS])

for bar in bars_base:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in bars_aug:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(SCENARIOS, rotation=10)
ax.set_ylabel("MAE (kWh)")
ax.set_title("Ablation: Model A vs Model C\n"
             "(C trained with matching scenario data; does targeted augmentation help?)",
             fontsize=11)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot2_ablation_mae.png"), dpi=150)
plt.close()
print("✓ Plot 2: Ablation MAE")

# =============================================================================
# Plot 3: Prediction vs actual — real test set (do A and B differ on normal days?)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
y_true = results_A["real_test"]["y_true"]
for ax, model_label, res in zip(
    axes,
    ["Model A (real only)", "Model B (real + all)"],
    [results_A["real_test"], results_B["real_test"]]
):
    ax.scatter(y_true, res["y_pred"], alpha=0.3, s=8,
               color=COLORS[model_label])
    lims = [min(y_true.min(), res["y_pred"].min()),
            max(y_true.max(), res["y_pred"].max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_title(f"{model_label}\nMAE={res['mae']:.2f}  R²={res['r2']:.3f}",
                 fontsize=10)
    ax.set_xlabel("Actual demand_kwh")
    ax.set_ylabel("Predicted demand_kwh")
    ax.grid(alpha=0.3)
plt.suptitle("Prediction vs Actual on Real Test Set", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot3_pred_vs_actual_real.png"), dpi=150)
plt.close()
print("✓ Plot 3: Pred vs actual (real test)")

# =============================================================================
# Plot 4: MAE improvement — B vs A, per extreme scenario test set
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
improvements = [results_A[sc]["mae"] - results_B[sc]["mae"] for sc in SCENARIOS]
bar_colors = ["#2ca25f" if v > 0 else "#de2d26" for v in improvements]
bars = ax.bar(SCENARIOS, improvements, color=bar_colors, alpha=0.85)
for bar, val in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.2 if val >= 0 else -1.2),
            f"{val:+.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("MAE improvement (kWh)\npositive = Model B is better")
ax.set_title("MAE Improvement: Model B vs Model A\non Extreme Scenario Test Sets",
             fontsize=11)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot4_mae_improvement.png"), dpi=150)
plt.close()
print("✓ Plot 4: MAE improvement B vs A")

print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
