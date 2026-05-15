# =============================================================================
# Use Case 4 – Part 1: EDA + Quality Assessment
# Three layers:
#   Layer 1: Distribution visualization
#   Layer 2: Physical consistency check
#   Layer 3: Internal consistency test
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset4_augmentation.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase4_eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = ["real", "cold_snap", "ev_fleet_event", "grid_outage", "price_spike"]
COLORS = {
    "real":           "#2c7bb6",
    "cold_snap":      "#d7191c",
    "ev_fleet_event": "#fdae61",
    "grid_outage":    "#1a9641",
    "price_spike":    "#9e0142",
}
FEATURES = ["temperature_f", "hour", "day_of_week", "month", "is_weekend"]
TARGET   = "demand_kwh"
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["scenario"] = df.apply(
    lambda r: r["augmentation_scenario"] if r["source"] == "synthetic" else "real",
    axis=1
)

# =============================================================================
# LAYER 1: Distribution Visualization
# =============================================================================
print("=" * 60)
print("LAYER 1: Distribution Visualization")
print("=" * 60)

# Plot 1a: KDE of demand by scenario
fig, ax = plt.subplots(figsize=(10, 5))
for sc in SCENARIOS:
    sub = df[df["scenario"] == sc][TARGET]
    sub.plot.kde(ax=ax, label=sc, color=COLORS[sc], linewidth=2)
ax.set_title("Demand Distribution by Scenario (KDE)", fontsize=13)
ax.set_xlabel("demand_kwh")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot1a_demand_kde.png"), dpi=150)
plt.close()
print("✓ Plot 1a: demand KDE")

# Plot 1b: Hourly demand profile
fig, ax = plt.subplots(figsize=(11, 5))
for sc in SCENARIOS:
    sub = df[df["scenario"] == sc]
    hourly = sub.groupby("hour")[TARGET].mean()
    lw = 2.5 if sc == "real" else 1.8
    ls = "-"  if sc == "real" else "--"
    ax.plot(hourly.index, hourly.values,
            label=sc, color=COLORS[sc], linewidth=lw, linestyle=ls)
ax.set_title("Average Hourly Demand Profile by Scenario", fontsize=13)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean demand_kwh")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot1b_hourly_profile.png"), dpi=150)
plt.close()
print("✓ Plot 1b: hourly profile")

# Summary stats
print("\n--- Demand Summary ---")
summary = df.groupby("scenario")[TARGET].agg(
    mean="mean", std="std", min="min", max="max",
    p95=lambda x: x.quantile(0.95)
).round(2).loc[SCENARIOS]
print(summary)

# =============================================================================
# LAYER 2: Physical Consistency Check
# =============================================================================
print("\n" + "=" * 60)
print("LAYER 2: Physical Consistency Check")
print("=" * 60)
print("Checking: does each scenario behave in the right direction?")
print("Expected:")
print("  cold_snap      → lower temp, higher demand (battery heating)")
print("  ev_fleet_event → highest demand (concurrent charging)")
print("  grid_outage    → lower mean, high variance (stations unavailable)")
print("  price_spike    → similar mean, demand shifts away from peak hours")

# Check 1: Mean temperature per scenario
print("\n--- Mean Temperature by Scenario ---")
temp_summary = df.groupby("scenario")["temperature_f"].mean().round(2).loc[SCENARIOS]
print(temp_summary)

# Check 2: Temperature → demand slope (linear regression coefficient)
print("\n--- Temperature-Demand Slope (should be negative for cold_snap) ---")
for sc in SCENARIOS:
    sub = df[df["scenario"] == sc]
    if sub["temperature_f"].std() > 0:
        z = np.polyfit(sub["temperature_f"], sub[TARGET], 1)
        print(f"  {sc:<18}: slope = {z[0]:+.3f}  "
              f"(demand changes {z[0]:+.2f} kWh per 1°F)")

# Check 3: Demand variance (grid_outage should be highest)
print("\n--- Demand Std Dev by Scenario (grid_outage should be highest) ---")
std_summary = df.groupby("scenario")[TARGET].std().round(2).loc[SCENARIOS]
print(std_summary)

# Check 4: Peak hour concentration (fleet event should be more concentrated)
print("\n--- Peak Hour Concentration (CV of hourly mean demand) ---")
print("  Higher CV = more concentrated in certain hours")
for sc in SCENARIOS:
    sub = df[df["scenario"] == sc]
    hourly = sub.groupby("hour")[TARGET].mean()
    cv = hourly.std() / hourly.mean()
    print(f"  {sc:<18}: CV = {cv:.3f}")

# Plot 2: Temperature vs Demand scatter for real vs cold_snap
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, sc in zip(axes, ["real", "cold_snap"]):
    sub = df[df["scenario"] == sc]
    ax.scatter(sub["temperature_f"], sub[TARGET],
               alpha=0.15, s=6, color=COLORS[sc])
    z = np.polyfit(sub["temperature_f"], sub[TARGET], 1)
    xs = np.linspace(sub["temperature_f"].min(), sub["temperature_f"].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), color="black", linewidth=1.5, linestyle="--",
            label=f"slope={z[0]:+.2f}")
    ax.set_title(f"{sc}: Temperature vs Demand", fontsize=12)
    ax.set_xlabel("temperature_f")
    ax.set_ylabel("demand_kwh")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot2_temp_demand_scatter.png"), dpi=150)
plt.close()
print("\n✓ Plot 2: temperature vs demand scatter")

# =============================================================================
# LAYER 3: Internal Consistency Test
# =============================================================================
print("\n" + "=" * 60)
print("LAYER 3: Internal Consistency Test")
print("=" * 60)
print("For each scenario: train RandomForest on 80%, test on 20%")
print("If data has internal structure, R² should be reasonably high")
print("If data is random noise, R² ≈ 0\n")

results = []
for sc in SCENARIOS:
    sub = df[df["scenario"] == sc][FEATURES + [TARGET]].dropna()
    X = sub[FEATURES].values
    y = sub[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    results.append({"scenario": sc, "MAE": round(mae, 2), "R²": round(r2, 3),
                    "n_samples": len(sub)})
    print(f"  {sc:<18}: MAE={mae:.2f} kWh,  R²={r2:.3f}  (n={len(sub)})")

# Plot 3: R² comparison across scenarios
results_df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(results_df["scenario"], results_df["R²"],
              color=[COLORS[sc] for sc in results_df["scenario"]], alpha=0.8)
ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, label="R²=0.7 reference")
for bar, val in zip(bars, results_df["R²"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_title("Internal Consistency: R² by Scenario\n(higher = more learnable structure in synthetic data)",
             fontsize=12)
ax.set_ylabel("R²")
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', rotation=15)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot3_internal_consistency_r2.png"), dpi=150)
plt.close()
print("\n✓ Plot 3: internal consistency R²")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("PART 1 SUMMARY")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files saved:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
