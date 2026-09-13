"""
Use Case 8: Analysis & Visualization
======================================
Input:
  - usecase8_zeroshot_v3_results.csv   (full-year predictions for 5 cities)
  - usecase8_ablation_results.csv      (ablation experiment results)
Output: 4 figures, saved to outputs/usecase8_zeroshot/

Fig 1: Annual seasonal curves — predicted values for 5 cities vs State College actual
Fig 2: City characteristics vs average transfer ratio scatter plot
Fig 3: Ablation heatmap — 5 conditions × 2 cities, avg ratio
Fig 4: Monthly ratio stability for Altoona and Harrisburg (variation across conditions per month)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OUTPUT_DIR   = "/home/xzh5180/Research/llm-evprediction/outputs/usecase8_zeroshot"
V3_FILE      = os.path.join(OUTPUT_DIR, "usecase8_zeroshot_v3_results.csv")
ABLATION_FILE= os.path.join(OUTPUT_DIR, "usecase8_ablation_results.csv")
FIG_DIR      = OUTPUT_DIR

MONTH_ORDER  = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]
CONDITIONS   = ["full","no_desc","no_notes","no_source","numeric_only"]
COND_LABELS  = {
    "full"        : "Full prompt",
    "no_desc"     : "No description",
    "no_notes"    : "No transfer notes",
    "no_source"   : "No source data",
    "numeric_only": "Numeric only",
}

# Short city labels for plots
CITY_SHORT = {
    "Altoona - Chestnut Ave Station"       : "Altoona",
    "Harrisburg - Market St Station"       : "Harrisburg",
    "Erie - Peach St Station"              : "Erie",
    "Bethlehem - Main St Station"          : "Bethlehem",
    "Philadelphia - University City Station": "Philadelphia",
}

COLORS = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B"]

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading results...")
v3_exists  = os.path.exists(V3_FILE)
abl_exists = os.path.exists(ABLATION_FILE)

if not v3_exists:
    print(f"  [WARN] v3 results not found: {V3_FILE}")
if not abl_exists:
    print(f"  [WARN] ablation results not found: {ABLATION_FILE}")

if v3_exists:
    df_v3 = pd.read_csv(V3_FILE)
    df_v3["month_name"] = pd.Categorical(df_v3["month_name"], categories=MONTH_ORDER, ordered=True)
    df_v3 = df_v3.sort_values(["site_name","month"])
    df_v3["city_short"] = df_v3["site_name"].map(CITY_SHORT)
    print(f"  v3: {len(df_v3)} rows, {df_v3['predicted_kwh'].notna().sum()} extracted")

if abl_exists:
    df_abl = pd.read_csv(ABLATION_FILE)
    df_abl["month_name"] = pd.Categorical(df_abl["month_name"], categories=MONTH_ORDER, ordered=True)
    df_abl["city_short"] = df_abl["site_name"].map(CITY_SHORT)
    df_abl["condition"]  = pd.Categorical(df_abl["condition"], categories=CONDITIONS, ordered=True)
    print(f"  ablation: {len(df_abl)} rows, {df_abl['predicted_kwh'].notna().sum()} extracted")

print()

# ─────────────────────────────────────────────
# Fig 1: Seasonal curves — predicted vs SC source
# ─────────────────────────────────────────────
if v3_exists:
    fig, ax = plt.subplots(figsize=(11, 5))

    # State College ground truth (same for all cities, just take from any city)
    sc_ref = df_v3.groupby("month")["source_avg_daily_demand_kwh"].first().reset_index()
    sc_ref = sc_ref.sort_values("month")
    ax.plot(
        range(1, 13), sc_ref["source_avg_daily_demand_kwh"],
        color="black", linewidth=2.5, linestyle="--",
        marker="s", markersize=5, label="State College (source, actual)", zorder=5
    )

    # Each target city
    cities = df_v3["site_name"].unique()
    for i, city in enumerate(cities):
        city_df = df_v3[df_v3["site_name"] == city].dropna(subset=["predicted_kwh"])
        city_df = city_df.sort_values("month")
        label   = CITY_SHORT.get(city, city)
        ax.plot(
            city_df["month"], city_df["predicted_kwh"],
            color=COLORS[i % len(COLORS)], linewidth=1.8,
            marker="o", markersize=4, label=f"{label} (predicted)", alpha=0.85
        )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([m[:3] for m in MONTH_ORDER], fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Daily Demand (kWh)")
    ax.set_title("Fig 1: LLM Zero-Shot Transfer — Seasonal Demand Curves\n"
                 "State College (actual) vs 5 Target Cities (LLM predicted)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    path1 = os.path.join(FIG_DIR, "fig1_seasonal_curves.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved: {path1}")

# ─────────────────────────────────────────────
# Fig 2: City characteristics vs avg transfer ratio
# ─────────────────────────────────────────────
if v3_exists:
    city_summary = df_v3.groupby("site_name").agg(
        avg_ratio       = ("transfer_ratio", "mean"),
        population      = ("population", "first"),
        ev_penetration  = ("ev_penetration_pct", "first"),
        avg_income_k    = ("avg_income_k", "first"),
    ).reset_index()
    city_summary["city_short"] = city_summary["site_name"].map(CITY_SHORT)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    features = [
        ("population",     "Population",          lambda x: f"{x/1000:.0f}k"),
        ("ev_penetration", "EV Penetration (%)",  lambda x: f"{x:.1f}%"),
        ("avg_income_k",   "Avg Income ($k)",     lambda x: f"${x}k"),
    ]

    for ax, (feat, xlabel, fmt) in zip(axes, features):
        for i, row in city_summary.iterrows():
            ax.scatter(row[feat], row["avg_ratio"],
                       color=COLORS[i % len(COLORS)], s=100, zorder=3)
            ax.annotate(row["city_short"],
                        (row[feat], row["avg_ratio"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=8)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                   label="ratio = 1.0 (= State College)")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Avg Transfer Ratio", fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt(x)))

    axes[0].set_title("Population vs Ratio")
    axes[1].set_title("EV Penetration vs Ratio")
    axes[2].set_title("Income vs Ratio")
    fig.suptitle("Fig 2: City Characteristics vs Avg Transfer Ratio", fontsize=11)
    plt.tight_layout()
    path2 = os.path.join(FIG_DIR, "fig2_characteristics_vs_ratio.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved: {path2}")

# ─────────────────────────────────────────────
# Fig 3: Ablation heatmap
# ─────────────────────────────────────────────
if abl_exists:
    pivot = df_abl.pivot_table(
        values="transfer_ratio",
        index="condition",
        columns="city_short",
        aggfunc="mean"
    ).reindex(CONDITIONS)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=1.5)
    plt.colorbar(im, ax=ax, label="Avg Transfer Ratio")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([COND_LABELS[c] for c in pivot.index], fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black" if 0.4 < val < 1.3 else "white")

    ax.set_title("Fig 3: Ablation — Avg Transfer Ratio by Condition × City\n"
                 "(green = closer to SC, red = further)", fontsize=10)
    plt.tight_layout()
    path3 = os.path.join(FIG_DIR, "fig3_ablation_heatmap.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Saved: {path3}")

# ─────────────────────────────────────────────
# Fig 4: Month-level ratio stability across conditions
# ─────────────────────────────────────────────
if abl_exists:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    ablation_cities = ["Altoona", "Harrisburg"]

    for ax, city_short in zip(axes, ablation_cities):
        city_df = df_abl[df_abl["city_short"] == city_short]

        for i, cond in enumerate(CONDITIONS):
            cond_df = city_df[city_df["condition"] == cond].dropna(subset=["transfer_ratio"])
            cond_df = cond_df.sort_values("month")
            ax.plot(
                cond_df["month"], cond_df["transfer_ratio"],
                label=COND_LABELS[cond],
                color=COLORS[i], linewidth=1.6,
                marker="o", markersize=3.5, alpha=0.85
            )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([m[:3] for m in MONTH_ORDER], fontsize=8)
        ax.set_xlabel("Month")
        ax.set_ylabel("Transfer Ratio")
        ax.set_title(f"{city_short}: Ratio per Month × Condition", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Fig 4: Monthly Ratio Stability — Do Ablation Conditions Change the Pattern?",
                 fontsize=11)
    plt.tight_layout()
    path4 = os.path.join(FIG_DIR, "fig4_monthly_stability.png")
    plt.savefig(path4, dpi=150)
    plt.close()
    print(f"Saved: {path4}")

# ─────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
if v3_exists:
    print("V3 CITY SUMMARY (avg ratio, all months):")
    cs = df_v3.groupby("city_short")["transfer_ratio"].mean().round(3)
    print(cs.to_string())

if abl_exists:
    print("\nABLATION SUMMARY (avg ratio by condition × city):")
    print(pivot.round(3).to_string())

print("\nDone. Check figures in:", FIG_DIR)
