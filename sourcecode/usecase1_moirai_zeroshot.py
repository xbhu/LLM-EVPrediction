"""
Use Case 1: EV Charging Demand Forecasting with MOIRAI (Zero-shot)
Salesforce's pretrained time series foundation model

Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset1_timeseries.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase1_moirai_zeroshot/"
N_EVAL     = 200
PRED_LEN   = 6
CTX_LEN    = 24

# 之前的结果（用于对比）
CHRONOS_MAE  = [9.69, 13.96, 15.79, 16.60, 16.29, 15.70]
CHRONOS_MAPE = 21.3
TIMESFM_MAE  = [10.18, 14.34, 15.28, 16.23, 17.27, 20.06]
TIMESFM_MAPE = 22.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Use Case 1: EV Demand Forecasting with MOIRAI (Zero-shot)")
print("=" * 60)

# ── Step 1: 导入 MOIRAI ───────────────────────────────────────────────────────
print("\n[Step 1] 导入 MOIRAI...")
print("  如果报错，先运行：pip install uni2ts")

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from einops import rearrange
except ImportError:
    print("\n  ❌ uni2ts 未安装，请先运行：")
    print("     pip install uni2ts")
    exit(1)

print("  ✅ MOIRAI 导入成功")

# ── Step 2: 加载数据 ──────────────────────────────────────────────────────────
print("\n[Step 2] 加载数据...")
df = pd.read_csv(DATA_PATH)

history_cols = [f"demand_t-{i}" for i in range(24, 0, -1)]
target_cols  = [f"target_t+{i}" for i in range(1, 7)]

np.random.seed(42)
eval_idx  = np.random.choice(len(df), size=N_EVAL, replace=False)
eval_df   = df.iloc[eval_idx].reset_index(drop=True)
histories = eval_df[history_cols].values.astype(np.float32)   # (200, 24)
targets   = eval_df[target_cols].values.astype(np.float32)    # (200, 6)

print(f"  评估样本数: {N_EVAL}")
print(f"  输入: {CTX_LEN}小时 → 预测: {PRED_LEN}小时")

# ── Step 3: 加载 MOIRAI 模型 ──────────────────────────────────────────────────
print("\n[Step 3] 加载 MOIRAI 模型...")
print("  模型: Salesforce/moirai-1.0-R-small (91M params)")
print("  第一次运行会从 HuggingFace 下载权重...")

model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
    prediction_length=PRED_LEN,
    context_length=CTX_LEN,
    patch_size="auto",       # 自动选择最合适的 patch 大小
    num_samples=100,         # 生成 100 个样本，取中位数
    target_dim=1,            # 单变量（只用需求）
    feat_dynamic_real_dim=0, # 无额外动态特征（zero-shot 先不用）
    past_feat_dynamic_real_dim=0,
)
model.eval()
print("  ✅ MOIRAI 加载成功")

# ── Step 4: 运行 MOIRAI 预测 ──────────────────────────────────────────────────
print("\n[Step 4] 开始预测...")

all_predictions = []
BATCH_SIZE = 32

for start in range(0, len(histories), BATCH_SIZE):
    end   = min(start + BATCH_SIZE, len(histories))
    batch = histories[start:end]   # (B, 24)
    B     = len(batch)

    # MOIRAI 输入格式：(batch, time, variate)
    past_target = torch.tensor(batch).unsqueeze(-1)   # (B, 24, 1)

    # 构造 observed mask（全部为 True，表示数据完整）
    past_observed_target = torch.ones(B, CTX_LEN, 1, dtype=torch.bool)

    with torch.no_grad():
        # forecast 返回 (batch, num_samples, prediction_length, variate)
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
        )

    # 取中位数，shape: (B, PRED_LEN)
    median_pred = forecast.median(dim=1).values.squeeze(-1).numpy()
    all_predictions.append(median_pred)

    print(f"  进度: {end}/{len(histories)}")

predictions = np.vstack(all_predictions)   # (200, 6)
print(f"  ✅ 预测完成，输出 shape: {predictions.shape}")

# ── Step 5: 评估精度 ──────────────────────────────────────────────────────────
print("\n[Step 5] 评估预测精度...")

mae_per_h  = []
rmse_per_h = []

for h in range(PRED_LEN):
    mae  = mean_absolute_error(targets[:, h], predictions[:, h])
    rmse = mean_squared_error(targets[:, h], predictions[:, h]) ** 0.5
    mae_per_h.append(mae)
    rmse_per_h.append(rmse)
    print(f"  t+{h+1}h  MAE: {mae:.2f} kWh   RMSE: {rmse:.2f} kWh")

overall_mae  = mean_absolute_error(targets.flatten(), predictions.flatten())
overall_rmse = mean_squared_error(targets.flatten(), predictions.flatten()) ** 0.5
mape         = overall_mae / targets.mean() * 100

print(f"\n  总体 MAE:  {overall_mae:.2f} kWh")
print(f"  总体 RMSE: {overall_rmse:.2f} kWh")
print(f"  MAPE:      {mape:.1f}%")
print(f"\n  三模型对比：")
print(f"  Chronos  MAPE: {CHRONOS_MAPE}%")
print(f"  TimesFM  MAPE: {TIMESFM_MAPE}%")
print(f"  MOIRAI   MAPE: {mape:.1f}%")

# ── Step 6: 画图 ──────────────────────────────────────────────────────────────
print("\n[Step 6] 生成图表...")

# 图1：4个预测示例
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("EV Charging Demand Forecasting — MOIRAI (Zero-shot)\nSmart Mobility Lab, Penn State",
             fontsize=13, fontweight="bold")

for i in range(4):
    ax  = axes[i // 2][i % 2]
    idx = i * 50
    ax.plot(range(-24, 0), histories[idx], color="steelblue",
            lw=1.5, label="History (24h)")
    ax.plot(range(0, 6), targets[idx], color="green",
            lw=2, marker="o", label="Ground Truth")
    ax.plot(range(0, 6), predictions[idx], color="tomato",
            lw=2, marker="s", linestyle="--", label="MOIRAI Forecast")
    ax.axvline(x=0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Hours (0 = forecast start)")
    ax.set_ylabel("Demand (kWh)")
    ax.set_title(f"Example {i+1}  |  {eval_df['timestamp'].iloc[idx]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "predictions.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}predictions.png")

# 图2：三模型 MAE 对比
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Zero-shot Comparison: Chronos vs TimesFM vs MOIRAI\nSmart Mobility Lab, Penn State",
              fontsize=12, fontweight="bold")

x = np.arange(1, 7)
w = 0.25
ax1.bar(x - w,   CHRONOS_MAE, w, label="Chronos",  color="steelblue", alpha=0.85)
ax1.bar(x,       TIMESFM_MAE, w, label="TimesFM",  color="orange",    alpha=0.85)
ax1.bar(x + w,   mae_per_h,   w, label="MOIRAI",   color="tomato",    alpha=0.85)
ax1.set_xlabel("Forecast Horizon (hours ahead)")
ax1.set_ylabel("MAE (kWh)")
ax1.set_title("MAE by Forecast Horizon")
ax1.set_xticks(x)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# 总体 MAPE 对比柱状图
models = ["Chronos", "TimesFM", "MOIRAI"]
mapes  = [CHRONOS_MAPE, TIMESFM_MAPE, mape]
colors = ["steelblue", "orange", "tomato"]
ax2.bar(models, mapes, color=colors, alpha=0.85, width=0.5)
ax2.set_ylabel("MAPE (%)")
ax2.set_title("Overall MAPE Comparison")
ax2.set_ylim(0, max(mapes) * 1.3)
for i, (m, v) in enumerate(zip(models, mapes)):
    ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "comparison.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}comparison.png")

print("\n" + "=" * 60)
print("✅ Use Case 1 MOIRAI Zero-shot 完成")
print(f"   Chronos MAPE:  {CHRONOS_MAPE}%")
print(f"   TimesFM MAPE:  {TIMESFM_MAPE}%")
print(f"   MOIRAI  MAPE:  {mape:.1f}%")
print(f"   输出目录: {OUTPUT_DIR}")
print("=" * 60)
