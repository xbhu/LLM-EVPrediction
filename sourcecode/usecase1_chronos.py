"""
Use Case 1: EV Charging Demand Forecasting with Chronos
Zero-shot time series prediction using Amazon's pretrained Chronos model

Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset1_timeseries.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase1_chronos/"
MODEL_NAME = "amazon/chronos-t5-small"
N_EVAL     = 200
N_SAMPLES  = 20
BATCH_SIZE = 20

# 自动创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Use Case 1: EV Demand Forecasting with Chronos")
print("=" * 60)

# ── Step 1: 加载数据 ──────────────────────────────────────────────────────────
print("\n[Step 1] 加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"  数据集大小: {len(df)} 个样本")
print(f"  每个样本: 过去24小时需求 → 预测未来6小时需求")

# ── Step 2: 加载 Chronos 模型 ─────────────────────────────────────────────────
print("\n[Step 2] 加载 Chronos 模型...")
print(f"  模型: {MODEL_NAME}")
print("  第一次运行会从 HuggingFace 下载模型权重，需要几分钟...")

from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
print("  ✅ 模型加载成功")

# ── Step 3: 准备评估数据 ──────────────────────────────────────────────────────
print("\n[Step 3] 准备评估样本...")

np.random.seed(42)
eval_idx = np.random.choice(len(df), size=N_EVAL, replace=False)
eval_df  = df.iloc[eval_idx].reset_index(drop=True)

history_cols = [f"demand_t-{i}" for i in range(24, 0, -1)]
target_cols  = [f"target_t+{i}" for i in range(1, 7)]

histories = eval_df[history_cols].values   # (200, 24)
targets   = eval_df[target_cols].values    # (200, 6)

print(f"  评估样本数: {N_EVAL}")
print(f"  输入长度: 24小时 → 预测: 6小时")

# ── Step 4: 运行 Chronos 预测 ─────────────────────────────────────────────────
print("\n[Step 4] 开始预测...")
print(f"  每个样本生成 {N_SAMPLES} 个预测，取中位数作为最终结果")

all_predictions = []

for start in range(0, len(histories), BATCH_SIZE):
    end     = min(start + BATCH_SIZE, len(histories))
    batch   = histories[start:end]
    context = [torch.tensor(h, dtype=torch.float32) for h in batch]

    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            prediction_length=6,
            num_samples=N_SAMPLES,
        )

    median_pred = np.median(forecast.numpy(), axis=1)
    all_predictions.append(median_pred)
    print(f"  进度: {end}/{len(histories)}")

predictions = np.vstack(all_predictions)  # (200, 6)
print("  ✅ 预测完成")

# ── Step 5: 评估精度 ──────────────────────────────────────────────────────────
print("\n[Step 5] 评估预测精度...")

mae_per_h  = []
rmse_per_h = []

for h in range(6):
    mae  = mean_absolute_error(targets[:, h], predictions[:, h])
    rmse = mean_squared_error(targets[:, h], predictions[:, h]) ** 0.5
    mae_per_h.append(mae)
    rmse_per_h.append(rmse)
    print(f"  t+{h+1}h  MAE: {mae:.2f} kWh   RMSE: {rmse:.2f} kWh")

overall_mae  = mean_absolute_error(targets.flatten(), predictions.flatten())
overall_rmse = mean_squared_error(targets.flatten(), predictions.flatten()) ** 0.5
mean_demand  = targets.mean()
mape         = overall_mae / mean_demand * 100

print(f"\n  总体 MAE:  {overall_mae:.2f} kWh")
print(f"  总体 RMSE: {overall_rmse:.2f} kWh")
print(f"  平均需求:  {mean_demand:.2f} kWh")
print(f"  MAPE:      {mape:.1f}%")

# ── Step 6: 画图 ──────────────────────────────────────────────────────────────
print("\n[Step 6] 生成图表...")

# 图1：4个预测示例
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("EV Charging Demand Forecasting — Chronos (Zero-shot)\nSmart Mobility Lab, Penn State",
             fontsize=13, fontweight="bold")

for i in range(4):
    ax   = axes[i // 2][i % 2]
    idx  = i * 50
    hist = histories[idx]
    true = targets[idx]
    pred = predictions[idx]

    ax.plot(range(-24, 0), hist, color="steelblue", lw=1.5, label="History (24h)")
    ax.plot(range(0, 6),   true, color="green",     lw=2, marker="o", label="Ground Truth")
    ax.plot(range(0, 6),   pred, color="tomato",    lw=2, marker="s",
            linestyle="--", label="Chronos Forecast")
    ax.axvline(x=0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Hours (0 = forecast start)")
    ax.set_ylabel("Demand (kWh)")
    ax.set_title(f"Example {i+1}  |  {eval_df['timestamp'].iloc[idx]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path1 = OUTPUT_DIR + "predictions.png"
plt.savefig(path1, dpi=150, bbox_inches="tight")
print(f"  保存: {path1}")

# 图2：误差分析
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Forecast Error Analysis — Chronos", fontsize=12, fontweight="bold")

ax1.bar(range(1, 7), mae_per_h, color="steelblue", alpha=0.8)
ax1.set_xlabel("Forecast Horizon (hours ahead)")
ax1.set_ylabel("MAE (kWh)")
ax1.set_title("MAE by Forecast Horizon")
ax1.set_xticks(range(1, 7))
ax1.grid(True, alpha=0.3, axis="y")

ax2.scatter(targets.flatten(), predictions.flatten(), alpha=0.15, s=8, color="steelblue")
lim = [min(targets.min(), predictions.min()) - 5,
       max(targets.max(), predictions.max()) + 5]
ax2.plot(lim, lim, "r--", lw=1.5, label="Perfect forecast")
ax2.set_xlabel("Ground Truth (kWh)")
ax2.set_ylabel("Predicted (kWh)")
ax2.set_title(f"Predicted vs Actual  (MAPE: {mape:.1f}%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path2 = OUTPUT_DIR + "error_analysis.png"
plt.savefig(path2, dpi=150, bbox_inches="tight")
print(f"  保存: {path2}")

print("\n" + "=" * 60)
print("✅ Use Case 1 完成")
print(f"   MAE:  {overall_mae:.2f} kWh  ({mape:.1f}% of mean demand)")
print(f"   RMSE: {overall_rmse:.2f} kWh")
print(f"   输出目录: {OUTPUT_DIR}")
print("=" * 60)
