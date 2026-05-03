"""
Use Case 1: EV Charging Demand Forecasting with TimesFM (Zero-shot)
Google DeepMind's pretrained time series foundation model

Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset1_timeseries.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase1_timesfm_zeroshot/"
N_EVAL     = 200
PRED_LEN   = 6
CTX_LEN    = 24

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chronos zero-shot 结果（用于对比）
CHRONOS_MAE = [9.69, 13.96, 15.79, 16.60, 16.29, 15.70]
CHRONOS_MAPE = 21.3

print("=" * 60)
print("Use Case 1: EV Demand Forecasting with TimesFM (Zero-shot)")
print("=" * 60)

# ── Step 1: 安装并导入 TimesFM ────────────────────────────────────────────────
print("\n[Step 1] 导入 TimesFM...")
print("  如果报错 ModuleNotFoundError，先运行：")
print("  pip install timesfm")

try:
    import timesfm
except ImportError:
    print("\n  ❌ TimesFM 未安装，请先运行：")
    print("     pip install timesfm")
    exit(1)

print("  ✅ TimesFM 导入成功")

# ── Step 2: 加载数据 ──────────────────────────────────────────────────────────
print("\n[Step 2] 加载数据...")
df = pd.read_csv(DATA_PATH)

history_cols = [f"demand_t-{i}" for i in range(24, 0, -1)]
target_cols  = [f"target_t+{i}" for i in range(1, 7)]

np.random.seed(42)
eval_idx  = np.random.choice(len(df), size=N_EVAL, replace=False)
eval_df   = df.iloc[eval_idx].reset_index(drop=True)
histories = eval_df[history_cols].values   # (200, 24)
targets   = eval_df[target_cols].values    # (200, 6)

print(f"  评估样本数: {N_EVAL}")
print(f"  输入: 24小时 → 预测: 6小时")

# ── Step 3: 加载 TimesFM 模型 ─────────────────────────────────────────────────
print("\n[Step 3] 加载 TimesFM 模型...")
print("  模型: google/timesfm-1.0-200m")
print("  第一次运行会从 HuggingFace 下载权重（约800MB）...")

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",           # 使用 GPU
        per_core_batch_size=32,
        horizon_len=PRED_LEN,    # 预测步长
        context_len=CTX_LEN,     # 输入长度
        num_layers=20,
        model_dims=1280,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m"
    ),
)

print("  ✅ TimesFM 加载成功")

# ── Step 4: 运行 TimesFM 预测 ─────────────────────────────────────────────────
print("\n[Step 4] 开始预测...")

# TimesFM 输入格式：list of 1D numpy arrays
inputs     = [histories[i] for i in range(len(histories))]
freq_input = [0] * len(inputs)   # 0 = 高频数据（小时级别）

# 预测，返回 (point_forecast, experimental_quantile_preds)
point_forecasts, _ = tfm.forecast(
    inputs,
    freq=freq_input,
)

# point_forecasts shape: (N_EVAL, horizon_len)
predictions = np.array(point_forecasts)[:, :PRED_LEN]
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
mean_demand  = targets.mean()
mape         = overall_mae / mean_demand * 100

print(f"\n  总体 MAE:  {overall_mae:.2f} kWh")
print(f"  总体 RMSE: {overall_rmse:.2f} kWh")
print(f"  MAPE:      {mape:.1f}%")
print(f"\n  对比：Chronos MAPE {CHRONOS_MAPE}%  →  TimesFM MAPE {mape:.1f}%")

# ── Step 6: 画图 ──────────────────────────────────────────────────────────────
print("\n[Step 6] 生成图表...")

# 图1：4个预测示例
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("EV Charging Demand Forecasting — TimesFM (Zero-shot)\nSmart Mobility Lab, Penn State",
             fontsize=13, fontweight="bold")

for i in range(4):
    ax   = axes[i // 2][i % 2]
    idx  = i * 50
    ax.plot(range(-24, 0), histories[idx], color="steelblue",
            lw=1.5, label="History (24h)")
    ax.plot(range(0, 6), targets[idx], color="green",
            lw=2, marker="o", label="Ground Truth")
    ax.plot(range(0, 6), predictions[idx], color="tomato",
            lw=2, marker="s", linestyle="--", label="TimesFM Forecast")
    ax.axvline(x=0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Hours (0 = forecast start)")
    ax.set_ylabel("Demand (kWh)")
    ax.set_title(f"Example {i+1}  |  {eval_df['timestamp'].iloc[idx]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "predictions.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}predictions.png")

# 图2：TimesFM vs Chronos MAE 对比
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("TimesFM vs Chronos — Zero-shot Comparison\nSmart Mobility Lab, Penn State",
              fontsize=12, fontweight="bold")

x = np.arange(1, 7)
w = 0.35
ax1.bar(x - w/2, CHRONOS_MAE, w, label="Chronos", color="steelblue", alpha=0.8)
ax1.bar(x + w/2, mae_per_h,   w, label="TimesFM", color="tomato",    alpha=0.8)
ax1.set_xlabel("Forecast Horizon (hours ahead)")
ax1.set_ylabel("MAE (kWh)")
ax1.set_title("MAE by Forecast Horizon")
ax1.set_xticks(x)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

ax2.scatter(targets.flatten(), predictions.flatten(),
            alpha=0.15, s=8, color="tomato")
lim = [min(targets.min(), predictions.min()) - 5,
       max(targets.max(), predictions.max()) + 5]
ax2.plot(lim, lim, "r--", lw=1.5, label="Perfect forecast")
ax2.set_xlabel("Ground Truth (kWh)")
ax2.set_ylabel("Predicted (kWh)")
ax2.set_title(f"Predicted vs Actual  (MAPE: {mape:.1f}%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "comparison.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}comparison.png")

print("\n" + "=" * 60)
print("✅ Use Case 1 TimesFM Zero-shot 完成")
print(f"   TimesFM MAPE:  {mape:.1f}%")
print(f"   Chronos MAPE:  {CHRONOS_MAPE}%")
print(f"   输出目录: {OUTPUT_DIR}")
print("=" * 60)
