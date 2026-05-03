"""
Use Case 1: MOIRAI + Residual Correction Network for EV Charging Demand
Approach: Freeze MOIRAI, train a small correction network on top
- MOIRAI (frozen) makes base predictions using demand + temperature
- Correction network learns domain-specific adjustments
- Only correction network is trained (differentiable)

Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset1_timeseries.csv"
OUTPUT_DIR  = "/home/xzh5180/Research/llm-evprediction/outputs/usecase1_moirai_finetune/"
PRED_LEN    = 6
CTX_LEN     = 24
PATCH_SIZE  = 8
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 5
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1

CHRONOS_MAPE    = 21.3
CHRONOS_FT_MAPE = 19.5
TIMESFM_MAPE    = 22.6
MOIRAI_ZS_MAPE  = 42.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Use Case 1: MOIRAI + Residual Correction Network")
print("Method: Frozen MOIRAI + trainable correction network")
print("=" * 60)

# ── Step 1: 加载数据 ───────────────────────────────────────────────────────────
print("\n[Step 1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']       = df['timestamp'].dt.hour
df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(float)

demand_cols = [f"demand_t-{i}" for i in range(24, 0, -1)]
target_cols = [f"target_t+{i}" for i in range(1, 7)]
temp_cols   = [f"temp_t-{i}"   for i in range(24, 0, -1)]

X_demand = df[demand_cols].values.astype(np.float32)
X_temp   = df[temp_cols].values.astype(np.float32)
y        = df[target_cols].values.astype(np.float32)

future_temp = X_temp[:, -1:] * np.ones((len(X_temp), PRED_LEN), dtype=np.float32)
X_temp_full = np.concatenate([X_temp, future_temp], axis=1)   # (N, 30)

# 额外特征：最后一步需求、小时、是否周末 → 辅助修正网络
hour_arr      = df['hour'].values.astype(np.float32) / 23.0         # 归一化到 0-1
weekend_arr   = df['is_weekend'].values.astype(np.float32)
last_demand   = X_demand[:, -1]                                       # 最近一步需求
last_temp     = X_temp[:, -1]                                         # 最近一步温度
X_meta = np.stack([last_demand/150.0, last_temp/110.0,
                   hour_arr, weekend_arr], axis=1).astype(np.float32) # (N, 4)

print(f"  需求 shape: {X_demand.shape}")
print(f"  温度 shape: {X_temp_full.shape}")
print(f"  元特征 shape: {X_meta.shape}  (last_demand, last_temp, hour, is_weekend)")

# ── Step 2: 数据切分 ───────────────────────────────────────────────────────────
n       = len(X_demand)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * (TRAIN_RATIO + VAL_RATIO))

X_d_tr, X_d_val, X_d_te     = X_demand[:n_train], X_demand[n_train:n_val], X_demand[n_val:]
X_t_tr, X_t_val, X_t_te     = X_temp_full[:n_train], X_temp_full[n_train:n_val], X_temp_full[n_val:]
X_m_tr, X_m_val, X_m_te     = X_meta[:n_train], X_meta[n_train:n_val], X_meta[n_val:]
y_tr,   y_val,   y_te        = y[:n_train], y[n_train:n_val], y[n_val:]

print(f"\n[Step 2] 数据切分: 训练 {len(X_d_tr)} | 验证 {len(X_d_val)} | 测试 {len(X_d_te)}")

# ── Step 3: 加载冻结的 MOIRAI ─────────────────────────────────────────────────
print("\n[Step 3] 加载 MOIRAI（冻结，不训练）...")

moirai = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
    prediction_length=PRED_LEN,
    context_length=CTX_LEN,
    patch_size=PATCH_SIZE,
    num_samples=50,
    target_dim=1,
    feat_dynamic_real_dim=1,
    past_feat_dynamic_real_dim=0,
)
moirai.eval()
for p in moirai.parameters():
    p.requires_grad = False
print("  ✅ MOIRAI 加载完成，全部参数已冻结")

# ── Step 4: 定义残差修正网络 ──────────────────────────────────────────────────
print("\n[Step 4] 定义残差修正网络...")

class ResidualCorrectionNet(nn.Module):
    """
    输入：MOIRAI 的基础预测（PRED_LEN=6）+ 元特征（4维）
    输出：残差修正量（PRED_LEN=6）
    最终预测 = MOIRAI 基础预测 + 修正量
    """
    def __init__(self, pred_len=6, meta_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pred_len + meta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pred_len),
        )
        # 初始化输出层为接近0，让修正量一开始很小
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_pred, meta):
        x = torch.cat([base_pred, meta], dim=-1)
        return self.net(x)

correction_net = ResidualCorrectionNet(pred_len=PRED_LEN, meta_dim=4, hidden=64)
n_trainable = sum(p.numel() for p in correction_net.parameters())
print(f"  修正网络参数量: {n_trainable:,}（极少，只占 MOIRAI 的 {n_trainable/13827528*100:.2f}%）")

# ── Step 5: 先获取 MOIRAI 的全部基础预测（预计算，节省训练时间） ──────────────────
print("\n[Step 5] 预计算 MOIRAI 基础预测（所有样本）...")

def get_moirai_predictions(X_d, X_t, batch_size=32):
    """用冻结的 MOIRAI 预测所有样本，结果存到 numpy array"""
    preds = []
    for s in range(0, len(X_d), batch_size):
        e = min(s + batch_size, len(X_d))
        B = e - s
        pt   = torch.tensor(X_d[s:e]).unsqueeze(-1)
        pot  = torch.ones(B, CTX_LEN, 1, dtype=torch.bool)
        pip  = torch.zeros(B, CTX_LEN, dtype=torch.bool)
        fdr  = torch.tensor(X_t[s:e]).unsqueeze(-1)
        ofdr = torch.cat([
            torch.ones(B, CTX_LEN, 1, dtype=torch.bool),
            torch.zeros(B, PRED_LEN, 1, dtype=torch.bool)
        ], dim=1)
        with torch.no_grad():
            fc = moirai(past_target=pt, past_observed_target=pot,
                        past_is_pad=pip, feat_dynamic_real=fdr,
                        observed_feat_dynamic_real=ofdr)
        preds.append(fc.median(dim=1).values.squeeze(-1).numpy())
    return np.vstack(preds)

print("  计算训练集基础预测...")
base_tr  = get_moirai_predictions(X_d_tr,  X_t_tr)
print("  计算验证集基础预测...")
base_val = get_moirai_predictions(X_d_val, X_t_val)
print("  计算测试集基础预测...")
base_te  = get_moirai_predictions(X_d_te,  X_t_te)

# 计算零样本 MAPE
zs_mape = mean_absolute_error(y_te.flatten(), base_te.flatten()) / y_te.mean() * 100
print(f"\n  多变量 Zero-shot MAPE（预计算验证）: {zs_mape:.1f}%")

# ── Step 6: Dataset（用预计算的基础预测） ──────────────────────────────────────
class CorrectionDataset(Dataset):
    def __init__(self, base_pred, meta, y):
        self.base = torch.tensor(base_pred)
        self.meta = torch.tensor(meta)
        self.y    = torch.tensor(y)
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i], self.meta[i], self.y[i]

train_loader = DataLoader(CorrectionDataset(base_tr,  X_m_tr,  y_tr),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(CorrectionDataset(base_val, X_m_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── Step 7: 训练修正网络 ───────────────────────────────────────────────────────
print("\n[Step 6] 训练残差修正网络...")
print("-" * 60)

optimizer = AdamW(correction_net.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn   = nn.MSELoss()

train_losses  = []
val_losses    = []
best_val_loss = float("inf")
best_epoch    = 0
patience_ctr  = 0

for epoch in range(1, EPOCHS + 1):

    # 训练
    correction_net.train()
    ep_train = 0.0
    for base_pred, meta, batch_y in train_loader:
        optimizer.zero_grad()
        correction = correction_net(base_pred, meta)
        final_pred = base_pred + correction
        loss = loss_fn(final_pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(correction_net.parameters(), 1.0)
        optimizer.step()
        ep_train += loss.item()
    avg_train = ep_train / len(train_loader)

    # 验证
    correction_net.eval()
    ep_val = 0.0
    with torch.no_grad():
        for base_pred, meta, batch_y in val_loader:
            correction = correction_net(base_pred, meta)
            final_pred = base_pred + correction
            ep_val += loss_fn(final_pred, batch_y).item()
    avg_val = ep_val / len(val_loader)

    scheduler.step()
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"  Epoch {epoch:2d}/{EPOCHS}  "
          f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
          f"LR: {scheduler.get_last_lr()[0]:.2e}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_epoch    = epoch
        patience_ctr  = 0
        torch.save(correction_net.state_dict(), OUTPUT_DIR + "best_correction.pt")
        print(f"             ✅ 保存最佳修正网络")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  Early Stopping（epoch {epoch}），最佳在 epoch {best_epoch}")
            break

print("-" * 60)

# ── Step 8: 测试集评估 ─────────────────────────────────────────────────────────
print("\n[Step 7] 加载最佳修正网络，测试集评估...")
correction_net.load_state_dict(torch.load(OUTPUT_DIR + "best_correction.pt"))
correction_net.eval()

with torch.no_grad():
    base_te_t  = torch.tensor(base_te)
    meta_te_t  = torch.tensor(X_m_te)
    correction = correction_net(base_te_t, meta_te_t)
    ft_preds   = (base_te_t + correction).numpy()

mae_per_h  = []
rmse_per_h = []
print("\n  Fine-tuned（MOIRAI + 修正网络）结果：")
for h in range(PRED_LEN):
    mae  = mean_absolute_error(y_te[:, h], ft_preds[:, h])
    rmse = mean_squared_error(y_te[:, h],  ft_preds[:, h]) ** 0.5
    mae_per_h.append(mae)
    rmse_per_h.append(rmse)
    print(f"  t+{h+1}h  MAE: {mae:.2f}  RMSE: {rmse:.2f}")

ft_mape = mean_absolute_error(y_te.flatten(), ft_preds.flatten()) / y_te.mean() * 100

print(f"\n  ── 全部模型 MAPE 汇总 ──")
print(f"  Chronos zero-shot:                {CHRONOS_MAPE}%")
print(f"  Chronos fine-tuned:               {CHRONOS_FT_MAPE}%")
print(f"  TimesFM zero-shot:                {TIMESFM_MAPE}%")
print(f"  MOIRAI zero-shot（单变量）:        {MOIRAI_ZS_MAPE}%")
print(f"  MOIRAI zero-shot（多变量）:        {zs_mape:.1f}%")
print(f"  MOIRAI + 残差修正（多变量）:       {ft_mape:.1f}%")

# ── Step 9: 画图 ───────────────────────────────────────────────────────────────
print("\n[Step 8] 生成图表...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("MOIRAI + Residual Correction Network — EV Charging Demand\nSmart Mobility Lab, Penn State",
             fontsize=12, fontweight="bold")

# 训练曲线
ax = axes[0]
ax.plot(range(1, len(train_losses)+1), train_losses,
        color="steelblue", marker="o", label="Train Loss")
ax.plot(range(1, len(val_losses)+1), val_losses,
        color="tomato", marker="s", label="Val Loss")
ax.axvline(x=best_epoch, color="green", linestyle="--",
           label=f"Best Epoch {best_epoch}")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Correction Network Training Curve")
ax.legend(); ax.grid(True, alpha=0.3)

# MAE by horizon
ax = axes[1]
x = np.arange(1, 7); w = 0.2
zs_mae_h = [mean_absolute_error(y_te[:, h], base_te[:, h]) for h in range(6)]
ax.bar(x - w,   [9.69,13.96,15.79,16.60,16.29,15.70], w,
       label="Chronos ZS",   color="steelblue",      alpha=0.8)
ax.bar(x,       zs_mae_h,  w, label="MOIRAI ZS multi", color="orange",    alpha=0.8)
ax.bar(x + w,   mae_per_h, w, label="MOIRAI+Correction", color="tomato",  alpha=0.8)
ax.set_xlabel("Forecast Horizon"); ax.set_ylabel("MAE (kWh)")
ax.set_title("MAE by Horizon"); ax.set_xticks(x)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# MAPE 总览
ax = axes[2]
labels = ["Chronos\nZS", "Chronos\nFT", "TimesFM\nZS",
          "MOIRAI\nZS\n单变量", "MOIRAI\nZS\n多变量", "MOIRAI+\n修正网络"]
mapes  = [CHRONOS_MAPE, CHRONOS_FT_MAPE, TIMESFM_MAPE,
          MOIRAI_ZS_MAPE, zs_mape, ft_mape]
colors = ["steelblue","cornflowerblue","orange","salmon","tomato","darkred"]
bars   = ax.bar(labels, mapes, color=colors, alpha=0.85, width=0.6)
for bar, v in zip(bars, mapes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
            f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("MAPE (%)"); ax.set_title("Overall MAPE — All Models")
ax.set_ylim(0, max(mapes) * 1.3)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "results.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}results.png")

# 预测示例图
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle("MOIRAI + Residual Correction — Prediction Examples\nSmart Mobility Lab, Penn State",
              fontsize=13, fontweight="bold")
for i in range(4):
    ax  = axes2[i // 2][i % 2]
    idx = i * (len(X_d_te) // 4)
    ax.plot(range(-24, 0), X_d_te[idx], color="steelblue", lw=1.5, label="History")
    ax.plot(range(0, 6),   y_te[idx],   color="green",     lw=2, marker="o", label="Ground Truth")
    ax.plot(range(0, 6),   base_te[idx],color="orange",    lw=1.5, marker="^",
            linestyle=":", label="MOIRAI ZS")
    ax.plot(range(0, 6),   ft_preds[idx], color="tomato",  lw=2, marker="s",
            linestyle="--", label="MOIRAI+Correction")
    ax.axvline(x=0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Hours"); ax.set_ylabel("Demand (kWh)")
    ax.set_title(f"Test Example {i+1}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "predictions.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}predictions.png")

print("\n" + "=" * 60)
print("✅ MOIRAI + 残差修正网络 完成")
print(f"   MOIRAI ZS 单变量:          {MOIRAI_ZS_MAPE}%")
print(f"   MOIRAI ZS 多变量:          {zs_mape:.1f}%")
print(f"   MOIRAI + 残差修正 多变量:  {ft_mape:.1f}%")
print(f"   最佳修正网络: {OUTPUT_DIR}best_correction.pt")
print("=" * 60)
