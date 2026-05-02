"""
Use Case 1: Fine-tuning Chronos on EV Charging Demand Data
Full parameter fine-tuning - Chronos 2.x compatible version

Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from chronos import BaseChronosPipeline
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH     = "/home/xzh5180/Research/llm-evprediction/datasets/dataset1_timeseries.csv"
OUTPUT_DIR    = "/home/xzh5180/Research/llm-evprediction/outputs/usecase1_finetune/"
MODEL_NAME    = "amazon/chronos-t5-small"
EPOCHS        = 10
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
PRED_LEN      = 6
CTX_LEN       = 24

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Use Case 1: Fine-tuning Chronos on EV Charging Data")
print("=" * 60)

# ── Step 1: 加载数据 ──────────────────────────────────────────────────────────
print("\n[Step 1] 加载数据...")
df = pd.read_csv(DATA_PATH)

history_cols = [f"demand_t-{i}" for i in range(24, 0, -1)]
target_cols  = [f"target_t+{i}" for i in range(1, 7)]
X = df[history_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

n       = len(X)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
X_test,  y_test  = X[n_val:], y[n_val:]

print(f"  训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ── Step 2: Dataset ───────────────────────────────────────────────────────────
class EVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(EVDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EVDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

# ── Step 3: 加载模型 ──────────────────────────────────────────────────────────
print("\n[Step 2] 加载 Chronos 预训练权重...")

pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype=torch.float32,
)
device    = torch.device("cuda")
tokenizer = pipeline.tokenizer
t5_model  = pipeline.model.model   # T5ForConditionalGeneration

print(f"  参数量: {sum(p.numel() for p in t5_model.parameters()):,}")

# ── Step 4: 检查 tokenizer 的 boundaries ──────────────────────────────────────
# Chronos tokenizer 把浮点数量化成 token ID
# 我们需要 boundaries 来对 target 做同样的量化
print("\n[Step 3] 检查 tokenizer API...")

def get_boundaries(tokenizer):
    for attr in ["boundaries", "bin_edges", "centers", "low", "bins"]:
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            if isinstance(val, torch.Tensor) and val.numel() > 10:
                print(f"  找到 boundaries: tokenizer.{attr}, shape={val.shape}")
                return val
    # fallback: 从 tokenizer 源码推断（Chronos 默认 4096 bins, 范围约 -10~10）
    print("  未找到 boundaries 属性，使用默认 linspace 估算")
    return torch.linspace(-15.0, 15.0, 4095)  # 4095 个边界 → 4096 个 bin

boundaries = get_boundaries(tokenizer)

def tokenize_sequence(seq_cpu, boundaries):
    """
    把原始时间序列 (B, T) 量化成 token ID (B, T)
    同时返回 scale，用于后续解码
    1. scale = mean(|seq|) per sample
    2. normalized = seq / scale
    3. token_id = bucketize(normalized, boundaries)
    """
    scale = seq_cpu.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    norm  = seq_cpu / scale
    ids   = torch.bucketize(norm, boundaries.to(seq_cpu.device))
    return ids, scale

def decode_tokens(token_ids, scale, boundaries):
    """
    把 token ID 解码回浮点值
    取每个 bin 的中心值，再乘以 scale
    """
    centers = torch.cat([
        boundaries[:1] - (boundaries[1] - boundaries[0]),
        (boundaries[:-1] + boundaries[1:]) / 2,
        boundaries[-1:] + (boundaries[-1] - boundaries[-2])
    ])
    values = centers[token_ids.cpu()] * scale
    return values

# ── Step 5: 验证 tokenize/decode 的正确性 ─────────────────────────────────────
print("\n[Step 4] 验证量化/反量化...")
test_seq   = torch.tensor(X_train[:3], dtype=torch.float32)
test_ids, test_scale = tokenize_sequence(test_seq, boundaries)
test_recon = decode_tokens(test_ids, test_scale, boundaries)
recon_err  = (test_seq - test_recon).abs().mean().item()
print(f"  量化重建误差（应接近0）: {recon_err:.4f} kWh")

# ── Step 6: 优化器 ────────────────────────────────────────────────────────────
print("\n[Step 5] 设置优化器...")
optimizer = AdamW(t5_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Step 7: 训练循环 ──────────────────────────────────────────────────────────
print("\n[Step 6] 开始训练...")
print("-" * 60)

train_losses  = []
val_losses    = []
best_val_loss = float("inf")
best_epoch    = 0
patience_ctr  = 0
PATIENCE      = 3

for epoch in range(1, EPOCHS + 1):

    # ── 训练 ──
    t5_model.train()
    ep_train = 0.0

    for batch_X, batch_y in train_loader:
        # 1. 把 context + target 拼成完整序列，用同一个 scale 量化
        full_seq   = torch.cat([batch_X, batch_y], dim=1)       # (B, 30)
        full_ids, scale = tokenize_sequence(full_seq, boundaries)  # (B, 30)

        ctx_ids = full_ids[:, :CTX_LEN].to(device)             # (B, 24)
        tgt_ids = full_ids[:, CTX_LEN:].to(device)             # (B, 6)

        # 2. T5 seq2seq: encoder 看 context，decoder 预测 target
        #    T5ForConditionalGeneration.forward(input_ids, labels) 返回 cross-entropy loss
        optimizer.zero_grad()
        outputs = t5_model(input_ids=ctx_ids, labels=tgt_ids)
        loss    = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(t5_model.parameters(), 1.0)
        optimizer.step()
        ep_train += loss.item()

    avg_train = ep_train / len(train_loader)

    # ── 验证 ──
    t5_model.eval()
    ep_val = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            full_seq = torch.cat([batch_X, batch_y], dim=1)
            full_ids, _ = tokenize_sequence(full_seq, boundaries)
            ctx_ids  = full_ids[:, :CTX_LEN].to(device)
            tgt_ids  = full_ids[:, CTX_LEN:].to(device)
            outputs  = t5_model(input_ids=ctx_ids, labels=tgt_ids)
            ep_val  += outputs.loss.item()

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
        torch.save(t5_model.state_dict(), OUTPUT_DIR + "best_model.pt")
        print(f"             ✅ 保存最佳模型")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  Early Stopping（epoch {epoch}），最佳在 epoch {best_epoch}")
            break

print("-" * 60)

# ── Step 8: 测试集评估 ────────────────────────────────────────────────────────
print("\n[Step 7] 加载最佳模型，在测试集上评估...")
t5_model.load_state_dict(torch.load(OUTPUT_DIR + "best_model.pt"))
t5_model.eval()

all_preds = []
EVAL_BS   = 32

for start in range(0, len(X_test), EVAL_BS):
    end      = min(start + EVAL_BS, len(X_test))
    batch_X  = torch.tensor(X_test[start:end])
    ctx_ids, scale = tokenize_sequence(batch_X, boundaries)
    ctx_ids  = ctx_ids.to(device)

    with torch.no_grad():
        # 用 T5 greedy decode 生成 target tokens
        gen_ids = t5_model.generate(
            input_ids=ctx_ids,
            max_new_tokens=PRED_LEN,
            do_sample=False,
        )
        # generate 输出包含 decoder_start_token，去掉第一个 token
        pred_ids = gen_ids[:, 1:PRED_LEN+1].cpu()

    # 解码回 kWh
    preds = decode_tokens(pred_ids, scale, boundaries)
    all_preds.append(preds.numpy())

predictions = np.vstack(all_preds)

print("\n  Fine-tuned 模型结果：")
mae_per_h  = []
rmse_per_h = []
for h in range(PRED_LEN):
    mae  = mean_absolute_error(y_test[:, h], predictions[:, h])
    rmse = mean_squared_error(y_test[:, h], predictions[:, h]) ** 0.5
    mae_per_h.append(mae)
    rmse_per_h.append(rmse)
    print(f"  t+{h+1}h  MAE: {mae:.2f}  RMSE: {rmse:.2f}")

overall_mae  = mean_absolute_error(y_test.flatten(), predictions.flatten())
overall_rmse = mean_squared_error(y_test.flatten(), predictions.flatten()) ** 0.5
mape         = overall_mae / y_test.mean() * 100
print(f"\n  总体 MAE: {overall_mae:.2f} kWh  MAPE: {mape:.1f}%")
print(f"  对比 Zero-shot MAPE: 21.3%  →  Fine-tuned MAPE: {mape:.1f}%")

# ── Step 9: 画图 ──────────────────────────────────────────────────────────────
print("\n[Step 8] 生成图表...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fine-tuning Chronos — EV Charging Demand\nSmart Mobility Lab, Penn State",
             fontsize=12, fontweight="bold")

ax1.plot(range(1, len(train_losses)+1), train_losses,
         color="steelblue", marker="o", label="Train Loss")
ax1.plot(range(1, len(val_losses)+1), val_losses,
         color="tomato", marker="s", label="Val Loss")
ax1.axvline(x=best_epoch, color="green", linestyle="--",
            label=f"Best Epoch {best_epoch}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

zero_shot_mae = [9.69, 13.96, 15.79, 16.60, 16.29, 15.70]
x = np.arange(1, 7); w = 0.35
ax2.bar(x - w/2, zero_shot_mae, w, label="Zero-shot", color="steelblue", alpha=0.8)
ax2.bar(x + w/2, mae_per_h,     w, label="Fine-tuned", color="tomato",    alpha=0.8)
ax2.set_xlabel("Forecast Horizon (hours ahead)")
ax2.set_ylabel("MAE (kWh)")
ax2.set_title("Zero-shot vs Fine-tuned MAE")
ax2.set_xticks(x); ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "results.png", dpi=150, bbox_inches="tight")
print(f"  保存: {OUTPUT_DIR}results.png")

print("\n" + "=" * 60)
print("✅ Fine-tuning 完成")
print(f"   Zero-shot MAPE: 21.3%  →  Fine-tuned MAPE: {mape:.1f}%")
print(f"   最佳模型: {OUTPUT_DIR}best_model.pt")
print("=" * 60)
