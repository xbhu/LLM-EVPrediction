"""
Use Case 2: EV Charging Demand Prediction
Flan-T5-base — Fine-Tune Regression

训练流程：
  1. 数据划分：80% 训练 / 10% 验证 / 10% 测试
  2. Fine-tune：Encoder（Flan-T5）+ 回归头（线性层）一起训练
  3. 验证集监控：每个 epoch 结束后评估，保存最佳模型
  4. 测试集评估：训练结束后用最佳模型评估，和 zero-shot 对比

Author: XB Hu / Smart Mobility Lab, Penn State
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, T5EncoderModel

# ─────────────────────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
OUTPUT_DIR  = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_flant5_finetune/"
MODEL_NAME  = "google/flan-t5-base"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 2e-4     # 学习率
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("Use Case 2: Flan-T5-base Fine-Tune Regression")
print("=" * 60)
print(f"  Device     : {DEVICE}")
print(f"  Model      : {MODEL_NAME}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 加载和划分数据
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] 加载数据...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

n       = len(df)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

train_df = df.iloc[:n_train].reset_index(drop=True)
val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

print(f"    总数据量 : {n} 行")
print(f"    训练集   : {len(train_df)} 行")
print(f"    验证集   : {len(val_df)} 行")
print(f"    测试集   : {len(test_df)} 行")

# 计算训练集的均值和标准差，用于归一化标签
# 归一化的目的：让目标值在 0 附近，让回归头更容易学习
label_mean = train_df["next_day_demand"].mean()
label_std  = train_df["next_day_demand"].std()
print(f"\n    标签均值 : {label_mean:.1f} kWh")
print(f"    标签标准差: {label_std:.1f} kWh")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset 和 DataLoader
#
# PyTorch 需要把数据包装成 Dataset 对象，才能批量加载
# ─────────────────────────────────────────────────────────────────────────────
class EVDemandDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, label_mean, label_std):
        self.texts  = df["context_text"].tolist()
        # 归一化标签：(真实值 - 均值) / 标准差
        self.labels = ((df["next_day_demand"] - label_mean) / label_std).tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.float)
        }

print("\n[2] 初始化 Tokenizer 和 DataLoader...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EVDemandDataset(train_df, tokenizer, MAX_LENGTH, label_mean, label_std)
val_dataset   = EVDemandDataset(val_df,   tokenizer, MAX_LENGTH, label_mean, label_std)
test_dataset  = EVDemandDataset(test_df,  tokenizer, MAX_LENGTH, label_mean, label_std)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"    训练 batches : {len(train_loader)}")
print(f"    验证 batches : {len(val_loader)}")
print(f"    测试 batches : {len(test_loader)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 模型：Encoder + 回归头
# ─────────────────────────────────────────────────────────────────────────────
class FlanT5Regressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder        = T5EncoderModel.from_pretrained(model_name)
        hidden_size         = self.encoder.config.d_model  # 768
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        # 文字 → 向量序列
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 取 [CLS]（第一个token）的向量作为整句摘要
        cls_vector = encoder_output.last_hidden_state[:, 0, :]  # [batch, 768]
        # 向量 → 数字
        return self.regression_head(cls_vector).squeeze(-1)     # [batch]

print("\n[3] 加载模型...")
model = FlanT5Regressor(MODEL_NAME).to(DEVICE)
print(f"    模型加载完成")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 训练配置
# ─────────────────────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch学习率减半
criterion = nn.MSELoss()  # 均方误差，标准回归损失函数

# ─────────────────────────────────────────────────────────────────────────────
# 5. 训练循环
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4] 开始训练（{EPOCHS} epochs）...")
print("-" * 60)

train_losses = []
val_losses   = []
best_val_loss = float("inf")
best_model_path = OUTPUT_DIR + "best_model.pt"

for epoch in range(EPOCHS):
    # ── 训练阶段 ──────────────────────────────────────────────
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ── 验证阶段 ──────────────────────────────────────────────
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    scheduler.step()

    # 保存验证集最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        flag = " ← 最佳"
    else:
        flag = ""

    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}{flag}")

print(f"\n  训练完成，最佳模型已保存: {best_model_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 测试集评估（使用最佳模型）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] 测试集评估...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        preds = model(input_ids, attention_mask)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 反归一化：还原成真实的 kWh 单位
all_preds  = np.array(all_preds)  * label_std + label_mean
all_labels = np.array(all_labels) * label_std + label_mean

mae  = mean_absolute_error(all_labels, all_preds)
rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-6))) * 100
mae_baseline = mean_absolute_error(all_labels, np.full_like(all_labels, all_labels.mean()))

print("\n" + "=" * 60)
print("  结果对比")
print("=" * 60)
print(f"  Zero-Shot（随机回归头）→ MAE: 1588.5 kWh  MAPE: 100.0%")
print(f"  Fine-Tune              → MAE: {mae:.1f} kWh  MAPE: {mape:.1f}%")
print(f"  基线（均值预测）        → MAE: {mae_baseline:.1f} kWh")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 7. 可视化
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"Flan-T5-base Fine-Tune  |  MAE: {mae:.0f} kWh  MAPE: {mape:.1f}%",
             fontsize=12, fontweight="bold")

# (A) 训练曲线
ax = axes[0]
ax.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="steelblue")
ax.plot(range(1, EPOCHS+1), val_losses,   label="Val Loss",   color="darkorange")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss（归一化空间）")
ax.set_title("训练曲线")
ax.legend()
ax.grid(True, alpha=0.3)

# (B) 预测 vs 真实（时间序列）
ax = axes[1]
ax.plot(all_labels, label="真实值", color="steelblue", lw=1.5)
ax.plot(all_preds,  label="预测值", color="darkorange", lw=1.5, linestyle="--")
ax.set_xlabel("样本序号")
ax.set_ylabel("需求 (kWh)")
ax.set_title("预测 vs 真实")
ax.legend()
ax.grid(True, alpha=0.3)

# (C) 散点图
ax = axes[2]
lim = [min(all_labels.min(), all_preds.min()) * 0.95,
       max(all_labels.max(), all_preds.max()) * 1.05]
ax.scatter(all_labels, all_preds, alpha=0.5, s=20, color="steelblue")
ax.plot(lim, lim, "r--", lw=1.5, label="理想预测线")
ax.set_xlabel("真实值 (kWh)")
ax.set_ylabel("预测值 (kWh)")
ax.set_title(f"散点图  (MAE: {mae:.0f} kWh)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR + "finetune_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  图表已保存: {plot_path}")

print("\n✅ Fine-tune 完成")
print(f"   MAE  : {mae:.1f} kWh  ({mae/all_labels.mean()*100:.1f}% of mean)")
print(f"   RMSE : {rmse:.1f} kWh")
print(f"   MAPE : {mape:.1f}%")
