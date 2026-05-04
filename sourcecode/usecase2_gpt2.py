"""
Use Case 2: EV Charging Demand Prediction
GPT-2 Medium — Zero-Shot + Fine-Tune

阶段一：Zero-Shot  — 直接用预训练 GPT-2 推理，不做任何训练
阶段二：Fine-Tune  — 把 context_text + demand 拼成完整文本训练
最终：三方对比    — Zero-Shot vs Fine-Tune vs 均值基线

Author: XB Hu / Smart Mobility Lab, Penn State
"""

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ─────────────────────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_gpt2/"
MODEL_NAME = "gpt2-medium"
MAX_LENGTH = 148       # context_text token数 + demand 数字的空间
BATCH_SIZE = 8         # GPT-2 比 T5 占显存多，batch size 适当减小
EPOCHS     = 20
LR         = 2e-5      # 生成式 fine-tune 学习率通常比回归式小一个量级
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
N_ZEROSHOT  = 100      # zero-shot 评估样本数

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("Use Case 2: GPT-2 Medium Zero-Shot + Fine-Tune")
print("=" * 60)
print(f"  Device     : {DEVICE}")
print(f"  Model      : {MODEL_NAME}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 加载数据
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载 Tokenizer 和模型
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] 加载 GPT-2 Medium...")
print(f"    第一次运行会从 Hugging Face 下载模型（约 1.5GB）...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# GPT-2 没有 padding token，用 eos_token 代替
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.pad_token_id = tokenizer.eos_token_id

print(f"    模型加载完成")
print(f"    参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 辅助函数：构建 prompt 和解析数字
#
# 生成式的关键设计：把输入输出拼成一段完整文本
#
# 训练时：
#   "Context: Winter weekday, 32.5°F... Predicted demand: 1543"
#   模型学习：看到 context 后，能续写出正确数字
#
# 推理时：
#   只给 "Context: Winter weekday, 32.5°F... Predicted demand:"
#   让模型续写出数字
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = "Context: {context}\nPredicted demand: "
TRAIN_TEMPLATE  = "Context: {context}\nPredicted demand: {demand}"

def build_inference_prompt(context_text: str) -> str:
    return PROMPT_TEMPLATE.format(context=context_text.strip())

def build_training_text(context_text: str, demand: float) -> str:
    return TRAIN_TEMPLATE.format(
        context=context_text.strip(),
        demand=f"{demand:.0f}"
    )

def parse_demand(text: str, prompt: str) -> float | None:
    """从模型生成的完整文本中提取数字"""
    # 去掉 prompt 部分，只看续写的内容
    generated = text[len(prompt):].strip()
    # 提取第一个整数或浮点数
    match = re.search(r'\b(\d{3,5}(?:\.\d+)?)\b', generated)
    if match:
        return float(match.group(1))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 阶段一：Zero-Shot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  阶段一：Zero-Shot")
print("=" * 60)

# 用测试集做 zero-shot 评估
zs_preds  = []
zs_labels = []
zs_failures = 0

model.eval()
with torch.no_grad():
    for i, row in test_df.iterrows():
        prompt = build_inference_prompt(row["context_text"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH - 10
        ).to(DEVICE)

        output = model.generate(
            **inputs,
            max_new_tokens=10,      # 只需要生成几个数字字符
            do_sample=False,        # 贪婪解码，结果确定
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = parse_demand(generated_text, prompt)

        if pred is None or pred < 100 or pred > 5000:
            zs_failures += 1
            pred = test_df["next_day_demand"].mean()  # 解析失败用均值兜底

        zs_preds.append(pred)
        zs_labels.append(row["next_day_demand"])

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_df)} 完成  — 生成: {generated_text[-30:].strip()!r}")

zs_preds  = np.array(zs_preds)
zs_labels = np.array(zs_labels)
zs_mae    = mean_absolute_error(zs_labels, zs_preds)
zs_mape   = np.mean(np.abs((zs_labels - zs_preds) / (zs_labels + 1e-6))) * 100

print(f"\n  Zero-Shot 结果:")
print(f"    MAE : {zs_mae:.1f} kWh  |  MAPE : {zs_mape:.1f}%")
print(f"    解析失败次数 : {zs_failures}/{len(test_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 阶段二：Fine-Tune
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  阶段二：Fine-Tune")
print("=" * 60)

# ── Dataset ──────────────────────────────────────────────────────────────────
class GPT2DemandDataset(Dataset):
    """
    生成式 fine-tune 的数据格式：
    把 context_text 和 demand 拼成完整文本，让模型学习整段文本的生成。
    损失函数只计算 demand 部分（用 labels mask 实现）。
    """
    def __init__(self, df, tokenizer, max_length):
        self.samples    = []
        self.tokenizer  = tokenizer
        self.max_length = max_length

        for _, row in df.iterrows():
            full_text  = build_training_text(row["context_text"], row["next_day_demand"])
            prompt     = build_inference_prompt(row["context_text"])
            self.samples.append((full_text, len(tokenizer.encode(prompt))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_text, prompt_len = self.samples[idx]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels：prompt 部分设为 -100（不计算损失），只计算 demand 部分的损失
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "labels"         : labels
        }

print("\n[3] 初始化 Dataset...")
train_dataset = GPT2DemandDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = GPT2DemandDataset(val_df,   tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"    训练 batches : {len(train_loader)}")
print(f"    验证 batches : {len(val_loader)}")

# ── 训练循环 ──────────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

train_losses  = []
val_losses    = []
best_val_loss = float("inf")
best_model_path = OUTPUT_DIR + "best_model.pt"

print(f"\n[4] 开始训练（{EPOCHS} epochs）...")
print("-" * 60)

for epoch in range(EPOCHS):
    # 训练
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    scheduler.step()

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

# ── 测试集评估 ────────────────────────────────────────────────────────────────
print("\n[5] 测试集评估（Fine-Tune）...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

ft_preds    = []
ft_labels   = []
ft_failures = 0

with torch.no_grad():
    for _, row in test_df.iterrows():
        prompt = build_inference_prompt(row["context_text"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH - 10
        ).to(DEVICE)

        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = parse_demand(generated_text, prompt)

        if pred is None or pred < 100 or pred > 5000:
            ft_failures += 1
            pred = test_df["next_day_demand"].mean()

        ft_preds.append(pred)
        ft_labels.append(row["next_day_demand"])

ft_preds  = np.array(ft_preds)
ft_labels = np.array(ft_labels)
ft_mae    = mean_absolute_error(ft_labels, ft_preds)
ft_rmse   = np.sqrt(mean_squared_error(ft_labels, ft_preds))
ft_mape   = np.mean(np.abs((ft_labels - ft_preds) / (ft_labels + 1e-6))) * 100
mae_base  = mean_absolute_error(ft_labels, np.full_like(ft_labels, ft_labels.mean()))

# ─────────────────────────────────────────────────────────────────────────────
# 最终结果对比
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  最终结果对比")
print("=" * 60)
print(f"  GPT-2 Zero-Shot  → MAE: {zs_mae:.1f} kWh  |  MAPE: {zs_mape:.1f}%")
print(f"  GPT-2 Fine-Tune  → MAE: {ft_mae:.1f} kWh  |  MAPE: {ft_mape:.1f}%")
print(f"  基线（均值预测）  → MAE: {mae_base:.1f} kWh")
print(f"\n  参考：Flan-T5 Fine-Tune → MAE: 149.2 kWh  |  MAPE: 12.2%")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"GPT-2 Medium  |  Zero-Shot MAE: {zs_mae:.0f}  Fine-Tune MAE: {ft_mae:.0f} kWh",
             fontsize=12, fontweight="bold")

# (A) 训练曲线
ax = axes[0, 0]
ax.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="steelblue")
ax.plot(range(1, EPOCHS+1), val_losses,   label="Val Loss",   color="darkorange")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("训练曲线")
ax.legend()
ax.grid(True, alpha=0.3)

# (B) Zero-Shot 预测 vs 真实
ax = axes[0, 1]
ax.plot(zs_labels, label="真实值", color="steelblue", lw=1.5)
ax.plot(zs_preds,  label="Zero-Shot 预测", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"Zero-Shot  (MAE: {zs_mae:.0f} kWh)")
ax.set_xlabel("样本序号")
ax.set_ylabel("需求 (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

# (C) Fine-Tune 预测 vs 真实
ax = axes[1, 0]
ax.plot(ft_labels, label="真实值",       color="steelblue", lw=1.5)
ax.plot(ft_preds,  label="Fine-Tune 预测", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"Fine-Tune  (MAE: {ft_mae:.0f} kWh)")
ax.set_xlabel("样本序号")
ax.set_ylabel("需求 (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

# (D) MAE 横向对比柱状图（含 Flan-T5 参考）
ax = axes[1, 1]
labels_bar = ["GPT-2\nZero-Shot", "GPT-2\nFine-Tune", "均值\n基线", "Flan-T5\nFine-Tune"]
maes_bar   = [zs_mae, ft_mae, mae_base, 149.2]
colors_bar = ["lightcoral", "steelblue", "lightgray", "darkorange"]
bars = ax.bar(labels_bar, maes_bar, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, maes_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("MAE (kWh)")
ax.set_title("MAE 横向对比")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = OUTPUT_DIR + "gpt2_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  图表已保存: {plot_path}")

print("\n✅ GPT-2 实验完成")
print(f"   Zero-Shot MAE : {zs_mae:.1f} kWh")
print(f"   Fine-Tune MAE : {ft_mae:.1f} kWh  RMSE: {ft_rmse:.1f}  MAPE: {ft_mape:.1f}%")
