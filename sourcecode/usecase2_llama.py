"""
Use Case 2: EV Charging Demand Prediction
Llama 3.1 8B — Zero-Shot + QLoRA Fine-Tune

前置条件：
  1. pip install peft bitsandbytes accelerate transformers
  2. huggingface-cli login （需要 Llama 访问权限）

QLoRA = 4-bit 量化 + LoRA
  - 4-bit 量化：把模型权重从 float16 压缩到 4-bit，显存从 ~16GB 降到 ~6GB
  - LoRA：只训练旁路矩阵，不动原始权重
  - 合计显存需求：约 8–10GB，12GB 显卡可跑

Author: XB Hu / Smart Mobility Lab, Penn State
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_llama/"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_LENGTH = 200
BATCH_SIZE = 2         # QLoRA 显存紧张，batch size 要小
GRAD_ACCUM = 8         # 梯度累积：等效 batch size = 2 × 8 = 16
EPOCHS     = 10        # Llama 比 GPT-2 收敛快，epoch 不需要太多
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# LoRA 超参数
LORA_R        = 16    # 旁路矩阵的秩，越大表达能力越强，显存越多
LORA_ALPHA    = 32    # LoRA 缩放系数，通常设为 r 的 2 倍
LORA_DROPOUT  = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("Use Case 2: Llama 3.1 8B QLoRA Zero-Shot + Fine-Tune")
print("=" * 60)
print(f"  Device      : {DEVICE}")
print(f"  Model       : {MODEL_NAME}")
print(f"  LoRA r      : {LORA_R}")
print(f"  LoRA alpha  : {LORA_ALPHA}")
print(f"  Epochs      : {EPOCHS}")
print(f"  Batch size  : {BATCH_SIZE} × {GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  LR          : {LR}")

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
# 2. 加载 Tokenizer 和 4-bit 量化模型
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] 加载 Llama 3.1 8B（4-bit 量化）...")
print(f"    第一次运行会从 Hugging Face 下载模型（约 16GB）...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 用 4-bit 加载
    bnb_4bit_quant_type="nf4",               # NF4 量化，精度损失最小
    bnb_4bit_compute_dtype=torch.float16,    # 计算时用 float16
    bnb_4bit_use_double_quant=True           # 双重量化，进一步节省显存
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",                       # 自动分配到可用 GPU
    trust_remote_code=True
)
base_model.config.pad_token_id = tokenizer.eos_token_id

print(f"    基础模型加载完成")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 添加 LoRA 旁路矩阵
#
# LoRA 的核心：不改动原始权重 W，而是在旁边加一个小矩阵 A×B
# 训练时只更新 A 和 B，原始 W 完全冻结
#
# target_modules：对哪些层加 LoRA（Llama 的注意力层）
# ─────────────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[                         # Llama 的 attention 权重矩阵
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none"
)

model = get_peft_model(base_model, lora_config)

# 打印可训练参数量
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\n    总参数量     : {total/1e6:.0f}M")
print(f"    可训练参数量 : {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")
print(f"    （LoRA 只训练 {trainable/total*100:.2f}% 的参数，其余全部冻结）")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Prompt 格式
#
# Llama 3.1 Instruct 有特定的对话格式，遵守格式能发挥最大效果
# ─────────────────────────────────────────────────────────────────────────────
def build_inference_prompt(context_text: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are an EV charging demand forecaster. "
        f"Given a daily context description, predict the next-day total EV charging demand in kWh. "
        f"Respond with ONLY the integer number, nothing else.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{context_text.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def build_training_text(context_text: str, demand: float) -> str:
    prompt = build_inference_prompt(context_text)
    return prompt + f"{demand:.0f}<|eot_id|>"

def parse_demand(text: str, prompt: str) -> float | None:
    generated = text[len(prompt):].strip()
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

zs_preds    = []
zs_labels   = []
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
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = parse_demand(generated_text, prompt)

        if pred is None or pred < 100 or pred > 5000:
            zs_failures += 1
            pred = test_df["next_day_demand"].mean()

        zs_preds.append(pred)
        zs_labels.append(row["next_day_demand"])

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_df)} 完成")

zs_preds  = np.array(zs_preds)
zs_labels = np.array(zs_labels)
zs_mae    = mean_absolute_error(zs_labels, zs_preds)
zs_mape   = np.mean(np.abs((zs_labels - zs_preds) / (zs_labels + 1e-6))) * 100

print(f"\n  Zero-Shot 结果:")
print(f"    MAE : {zs_mae:.1f} kWh  |  MAPE : {zs_mape:.1f}%")
print(f"    解析失败 : {zs_failures}/{len(test_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 阶段二：QLoRA Fine-Tune
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  阶段二：QLoRA Fine-Tune")
print("=" * 60)

class LlamaDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.samples   = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for _, row in df.iterrows():
            full_text  = build_training_text(row["context_text"], row["next_day_demand"])
            prompt     = build_inference_prompt(row["context_text"])
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            self.samples.append((full_text, prompt_len))

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
        labels         = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "labels"         : labels
        }

print("\n[3] 初始化 Dataset...")
train_dataset = LlamaDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = LlamaDataset(val_df,   tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
print(f"    训练 batches : {len(train_loader)}")
print(f"    验证 batches : {len(val_loader)}")

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)
scheduler     = StepLR(optimizer, step_size=3, gamma=0.5)
best_val_loss = float("inf")
best_model_path = OUTPUT_DIR + "best_lora"
train_losses  = []
val_losses    = []

print(f"\n[4] 开始训练（{EPOCHS} epochs）...")
print("-" * 60)

for epoch in range(EPOCHS):
    # 训练
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss / GRAD_ACCUM   # 梯度累积
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += outputs.loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # LoRA 只保存旁路矩阵（很小，几十MB），不保存整个模型
        model.save_pretrained(best_model_path)
        flag = " ← 最佳"
    else:
        flag = ""

    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}{flag}")

print(f"\n  训练完成，最佳 LoRA 权重已保存: {best_model_path}")

# ── 测试集评估 ────────────────────────────────────────────────────────────────
print("\n[5] 测试集评估（QLoRA Fine-Tune）...")
model = PeftModel.from_pretrained(base_model, best_model_path)
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
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
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
# 最终结果
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  最终结果对比")
print("=" * 60)
print(f"  Llama Zero-Shot      → MAE: {zs_mae:.1f} kWh  |  MAPE: {zs_mape:.1f}%")
print(f"  Llama QLoRA Fine-Tune→ MAE: {ft_mae:.1f} kWh  |  MAPE: {ft_mape:.1f}%")
print(f"  基线（均值预测）      → MAE: {mae_base:.1f} kWh")
print(f"\n  参考：Flan-T5 Fine-Tune → MAE: 149.2 kWh  |  MAPE: 12.2%")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"Llama 3.1 8B QLoRA  |  Zero-Shot MAE: {zs_mae:.0f}  Fine-Tune MAE: {ft_mae:.0f} kWh",
    fontsize=12, fontweight="bold"
)

ax = axes[0, 0]
ax.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="steelblue")
ax.plot(range(1, EPOCHS+1), val_losses,   label="Val Loss",   color="darkorange")
ax.set_title("训练曲线")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(zs_labels, label="真实值",        color="steelblue",  lw=1.5)
ax.plot(zs_preds,  label="Zero-Shot 预测", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"Zero-Shot  (MAE: {zs_mae:.0f} kWh)")
ax.set_xlabel("样本序号")
ax.set_ylabel("需求 (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(ft_labels, label="真实值",          color="steelblue",  lw=1.5)
ax.plot(ft_preds,  label="Fine-Tune 预测",  color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"QLoRA Fine-Tune  (MAE: {ft_mae:.0f} kWh)")
ax.set_xlabel("样本序号")
ax.set_ylabel("需求 (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
labels_bar = ["Llama\nZero-Shot", "Llama\nQLoRA", "均值\n基线", "Flan-T5\nFine-Tune"]
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
plot_path = OUTPUT_DIR + "llama_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  图表已保存: {plot_path}")

print("\n✅ Llama 3.1 8B QLoRA 实验完成")
print(f"   Zero-Shot MAE  : {zs_mae:.1f} kWh")
print(f"   Fine-Tune MAE  : {ft_mae:.1f} kWh  RMSE: {ft_rmse:.1f}  MAPE: {ft_mape:.1f}%")
