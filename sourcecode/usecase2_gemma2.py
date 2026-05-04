"""
Use Case 2: EV Charging Demand Prediction
Gemma 2 9B — Zero-Shot + QLoRA Fine-Tune

推理修复：直接 decode 新生成的 token，不用字符串切片

SKIP_TRAINING = True  → 跳过训练，直接加载已有 LoRA 权重测试
SKIP_TRAINING = False → 重新训练

前置条件：
  pip install peft bitsandbytes accelerate transformers

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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
OUTPUT_DIR   = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_gemma2/"
MODEL_NAME   = "google/gemma-2-9b-it"
MAX_LENGTH   = 200
BATCH_SIZE   = 2
GRAD_ACCUM   = 8
EPOCHS       = 10
LR           = 2e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED  = 42
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

SKIP_TRAINING = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("Use Case 2: Gemma 2 9B QLoRA Zero-Shot + Fine-Tune")
print("=" * 60)
print(f"  SKIP_TRAINING : {SKIP_TRAINING}")
print(f"  Device        : {DEVICE}")
print(f"  Model         : {MODEL_NAME}")

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
# 2. 加载模型
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] 加载 Gemma 2 9B（4-bit 量化）...")
print(f"    第一次运行会从 Hugging Face 下载模型（约 18GB）...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Gemma 2 推荐 bfloat16
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model.config.pad_token_id = tokenizer.eos_token_id
print(f"    基础模型加载完成")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt 格式
#
# Gemma 2 Instruct 使用 <start_of_turn> / <end_of_turn> 格式
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are an EV charging demand forecaster. "
    "Given a daily context description, predict the next-day total EV charging demand in kWh. "
    "Respond with ONLY the integer number, nothing else."
)

def build_inference_prompt(context_text: str) -> str:
    return (
        f"<bos><start_of_turn>user\n"
        f"{SYSTEM_MSG}\n\n"
        f"{context_text.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def build_training_text(context_text: str, demand: float) -> str:
    return build_inference_prompt(context_text) + f"{demand:.0f}<end_of_turn>"

def run_inference(model, tokenizer, context_text: str) -> float | None:
    prompt = build_inference_prompt(context_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH - 10
    ).to(DEVICE)

    prompt_len = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    new_tokens    = output[0][prompt_len:]
    generated_str = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    match = re.search(r'\b(\d{3,5}(?:\.\d+)?)\b', generated_str)
    if match:
        val = float(match.group(1))
        if 100 < val < 5000:
            return val
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

base_model.eval()
with torch.no_grad():
    for i, row in test_df.iterrows():
        pred = run_inference(base_model, tokenizer, row["context_text"])

        if pred is None:
            zs_failures += 1
            pred = test_df["next_day_demand"].mean()

        zs_preds.append(pred)
        zs_labels.append(row["next_day_demand"])

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_df)} 完成  pred={pred:.0f}")

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
best_model_path = OUTPUT_DIR + "best_lora"
train_losses    = []
val_losses      = []

if SKIP_TRAINING:
    print("\n" + "=" * 60)
    print("  阶段二：跳过训练，加载已有 LoRA 权重")
    print("=" * 60)
    print(f"    加载: {best_model_path}")

else:
    print("\n" + "=" * 60)
    print("  阶段二：QLoRA Fine-Tune")
    print("=" * 60)

    # Gemma 2 的 attention 层名称
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")

    class GemmaDataset(Dataset):
        def __init__(self, df, tokenizer, max_length):
            self.samples = []
            for _, row in df.iterrows():
                full_text  = build_training_text(row["context_text"], row["next_day_demand"])
                prompt_len = len(tokenizer.encode(
                    build_inference_prompt(row["context_text"]), add_special_tokens=False))
                self.samples.append((full_text, prompt_len))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            full_text, prompt_len = self.samples[idx]
            enc = tokenizer(full_text, max_length=MAX_LENGTH, truncation=True,
                            padding="max_length", return_tensors="pt")
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels         = input_ids.clone()
            labels[:prompt_len]         = -100
            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(GemmaDataset(train_df, tokenizer, MAX_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GemmaDataset(val_df, tokenizer, MAX_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=False)

    optimizer     = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=LR, weight_decay=0.01)
    scheduler     = StepLR(optimizer, step_size=3, gamma=0.5)
    best_val_loss = float("inf")

    print(f"\n[4] 开始训练（{EPOCHS} epochs）...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss           = outputs.loss / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_train_loss += outputs.loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"].to(DEVICE),
                                attention_mask=batch["attention_mask"].to(DEVICE),
                                labels=batch["labels"].to(DEVICE))
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(best_model_path)
            flag = " ← 最佳"
        else:
            flag = ""

        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
              f"Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}{flag}")

# ── 测试集评估 ────────────────────────────────────────────────────────────────
print("\n[5] 测试集评估（QLoRA Fine-Tune）...")
ft_model = PeftModel.from_pretrained(base_model, best_model_path)
ft_model.eval()

ft_preds    = []
ft_labels   = []
ft_failures = 0

with torch.no_grad():
    for i, row in test_df.iterrows():
        pred = run_inference(ft_model, tokenizer, row["context_text"])

        if pred is None:
            ft_failures += 1
            pred = test_df["next_day_demand"].mean()

        ft_preds.append(pred)
        ft_labels.append(row["next_day_demand"])

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_df)} 完成  pred={pred:.0f}")

ft_preds  = np.array(ft_preds)
ft_labels = np.array(ft_labels)
ft_mae    = mean_absolute_error(ft_labels, ft_preds)
ft_rmse   = np.sqrt(mean_squared_error(ft_labels, ft_preds))
ft_mape   = np.mean(np.abs((ft_labels - ft_preds) / (ft_labels + 1e-6))) * 100
mae_base  = mean_absolute_error(ft_labels, np.full_like(ft_labels, ft_labels.mean()))

print("\n" + "=" * 60)
print("  最终结果对比")
print("=" * 60)
print(f"  Gemma2 Zero-Shot       → MAE: {zs_mae:.1f} kWh  |  MAPE: {zs_mape:.1f}%")
print(f"  Gemma2 QLoRA Fine-Tune → MAE: {ft_mae:.1f} kWh  |  MAPE: {ft_mape:.1f}%")
print(f"  基线（均值预测）        → MAE: {mae_base:.1f} kWh")
print(f"\n  参考：Mistral QLoRA  → MAE: 42.3 kWh")
print(f"  参考：Llama QLoRA    → MAE: 73.5 kWh")
print(f"  参考：Flan-T5        → MAE: 149.2 kWh")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 可视화
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"Gemma 2 9B QLoRA  |  Zero-Shot MAE: {zs_mae:.0f}  Fine-Tune MAE: {ft_mae:.0f} kWh",
    fontsize=12, fontweight="bold"
)

ax = axes[0, 0]
if train_losses:
    ax.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", color="steelblue")
    ax.plot(range(1, len(val_losses)+1),   val_losses,   label="Val Loss",   color="darkorange")
    ax.legend()
else:
    ax.text(0.5, 0.5, "Training skipped\nUsing saved weights", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)
ax.set_title("Training Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(zs_labels, label="Actual",    color="steelblue",  lw=1.5)
ax.plot(zs_preds,  label="Zero-Shot", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"Zero-Shot  (MAE: {zs_mae:.0f} kWh)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Demand (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(ft_labels, label="Actual",    color="steelblue",  lw=1.5)
ax.plot(ft_preds,  label="Fine-Tune", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"QLoRA Fine-Tune  (MAE: {ft_mae:.0f} kWh)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Demand (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
labels_bar = ["Gemma2\nZero-Shot", "Gemma2\nQLoRA", "Mean\nBaseline",
              "Mistral\nQLoRA", "Llama\nQLoRA", "Flan-T5\nFT"]
maes_bar   = [zs_mae, ft_mae, mae_base, 42.3, 73.5, 149.2]
colors_bar = ["lightcoral", "steelblue", "lightgray", "darkorange", "mediumpurple", "seagreen"]
bars = ax.bar(labels_bar, maes_bar, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, maes_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val:.0f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("MAE (kWh)")
ax.set_title("MAE Comparison")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = OUTPUT_DIR + "gemma2_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  图表已保存: {plot_path}")
print(f"  Zero-Shot 解析失败: {zs_failures}/{len(test_df)}")
print(f"  Fine-Tune 解析失败: {ft_failures}/{len(test_df)}")
print(f"  Fine-Tune RMSE: {ft_rmse:.1f} kWh")
print("\n✅ Gemma 2 9B QLoRA 完成")
