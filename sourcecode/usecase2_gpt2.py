"""
Use Case 2: EV Charging Demand Prediction
GPT-2 Medium — Zero-Shot + Fine-Tune

Phase 1: Zero-Shot  — directly run pretrained GPT-2 inference with no training
Phase 2: Fine-Tune  — train on context_text + demand concatenated as full text
Final: three-way comparison — Zero-Shot vs Fine-Tune vs mean baseline

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
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_gpt2/"
MODEL_NAME = "gpt2-medium"
MAX_LENGTH = 148       # space for context_text tokens + demand number
BATCH_SIZE = 8         # GPT-2 uses more VRAM than T5; batch size reduced accordingly
EPOCHS     = 20
LR         = 2e-5      # generative fine-tune LR is typically one order of magnitude smaller than regression
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
N_ZEROSHOT  = 100      # number of zero-shot evaluation samples

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
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

n       = len(df)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

train_df = df.iloc[:n_train].reset_index(drop=True)
val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

print(f"    Total data : {n} rows")
print(f"    Train set  : {len(train_df)} rows")
print(f"    Val set    : {len(val_df)} rows")
print(f"    Test set   : {len(test_df)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Tokenizer and model
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Loading GPT-2 Medium...")
print(f"    First run will download model from Hugging Face (~1.5GB)...")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no padding token; use eos_token as substitute
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.pad_token_id = tokenizer.eos_token_id

print(f"    Model loaded successfully")
print(f"    Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper functions: build prompt and parse number
#
# Key design for generative approach: concatenate input and output into one text
#
# During training:
#   "Context: Winter weekday, 32.5°F... Predicted demand: 1543"
#   The model learns to generate the correct number given the context
#
# During inference:
#   Only provide "Context: Winter weekday, 32.5°F... Predicted demand:"
#   Let the model continue and generate the number
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
    """Extract the demand number from the model's generated text"""
    # Remove the prompt portion; only look at the continuation
    generated = text[len(prompt):].strip()
    # Extract the first integer or decimal number
    match = re.search(r'\b(\d{3,5}(?:\.\d+)?)\b', generated)
    if match:
        return float(match.group(1))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Zero-Shot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 1: Zero-Shot")
print("=" * 60)

# Run zero-shot evaluation on the test set
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
            max_new_tokens=10,      # only a few digit characters need to be generated
            do_sample=False,        # greedy decoding for deterministic output
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = parse_demand(generated_text, prompt)

        if pred is None or pred < 100 or pred > 5000:
            zs_failures += 1
            pred = test_df["next_day_demand"].mean()  # fall back to mean if parse fails

        zs_preds.append(pred)
        zs_labels.append(row["next_day_demand"])

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_df)} done  — generated: {generated_text[-30:].strip()!r}")

zs_preds  = np.array(zs_preds)
zs_labels = np.array(zs_labels)
zs_mae    = mean_absolute_error(zs_labels, zs_preds)
zs_mape   = np.mean(np.abs((zs_labels - zs_preds) / (zs_labels + 1e-6))) * 100

print(f"\n  Zero-Shot results:")
print(f"    MAE : {zs_mae:.1f} kWh  |  MAPE : {zs_mape:.1f}%")
print(f"    Parse failures : {zs_failures}/{len(test_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Fine-Tune
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 2: Fine-Tune")
print("=" * 60)

# ── Dataset ──────────────────────────────────────────────────────────────────
class GPT2DemandDataset(Dataset):
    """
    Data format for generative fine-tuning:
    Concatenate context_text and demand into a full text; model learns to generate the whole sequence.
    Loss is computed only over the demand portion (via labels masking).
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

        # Labels: set prompt positions to -100 (excluded from loss); only compute loss over demand
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "labels"         : labels
        }

print("\n[3] Initializing Dataset...")
train_dataset = GPT2DemandDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = GPT2DemandDataset(val_df,   tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"    Train batches : {len(train_loader)}")
print(f"    Val batches   : {len(val_loader)}")

# ── Training loop ────────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

train_losses  = []
val_losses    = []
best_val_loss = float("inf")
best_model_path = OUTPUT_DIR + "best_model.pt"

print(f"\n[4] Starting training ({EPOCHS} epochs)...")
print("-" * 60)

for epoch in range(EPOCHS):
    # Training
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
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
        flag = " <- Best"
    else:
        flag = ""

    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}{flag}")

print(f"\n  Training complete, best model saved: {best_model_path}")

# ── Test set evaluation ─────────────────────────────────────────────────────────
print("\n[5] Test set evaluation (Fine-Tune)...")
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
# Final results comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Final results comparison")
print("=" * 60)
print(f"  GPT-2 Zero-Shot  → MAE: {zs_mae:.1f} kWh  |  MAPE: {zs_mape:.1f}%")
print(f"  GPT-2 Fine-Tune  → MAE: {ft_mae:.1f} kWh  |  MAPE: {ft_mape:.1f}%")
print(f"  Baseline (mean prediction) -> MAE: {mae_base:.1f} kWh")
print(f"\n  Reference: Flan-T5 Fine-Tune -> MAE: 149.2 kWh  |  MAPE: 12.2%")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"GPT-2 Medium  |  Zero-Shot MAE: {zs_mae:.0f}  Fine-Tune MAE: {ft_mae:.0f} kWh",
             fontsize=12, fontweight="bold")

# (A) Training curve
ax = axes[0, 0]
ax.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="steelblue")
ax.plot(range(1, EPOCHS+1), val_losses,   label="Val Loss",   color="darkorange")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curve")
ax.legend()
ax.grid(True, alpha=0.3)

# (B) Zero-Shot predicted vs actual
ax = axes[0, 1]
ax.plot(zs_labels, label="Actual", color="steelblue", lw=1.5)
ax.plot(zs_preds,  label="Zero-Shot", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"Zero-Shot  (MAE: {zs_mae:.0f} kWh)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Demand (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

# (C) Fine-Tune predicted vs actual
ax = axes[1, 0]
ax.plot(ft_labels, label="Actual",   color="steelblue", lw=1.5)
ax.plot(ft_preds,  label="Fine-Tune", color="darkorange", lw=1.5, linestyle="--")
ax.set_title(f"Fine-Tune  (MAE: {ft_mae:.0f} kWh)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Demand (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

# (D) MAE bar chart comparison (including Flan-T5 reference)
ax = axes[1, 1]
labels_bar = ["GPT-2\nZero-Shot", "GPT-2\nFine-Tune", "Mean\nBaseline", "Flan-T5\nFine-Tune"]
maes_bar   = [zs_mae, ft_mae, mae_base, 149.2]
colors_bar = ["lightcoral", "steelblue", "lightgray", "darkorange"]
bars = ax.bar(labels_bar, maes_bar, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, maes_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("MAE (kWh)")
ax.set_title("MAE Comparison")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = OUTPUT_DIR + "gpt2_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved: {plot_path}")

print("\n✅ GPT-2 experiment complete")
print(f"   Zero-Shot MAE : {zs_mae:.1f} kWh")
print(f"   Fine-Tune MAE : {ft_mae:.1f} kWh  RMSE: {ft_rmse:.1f}  MAPE: {ft_mape:.1f}%")
