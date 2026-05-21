"""
Use Case 2: EV Charging Demand Prediction
Flan-T5-base — Fine-Tune Regression

Training workflow:
  1. Data split: 80% train / 10% validation / 10% test
  2. Fine-tune: Encoder (Flan-T5) + regression head (linear layer) trained together
  3. Validation monitoring: evaluate after each epoch, save best model
  4. Test evaluation: evaluate with best model at end, compare with zero-shot

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
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
OUTPUT_DIR  = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_flant5_finetune/"
MODEL_NAME  = "google/flan-t5-base"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 2e-4     # learning rate
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
# 1. Load and split data
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

print(f"    Total data  : {n} rows")
print(f"    Train set   : {len(train_df)} rows")
print(f"    Val set     : {len(val_df)} rows")
print(f"    Test set    : {len(test_df)} rows")

# Compute mean and std of training set for label normalization
# Normalization goal: keep targets near 0, making the regression head easier to train
label_mean = train_df["next_day_demand"].mean()
label_std  = train_df["next_day_demand"].std()
print(f"\n    Label mean  : {label_mean:.1f} kWh")
print(f"    Label std   : {label_std:.1f} kWh")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset and DataLoader
#
# PyTorch requires data wrapped in a Dataset object for batch loading
# ─────────────────────────────────────────────────────────────────────────────
class EVDemandDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, label_mean, label_std):
        self.texts  = df["context_text"].tolist()
        # Normalize labels: (actual - mean) / std
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

print("\n[2] Initializing Tokenizer and DataLoader...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EVDemandDataset(train_df, tokenizer, MAX_LENGTH, label_mean, label_std)
val_dataset   = EVDemandDataset(val_df,   tokenizer, MAX_LENGTH, label_mean, label_std)
test_dataset  = EVDemandDataset(test_df,  tokenizer, MAX_LENGTH, label_mean, label_std)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"    Train batches : {len(train_loader)}")
print(f"    Val batches   : {len(val_loader)}")
print(f"    Test batches  : {len(test_loader)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model: Encoder + regression head
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
        # text -> vector sequence
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # take [CLS] (first token) vector as sentence summary
        cls_vector = encoder_output.last_hidden_state[:, 0, :]  # [batch, 768]
        # vector -> number
        return self.regression_head(cls_vector).squeeze(-1)     # [batch]

print("\n[3] Loading model...")
model = FlanT5Regressor(MODEL_NAME).to(DEVICE)
print(f"    Model loaded successfully")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Training configuration
# ─────────────────────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # halve learning rate every 5 epochs
criterion = nn.MSELoss()  # mean squared error, standard regression loss

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4] Starting training ({EPOCHS} epochs)...")
print("-" * 60)

train_losses = []
val_losses   = []
best_val_loss = float("inf")
best_model_path = OUTPUT_DIR + "best_model.pt"

for epoch in range(EPOCHS):
    # ── Training phase ────────────────────────────────────────────
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

    # ── Validation phase ──────────────────────────────────────────
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

    # Save best model on validation set
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

# ─────────────────────────────────────────────────────────────────────────────
# 6. Test set evaluation (using best model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Test set evaluation...")
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

# Denormalize: convert back to real kWh units
all_preds  = np.array(all_preds)  * label_std + label_mean
all_labels = np.array(all_labels) * label_std + label_mean

mae  = mean_absolute_error(all_labels, all_preds)
rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-6))) * 100
mae_baseline = mean_absolute_error(all_labels, np.full_like(all_labels, all_labels.mean()))

print("\n" + "=" * 60)
print("  Results comparison")
print("=" * 60)
print(f"  Zero-Shot (random regression head) -> MAE: 1588.5 kWh  MAPE: 100.0%")
print(f"  Fine-Tune              → MAE: {mae:.1f} kWh  MAPE: {mape:.1f}%")
print(f"  Baseline (mean prediction) -> MAE: {mae_baseline:.1f} kWh")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"Flan-T5-base Fine-Tune  |  MAE: {mae:.0f} kWh  MAPE: {mape:.1f}%",
             fontsize=12, fontweight="bold")

# (A) Training curve
ax = axes[0]
ax.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="steelblue")
ax.plot(range(1, EPOCHS+1), val_losses,   label="Val Loss",   color="darkorange")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (normalized space)")
ax.set_title("Training Curve")
ax.legend()
ax.grid(True, alpha=0.3)

# (B) Predicted vs actual (time series)
ax = axes[1]
ax.plot(all_labels, label="Actual", color="steelblue", lw=1.5)
ax.plot(all_preds,  label="Predicted", color="darkorange", lw=1.5, linestyle="--")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Demand (kWh)")
ax.set_title("Predicted vs Actual")
ax.legend()
ax.grid(True, alpha=0.3)

# (C) Scatter plot
ax = axes[2]
lim = [min(all_labels.min(), all_preds.min()) * 0.95,
       max(all_labels.max(), all_preds.max()) * 1.05]
ax.scatter(all_labels, all_preds, alpha=0.5, s=20, color="steelblue")
ax.plot(lim, lim, "r--", lw=1.5, label="Ideal forecast")
ax.set_xlabel("Actual (kWh)")
ax.set_ylabel("Predicted (kWh)")
ax.set_title(f"Scatter plot  (MAE: {mae:.0f} kWh)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR + "finetune_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved: {plot_path}")

print("\n✅ Fine-tuning complete")
print(f"   MAE  : {mae:.1f} kWh  ({mae/all_labels.mean()*100:.1f}% of mean)")
print(f"   RMSE : {rmse:.1f} kWh")
print(f"   MAPE : {mape:.1f}%")
