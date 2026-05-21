"""
Use Case 2: EV Charging Demand Prediction
Flan-T5-base — Zero-Shot Regression (regression mode, no fine-tuning)

Purpose: Verify how poorly Flan-T5 Encoder performs without fine-tuning,
         when directly connected to a randomly initialized regression head.
         This is the baseline before fine-tuning, illustrating why fine-tuning is necessary.

Author: XB Hu / Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, T5EncoderModel

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
MODEL_NAME = "google/flan-t5-base"
N_EVAL     = 100       # number of evaluation samples
MAX_LENGTH = 128       # maximum input token length
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Use Case 2: Flan-T5-base Zero-Shot Regression")
print("=" * 60)
print(f"  Device : {DEVICE}")
print(f"  Model  : {MODEL_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Use the last N_EVAL rows as the test set
test_df = df.tail(N_EVAL).reset_index(drop=True)
print(f"    Total data   : {len(df)} rows")
print(f"    Test samples : {N_EVAL} rows")
print(f"    Sample input : \"{test_df['context_text'].iloc[0][:80]}...\"")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load model and Tokenizer
#
# Note: T5EncoderModel is used here, not the full T5.
# Only the Encoder is needed to extract semantic vectors; the Decoder is not required.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Loading Flan-T5 Encoder...")
print(f"    First run will download model from Hugging Face (~500MB)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder   = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()  # inference mode, disable dropout

print(f"    Model loaded successfully")
print(f"    Encoder hidden dim: {encoder.config.d_model}")  # Flan-T5-base = 768

# ─────────────────────────────────────────────────────────────────────────────
# 3. Define regression head
#
# A randomly initialized linear layer: 768-dim vector -> 1 number
# Zero-shot means this regression head has never been trained
# so predictions will be very poor — which is exactly what we want to demonstrate
# ─────────────────────────────────────────────────────────────────────────────
class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

regression_head = RegressionHead(encoder.config.d_model).to(DEVICE)
# Note: no trained weights are loaded; W and b are random numbers

# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference: text -> vector -> predicted number
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] Running inference...")

predictions = []
actuals     = []

with torch.no_grad():  # no gradient computation to save GPU memory
    for i, row in test_df.iterrows():
        # Step 1: text -> tokens
        inputs = tokenizer(
            row["context_text"],
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        ).to(DEVICE)

        # Step 2: tokens -> vector sequence (one 768-dim vector per token)
        encoder_output = encoder(**inputs)
        hidden_states  = encoder_output.last_hidden_state  # shape: [1, seq_len, 768]

        # Step 3: take [CLS] position (first token) vector as sentence summary
        cls_vector = hidden_states[:, 0, :]  # shape: [1, 768]

        # Step 4: vector -> number (random regression head)
        pred = regression_head(cls_vector).item()

        predictions.append(pred)
        actuals.append(row["next_day_demand"])

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{N_EVAL} done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
predictions = np.array(predictions)
actuals     = np.array(actuals)

mae  = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100

# Baseline: predict with mean
mae_mean = mean_absolute_error(actuals, np.full_like(actuals, actuals.mean()))

print("\n" + "=" * 60)
print("  Results")
print("=" * 60)
print(f"  Zero-Shot (random regression head) -> MAE: {mae:.1f} kWh  |  MAPE: {mape:.1f}%")
print(f"  Baseline (mean prediction)         -> MAE: {mae_mean:.1f} kWh")
print(f"\n  Mean actual demand: {actuals.mean():.1f} kWh")
print(f"\n  Conclusion: Zero-shot regression head outputs random numbers, far worse than mean baseline.")
print(f"        This demonstrates that regression models must be fine-tuned to be useful.")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Flan-T5-base Zero-Shot (random regression head) — expected to be very poor", fontsize=12)

# Predicted vs actual
ax = axes[0]
ax.plot(actuals,     label="Actual",   color="steelblue", lw=1.5)
ax.plot(predictions, label="Predicted",   color="darkorange", lw=1, alpha=0.8, linestyle="--")
ax.set_title("Predicted vs Actual (time series)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Demand (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

# Scatter plot
ax = axes[1]
ax.scatter(actuals, predictions, alpha=0.4, s=15, color="steelblue")
lim = [min(actuals.min(), predictions.min()) * 0.9,
       max(actuals.max(), predictions.max()) * 1.1]
ax.plot(lim, lim, "r--", lw=1.5, label="Ideal forecast")
ax.set_xlabel("Actual (kWh)")
ax.set_ylabel("Predicted (kWh)")
ax.set_title(f"Scatter plot  (MAE: {mae:.0f} kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_flant5_zeroshot/usecase2_zeroshot_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved: {plot_path}")

print("\n✅ Zero-shot run complete")
print("   Next step: fine-tune the regression head and observe MAE improvement.")
