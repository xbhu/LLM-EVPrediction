# ============================================================
# usecase5_route_b_finetune.py
# Use Case 5 - Route B: Fine-tuned Seq2Seq Explainer
# Model: Flan-T5-large (780M, encoder-decoder)
#
# Fix log (relative to initial version):
#   1. evaluation_strategy -> eval_strategy (renamed in newer transformers)
#   2. as_target_tokenizer() removed -> use text_target= parameter instead
#   3. Seq2SeqTrainer uses processing_class= instead of tokenizer=
# ============================================================

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset5_anomaly.csv"
OUTPUT_DIR     = "/home/xzh5180/Research/llm-evprediction/outputs/usecase5_route_b"
MODEL_NAME     = "google/flan-t5-large"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 64
TRAIN_RATIO    = 0.8
NUM_EPOCHS     = 10       # intentionally run extra epochs to let overfitting fully emerge
BATCH_SIZE     = 4
LEARNING_RATE  = 5e-4
SEED           = 42

# ============================================================
import os, torch, numpy as np, pandas as pd, random
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset
from rouge_score import rouge_scorer as rouge_lib
import bert_score

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1. Load data & build prompts
# ============================================================
df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])

normal_df  = df[df["is_anomaly"] == 0].copy()
anomaly_df = df[df["is_anomaly"] == 1].copy().reset_index(drop=True)
print(f"Total: {len(df)} | Normal: {len(normal_df)} | Anomaly: {len(anomaly_df)}")

hour_baseline = normal_df.groupby("hour")["demand_kwh"].mean().to_dict()
anomaly_df["expected_kwh"]  = anomaly_df["hour"].map(hour_baseline)
anomaly_df["deviation_kwh"] = anomaly_df["demand_kwh"] - anomaly_df["expected_kwh"]
anomaly_df["deviation_pct"] = (
    anomaly_df["deviation_kwh"] / anomaly_df["expected_kwh"] * 100
).round(1)

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"]

def build_prompt(row):
    direction = "above" if row["deviation_kwh"] > 0 else "below"
    return (
        "You are an EV charging demand analyst. "
        "A significant deviation has been detected at a charging station.\n\n"
        f"Timestamp       : {row['timestamp']}\n"
        f"Hour of day     : {int(row['hour'])}:00\n"
        f"Day of week     : {DAY_NAMES[int(row['day_of_week'])]}\n"
        f"Month           : {int(row['month'])}\n"
        f"Weekend         : {'Yes' if row['is_weekend'] else 'No'}\n"
        f"Scheduled event : {row['event_type']}\n"
        f"Temperature     : {row['temperature_f']}°F\n\n"
        f"Expected demand (historical average for this hour) : {row['expected_kwh']:.1f} kWh\n"
        f"Actual observed demand                            : {row['demand_kwh']:.1f} kWh\n"
        f"Deviation       : {abs(row['deviation_kwh']):.1f} kWh "
        f"{direction} expected ({abs(row['deviation_pct'])}%)\n\n"
        "What is the most likely cause of this anomaly? "
        "Give a brief one-sentence explanation."
    )

anomaly_df["prompt"] = anomaly_df.apply(build_prompt, axis=1)

# ============================================================
# 2. Train / Val split (80/20)
# ============================================================
n       = len(anomaly_df)
indices = list(range(n))
random.shuffle(indices)

train_end = int(n * TRAIN_RATIO)
train_idx = indices[:train_end]
val_idx   = indices[train_end:]

train_df = anomaly_df.iloc[train_idx].reset_index(drop=True)
val_df   = anomaly_df.iloc[val_idx].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} samples | Val: {len(val_df)} samples")
print("\nTrain anomaly type distribution:")
print(train_df["anomaly_type"].value_counts().to_string())
print("\nVal anomaly type distribution:")
print(val_df["anomaly_type"].value_counts().to_string())

# ============================================================
# 3. PyTorch Dataset
#
# Seq2Seq tokenization key points:
#   - Input tokenized normally
#   - Output uses text_target= parameter (new API, replaces removed as_target_tokenizer())
#   - Padding is handled uniformly by DataCollator; not done here
# ============================================================
class AnomalyDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data      = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize input
        model_inputs = self.tokenizer(
            row["prompt"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )

        # Tokenize output (text_target= is the new API)
        labels = self.tokenizer(
            text_target=row["llm_explanation"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ============================================================
# 4. Load tokenizer & model
# ============================================================
print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
print("Model loaded.")

# ============================================================
# 5. Dataset & DataCollator
#
# Role of DataCollatorForSeq2Seq:
#   - Pad sequences of different lengths within a batch to the same length
#   - Set padding positions in labels to -100 (automatically ignored during loss computation)
# ============================================================
train_dataset = AnomalyDataset(train_df, tokenizer)
val_dataset   = AnomalyDataset(val_df,   tokenizer)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ============================================================
# 6. Training arguments
#    eval_strategy is the new API (old name was evaluation_strategy; renamed)
# ============================================================
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LEARNING_RATE,
    bf16                        = True,
    predict_with_generate       = True,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    save_total_limit            = 1,
    logging_strategy            = "epoch",
    report_to                   = "none",
    seed                        = SEED,
)

# ============================================================
# 7. Trainer
#    processing_class= is the new API (old name was tokenizer=)
# ============================================================
trainer = Seq2SeqTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_dataset,
    eval_dataset     = val_dataset,
    processing_class = tokenizer,
    data_collator    = data_collator,
)

print("\nStarting training ...")
print(f"  Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
trainer.train()
print("Training complete.")

# ============================================================
# 8. Generate predictions on val set
# ============================================================
print("\nGenerating predictions on validation set ...")
model.eval()

predictions = []
for _, row in val_df.iterrows():
    inputs = tokenizer(
        row["prompt"],
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_TARGET_LEN)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    predictions.append(decoded)

val_df = val_df.copy()
val_df["prediction"] = predictions

# ============================================================
# 9. Evaluate: ROUGE-L + BERTScore
# ============================================================
references = val_df["llm_explanation"].tolist()

scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = [scorer.score(ref, pred)["rougeL"].fmeasure
           for ref, pred in zip(references, predictions)]
val_df["rouge_l"] = rouge_l

print("\nComputing BERTScore ...")
P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=False)
val_df["bertscore_f1"] = F1.numpy()

# ============================================================
# 10. Print results
# ============================================================
print("\n" + "="*60)
print("VALIDATION SET EVALUATION (fine-tuned Flan-T5-large)")
print("="*60)
print(f"  Samples     : {len(val_df)}")
print(f"  ROUGE-L     : {np.mean(rouge_l):.4f}")
print(f"  BERTScore F1: {F1.mean().item():.4f}")

print("\n" + "="*60)
print("SAMPLE OUTPUTS")
print("="*60)
for _, row in val_df.iterrows():
    print(f"\n[{row['anomaly_type']}]")
    print(f"  Ground truth : {row['llm_explanation']}")
    print(f"  Prediction   : {row['prediction']}")
    print(f"  ROUGE-L      : {row['rouge_l']:.4f}")
    print(f"  BERTScore F1 : {row['bertscore_f1']:.4f}")

# ============================================================
# 11. Compare with Route A baseline
# ============================================================
print("\n" + "="*60)
print("COMPARISON WITH ROUTE A (zero-shot)")
print("="*60)
print(f"  Flan-T5-base  zero-shot  → ROUGE-L: 0.0085 | BERTScore: 0.7942")
print(f"  Qwen3-4B      zero-shot  → ROUGE-L: 0.0449 | BERTScore: 0.8604")
print(f"  Flan-T5-large fine-tuned → ROUGE-L: {np.mean(rouge_l):.4f} | BERTScore: {F1.mean().item():.4f}")

# ============================================================
# 12. Save
# ============================================================
save_cols = [
    "timestamp","anomaly_type","anomaly_magnitude",
    "expected_kwh","demand_kwh","deviation_pct",
    "llm_explanation","prediction","rouge_l","bertscore_f1"
]
out_path = os.path.join(OUTPUT_DIR, "route_b_results.csv")
val_df[save_cols].to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")
