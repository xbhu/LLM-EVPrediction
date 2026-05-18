"""
Use Case 6 - Stage 1: Zero-shot Decision Support
Model: Qwen3-4B (BF16, enable_thinking=False)
Task: Given operational context, generate charging management recommendation
"""

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer as rouge_scorer_lib
from tqdm import tqdm
from datetime import datetime

# ============================================================
# Config
# ============================================================
MODEL_NAME        = "Qwen/Qwen3-4B"
DATA_PATH         = "/home/xzh5180/Research/llm-evprediction/datasets/dataset6_decision_support.csv"
OUTPUT_DIR        = "/home/xzh5180/Research/llm-evprediction/outputs/usecase6_stage1_zeroshot"
TRAIN_SIZE        = 292          # first 292 rows → train (used for Stage 2/3)
MAX_NEW_TOKENS    = 128
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load & Split Data
# ============================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df.iloc[:TRAIN_SIZE].copy()
test_df  = df.iloc[TRAIN_SIZE:].copy().reset_index(drop=True)

print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
print(f"Test : {len(test_df)} rows ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")

# ============================================================
# Load Model
# ============================================================
print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded.")

# ============================================================
# Inference Function
# ============================================================
def generate_decision(prompt_text: str) -> str:
    """
    Takes the pre-formatted decision_prompt from the dataset
    and returns the model's recommendation.
    """
    messages = [{"role": "user", "content": prompt_text}]

    # enable_thinking=False: skip chain-of-thought, output clean action directly
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy — reproducible
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

    return generated

# ============================================================
# Run Inference on Test Set
# ============================================================
print(f"\nRunning zero-shot inference on {len(test_df)} test samples...")

predictions = []
references  = []
dates       = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    pred = generate_decision(row["decision_prompt"])
    predictions.append(pred)
    references.append(row["reference_action"])
    dates.append(str(row["date"].date()))

# ============================================================
# Evaluation: ROUGE-1, ROUGE-2, ROUGE-L
# ============================================================
print("\nCalculating ROUGE scores...")
scorer = rouge_scorer_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge1_list, rouge2_list, rougeL_list = [], [], []

for pred, ref in zip(predictions, references):
    scores = scorer.score(ref, pred)
    rouge1_list.append(scores["rouge1"].fmeasure)
    rouge2_list.append(scores["rouge2"].fmeasure)
    rougeL_list.append(scores["rougeL"].fmeasure)

avg_rouge1 = sum(rouge1_list) / len(rouge1_list)
avg_rouge2 = sum(rouge2_list) / len(rouge2_list)
avg_rougeL = sum(rougeL_list) / len(rougeL_list)

print(f"\n{'='*50}")
print(f"  Zero-shot Evaluation Results (n={len(test_df)})")
print(f"{'='*50}")
print(f"  ROUGE-1 : {avg_rouge1:.4f}")
print(f"  ROUGE-2 : {avg_rouge2:.4f}")
print(f"  ROUGE-L : {avg_rougeL:.4f}")
print(f"{'='*50}")

# ============================================================
# Save Results
# ============================================================

# 1. Per-sample detail CSV
results_df = pd.DataFrame({
    "date"            : dates,
    "reference_action": references,
    "predicted_action": predictions,
    "rouge1"          : rouge1_list,
    "rouge2"          : rouge2_list,
    "rougeL"          : rougeL_list,
})
results_csv = os.path.join(OUTPUT_DIR, "stage1_predictions.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nPer-sample results saved to: {results_csv}")

# 2. Summary JSON
summary = {
    "stage"      : "1_zeroshot",
    "model"      : MODEL_NAME,
    "test_size"  : len(test_df),
    "rouge1"     : round(avg_rouge1, 4),
    "rouge2"     : round(avg_rouge2, 4),
    "rougeL"     : round(avg_rougeL, 4),
    "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
summary_path = os.path.join(OUTPUT_DIR, "stage1_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_path}")

# 3. Print a few sample comparisons for qualitative inspection
print(f"\n{'='*50}")
print("  Sample Comparisons (first 5 test cases)")
print(f"{'='*50}")
for i in range(min(5, len(test_df))):
    print(f"\n[{dates[i]}]")
    print(f"  Reference : {references[i]}")
    print(f"  Predicted : {predictions[i]}")
    print(f"  ROUGE-L   : {rougeL_list[i]:.4f}")

print("\nStage 1 complete.")
