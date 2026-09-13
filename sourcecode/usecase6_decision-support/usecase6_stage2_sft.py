"""
Use Case 6 - Stage 2: SFT (Instruction Tuning)
Model: Qwen3-4B (BF16 + LoRA)
"""

import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer as rouge_scorer_lib
from tqdm import tqdm
from datetime import datetime

# ============================================================
# Config
# ============================================================
MODEL_NAME        = "Qwen/Qwen3-4B"
DATA_PATH         = "/home/xzh5180/Research/llm-evprediction/datasets/dataset6_decision_support.csv"
OUTPUT_DIR        = "/home/xzh5180/Research/llm-evprediction/outputs/usecase6_stage2_sft"
CHECKPOINT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
TRAIN_SIZE        = 292
MAX_NEW_TOKENS    = 128
MAX_SEQ_LENGTH    = 512
NUM_EPOCHS        = 5
PER_DEVICE_BATCH  = 4
GRAD_ACCUM_STEPS  = 4
LEARNING_RATE     = 2e-4
WARMUP_RATIO      = 0.1
LORA_R            = 8
LORA_ALPHA        = 16
LORA_DROPOUT      = 0.05
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Load & Split Data
# ============================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df.iloc[:TRAIN_SIZE].copy().reset_index(drop=True)
test_df  = df.iloc[TRAIN_SIZE:].copy().reset_index(drop=True)

print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
print(f"Test : {len(test_df)} rows ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")

# ============================================================
# Load Tokenizer & Model
# ============================================================
print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)

# ============================================================
# LoRA Setup
# ============================================================
print("Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules="all-linear",
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# Format Dataset
# ============================================================
def format_sample(row):
    messages = [
        {"role": "user",      "content": row["decision_prompt"]},
        {"role": "assistant", "content": row["reference_action"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    return {"text": text}

print("\nFormatting training data...")
train_formatted = [format_sample(row) for _, row in train_df.iterrows()]
train_dataset   = Dataset.from_list(train_formatted)

# ============================================================
# Training Arguments
# ============================================================
training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    load_best_model_at_end=False,
    save_total_limit=1,
    report_to="none",
    dataloader_num_workers=0,
)

# ============================================================
# Train
# ============================================================
print("\nStarting SFT training...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()
print("Training complete.")

adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved to: {adapter_path}")

# ============================================================
# Inference on Test Set
# ============================================================
print(f"\nRunning inference on {len(test_df)} test samples...")
model.eval()

def generate_decision(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
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
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

predictions, references, dates = [], [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    pred = generate_decision(row["decision_prompt"])
    predictions.append(pred)
    references.append(row["reference_action"])
    dates.append(str(row["date"].date()))

# ============================================================
# Evaluation
# ============================================================
print("\nCalculating ROUGE scores...")
scorer = rouge_scorer_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge1_list, rouge2_list, rougeL_list = [], [], []
for pred, ref in zip(predictions, references):
    s = scorer.score(ref, pred)
    rouge1_list.append(s["rouge1"].fmeasure)
    rouge2_list.append(s["rouge2"].fmeasure)
    rougeL_list.append(s["rougeL"].fmeasure)

avg_rouge1 = sum(rouge1_list) / len(rouge1_list)
avg_rouge2 = sum(rouge2_list) / len(rouge2_list)
avg_rougeL = sum(rougeL_list) / len(rougeL_list)

print(f"\n{'='*50}")
print(f"  SFT Evaluation Results (n={len(test_df)})")
print(f"{'='*50}")
print(f"  ROUGE-1 : {avg_rouge1:.4f}")
print(f"  ROUGE-2 : {avg_rouge2:.4f}")
print(f"  ROUGE-L : {avg_rougeL:.4f}")
print(f"{'='*50}")

# ============================================================
# Save Results
# ============================================================
results_df = pd.DataFrame({
    "date"            : dates,
    "reference_action": references,
    "predicted_action": predictions,
    "rouge1"          : rouge1_list,
    "rouge2"          : rouge2_list,
    "rougeL"          : rougeL_list,
})
results_csv = os.path.join(OUTPUT_DIR, "stage2_predictions.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nPer-sample results saved to: {results_csv}")

summary = {
    "stage"      : "2_sft",
    "model"      : MODEL_NAME,
    "lora_r"     : LORA_R,
    "lora_alpha" : LORA_ALPHA,
    "train_size" : TRAIN_SIZE,
    "num_epochs" : NUM_EPOCHS,
    "test_size"  : len(test_df),
    "rouge1"     : round(avg_rouge1, 4),
    "rouge2"     : round(avg_rouge2, 4),
    "rougeL"     : round(avg_rougeL, 4),
    "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
summary_path = os.path.join(OUTPUT_DIR, "stage2_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_path}")

print(f"\n{'='*50}")
print("  Sample Comparisons (first 5 test cases)")
print(f"{'='*50}")
for i in range(min(5, len(test_df))):
    print(f"\n[{dates[i]}]")
    print(f"  Reference : {references[i]}")
    print(f"  Predicted : {predictions[i]}")
    print(f"  ROUGE-L   : {rougeL_list[i]:.4f}")

print("\nStage 2 complete.")
