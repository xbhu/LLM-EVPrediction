# ============================================================
# usecase5_route_a_qwen3.py
# Use Case 5 - Route A: Zero-shot Anomaly Explanation
# Model: Qwen3-4B (causal LM, no training)
#
# Main differences from the flan-t5 version:
#   1. AutoModelForCausalLM  replaces  AutoModelForSeq2SeqLM
#   2. apply_chat_template   wraps the prompt (required for chat models)
#   3. During inference, cut off input tokens and decode only newly generated parts
# ============================================================

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset5_anomaly.csv"
OUTPUT_DIR     = "/home/xzh5180/Research/llm-evprediction/outputs/usecase5_route_a"
MODEL_NAME     = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.2   # prevent repetition loops (an issue observed with flan-t5)

# ============================================================
import os, torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer as rouge_lib
import bert_score

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])

normal_df  = df[df["is_anomaly"] == 0].copy()
anomaly_df = df[df["is_anomaly"] == 1].copy()
print(f"Total: {len(df)} rows | Normal: {len(normal_df)} | Anomaly: {len(anomaly_df)}")

# ============================================================
# 2. Compute expected_kwh
#    mean of all normal rows at the same hour
# ============================================================
hour_baseline = normal_df.groupby("hour")["demand_kwh"].mean().to_dict()

anomaly_df["expected_kwh"]  = anomaly_df["hour"].map(hour_baseline)
anomaly_df["deviation_kwh"] = anomaly_df["demand_kwh"] - anomaly_df["expected_kwh"]
anomaly_df["deviation_pct"] = (
    anomaly_df["deviation_kwh"] / anomaly_df["expected_kwh"] * 100
).round(1)

# ============================================================
# 3. Build prompts (same as flan-t5 version; answer not revealed)
# ============================================================
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

# Print one example to verify prompt content
print("\n" + "="*60)
print("EXAMPLE PROMPT (first anomaly row):")
print("="*60)
print(anomaly_df["prompt"].iloc[0])

# ============================================================
# 4. Load Qwen3-4B
#
# Key difference from Flan-T5:
#   - AutoModelForCausalLM (autoregressive generation, suitable for open-ended text)
#   - AutoModelForSeq2SeqLM is encoder-decoder, better suited for structured QA
# ============================================================
print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16
)
model = model.to(device)
model.eval()
print("Model loaded.")

# ============================================================
# 5. Inference: process row by row
#
# Core difference between Causal LM and Seq2Seq inference:
#   - Seq2Seq: input -> encoder -> decoder generates (input and output are separate)
#   - Causal LM: input and output are concatenated; generate() returns
#     [input tokens + new tokens], so input portion must be sliced off
#
# Qwen3 is a chat model; wrapping the prompt with apply_chat_template
# yields better results (the model is trained on a conversational format)
# ============================================================
predictions = []
prompts = anomaly_df["prompt"].tolist()

print(f"\nRunning inference on {len(prompts)} anomaly samples ...")

for i, prompt in enumerate(prompts):

    # Wrap with chat template (enable_thinking=False disables reasoning mode)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                   # greedy decoding for reproducibility
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated tokens (slice off input portion)
    new_tokens = output_ids[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    predictions.append(decoded)

    if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
        print(f"  {i+1}/{len(prompts)} done")

anomaly_df["prediction"] = predictions

# ============================================================
# 6. Evaluate: ROUGE-L + BERTScore
# ============================================================
references = anomaly_df["llm_explanation"].tolist()

# ROUGE-L
scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = [scorer.score(ref, pred)["rougeL"].fmeasure
           for ref, pred in zip(references, predictions)]
anomaly_df["rouge_l"] = rouge_l

# BERTScore
print("\nComputing BERTScore ...")
P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=False)
anomaly_df["bertscore_f1"] = F1.numpy()

# ============================================================
# 7. Print results
# ============================================================
print("\n" + "="*60)
print("OVERALL EVALUATION")
print("="*60)
print(f"  ROUGE-L mean      : {np.mean(rouge_l):.4f}")
print(f"  BERTScore F1 mean : {F1.mean().item():.4f}")

print("\n" + "="*60)
print("ONE EXAMPLE PER ANOMALY TYPE")
print("="*60)
for atype in anomaly_df["anomaly_type"].unique():
    row = anomaly_df[anomaly_df["anomaly_type"] == atype].iloc[0]
    print(f"\n[{atype}]")
    print(f"  Ground truth  : {row['llm_explanation']}")
    print(f"  LLM output    : {row['prediction']}")
    print(f"  ROUGE-L       : {row['rouge_l']:.4f}")
    print(f"  BERTScore F1  : {row['bertscore_f1']:.4f}")

# ============================================================
# 8. Save
# ============================================================
save_cols = [
    "timestamp","anomaly_type","anomaly_magnitude",
    "expected_kwh","demand_kwh","deviation_pct",
    "llm_explanation","prediction","rouge_l","bertscore_f1"
]
out_path = os.path.join(OUTPUT_DIR, "route_a_qwen3_results.csv")
anomaly_df[save_cols].to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")
