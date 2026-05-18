"""
Use Case 6 - Stage 3: RAG (Retrieval-Augmented Generation)
Model: Qwen3-4B (BF16, no fine-tuning)
Retriever: sentence-transformers (all-MiniLM-L6-v2)
Task: For each test case, retrieve K most similar historical decisions
      from the training set, inject them into the prompt as examples,
      then generate a recommendation.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer as rouge_scorer_lib
from tqdm import tqdm
from datetime import datetime

# ============================================================
# Config
# ============================================================
MODEL_NAME        = "Qwen/Qwen3-4B"
EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH         = "/home/xzh5180/Research/llm-evprediction/datasets/dataset6_decision_support.csv"
OUTPUT_DIR        = "/home/xzh5180/Research/llm-evprediction/outputs/usecase6_stage3_rag"
TRAIN_SIZE        = 292
TOP_K             = 3           # number of retrieved examples to inject
MAX_NEW_TOKENS    = 128
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load & Split Data
# ============================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df.iloc[:TRAIN_SIZE].copy().reset_index(drop=True)
test_df  = df.iloc[TRAIN_SIZE:].copy().reset_index(drop=True)

print(f"Train (knowledge base): {len(train_df)} rows")
print(f"Test                  : {len(test_df)} rows")

# ============================================================
# Build Knowledge Base with Embeddings
# Embed each training sample's decision_prompt
# ============================================================
print(f"\nLoading embedding model: {EMBED_MODEL_NAME} ...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Encoding training prompts into vector knowledge base...")
train_texts = train_df["decision_prompt"].tolist()
train_embeddings = embed_model.encode(
    train_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True    # L2-normalize → cosine sim = dot product
)
# train_embeddings: shape (292, 384)
print(f"Knowledge base ready. Shape: {train_embeddings.shape}")

# ============================================================
# Retrieval Function
# For a given test prompt, find TOP_K most similar training cases
# Returns list of (date, prompt_snippet, reference_action) tuples
# ============================================================
def retrieve_similar_cases(query_prompt: str, top_k: int = TOP_K):
    query_emb = embed_model.encode(
        [query_prompt],
        normalize_embeddings=True
    )  # shape (1, 384)

    # Cosine similarity: since both are L2-normalized, sim = dot product
    sims = np.dot(train_embeddings, query_emb.T).flatten()  # (292,)

    top_indices = np.argsort(sims)[::-1][:top_k]

    cases = []
    for idx in top_indices:
        row = train_df.iloc[idx]
        cases.append({
            "date"       : str(row["date"].date()),
            "similarity" : float(sims[idx]),
            "action"     : row["reference_action"],
            # Use a compact summary of the operational state for the context block
            "context"    : (
                f"Date: {row['date'].date()} | "
                f"Total demand: {row['total_demand']:.0f} kWh | "
                f"Peak: {row['peak_demand']:.1f} kWh | "
                f"Price: ${row['electricity_price_per_kwh']:.3f}/kWh | "
                f"Grid: {row['grid_stress_level']} | "
                f"Queue: {row['ev_queue_length']} | "
                f"Event: {row['event_type']}"
            )
        })
    return cases

# ============================================================
# Build RAG-Augmented Prompt
# Injects retrieved examples before the actual question
# ============================================================
def build_rag_prompt(current_prompt: str, retrieved_cases: list) -> str:
    examples_block = ""
    for i, case in enumerate(retrieved_cases, 1):
        examples_block += (
            f"Example {i} (similarity: {case['similarity']:.3f}):\n"
            f"  Situation : {case['context']}\n"
            f"  Decision  : {case['action']}\n\n"
        )

    rag_prompt = (
        f"You are an EV charging station operations assistant.\n\n"
        f"Here are {len(retrieved_cases)} similar historical operational scenarios "
        f"and the decisions that were made:\n\n"
        f"{examples_block}"
        f"Now, based on the current operational status below and the patterns "
        f"from the historical examples, provide your operational recommendation.\n\n"
        f"--- CURRENT STATUS ---\n"
        f"{current_prompt}"
    )
    return rag_prompt

# ============================================================
# Load LLM
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

# ============================================================
# Run RAG Inference on Test Set
# ============================================================
print(f"\nRunning RAG inference on {len(test_df)} test samples (TOP_K={TOP_K})...")

predictions    = []
references     = []
dates          = []
retrieved_log  = []      # save retrieval details for analysis

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # Step 1: Retrieve similar historical cases
    cases = retrieve_similar_cases(row["decision_prompt"], top_k=TOP_K)

    # Step 2: Build augmented prompt
    rag_prompt = build_rag_prompt(row["decision_prompt"], cases)

    # Step 3: Generate
    pred = generate_decision(rag_prompt)

    predictions.append(pred)
    references.append(row["reference_action"])
    dates.append(str(row["date"].date()))
    retrieved_log.append({
        "date"      : str(row["date"].date()),
        "retrieved" : [{"date": c["date"], "sim": round(c["similarity"], 4),
                        "action": c["action"]} for c in cases]
    })

# ============================================================
# Evaluation: ROUGE
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
print(f"  RAG Evaluation Results (n={len(test_df)}, K={TOP_K})")
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
results_csv = os.path.join(OUTPUT_DIR, "stage3_predictions.csv")
results_df.to_csv(results_csv, index=False)
print(f"\nPer-sample results saved to: {results_csv}")

# Save retrieval log (useful for qualitative analysis)
retrieval_log_path = os.path.join(OUTPUT_DIR, "stage3_retrieval_log.json")
with open(retrieval_log_path, "w") as f:
    json.dump(retrieved_log, f, indent=2)
print(f"Retrieval log saved to: {retrieval_log_path}")

summary = {
    "stage"     : "3_rag",
    "model"     : MODEL_NAME,
    "retriever" : EMBED_MODEL_NAME,
    "top_k"     : TOP_K,
    "test_size" : len(test_df),
    "rouge1"    : round(avg_rouge1, 4),
    "rouge2"    : round(avg_rouge2, 4),
    "rougeL"    : round(avg_rougeL, 4),
    "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
summary_path = os.path.join(OUTPUT_DIR, "stage3_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_path}")

# Sample comparisons with retrieval context
print(f"\n{'='*50}")
print(f"  Sample Comparisons with Retrieved Context (first 3)")
print(f"{'='*50}")
for i in range(min(3, len(test_df))):
    print(f"\n[{dates[i]}]")
    print(f"  Reference : {references[i]}")
    print(f"  Predicted : {predictions[i]}")
    print(f"  ROUGE-L   : {rougeL_list[i]:.4f}")
    print(f"  Retrieved cases:")
    for c in retrieved_log[i]["retrieved"]:
        print(f"    {c['date']} (sim={c['sim']:.3f}) → {c['action']}")

print("\nStage 3 complete.")
