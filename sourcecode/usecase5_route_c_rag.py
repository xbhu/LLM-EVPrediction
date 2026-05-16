# ============================================================
# usecase5_route_c_rag.py
# Use Case 5 - Route C: RAG-based Anomaly Explanation
#
# 流程：
#   1. 用 sentence-transformers 把每个异常事件向量化
#   2. 用 FAISS 做向量检索，找最相似的历史案例
#   3. 把检索到的案例注入 prompt
#   4. 用 Qwen3-4B 结合历史案例生成解释
#
# 评估方式：Leave-one-out
#   对每个异常，从其余 67 个里检索，避免检索到自身
# ============================================================

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH   = "/home/xzh5180/Research/llm-evprediction/datasets/dataset5_anomaly.csv"
OUTPUT_DIR     = "/home/xzh5180/Research/llm-evprediction/outputs/usecase5_route_c"
EMBED_MODEL    = "all-MiniLM-L6-v2"   # sentence-transformers 模型，轻量快速
GEN_MODEL      = "Qwen/Qwen3-4B"
TOP_K          = 3                     # 检索最相似的 3 个历史案例
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.2

# ============================================================
import os, torch, numpy as np, pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
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

# ============================================================
# 2. 为每个异常建立"描述字符串"用于 embedding
#
# 不直接用 prompt（太长），而是提炼关键特征做成简短描述。
# 语义相近的事件，embedding 向量在空间里距离就近。
# ============================================================
def build_description(row):
    direction = "above" if row["deviation_kwh"] > 0 else "below"
    return (
        f"Hour {int(row['hour'])}, "
        f"{DAY_NAMES[int(row['day_of_week'])]}, "
        f"month {int(row['month'])}, "
        f"{'weekend' if row['is_weekend'] else 'weekday'}, "
        f"event: {row['event_type']}, "
        f"temperature {row['temperature_f']}F, "
        f"demand {abs(row['deviation_pct'])}% {direction} expected"
    )

anomaly_df["description"] = anomaly_df.apply(build_description, axis=1)

# ============================================================
# 3. Embedding
#
# SentenceTransformer 把每段描述文字转成固定维度的向量。
# normalize_embeddings=True：归一化到单位长度，
# 这样内积（inner product）就等于余弦相似度。
# ============================================================
print(f"\nLoading embedding model: {EMBED_MODEL} ...")
embed_model = SentenceTransformer(EMBED_MODEL)

descriptions = anomaly_df["description"].tolist()
embeddings   = embed_model.encode(
    descriptions,
    normalize_embeddings=True,   # 归一化，使内积 = 余弦相似度
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)

print(f"Embedding shape: {embeddings.shape}")  # (68, 384)

# ============================================================
# 4. 建立 FAISS 索引
#
# IndexFlatIP = 用内积（Inner Product）做相似度。
# 因为向量已经归一化，内积值越大 = 余弦相似度越高 = 越相似。
# ============================================================
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

# ============================================================
# 5. 查看检索效果（打印第一个异常的 top-3 邻居）
# ============================================================
print("\n" + "="*60)
print("RETRIEVAL EXAMPLE (first anomaly row):")
print("="*60)
query_vec = embeddings[0:1]
D, I = index.search(query_vec, TOP_K + 1)   # +1 因为第一名是自身

print(f"Query: {descriptions[0]}")
print(f"True label: {anomaly_df['anomaly_type'].iloc[0]}")
print("\nTop retrieved neighbors (excluding self):")
for rank, (score, idx) in enumerate(zip(D[0], I[0])):
    if idx == 0:   # 跳过自身
        continue
    print(f"  #{rank} score={score:.4f} | {descriptions[idx]}")
    print(f"         type={anomaly_df['anomaly_type'].iloc[idx]} | "
          f"explanation={anomaly_df['llm_explanation'].iloc[idx]}")

# ============================================================
# 6. 构造 RAG prompt
#
# 两部分组成：
#   - 当前异常的基本信息（和 Route A 相同）
#   - 检索到的历史案例（新增部分）
# ============================================================
def build_rag_prompt(row, retrieved_rows):
    direction = "above" if row["deviation_kwh"] > 0 else "below"

    # 当前异常描述
    current = (
        "You are an EV charging demand analyst. "
        "A significant deviation has been detected at a charging station.\n\n"
        f"Timestamp       : {row['timestamp']}\n"
        f"Hour of day     : {int(row['hour'])}:00\n"
        f"Day of week     : {DAY_NAMES[int(row['day_of_week'])]}\n"
        f"Month           : {int(row['month'])}\n"
        f"Weekend         : {'Yes' if row['is_weekend'] else 'No'}\n"
        f"Scheduled event : {row['event_type']}\n"
        f"Temperature     : {row['temperature_f']}°F\n\n"
        f"Expected demand : {row['expected_kwh']:.1f} kWh\n"
        f"Actual demand   : {row['demand_kwh']:.1f} kWh\n"
        f"Deviation       : {abs(row['deviation_kwh']):.1f} kWh "
        f"{direction} expected ({abs(row['deviation_pct'])}%)\n"
    )

    # 检索到的历史案例
    history = "\nSimilar historical anomaly cases for reference:\n"
    for i, r in enumerate(retrieved_rows, 1):
        r_dir = "above" if r["deviation_kwh"] > 0 else "below"
        history += (
            f"  Case {i}: "
            f"Hour {int(r['hour'])}, {DAY_NAMES[int(r['day_of_week'])]}, "
            f"month {int(r['month'])}, event={r['event_type']}, "
            f"demand {abs(r['deviation_pct']):.1f}% {r_dir} expected "
            f"→ Explanation: {r['llm_explanation']}\n"
        )

    question = (
        "\nBased on the current anomaly and the historical cases above, "
        "what is the most likely cause? "
        "Give a brief one-sentence explanation."
    )

    return current + history + question


# ============================================================
# 7. Load Qwen3-4B（与 Route A 完全相同）
# ============================================================
print(f"\nLoading generation model: {GEN_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.bfloat16)
gen_model = gen_model.to(device)
gen_model.eval()
print("Generation model loaded.")

# ============================================================
# 8. Leave-one-out 推理
#
# 对每个异常 i，从其余所有样本里检索 top-k，
# 避免检索到自身（自身相似度最高，检索到自身就等于作弊）。
# 实现方式：检索 top-(k+1)，跳过结果中 index == i 的那条。
# ============================================================
print(f"\nRunning RAG inference on {len(anomaly_df)} anomaly samples ...")
predictions = []

for i in range(len(anomaly_df)):
    query_vec = embeddings[i:i+1]

    # 检索 top-(k+1)，第一名通常是自身
    D, I = index.search(query_vec, TOP_K + 1)

    # 过滤掉自身，取前 TOP_K 个
    neighbors = [idx for idx in I[0] if idx != i][:TOP_K]
    retrieved_rows = [anomaly_df.iloc[idx].to_dict() for idx in neighbors]

    # 构造 RAG prompt
    row    = anomaly_df.iloc[i]
    prompt = build_rag_prompt(row, retrieved_rows)

    # Qwen3 chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs    = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    predictions.append(decoded)

    if (i + 1) % 10 == 0 or (i + 1) == len(anomaly_df):
        print(f"  {i+1}/{len(anomaly_df)} done")

anomaly_df["prediction"] = predictions

# ============================================================
# 9. Evaluate: ROUGE-L + BERTScore
# ============================================================
references = anomaly_df["llm_explanation"].tolist()

scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = [scorer.score(ref, pred)["rougeL"].fmeasure
           for ref, pred in zip(references, predictions)]
anomaly_df["rouge_l"] = rouge_l

print("\nComputing BERTScore ...")
P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=False)
anomaly_df["bertscore_f1"] = F1.numpy()

# ============================================================
# 10. Print results
# ============================================================
print("\n" + "="*60)
print("OVERALL EVALUATION (RAG, leave-one-out, all 68 samples)")
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
    print(f"  RAG output    : {row['prediction']}")
    print(f"  ROUGE-L       : {row['rouge_l']:.4f}")
    print(f"  BERTScore F1  : {row['bertscore_f1']:.4f}")

# ============================================================
# 11. Compare all three routes
# ============================================================
print("\n" + "="*60)
print("FULL COMPARISON: Route A / B / C")
print("="*60)
print(f"  Route A | Flan-T5-base  zero-shot  → ROUGE-L: 0.0085 | BERTScore: 0.7942")
print(f"  Route A | Qwen3-4B      zero-shot  → ROUGE-L: 0.0449 | BERTScore: 0.8604")
print(f"  Route C | Qwen3-4B      RAG        → ROUGE-L: {np.mean(rouge_l):.4f} | BERTScore: {F1.mean().item():.4f}")

# ============================================================
# 12. Save
# ============================================================
save_cols = [
    "timestamp","anomaly_type","anomaly_magnitude",
    "expected_kwh","demand_kwh","deviation_pct",
    "llm_explanation","prediction","rouge_l","bertscore_f1"
]
out_path = os.path.join(OUTPUT_DIR, "route_c_results.csv")
anomaly_df[save_cols].to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")
