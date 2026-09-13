"""
Use Case 7b - Step 2: LLM-as-Judge Evaluation of Negotiation Transcripts
=======================================================
Core concept:
  The LLM acts as an outside observer — not a negotiation participant.
  Reads the full transcript → outputs a structured JSON judgment → compares with ground truth.

Input:
  1. The 120 original dataset transcripts (primary evaluation target)
  2. If the Step 1 output file exists, evaluate those transcripts as well (optional)

Output:
  outputs/usecase7_judge/
    judge_results.jsonl      <- per-session results (incremental write, resume-safe)
    judge_summary.json       <- aggregated metrics
    judge_analysis.txt       <- human-readable report

Run: python sourcecode/usecase7b_step2_llm_judge.py
"""

import json
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME      = "Qwen/Qwen3-4B"
DATA_PATH       = "/home/xzh5180/Research/llm-evprediction/datasets/dataset7_multiagent.csv"
STEP1_OUTPUT    = "/home/xzh5180/Research/llm-evprediction/outputs/usecase7_zeroshot/step1_all_results.json"
OUTPUT_DIR      = "/home/xzh5180/Research/llm-evprediction/outputs/usecase7_judge"
RESULTS_FILE    = os.path.join(OUTPUT_DIR, "judge_results.jsonl")
SUMMARY_FILE    = os.path.join(OUTPUT_DIR, "judge_summary.json")
ANALYSIS_FILE   = os.path.join(OUTPUT_DIR, "judge_analysis.txt")
MAX_NEW_TOKENS  = 300   # judge needs to output JSON — allocate enough space
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# JUDGE PROMPT
# Design notes:
#   1. Clear role: observer, not a participant
#   2. Provide sufficient context: scenario + initial state of all parties + transcript
#   3. Require pure JSON output — no preamble or explanatory text
#   4. Field names and value ranges must exactly match those in the ground truth
# =============================================================================
JUDGE_SYSTEM = """You are an expert evaluator of multi-party EV charging negotiation transcripts.
Your task: read a negotiation transcript and provide an objective structured evaluation.
You are an outside observer — you were not part of the negotiation.

Output ONLY a JSON object with exactly these three fields. No preamble, no explanation, no markdown.
{
  "predicted_outcome": "full_agreement" or "partial_compromise",
  "estimated_satisfaction": <float between 0.0 and 1.0>,
  "reasoning": "<2-3 sentences explaining your judgment>"
}

Evaluation guidelines:
- full_agreement: all parties reached a concrete, executable plan with no unresolved conflicts
- partial_compromise: one or more parties made significant concessions but key needs are unmet,
  OR the agreement is vague / conditional with unresolved details
- estimated_satisfaction: reflects the EV user's likely satisfaction (0=very dissatisfied, 1=fully satisfied)
  Consider: did the user get what they needed? Did they have to make painful concessions?"""

def build_judge_user_prompt(row: pd.Series) -> str:
    """Build the judge's user message: scenario context + formatted transcript."""
    transcript = json.loads(row["negotiation_transcript"])
    transcript_text = "\n".join(
        f"  [{t['agent']}]: {t['message']}" for t in transcript
    )
    return f"""Scenario type: {row['scenario']}
Scenario description: {row['scenario_description']}
User need: {row['user_need']}
Station status: {row['station_status']}
Grid signal: {row['grid_signal']}

--- Negotiation Transcript ---
{transcript_text}
--- End of Transcript ---

Now provide your evaluation as a JSON object."""

def build_judge_user_prompt_from_transcript(
    scenario: str,
    scenario_desc: str,
    user_need: str,
    station_status: str,
    grid_signal: str,
    transcript: list,
) -> str:
    """Build a judge prompt from a Step 1 generated transcript (fields passed individually)."""
    transcript_text = "\n".join(
        f"  [{t['agent']}]: {t['message']}" for t in transcript
    )
    return f"""Scenario type: {scenario}
Scenario description: {scenario_desc}
User need: {user_need}
Station status: {station_status}
Grid signal: {grid_signal}

--- Negotiation Transcript ---
{transcript_text}
--- End of Transcript ---

Now provide your evaluation as a JSON object."""

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(model_name: str):
    print(f"\n[Loading] {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[Loaded]  dtype=bfloat16 | device={next(model.parameters()).device}")
    return tokenizer, model

# =============================================================================
# JUDGE INFERENCE
# =============================================================================
def run_judge(user_prompt: str, tokenizer, model) -> str:
    """Call the judge LLM and return its raw string output."""
    messages = [
        {"role": "user", "content": f"{JUDGE_SYSTEM}\n\n{user_prompt}"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =============================================================================
# JSON PARSING (robust)
# LLMs sometimes add extra text or markdown around the JSON — handle this gracefully
# =============================================================================
def parse_judge_output(raw: str) -> dict:
    """
    Three-layer parsing strategy:
      1. Try json.loads directly (ideal case)
      2. Use regex to extract the first {...} block and parse that
      3. If both fail, return None and record a parse failure
    """
    # Layer 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Layer 2: extract {...} block
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Layer 3: failure
    return None

# =============================================================================
# VALIDATE PARSED RESULT FIELDS
# =============================================================================
def validate_parsed(parsed: dict) -> bool:
    if parsed is None:
        return False
    if "predicted_outcome" not in parsed:
        return False
    if parsed["predicted_outcome"] not in ("full_agreement", "partial_compromise"):
        return False
    if "estimated_satisfaction" not in parsed:
        return False
    try:
        s = float(parsed["estimated_satisfaction"])
        if not (0.0 <= s <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True

# =============================================================================
# LOAD COMPLETED SESSION IDs (for resume support)
# =============================================================================
def load_completed_ids(results_file: str) -> set:
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completed.add(obj["session_id"])
            except Exception:
                pass
    return completed

# =============================================================================
# PART A: Evaluate the 120 original dataset transcripts
# =============================================================================
def evaluate_dataset_transcripts(df: pd.DataFrame, tokenizer, model):
    print("\n" + "="*60)
    print("PART A: Evaluate 120 original dataset transcripts")
    print("="*60)

    completed = load_completed_ids(RESULTS_FILE)
    if completed:
        print(f"[Resume] {len(completed)} sessions already completed, skipping...")

    with open(RESULTS_FILE, "a", encoding="utf-8") as f_out:
        for idx, row in df.iterrows():
            sid = f"dataset_{row['session_id']}"
            if sid in completed:
                continue

            user_prompt = build_judge_user_prompt(row)
            raw_output  = run_judge(user_prompt, tokenizer, model)
            parsed      = parse_judge_output(raw_output)
            valid       = validate_parsed(parsed)

            record = {
                "session_id":            sid,
                "source":                "dataset",
                "scenario":              row["scenario"],
                "gt_outcome":            row["outcome"],
                "gt_satisfaction":       row["user_satisfaction_score"],
                "gt_grid_stress":        row["grid_stress_reduction_pct"],
                "raw_output":            raw_output,
                "predicted_outcome":     parsed.get("predicted_outcome")     if valid else None,
                "estimated_satisfaction":parsed.get("estimated_satisfaction") if valid else None,
                "reasoning":             parsed.get("reasoning")             if valid else None,
                "parse_success":         valid,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            status = "✓" if valid else "✗ parse fail"
            pred   = record["predicted_outcome"] or "?"
            est    = f"{record['estimated_satisfaction']:.3f}" if record["estimated_satisfaction"] else "?"
            print(
                f"  [{idx+1:3d}/120] {row['scenario']:30s} | "
                f"GT={row['outcome']:20s} | Pred={pred:20s} | "
                f"GT_sat={row['user_satisfaction_score']:.3f} | Est_sat={est} | {status}"
            )

# =============================================================================
# PART B: Evaluate Step 1 generated transcripts (if the file exists)
# =============================================================================
def evaluate_step1_transcripts(step1_path: str, df: pd.DataFrame, tokenizer, model):
    if not os.path.exists(step1_path):
        print(f"\n[Skip Part B] Step 1 output file not found: {step1_path}")
        return

    print("\n" + "="*60)
    print("PART B: Evaluate Step 1 generated transcripts")
    print("="*60)

    with open(step1_path, "r", encoding="utf-8") as f:
        step1 = json.load(f)

    session_row = df[df["session_id"] == step1["session_id"]].iloc[0]
    completed   = load_completed_ids(RESULTS_FILE)

    # Build evaluation entries: step1a, step1b, step1c (three persona types)
    to_evaluate = {}
    if "step1a" in step1:
        to_evaluate["step1a"] = step1["step1a"]["transcript"]
    if "step1b" in step1:
        to_evaluate["step1b"] = step1["step1b"]["transcript"]
    if "step1c" in step1:
        for persona_name, val in step1["step1c"].items():
            to_evaluate[f"step1c_{persona_name}"] = val["transcript"]

    with open(RESULTS_FILE, "a", encoding="utf-8") as f_out:
        for label, transcript in to_evaluate.items():
            sid = f"step1_{label}"
            if sid in completed:
                print(f"  [Skip] {sid} already evaluated")
                continue

            user_prompt = build_judge_user_prompt_from_transcript(
                scenario       = session_row["scenario"],
                scenario_desc  = session_row["scenario_description"],
                user_need      = session_row["user_need"],
                station_status = session_row["station_status"],
                grid_signal    = session_row["grid_signal"],
                transcript     = transcript,
            )
            raw_output = run_judge(user_prompt, tokenizer, model)
            parsed     = parse_judge_output(raw_output)
            valid      = validate_parsed(parsed)

            record = {
                "session_id":            sid,
                "source":                label,
                "scenario":              session_row["scenario"],
                "gt_outcome":            step1["ground_truth"]["outcome"],
                "gt_satisfaction":       step1["ground_truth"]["user_satisfaction_score"],
                "gt_grid_stress":        step1["ground_truth"]["grid_stress_reduction_pct"],
                "raw_output":            raw_output,
                "predicted_outcome":     parsed.get("predicted_outcome")     if valid else None,
                "estimated_satisfaction":parsed.get("estimated_satisfaction") if valid else None,
                "reasoning":             parsed.get("reasoning")             if valid else None,
                "parse_success":         valid,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            status = "✓" if valid else "✗ parse fail"
            print(
                f"  [{sid:30s}] GT={record['gt_outcome']:20s} | "
                f"Pred={record['predicted_outcome'] or '?':20s} | {status}"
            )

# =============================================================================
# SUMMARY ANALYSIS
# =============================================================================
def compute_summary():
    print("\n" + "="*60)
    print("Summary Analysis")
    print("="*60)

    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df_r = pd.DataFrame(records)

    # Only analyze the original dataset portion (which has full ground truth)
    ds = df_r[df_r["source"] == "dataset"].copy()
    ds = ds[ds["parse_success"] == True]

    total     = len(df_r[df_r["source"] == "dataset"])
    parsed_ok = len(ds)
    parse_rate = parsed_ok / total if total > 0 else 0

    # Classification accuracy
    correct  = (ds["predicted_outcome"] == ds["gt_outcome"]).sum()
    accuracy = correct / len(ds) if len(ds) > 0 else 0

    # Satisfaction MAE
    ds["sat_error"] = (
        ds["estimated_satisfaction"].astype(float) - ds["gt_satisfaction"].astype(float)
    ).abs()
    sat_mae = ds["sat_error"].mean()

    # Break down by scenario
    scenario_stats = ds.groupby("scenario").apply(
        lambda g: pd.Series({
            "n":              len(g),
            "accuracy":       (g["predicted_outcome"] == g["gt_outcome"]).mean(),
            "sat_mae":        (g["estimated_satisfaction"].astype(float) -
                               g["gt_satisfaction"].astype(float)).abs().mean(),
            "gt_outcomes":    g["gt_outcome"].value_counts().to_dict(),
            "pred_outcomes":  g["predicted_outcome"].value_counts().to_dict(),
        })
    ).to_dict(orient="index")

    # Confusion matrix (text version)
    from collections import Counter
    cm = Counter(zip(ds["gt_outcome"], ds["predicted_outcome"]))

    summary = {
        "total_dataset_sessions": total,
        "parse_success_count":    parsed_ok,
        "parse_rate":             round(parse_rate, 4),
        "outcome_accuracy":       round(accuracy, 4),
        "satisfaction_mae":       round(float(sat_mae), 4),
        "confusion_matrix":       {str(k): v for k, v in cm.items()},
        "by_scenario":            scenario_stats,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable report
    lines = [
        "=" * 60,
        "Use Case 7b Step 2 - LLM-as-Judge Evaluation Report",
        "=" * 60,
        f"Total dataset sessions   : {total}",
        f"JSON parse success rate  : {parsed_ok}/{total} ({parse_rate:.1%})",
        f"Outcome classification   : {accuracy:.1%}  ({correct}/{parsed_ok})",
        f"Satisfaction est. MAE    : {sat_mae:.4f}",
        "",
        "Confusion Matrix (GT → Predicted):",
    ]
    for (gt, pred), cnt in sorted(cm.items()):
        lines.append(f"  GT={gt:22s} → Pred={pred:22s} : {cnt}")

    lines += ["", "By scenario:"]
    for sc, stats in scenario_stats.items():
        lines.append(
            f"  {sc:35s} | acc={stats['accuracy']:.1%} | sat_mae={stats['sat_mae']:.4f}"
        )

    # Part B results (if any)
    step1_rows = df_r[df_r["source"] != "dataset"]
    if len(step1_rows) > 0:
        lines += ["", "Judge verdicts for Step 1 generated transcripts:"]
        for _, r in step1_rows.iterrows():
            lines.append(
                f"  [{r['source']:25s}] Pred={r.get('predicted_outcome','?'):22s} "
                f"| Est_sat={r.get('estimated_satisfaction','?')} "
                f"| GT={r['gt_outcome']}"
            )

    report = "\n".join(lines)
    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    return summary

# =============================================================================
# MAIN
# =============================================================================
def main():
    df = pd.read_csv(DATA_PATH)
    tokenizer, model = load_model(MODEL_NAME)

    # Part A: 120 original dataset sessions
    evaluate_dataset_transcripts(df, tokenizer, model)

    # Part B: Step 1 generated transcripts (only if file exists)
    evaluate_step1_transcripts(STEP1_OUTPUT, df, tokenizer, model)

    # Summary
    compute_summary()
    print(f"\n[Saved] {SUMMARY_FILE}")
    print(f"[Saved] {ANALYSIS_FILE}")
    print("[Done]  Step 2 complete. Analyze the results, then proceed to Step 3.")


if __name__ == "__main__":
    main()
