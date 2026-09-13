"""
Use Case 7c - Step 3: SFT Grid Agent (LoRA fine-tuning)
========================================================
Core concept:
  Fine-tune the Grid Agent using the ground-truth Grid responses from the dataset,
  so its negotiation language in peak/V2G scenarios becomes more professional and specific.

Training data construction:
  120 sessions × 2 Grid turns = 240 samples
  Input:  scenario context + full conversation history before each Grid turn
  Output: ground-truth Grid response from the dataset

Post-training comparison:
  For peak_hour and V2G scenarios,
  run the same conversation history through both zero-shot and fine-tuned Grid,
  then observe how the wording differs.

Output directory:
  outputs/usecase7_sft/
    grid_agent_lora/          <- LoRA checkpoint
    sft_train_data.jsonl
    sft_eval_data.jsonl
    inference_comparison.json
    inference_comparison.txt  <- human-readable comparison report

Run: python sourcecode/usecase7_step3_sft_grid.py
"""

import json
import os
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME      = "Qwen/Qwen3-4B"
DATA_PATH       = "/home/xzh5180/Research/llm-evprediction/datasets/dataset7_multiagent.csv"
OUTPUT_DIR      = "/home/xzh5180/Research/llm-evprediction/outputs/usecase7_sft"
LORA_DIR        = os.path.join(OUTPUT_DIR, "grid_agent_lora")
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "sft_train_data.jsonl")
EVAL_DATA_PATH  = os.path.join(OUTPUT_DIR, "sft_eval_data.jsonl")
COMPARE_JSON    = os.path.join(OUTPUT_DIR, "inference_comparison.json")
COMPARE_TXT     = os.path.join(OUTPUT_DIR, "inference_comparison.txt")

# Training hyperparameters
TRAIN_SESSIONS      = 96    # first 96 sessions for training (192 samples)
# last 24 sessions for validation (48 samples)
NUM_EPOCHS          = 3
PER_DEVICE_BATCH    = 2
GRAD_ACCUM_STEPS    = 4     # effective batch size = 8
LEARNING_RATE       = 2e-4
WARMUP_RATIO        = 0.1
MAX_SEQ_LENGTH      = 512

# LoRA hyperparameters
LORA_R              = 8
LORA_ALPHA          = 16
LORA_TARGET_MODULES = "all-linear"
LORA_DROPOUT        = 0.05

# Sessions used for inference comparison (one peak_hour and one V2G)
COMPARE_SESSION_IDS = [1, 3]  # session 1=peak_hour, session 3=emergency_v2g
COMPARE_GRID_TURN   = 2       # compare Grid's first speaking turn (transcript index=2)
MAX_NEW_TOKENS_INFER = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# GRID AGENT SYSTEM PROMPT (consistent with Step 1b)
# This is the system prompt used during fine-tuning; must be identical at train and inference time
# =============================================================================
GRID_SYSTEM_PROMPT = """You are the regional grid operator responsible for power grid stability.
Persona: You manage demand response programs and use financial incentives to shift load.
Goal: Reduce peak demand during the 4–7 PM critical window by at least 30%.
Constraint: You cannot mandate curtailment — incentives only, no forced load reduction.
  Your rebate budget is limited; escalate only if the initial offer is refused.
Strategy: Lead with a specific dollar rebate tied to a clear delay condition.
  Acknowledge user urgency, but emphasize the grid emergency and the compensation value.
Keep your response to 2-3 sentences. Propose or respond to concrete actions."""

# Negotiation turn order (consistent with Step 1)
TURN_ORDER = ["EV_User", "Station", "Grid", "EV_User", "Station", "Grid"]

# =============================================================================
# STEP 1: Build SFT training data
# Extract the two Grid turns from each transcript and construct one training sample per turn
# =============================================================================
def build_user_content(scenario_desc: str, history: list, agent_role: str) -> str:
    """Build the user message: scenario + conversation history before the Grid turn."""
    if history:
        history_text = "\n".join(
            f"[{t['agent']}]: {t['message']}" for t in history
        )
        return (
            f"Scenario: {scenario_desc}\n\n"
            f"Full conversation so far:\n{history_text}\n\n"
            f"It is your turn as {agent_role}. Respond to the conversation above."
        )
    else:
        return (
            f"Scenario: {scenario_desc}\n\n"
            f"No messages yet. You ({agent_role}) speak first."
        )


def prepare_sft_data(df: pd.DataFrame, tokenizer) -> tuple[list, list]:
    """
    Extract Grid speaking turns from 120 transcripts and build training samples.

    Each sample messages format:
      system:    GRID_SYSTEM_PROMPT
      user:      scenario + conversation history before the turn
      assistant: Grid ground-truth response

    Uses tokenizer.apply_chat_template to convert messages to text;
    enable_thinking=False ensures no <think> tokens are produced during training.

    Returns: (train_examples, eval_examples)
    """
    all_examples = []

    for _, row in df.iterrows():
        transcript = json.loads(row["negotiation_transcript"])

        # Grid turn indices in TURN_ORDER: 2 and 5
        for grid_idx in [2, 5]:
            history_before  = transcript[:grid_idx]   # conversation history before the Grid turn
            grid_response   = transcript[grid_idx]["message"]

            user_content = build_user_content(
                scenario_desc=row["scenario_description"],
                history=history_before,
                agent_role="Grid",
            )

            messages = [
                {"role": "system",    "content": GRID_SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": grid_response},
            ]

            # Convert messages to text via apply_chat_template
            # add_generation_prompt=False: do not append a generation prompt during training; the assistant response is included directly
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            all_examples.append({
                "text":       text,
                "session_id": int(row["session_id"]),
                "scenario":   row["scenario"],
                "grid_turn":  grid_idx,
            })

    # Split by session_id for train/eval (not by sample, to keep both turns of a session together)
    train_examples = [e for e in all_examples if e["session_id"] <= TRAIN_SESSIONS]
    eval_examples  = [e for e in all_examples if e["session_id"] >  TRAIN_SESSIONS]

    print(f"[Data] Total samples: {len(all_examples)}")
    print(f"[Data] Train: {len(train_examples)} | Eval: {len(eval_examples)}")
    print(f"[Data] Train sessions: 1–{TRAIN_SESSIONS} | Eval sessions: {TRAIN_SESSIONS+1}–{df['session_id'].max()}")

    # Save to file for inspection
    with open(TRAIN_DATA_PATH, "w") as f:
        for e in train_examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(EVAL_DATA_PATH, "w") as f:
        for e in eval_examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[Saved] {TRAIN_DATA_PATH}")
    print(f"[Saved] {EVAL_DATA_PATH}")

    return train_examples, eval_examples


# =============================================================================
# STEP 2: LoRA Fine-tuning
# =============================================================================
def train_grid_agent(train_examples: list, eval_examples: list, tokenizer, model):
    """
    Fine-tune the Grid Agent using SFTTrainer + LoRA.

    Key design decisions:
      - 4B model trained directly in BF16 (no QLoRA needed)
      - target_modules="all-linear" covers all linear layers
      - dataset_text_field="text": use pre-processed text directly, skip internal template handling
      - processing_class=tokenizer (trl 1.4.0 convention)
    """
    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build HuggingFace Dataset
    train_dataset = Dataset.from_list([{"text": e["text"]} for e in train_examples])
    eval_dataset  = Dataset.from_list([{"text": e["text"]} for e in eval_examples])

    # SFTConfig (trl 1.4.0)
    sft_config = SFTConfig(
        output_dir=LORA_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        logging_steps=10,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\n[Training] Starting LoRA fine-tuning of Grid Agent...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"[Saved] LoRA adapter → {LORA_DIR}")

    return model


# =============================================================================
# STEP 3: Inference comparison (zero-shot vs fine-tuned)
# =============================================================================
def infer_grid_response(
    scenario_desc: str,
    history: list,
    system_prompt: str,
    tokenizer,
    model,
) -> str:
    """Generate a Grid response given the current history (used for inference comparison)."""
    user_content = build_user_content(scenario_desc, history, "Grid")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
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
            max_new_tokens=MAX_NEW_TOKENS_INFER,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference_comparison(df: pd.DataFrame, tokenizer, base_model, finetuned_model):
    """
    Compare zero-shot vs fine-tuned Grid responses for two scenarios.
    The input (conversation history) is fixed; only the model changes.

    Comparison scenarios:
      session 1 (peak_hour_conflict)     -> Grid's 1st response (transcript index=2)
      session 3 (emergency_v2g_request)  -> Grid's 1st response (transcript index=2)
    """
    results = []
    report_lines = ["=" * 70, "Zero-shot vs Fine-tuned Grid Agent Comparison Report", "=" * 70]

    for sid in COMPARE_SESSION_IDS:
        row        = df[df["session_id"] == sid].iloc[0]
        transcript = json.loads(row["negotiation_transcript"])

        # Take the conversation history before Grid's first turn (indices 0,1 = EV_User, Station)
        history_before = transcript[:COMPARE_GRID_TURN]
        gt_response    = transcript[COMPARE_GRID_TURN]["message"]

        print(f"\n[Comparing] Session {sid} | {row['scenario']}")
        print(f"  History context:")
        for t in history_before:
            print(f"    [{t['agent']}]: {t['message']}")

        # Zero-shot Grid（base model）
        zs_response = infer_grid_response(
            scenario_desc=row["scenario_description"],
            history=history_before,
            system_prompt=GRID_SYSTEM_PROMPT,
            tokenizer=tokenizer,
            model=base_model,
        )

        # Fine-tuned Grid
        ft_response = infer_grid_response(
            scenario_desc=row["scenario_description"],
            history=history_before,
            system_prompt=GRID_SYSTEM_PROMPT,
            tokenizer=tokenizer,
            model=finetuned_model,
        )

        print(f"  [GT         ]: {gt_response}")
        print(f"  [Zero-shot  ]: {zs_response}")
        print(f"  [Fine-tuned ]: {ft_response}")

        entry = {
            "session_id":       sid,
            "scenario":         row["scenario"],
            "history_context":  history_before,
            "ground_truth":     gt_response,
            "zero_shot":        zs_response,
            "fine_tuned":       ft_response,
        }
        results.append(entry)

        report_lines += [
            "",
            f"Session {sid} | {row['scenario']}",
            "-" * 50,
            "Context:",
        ]
        for t in history_before:
            report_lines.append(f"  [{t['agent']}]: {t['message']}")
        report_lines += [
            "",
            f"[Ground truth ]: {gt_response}",
            f"[Zero-shot    ]: {zs_response}",
            f"[Fine-tuned   ]: {ft_response}",
        ]

    report_lines += [
        "",
        "=" * 70,
        "Analysis prompts (judge for yourself before looking at the results):",
        "  1. Is the fine-tuned Grid's incentive language more specific (clear $/kWh numbers)?",
        "  2. Does the fine-tuned Grid more accurately distinguish peak vs V2G strategies?",
        "  3. Does the fine-tuned Grid recite training data, or has it truly generalized?",
        "  4. What is the main difference between zero-shot and fine-tuned Grid responses?",
    ]

    # Save results
    with open(COMPARE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(COMPARE_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[Saved] {COMPARE_JSON}")
    print(f"[Saved] {COMPARE_TXT}")
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    df = pd.read_csv(DATA_PATH)

    # ── Load base model (BF16; 4B model does not need QLoRA) ────────────
    print(f"\n[Loading] {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    print(f"[Loaded]  dtype=bfloat16 | device={next(base_model.parameters()).device}")

    # ── Build training data ───────────────────────────────────────────────
    train_examples, eval_examples = prepare_sft_data(df, tokenizer)

    # ── Training ──────────────────────────────────────────────────────────
    # Note: train_grid_agent internally calls get_peft_model, which modifies base_model.
    # After training, base_model becomes a PeftModel (with LoRA adapter attached).
    print("\n[Note] After training, base_model becomes a PeftModel.")
    print("[Note] For inference comparison, zero-shot uses the original base weights,")
    print("       and fine-tuned uses the merged adapter.")

    finetuned_model = train_grid_agent(train_examples, eval_examples, tokenizer, base_model)
    finetuned_model.eval()

    # ── Reload a clean base model for zero-shot comparison ───────────────
    # Reload without LoRA to ensure a fair comparison
    print(f"\n[Loading] Reloading base model for zero-shot comparison ...")
    base_model_clean = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model_clean.eval()

    # ── Inference comparison ──────────────────────────────────────────────
    run_inference_comparison(df, tokenizer, base_model_clean, finetuned_model)

    print("\n[Done] Step 3 complete.")
    print("Please review outputs/usecase7_sft/inference_comparison.txt,")
    print("and describe the differences you observe before analyzing the fine-tuning effect.")


if __name__ == "__main__":
    main()
