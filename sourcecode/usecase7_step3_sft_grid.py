"""
Use Case 7c - Step 3: SFT Grid Agent（LoRA 微调）
=================================================
核心概念：
  用数据集中 Grid 的真实回应来 fine-tune Grid Agent，
  使其在 peak/V2G 场景下的谈判措辞更专业、更具体。

训练数据构造：
  120 sessions × 2 Grid 轮次 = 240 个样本
  输入：scenario 上下文 + Grid 发言前的完整对话历史
  输出：数据集里 Grid 的 ground truth 回应

训练后对比：
  在 peak_hour 和 V2G 场景下，
  分别让 zero-shot Grid 和 fine-tuned Grid 接同一段历史各说一轮，
  观察措辞变化。

输出目录：
  outputs/usecase7_sft/
    grid_agent_lora/          ← LoRA checkpoint
    sft_train_data.jsonl
    sft_eval_data.jsonl
    inference_comparison.json
    inference_comparison.txt  ← 可读对比报告

运行：python sourcecode/usecase7_step3_sft_grid.py
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

# 训练超参数
TRAIN_SESSIONS      = 96    # 前96条 session 做训练（192个样本）
# 后24条 session 做验证（48个样本）
NUM_EPOCHS          = 3
PER_DEVICE_BATCH    = 2
GRAD_ACCUM_STEPS    = 4     # 有效 batch size = 8
LEARNING_RATE       = 2e-4
WARMUP_RATIO        = 0.1
MAX_SEQ_LENGTH      = 512

# LoRA 超参数
LORA_R              = 8
LORA_ALPHA          = 16
LORA_TARGET_MODULES = "all-linear"
LORA_DROPOUT        = 0.05

# 推理对比用的 session（peak_hour 和 V2G 各一条）
COMPARE_SESSION_IDS = [1, 3]  # session 1=peak_hour, session 3=emergency_v2g
COMPARE_GRID_TURN   = 2       # 对比 Grid 的第一轮发言（transcript index=2）
MAX_NEW_TOKENS_INFER = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# GRID AGENT SYSTEM PROMPT（和 Step 1b 保持一致）
# 这是 fine-tuning 时用的 system prompt，训练时和推理时要保持相同
# =============================================================================
GRID_SYSTEM_PROMPT = """You are the regional grid operator responsible for power grid stability.
Persona: You manage demand response programs and use financial incentives to shift load.
Goal: Reduce peak demand during the 4–7 PM critical window by at least 30%.
Constraint: You cannot mandate curtailment — incentives only, no forced load reduction.
  Your rebate budget is limited; escalate only if the initial offer is refused.
Strategy: Lead with a specific dollar rebate tied to a clear delay condition.
  Acknowledge user urgency, but emphasize the grid emergency and the compensation value.
Keep your response to 2-3 sentences. Propose or respond to concrete actions."""

# 谈判轮次顺序（和 Step 1 保持一致）
TURN_ORDER = ["EV_User", "Station", "Grid", "EV_User", "Station", "Grid"]

# =============================================================================
# STEP 1：构造 SFT 训练数据
# 从每条 transcript 中提取 Grid 的两轮发言，各自构造一个训练样本
# =============================================================================
def build_user_content(scenario_desc: str, history: list, agent_role: str) -> str:
    """构造 user message：场景 + Grid 发言前的对话历史"""
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
    从 120 条 transcript 中提取 Grid 的发言轮次，构造训练样本。

    每个样本的 messages 格式：
      system : GRID_SYSTEM_PROMPT
      user   : scenario + 发言前历史
      assistant: Grid 的 ground truth 回应

    用 tokenizer.apply_chat_template 把 messages 转成文本，
    enable_thinking=False 确保训练时不产生 <think> token。

    返回：(train_examples, eval_examples)
    """
    all_examples = []

    for _, row in df.iterrows():
        transcript = json.loads(row["negotiation_transcript"])

        # Grid 在 TURN_ORDER 中的索引：2 和 5
        for grid_idx in [2, 5]:
            history_before  = transcript[:grid_idx]   # Grid 发言前的历史
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

            # 把 messages 转成 text（apply_chat_template）
            # add_generation_prompt=False：训练时不加推理提示，直接包含 assistant 回应
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

    # 按 session_id 分 train/eval（不按样本随机分，避免同一 session 的两轮分开）
    train_examples = [e for e in all_examples if e["session_id"] <= TRAIN_SESSIONS]
    eval_examples  = [e for e in all_examples if e["session_id"] >  TRAIN_SESSIONS]

    print(f"[Data] Total samples: {len(all_examples)}")
    print(f"[Data] Train: {len(train_examples)} | Eval: {len(eval_examples)}")
    print(f"[Data] Train sessions: 1–{TRAIN_SESSIONS} | Eval sessions: {TRAIN_SESSIONS+1}–{df['session_id'].max()}")

    # 保存到文件（便于检查）
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
# STEP 2：LoRA Fine-tuning
# =============================================================================
def train_grid_agent(train_examples: list, eval_examples: list, tokenizer, model):
    """
    用 SFTTrainer + LoRA 微调 Grid Agent。

    关键设计：
      - 4B 模型用 BF16 直接训练（不需要 QLoRA）
      - target_modules="all-linear" 覆盖所有线性层
      - dataset_text_field="text"：直接用预处理好的文本，跳过内部 template 处理
      - processing_class=tokenizer（trl 1.4.0 规范）
    """
    # LoRA 配置
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

    # 构造 HuggingFace Dataset
    train_dataset = Dataset.from_list([{"text": e["text"]} for e in train_examples])
    eval_dataset  = Dataset.from_list([{"text": e["text"]} for e in eval_examples])

    # SFTConfig（trl 1.4.0）
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

    # 保存 LoRA adapter
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"[Saved] LoRA adapter → {LORA_DIR}")

    return model


# =============================================================================
# STEP 3：推理对比（zero-shot vs fine-tuned）
# =============================================================================
def infer_grid_response(
    scenario_desc: str,
    history: list,
    system_prompt: str,
    tokenizer,
    model,
) -> str:
    """让 Grid 根据当前历史生成回应（用于推理对比）"""
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
    对比 zero-shot vs fine-tuned Grid 在两个场景下的回应。
    固定输入（相同对话历史），只换模型，观察输出差异。

    对比场景：
      session 1（peak_hour_conflict） → Grid 的第1轮回应（transcript index=2）
      session 3（emergency_v2g_request）→ Grid 的第1轮回应（transcript index=2）
    """
    results = []
    report_lines = ["=" * 70, "Zero-shot vs Fine-tuned Grid Agent 对比报告", "=" * 70]

    for sid in COMPARE_SESSION_IDS:
        row        = df[df["session_id"] == sid].iloc[0]
        transcript = json.loads(row["negotiation_transcript"])

        # 取 Grid 第一轮发言前的历史（index=0,1 = EV_User, Station）
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
        "分析提示（跑完后自己先判断）：",
        "  1. Fine-tuned Grid 的激励措辞是否更具体（有明确 $/kWh 数字）？",
        "  2. Fine-tuned Grid 是否更准确地区分 peak vs V2G 场景的策略？",
        "  3. Fine-tuned Grid 的回应是否像在复述训练数据，还是真正泛化了？",
        "  4. Zero-shot Grid 和 Fine-tuned Grid 的主要差异在哪里？",
    ]

    # 保存结果
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

    # ── 加载 base model（BF16，4B 不需要 QLoRA）──────────────────────────
    print(f"\n[Loading] {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    print(f"[Loaded]  dtype=bfloat16 | device={next(base_model.parameters()).device}")

    # ── 构造训练数据 ──────────────────────────────────────────────────────
    train_examples, eval_examples = prepare_sft_data(df, tokenizer)

    # ── 训练 ──────────────────────────────────────────────────────────────
    # 注意：train_grid_agent 内部调用 get_peft_model，会修改 base_model
    # 训练完成后 base_model 已经是 peft model（带了 LoRA adapter）
    print("\n[Note] 训练后 base_model 会变成 PeftModel。")
    print("[Note] 推理对比时，zero-shot 用 merge 前的 base weights，")
    print("       fine-tuned 用 merge 后的 adapter。")

    finetuned_model = train_grid_agent(train_examples, eval_examples, tokenizer, base_model)
    finetuned_model.eval()

    # ── 加载纯净的 base model 用于 zero-shot 对比 ────────────────────────
    # 重新加载一个没有 LoRA 的 base model，保证对比公平
    print(f"\n[Loading] Reloading base model for zero-shot comparison ...")
    base_model_clean = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model_clean.eval()

    # ── 推理对比 ──────────────────────────────────────────────────────────
    run_inference_comparison(df, tokenizer, base_model_clean, finetuned_model)

    print("\n[Done] Step 3 complete。")
    print("请对比 outputs/usecase7_sft/inference_comparison.txt 里的输出，")
    print("先说说你观察到的差异，我们再一起分析 fine-tuning 的效果。")


if __name__ == "__main__":
    main()
