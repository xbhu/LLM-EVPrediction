"""
Use Case 3 — Model 3: Llama-3.2-3B-Instruct
=============================================
Architecture : Decoder-only (CausalLM)
Training     : LoRA fine-tune, BF16
Developer    : Meta
Released     : September 2024

Same architecture as Phi-4-mini (decoder-only + LoRA).
Key difference: Llama uses a different chat template and tokenizer.
This script is structurally identical to usecase3_phi4mini.py —
comparing results isolates the effect of model family / pretraining.

NOTE: Requires HuggingFace login + accepting Meta license at
      https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

Run:
    conda activate ev_llm
    huggingface-cli login   # only needed once
    python usecase3_llama32.py
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset3_qa_pairs.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase3_llama32"
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 8
BATCH_SIZE  = 2
GRAD_ACCUM  = 4          # effective batch = 8
LR          = 2e-4
MAX_LEN     = 320
MAX_OUT_LEN = 80
N_ZEROSHOT  = 10
SEED        = 42

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = (
    "You are an EV charging demand forecasting assistant. "
    "Respond with exactly: 'Predicted demand: [NUMBER] kWh. [ONE sentence explanation.]'"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_NAME}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── 1. Data ───────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
df["event_type"] = df["event_type"].fillna("none").str.strip()
print(f"\n[Data] {len(df)} samples")
print(f"[Data] Event types: {df['event_type'].value_counts().to_dict()}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)}")


# ── 2. Tokenizer & Model ──────────────────────────────────────────────────────

print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
model = model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters (total) : {n_params:.0f}M")
if DEVICE == "cuda":
    print(f"VRAM after load    : {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ── 3. LoRA ───────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    task_type     = TaskType.CAUSAL_LM,
    r             = LORA_R,
    lora_alpha    = LORA_ALPHA,
    lora_dropout  = LORA_DROPOUT,
    target_modules= "all-linear",
    bias          = "none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── 4. Parsing ────────────────────────────────────────────────────────────────

_PATTERNS = [
    r"[Pp]redicted\s+demand\s*[:\-]?\s*([\d,]+\.?\d*)\s*kWh",
    r"[Ff]orecast\s*[:\-]?\s*([\d,]+\.?\d*)\s*kWh",
    r"([\d,]+\.?\d*)\s*kWh",
    r"\b([\d]{3,5}\.?\d*)\b",
]

def parse_kwh(text):
    if not text or not isinstance(text, str):
        return None
    for pat in _PATTERNS:
        m = re.search(pat, text.strip())
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if 100 <= val <= 50_000:
                    return val
            except ValueError:
                continue
    return None


# ── 5. Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred_raw):
    y_pred  = np.array([v if v is not None else np.nan for v in y_pred_raw])
    y_true  = np.array(y_true)
    mask    = ~np.isnan(y_pred)
    n_valid = int(mask.sum())
    n_total = len(y_pred)
    if n_valid == 0:
        return dict(MAE=None, RMSE=None, MAPE=None,
                    parse_rate=0.0, n_valid=0, n_total=n_total)
    yt, yp = y_true[mask], y_pred[mask]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    nz   = yt != 0
    mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.any() else None
    return dict(
        MAE        = round(mae, 2),
        RMSE       = round(rmse, 2),
        MAPE       = round(mape, 2) if mape else None,
        parse_rate = round(n_valid / n_total * 100, 1),
        n_valid    = n_valid,
        n_total    = n_total,
    )


# ── 6. Prompt Formatting ──────────────────────────────────────────────────────

def format_prompt(user_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── 7. Zero-Shot Preview ──────────────────────────────────────────────────────

def run_zeroshot(model, tokenizer, df, n=N_ZEROSHOT):
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT PREVIEW  ({n} samples, before fine-tuning)")
    print(f"{'='*60}")
    model.eval()
    lines = []
    for i, row in df.head(n).iterrows():
        prompt_text = format_prompt(row["prompt"])
        enc = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=MAX_LEN
        ).to(DEVICE)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_OUT_LEN,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        val  = parse_kwh(text)
        gt   = row["ground_truth_demand_kwh"]
        err  = f"{abs(val - gt):.1f} kWh error" if val else "PARSE FAILED"
        line = (f"\n[{i+1}] GT={gt:.1f} kWh | event={row['event_type']}\n"
                f"  Output : {text}\n"
                f"  Parsed : {val}  -> {err}\n")
        print(line)
        lines.append(line)
    path = os.path.join(OUTPUT_DIR, "zeroshot_preview.txt")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[Saved] {path}")


# ── 8. Dataset ────────────────────────────────────────────────────────────────

class EVDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt_text   = format_prompt(row["prompt"])
        response_text = row["reference_answer"] + tokenizer.eos_token
        full_text     = prompt_text + response_text

        full_enc = self.tokenizer(
            full_text, max_length=MAX_LEN, truncation=True, padding=False,
        )
        prompt_enc = self.tokenizer(
            prompt_text, max_length=MAX_LEN, truncation=True, padding=False,
        )
        prompt_len = len(prompt_enc["input_ids"])
        input_ids  = full_enc["input_ids"]
        labels     = [-100] * prompt_len + input_ids[prompt_len:]
        labels     = labels[:len(input_ids)]

        return {
            "input_ids"     : torch.tensor(input_ids,               dtype=torch.long),
            "attention_mask": torch.tensor(full_enc["attention_mask"], dtype=torch.long),
            "labels"        : torch.tensor(labels,                  dtype=torch.long),
        }


# ── 9. Collator ───────────────────────────────────────────────────────────────

def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, attention_masks, labels_list = [], [], []
    for x in batch:
        pad_len = max_len - x["input_ids"].shape[0]
        input_ids.append(torch.cat([
            x["input_ids"],
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
        ]))
        attention_masks.append(torch.cat([
            x["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        labels_list.append(torch.cat([
            x["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))
    return {
        "input_ids"     : torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels"        : torch.stack(labels_list),
    }


# ── 10. Naive Baseline ────────────────────────────────────────────────────────

def naive_baseline(val_df):
    preds = []
    for _, row in val_df.iterrows():
        m = re.search(r"\[([\d\s,\.]+)\]", row["prompt"])
        if m:
            vals = [float(v.strip()) for v in m.group(1).split(",") if v.strip()]
            preds.append(np.mean(vals))
        else:
            preds.append(np.nan)
    m = compute_metrics(val_df["ground_truth_demand_kwh"].tolist(), preds)
    print(f"  Naive 7-day avg -> MAE={m['MAE']} kWh  MAPE={m['MAPE']}%")
    return m


# ── 11. Training ──────────────────────────────────────────────────────────────

def train_model(model, tokenizer, train_df, val_df):
    train_ds = EVDataset(train_df, tokenizer)
    val_ds   = EVDataset(val_df,   tokenizer)

    args = TrainingArguments(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = 0.1,
        bf16                        = True,
        gradient_checkpointing      = True,
        eval_strategy               = "epoch",
        save_strategy               = "no",
        load_best_model_at_end      = False,
        metric_for_best_model       = "eval_loss",
        logging_steps               = 20,
        seed                        = SEED,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = collate_fn,
        processing_class = tokenizer,
    )

    print(f"\n{'='*60}")
    print(f"FINE-TUNING  ({EPOCHS} epochs, BF16+LoRA, effective batch={BATCH_SIZE*GRAD_ACCUM})")
    print(f"{'='*60}")
    trainer.train()

    log      = trainer.state.log_history
    t_steps  = [x["step"]      for x in log if "loss"      in x and "eval_loss" not in x]
    t_losses = [x["loss"]      for x in log if "loss"      in x and "eval_loss" not in x]
    e_epochs = [x["epoch"]     for x in log if "eval_loss" in x]
    e_losses = [x["eval_loss"] for x in log if "eval_loss" in x]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t_steps, t_losses, alpha=0.6, color="steelblue", label="Train loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train loss", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(e_epochs, e_losses, marker="s", color="darkorange",
             linewidth=2, label="Val loss")
    ax2.set_ylabel("Val loss", color="darkorange")
    plt.title("Llama-3.2-3B-Instruct -- Training Curve (LoRA)")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"), dpi=150)
    plt.close()
    print(f"[Saved] training_curve.png")
    return trainer.model


# ── 12. Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, val_df):
    print(f"\n{'='*60}")
    print(f"EVALUATION  ({len(val_df)} samples)")
    print(f"{'='*60}")
    model.eval()
    pred_texts, pred_vals = [], []

    for _, row in val_df.iterrows():
        prompt_text = format_prompt(row["prompt"])
        enc = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=MAX_LEN
        ).to(DEVICE)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_OUT_LEN,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_texts.append(text)
        pred_vals.append(parse_kwh(text))

    overall = compute_metrics(val_df["ground_truth_demand_kwh"].tolist(), pred_vals)
    print(f"\n  Overall -> MAE={overall['MAE']} kWh  "
          f"RMSE={overall['RMSE']} kWh  "
          f"MAPE={overall['MAPE']}%  "
          f"ParseRate={overall['parse_rate']}%")

    by_event = {}
    for evt, grp in val_df.groupby("event_type"):
        pos       = [val_df.index.get_loc(i) for i in grp.index]
        evt_preds = [pred_vals[p] for p in pos]
        m         = compute_metrics(grp["ground_truth_demand_kwh"].tolist(), evt_preds)
        by_event[evt] = m
        print(f"  [{evt:12s}] n={m['n_total']:3d}  "
              f"MAE={m['MAE']}  MAPE={m['MAPE']}%  "
              f"ParseRate={m['parse_rate']}%")

    out_df = val_df[["date", "ground_truth_demand_kwh", "event_type"]].copy()
    out_df["pred_text"] = pred_texts
    out_df["pred_kwh"]  = pred_vals
    out_df["abs_error"] = (out_df["ground_truth_demand_kwh"] - out_df["pred_kwh"]).abs()
    out_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump({"overall": overall, "by_event": by_event}, f, indent=2)
    print(f"\n[Saved] predictions.csv  evaluation.json")
    return {"overall": overall, "by_event": by_event}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_zeroshot(model, tokenizer, df)

    print("\n[Baseline]")
    baseline = naive_baseline(val_df)

    best_model = train_model(model, tokenizer, train_df, val_df)

    results = evaluate(best_model, tokenizer, val_df)

    ov = results["overall"]
    print(f"\n{'='*60}")
    print(f"SUMMARY -- Llama-3.2-3B-Instruct (LoRA)")
    print(f"{'='*60}")
    print(f"  Naive baseline  ->  MAE = {baseline['MAE']} kWh  MAPE = {baseline['MAPE']}%")
    print(f"  Fine-tuned      ->  MAE = {ov['MAE']} kWh  "
          f"MAPE = {ov['MAPE']}%  ParseRate = {ov['parse_rate']}%")
    print(f"\nAll outputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
