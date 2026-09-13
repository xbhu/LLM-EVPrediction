"""
Use Case 3 — Model 1: Flan-T5-large
=====================================
Architecture : Encoder-Decoder (Seq2Seq)
Training     : Full parameter fine-tune, BF16, HuggingFace Trainer
Task         : EV charging demand prediction with natural language explanation

Run:
    conda activate ev_llm
    python usecase3_flant5.py
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "google/flan-t5-large"
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset3_qa_pairs.csv"
OUTPUT_DIR = "/home/xzh5180/Research/llm-evprediction/outputs/usecase3_flant5"
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 8
BATCH_SIZE  = 4
GRAD_ACCUM  = 2
LR          = 5e-5
MAX_IN_LEN  = 256
MAX_OUT_LEN = 64
N_ZEROSHOT  = 10
SEED        = 42

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
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
model = model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters : {n_params:.0f}M")
if DEVICE == "cuda":
    print(f"VRAM used  : {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ── 3. Parsing ────────────────────────────────────────────────────────────────

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


# ── 4. Metrics ────────────────────────────────────────────────────────────────

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


# ── 5. Zero-Shot Preview ──────────────────────────────────────────────────────

def run_zeroshot(model, tokenizer, df, n=N_ZEROSHOT):
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT PREVIEW  ({n} samples, before fine-tuning)")
    print(f"{'='*60}")
    model.eval()
    lines = []
    for i, row in df.head(n).iterrows():
        enc  = tokenizer(row["prompt"], return_tensors="pt",
                         max_length=MAX_IN_LEN, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_OUT_LEN)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
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


# ── 6. Dataset ────────────────────────────────────────────────────────────────

class EVDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        enc = self.tokenizer(
            row["prompt"], max_length=MAX_IN_LEN, truncation=True,
        )
        tgt = self.tokenizer(
            text_target=row["reference_answer"],
            max_length=MAX_OUT_LEN, truncation=True,
        )
        enc["labels"] = tgt["input_ids"]
        return enc


# ── 7. Naive Baseline ─────────────────────────────────────────────────────────

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


# ── 8. Training ───────────────────────────────────────────────────────────────

def train_model(model, tokenizer, train_df, val_df):
    train_ds = EVDataset(train_df, tokenizer)
    val_ds   = EVDataset(val_df,   tokenizer)
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )
    args = Seq2SeqTrainingArguments(
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
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        predict_with_generate       = True,
        generation_max_length       = MAX_OUT_LEN,
        logging_steps               = 20,
        seed                        = SEED,
        report_to                   = "none",
    )
    trainer = Seq2SeqTrainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
        processing_class = tokenizer,
    )
    print(f"\n{'='*60}")
    print(f"FINE-TUNING  ({EPOCHS} epochs, BF16, effective batch={BATCH_SIZE*GRAD_ACCUM})")
    print(f"{'='*60}")
    trainer.train()

    # Training curve
    log      = trainer.state.log_history
    t_steps  = [x["step"] for x in log if "loss" in x and "eval_loss" not in x]
    t_losses = [x["loss"] for x in log if "loss" in x and "eval_loss" not in x]
    e_epochs = [x["epoch"] for x in log if "eval_loss" in x]
    e_losses = [x["eval_loss"] for x in log if "eval_loss" in x]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t_steps, t_losses, label="Train loss", alpha=0.6, color="steelblue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train loss", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(e_epochs, e_losses, label="Val loss", marker="s",
             color="darkorange", linewidth=2)
    ax2.set_ylabel("Val loss", color="darkorange")
    plt.title("Flan-T5-large -- Training Curve")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"), dpi=150)
    plt.close()
    print(f"[Saved] training_curve.png")
    return trainer.model


# ── 9. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, val_df):
    print(f"\n{'='*60}")
    print(f"EVALUATION  ({len(val_df)} samples)")
    print(f"{'='*60}")
    model.eval()
    pred_texts, pred_vals = [], []
    for _, row in val_df.iterrows():
        enc  = tokenizer(row["prompt"], return_tensors="pt",
                         max_length=MAX_IN_LEN, truncation=True).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_OUT_LEN)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
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
    print(f"SUMMARY -- Flan-T5-large")
    print(f"{'='*60}")
    print(f"  Naive baseline  ->  MAE = {baseline['MAE']} kWh  MAPE = {baseline['MAPE']}%")
    print(f"  Fine-tuned      ->  MAE = {ov['MAE']} kWh  "
          f"MAPE = {ov['MAPE']}%  ParseRate = {ov['parse_rate']}%")
    print(f"\nAll outputs -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
