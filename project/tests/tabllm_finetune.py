"""
Step 2: Fine-tune FLAN-T5 on serialized tabular data — one model per dataset.

Usage:
    python tabllm_finetune.py --dataset ramen_ratings
    python tabllm_finetune.py --dataset coffee_ratings --epochs 5
    python tabllm_finetune.py --all   # train one model per dataset
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from tqdm import tqdm

root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

set_seed(42)

PREFIX = ""  # No extra prefix — the text already says "Stars (Good if ≥4.0):"


# ═══════════════════════════════════════════════════════════════════
# Dataset — loads a SINGLE dataset's JSONL
# ═══════════════════════════════════════════════════════════════════

def load_dataset_from_jsonl(dataset_dir: str, split: str):
    """Load samples from a single dataset's train.jsonl or test.jsonl."""
    filepath = os.path.join(dataset_dir, f"{split}.jsonl")
    samples = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
    return samples


class SingleDataset(Dataset):
    """HuggingFace Dataset wrapper for one dataset's JSONL."""

    def __init__(self, samples, tokenizer, max_input_length=512, max_output_length=16):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_enc = self.tokenizer(PREFIX + s["text"], max_length=self.max_input_length,
                                   truncation=True, padding=False)
        label_enc = self.tokenizer(s["label"], max_length=self.max_output_length,
                                   truncation=True, padding=False)
        return {
            "input_ids": input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels": label_enc["input_ids"],
        }


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred, tokenizer, label_to_id, id_to_label):
    """Compute accuracy and AUC from logits.

    When predict_with_generate=False, eval_pred contains (logits, labels).
    We take the first-token logits, softmax over the known label tokens → AUC.
    """
    logits, labels = eval_pred

    # Handle nested tuple from Seq2Seq evaluation loop
    if isinstance(logits, tuple):
        logits = logits[0]

    # logits: (batch, seq_len, vocab_size) — squeeze if needed
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)  # (batch, vocab) → (batch, 1, vocab)

    # Get the label token IDs
    label_token_ids = {}
    for lbl_text, lbl_id in label_to_id.items():
        tok_ids = tokenizer.encode(lbl_text, add_special_tokens=False)
        label_token_ids[lbl_id] = tok_ids[0]  # first token only

    # First position logits and true labels
    first_logits = logits[:, 0, :]  # (batch, vocab_size)
    first_labels = labels[:, 0]     # (batch,)

    # Filter out padding (-100)
    valid = first_labels != -100
    first_logits = first_logits[valid]
    first_labels = first_labels[valid]
    
    first_logits = torch.from_numpy(first_logits)

    if len(first_labels) == 0:
        return {"accuracy": 0.0, "eval_auc": 0.5}

    # Compute accuracy from argmax
    pred_ids = first_logits.argmax(dim=-1).cpu().numpy()
    true_ids = first_labels.cpu().numpy() if torch.is_tensor(first_labels) else first_labels
    acc = accuracy_score(true_ids, pred_ids)

    # Compute AUC: for binary, use softmax probability of class 1
    if len(label_to_id) == 2:
        id0, id1 = sorted(label_to_id.values())
        tok0, tok1 = label_token_ids[id0], label_token_ids[id1]

        scores_0 = first_logits[:, tok0]
        scores_1 = first_logits[:, tok1]

        stacked = torch.stack([scores_0, scores_1], dim=-1)
        probs = torch.softmax(stacked, dim=-1)
        prob_class1 = probs[:, 1].cpu().numpy()

        true_binary = (true_ids != id0).astype(int)
        # Check for single-class before calling roc_auc_score
        unique_classes = np.unique(true_binary)
        if len(unique_classes) < 2:
            auc = 0.5  # only one class present, AUC is undefined
        else:
            try:
                auc = roc_auc_score(true_binary, prob_class1)
            except ValueError:
                auc = 0.5
    else:
        auc = 0.5

    return {"accuracy": float(acc), "eval_auc": float(auc)}


# ═══════════════════════════════════════════════════════════════════
# Main — trains ONE model per dataset
# ═══════════════════════════════════════════════════════════════════

def eval_one_dataset(dataset_name, args, tokenizer):
    """Load a saved checkpoint and evaluate on test set."""
    model_dir = os.path.join(args.output_base, dataset_name, "final")
    if not os.path.exists(model_dir):
        # Try checkpoint dir
        checkpoints = [d for d in os.listdir(os.path.join(args.output_base, dataset_name))
                       if d.startswith("checkpoint-")]
        if checkpoints:
            model_dir = os.path.join(args.output_base, dataset_name, sorted(checkpoints)[-1])
        else:
            print(f"⚠️  {dataset_name}: no saved model found, skipping")
            return None

    print(f"📦 Loading {dataset_name} from {model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(args.device)

    dataset_dir = os.path.join(args.data_dir, dataset_name)
    test_samples = load_dataset_from_jsonl(dataset_dir, "test")
    if not test_samples:
        print(f"⚠️  {dataset_name}: no test data, skipping")
        return None

    all_labels = sorted(set(s["label"] for s in test_samples))
    label_to_id = {lbl: i for i, lbl in enumerate(all_labels)}
    id_to_label = {i: lbl for i, lbl in enumerate(all_labels)}

    test_ds = SingleDataset(test_samples, tokenizer, args.max_input_length, 16)
    print(f"📊 {dataset_name}: {len(test_samples)} test samples")

    # Minimal trainer just for evaluation
    eval_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/eval_tmp",
        per_device_eval_batch_size=args.batch_size * 2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=lambda p: compute_metrics(p, tokenizer, label_to_id, id_to_label),
    )

    metrics = trainer.evaluate()
    print(f"📈 {dataset_name}: acc={metrics.get('eval_accuracy', 'N/A'):.4f}, "
          f"auc={metrics.get('eval_auc', 'N/A'):.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Zero-shot / Few-shot evaluation (no fine-tuning, pure inference)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def zero_few_shot_eval(dataset_name, args, tokenizer, n_shots=0):
    """Evaluate pretrained model with zero-shot or few-shot prompting.

    n_shots=0 → zero-shot (just the serialized row as input)
    n_shots>0 → few-shot (include n_shots examples from train in prompt)
    """
    dataset_dir = os.path.join(args.data_dir, dataset_name)
    train_samples = load_dataset_from_jsonl(dataset_dir, "train")
    test_samples = load_dataset_from_jsonl(dataset_dir, "test")
    if not test_samples:
        print(f"⚠️  {dataset_name}: no test data, skipping")
        return None

    # Load fresh pretrained model
    print(f"📦 Loading pretrained {args.model} (no fine-tuning)")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    model.eval()

    all_labels = sorted(set(s["label"] for s in train_samples + test_samples))
    label_to_id = {lbl: i for i, lbl in enumerate(all_labels)}

    # Build few-shot example strings
    shot_examples = ""
    if n_shots > 0 and train_samples:
        # Balanced sampling
        by_label = {}
        for s in train_samples:
            by_label.setdefault(s["label"], []).append(s)
        per_label = n_shots // len(by_label)
        selected = []
        for lbl, samples in by_label.items():
            selected.extend(np.random.choice(samples, min(per_label, len(samples)), replace=False).tolist())
        np.random.shuffle(selected)
        selected = selected[:n_shots]

        shot_examples = "\n\n".join(
            f"{s['text']} {s['label']}" for s in selected
        ) + "\n\n"

    y_true, y_score = [], []
    # Truncate from LEFT: keep the query, drop oldest examples when prompt is too long
    tokenizer.truncation_side = "left"

    for s in tqdm(test_samples, desc=f" Evaluating {dataset_name}"):
        prompt = shot_examples + s["text"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_input_length).to(args.device)

        outputs = model(**inputs, decoder_input_ids=torch.tensor(
            [[model.config.decoder_start_token_id]], device=args.device))
        logits = outputs.logits[0, 0, :]

        # Probability of each label
        label_logits = {}
        for lbl_text, lbl_id in label_to_id.items():
            tok_ids = tokenizer.encode(lbl_text, add_special_tokens=False)
            label_logits[lbl_id] = logits[tok_ids[0]].item()

        # Softmax over labels
        ids_sorted = sorted(label_logits.keys())
        scores = np.array([label_logits[i] for i in ids_sorted])
        probs = np.exp(scores - np.max(scores))
        probs /= probs.sum()

        true_id = label_to_id[s["label"]]
        y_true.append(true_id)
        y_score.append(probs[list(ids_sorted).index(1)] if 1 in ids_sorted else probs[-1])

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    unique = np.unique(y_true)
    auc = roc_auc_score(y_true, y_score) if len(unique) >= 2 else 0.5

    mode = "Zero-shot" if n_shots == 0 else f"{n_shots}-shot"
    print(f"📈 {dataset_name} ({mode}): acc={acc:.4f}, auc={auc:.4f}")
    return {"accuracy": acc, "eval_auc": auc}


def train_one_dataset(dataset_name, args, tokenizer):
    """Fine-tune a model on a single dataset."""
    dataset_dir = os.path.join(args.data_dir, dataset_name)
    train_samples = load_dataset_from_jsonl(dataset_dir, "train")
    test_samples = load_dataset_from_jsonl(dataset_dir, "test")

    if not train_samples:
        print(f"⚠️  {dataset_name}: no train data, skipping")
        return None

    # Build label mapping
    all_labels = sorted(set(s["label"] for s in train_samples + test_samples))
    label_to_id = {lbl: i for i, lbl in enumerate(all_labels)}
    id_to_label = {i: lbl for i, lbl in enumerate(all_labels)}
    print(f"📋 {dataset_name}: labels={label_to_id}")
    print(f"   Train: {len(train_samples)}, Test: {len(test_samples)}")

    max_label_len = max(len(lbl) for lbl in all_labels)

    train_ds = SingleDataset(train_samples, tokenizer, args.max_input_length,
                             max(16, max_label_len + 4))
    test_ds = SingleDataset(test_samples, tokenizer, args.max_input_length,
                            max(16, max_label_len + 4))

    # Fresh model for each dataset
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    output_dir = os.path.join(args.output_base, dataset_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        bf16=torch.cuda.is_available(),
        report_to="none",
        predict_with_generate=False,  # ← False → get logits for AUC
        generation_max_length=max(16, max_label_len + 4),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=lambda p: compute_metrics(p, tokenizer, label_to_id, id_to_label),
    )

    print(f"\n🚀 Training {dataset_name}...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"📈 {dataset_name}: acc={metrics.get('eval_accuracy', 'N/A'):.4f}, "
          f"auc={metrics.get('eval_auc', 'N/A'):.4f}")

    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset name (e.g., ramen_ratings)")
    parser.add_argument("--all", action="store_true",
                        help="Train one model per dataset")
    parser.add_argument("--model", type=str, default="google/flan-t5-large")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(root_dir, "serialized_data"))
    parser.add_argument("--output_base", type=str,
                        default=os.path.join(root_dir, "finetuned_models"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate saved checkpoints, no training")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-shot: pretrained model, no training, no examples")
    parser.add_argument("--few_shot", type=int, default=None,
                        help="Few-shot: pretrained model with N examples in prompt (e.g., --few_shot 4)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {args.device}  |  Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.eval_only:
        # ── Evaluate saved checkpoint ──
        _run_on_datasets(args, tokenizer, eval_one_dataset, mode_label="Checkpoint eval")

    elif args.zero_shot or args.few_shot is not None:
        # ── Zero-shot / Few-shot (pretrained model, no training) ──
        n_shots = args.few_shot if args.few_shot is not None else 0
        mode_label = "Zero-shot" if n_shots == 0 else f"{n_shots}-shot"
        print(f"\n🧪 {mode_label} evaluation (no fine-tuning)\n")

        _run_on_datasets(args, tokenizer,
                         lambda ds, a, tok: zero_few_shot_eval(ds, a, tok, n_shots),
                         mode_label=mode_label)

    elif args.dataset or args.all:
        # ── Fine-tune ──
        _run_on_datasets(args, tokenizer, train_one_dataset, mode_label="Fine-tune")

    else:
        print("❌ Usage:")
        print("   Train:      python tabllm_finetune.py --dataset ramen_ratings")
        print("   Zero-shot:  python tabllm_finetune.py --zero_shot --dataset ramen_ratings")
        print("   Few-shot:   python tabllm_finetune.py --few_shot 4 --dataset ramen_ratings")
        print("   Eval ckpt:  python tabllm_finetune.py --eval_only --dataset ramen_ratings")
        print("   All:        add --all to any mode")


def _run_on_datasets(args, tokenizer, fn, mode_label=""):
    """Run fn on a single dataset or all datasets, print mean AUC."""
    if args.dataset:
        fn(args.dataset, args, tokenizer)
    elif args.all:
        dataset_names = sorted(
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        )
        print(f"\n📊 {mode_label} on {len(dataset_names)} datasets\n")
        all_aucs = []
        for ds_name in dataset_names:
            metrics = fn(ds_name, args, tokenizer)
            if metrics:
                all_aucs.append(metrics.get("eval_auc", 0.0))
        if all_aucs:
            print(f"\n{'='*50}")
            print(f"📊 MEAN AUC ({mode_label}): {sum(all_aucs)/len(all_aucs):.4f}")
            for ds_name, auc in zip(dataset_names, all_aucs):
                print(f"   {ds_name}: {auc:.4f}")


if __name__ == "__main__":
    main()
