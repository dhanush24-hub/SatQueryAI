#!/usr/bin/env python3
"""
Stage 8.5B: Validation Recovery Experiments on Authentic BigEarthNet.txt.
Runs a 3-configuration validation-only ablation:
  - Experiment A: LoRA baseline (LR=1e-4, standard sampler, r=8)
  - Experiment B: LoRA low-LR (LR=3e-5, standard sampler, r=8)
  - Experiment C: LoRA low-LR + Stratified Task/Answer-Balanced Sampler (LR=3e-5, r=8)

Evaluates on VALIDATION ONLY (data/manifests/bigearthnet_txt_stage8_5b_val.json).
Records:
  - Validation accuracy & balanced accuracy
  - Precision, Recall, F1 for YES and NO
  - Per-task accuracy (Presence, Area, Counting, Adjacency)
  - Validation loss & prediction distribution
STRICT RULE: Benchmark split is NEVER loaded or evaluated during recovery.
"""

import os
import sys
import json
import time
import math
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BlipProcessor, BlipForQuestionAnswering, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "Salesforce/blip-vqa-base"
TRAIN_MANIFEST = "data/manifests/bigearthnet_txt_stage8_5b_train.json"
VAL_MANIFEST = "data/manifests/bigearthnet_txt_stage8_5b_val.json"
OUTPUT_DIR = "models/stage8_5b_recovery"
ABLATION_REPORT = "docs/STAGE8_5B_VALIDATION_ABLATION.json"

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2  # Effective batch size = 16
STEPS_PER_EXP = 450   # ~1 full epoch over train set


class BigEarthNetRecoveryDataset(Dataset):
    def __init__(self, manifest_path: str, max_samples: int = None):
        with open(manifest_path, "r") as f:
            records = json.load(f)
        # Filter existing rasters
        self.records = [r for r in records if os.path.exists(r["local_raster_path"])]
        if max_samples:
            self.records = self.records[:max_samples]
        print(f"Loaded {len(self.records)} valid records from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return rec


def create_collate_fn(processor):
    def collate_fn(batch):
        images = [Image.open(b["local_raster_path"]).convert("RGB") for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"].strip().lower() for b in batch]
        tasks = [b["task"] for b in batch]

        # Process vision + question inputs
        inputs = processor(images=images, text=questions, padding=True, return_tensors="pt")

        # Process decoder targets with proper ignore_index=-100 masking
        ans_inputs = processor(text=answers, padding=True, return_tensors="pt")
        labels = ans_inputs.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs.pixel_values,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "decoder_input_ids": ans_inputs.input_ids,
            "labels": labels,
            "ground_truths": answers,
            "tasks": tasks
        }
    return collate_fn


def evaluate_on_validation(model, processor, val_loader, max_eval_batches=30):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0

    y_true = []
    y_pred = []
    task_results = {"presence": {"c": 0, "t": 0}, "area": {"c": 0, "t": 0}, "count": {"c": 0, "t": 0}, "adjacency": {"c": 0, "t": 0}}

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_eval_batches and b_idx >= max_eval_batches:
                break

            pv = batch["pixel_values"].to(DEVICE)
            ii = batch["input_ids"].to(DEVICE)
            am = batch["attention_mask"].to(DEVICE)
            d_ii = batch["decoder_input_ids"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            gts = batch["ground_truths"]
            tasks = batch["tasks"]

            # Compute validation loss
            out = model(
                pixel_values=pv,
                input_ids=ii,
                attention_mask=am,
                decoder_input_ids=d_ii,
                labels=lbl
            )
            val_loss += out.loss.item() * len(gts)

            # Generate predictions
            gen_out = model.generate(
                pixel_values=pv,
                input_ids=ii,
                attention_mask=am,
                max_new_tokens=5
            )

            for i in range(len(gts)):
                pred_raw = processor.decode(gen_out[i], skip_special_tokens=True).strip().lower()
                gt = gts[i]
                t = tasks[i]

                pred_bin = "yes" if "yes" in pred_raw and "no" not in pred_raw else ("no" if "no" in pred_raw and "yes" not in pred_raw else pred_raw)

                y_true.append(gt)
                y_pred.append(pred_bin)

                is_correct = (pred_bin == gt)
                if is_correct:
                    correct += 1
                total += 1

                if t in task_results:
                    task_results[t]["t"] += 1
                    if is_correct:
                        task_results[t]["c"] += 1

    avg_loss = round(val_loss / total, 4) if total > 0 else 0.0
    acc = round(correct / total, 4) if total > 0 else 0.0

    # Yes/No metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "yes" and yp == "yes")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "no" and yp == "yes")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "yes" and yp == "no")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "no" and yp == "no")

    prec_yes = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    rec_yes = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1_yes = round(2 * prec_yes * rec_yes / (prec_yes + rec_yes), 4) if (prec_yes + rec_yes) > 0 else 0.0

    prec_no = round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0.0
    rec_no = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    f1_no = round(2 * prec_no * rec_no / (prec_no + rec_no), 4) if (prec_no + rec_no) > 0 else 0.0

    balanced_acc = round((rec_yes + rec_no) / 2.0, 4)

    task_accs = {
        k: round(v["c"] / v["t"], 4) if v["t"] > 0 else 0.0
        for k, v in task_results.items()
    }

    pred_dist = {
        "yes": sum(1 for p in y_pred if p == "yes"),
        "no": sum(1 for p in y_pred if p == "no"),
        "other": sum(1 for p in y_pred if p not in ["yes", "no"])
    }

    return {
        "samples_evaluated": total,
        "val_loss": avg_loss,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision_yes": prec_yes,
        "recall_yes": rec_yes,
        "f1_yes": f1_yes,
        "precision_no": prec_no,
        "recall_no": rec_no,
        "f1_no": f1_no,
        "per_task_accuracy": task_accs,
        "prediction_distribution": pred_dist
    }


def train_single_experiment(exp_name, lr, use_balanced_sampler, train_ds, val_ds, processor):
    print(f"\n=======================================================", flush=True)
    print(f"STARTING {exp_name} | LR={lr} | BalancedSampler={use_balanced_sampler}", flush=True)
    print(f"=======================================================", flush=True)

    collate_fn = create_collate_fn(processor)

    if use_balanced_sampler:
        # Create balanced sampling weights across (task, answer)
        group_counts = {}
        for r in train_ds.records:
            key = (r["task"], r["answer"])
            group_counts[key] = group_counts.get(key, 0) + 1

        weights = [1.0 / group_counts[(r["task"], r["answer"])] for r in train_ds.records]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model with fresh base weights + LoRA
    base_model = BlipForQuestionAnswering.from_pretrained(MODEL_ID)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)

    # Pre-training validation baseline
    print("Evaluating pre-training validation baseline...", flush=True)
    baseline_val = evaluate_on_validation(model, processor, val_loader, max_eval_batches=30)
    print(f"Pre-training Val Acc: {baseline_val['accuracy']:.4f} | Balanced: {baseline_val['balanced_accuracy']:.4f} | Preds: {baseline_val['prediction_distribution']}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=30, num_training_steps=STEPS_PER_EXP)

    model.train()
    step = 0
    accum_loss = 0.0
    t0 = time.time()

    data_iter = iter(train_loader)
    optimizer.zero_grad()

    while step < STEPS_PER_EXP:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pv = batch["pixel_values"].to(DEVICE)
        ii = batch["input_ids"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        d_ii = batch["decoder_input_ids"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)

        out = model(
            pixel_values=pv,
            input_ids=ii,
            attention_mask=am,
            decoder_input_ids=d_ii,
            labels=lbl
        )
        loss = out.loss / GRAD_ACCUM_STEPS
        loss.backward()
        accum_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        if step % 75 == 0 or step == STEPS_PER_EXP:
            avg_step_loss = accum_loss / 75.0 if step % 75 == 0 else accum_loss / (step % 75 or 1)
            accum_loss = 0.0
            print(f"  [{exp_name}] Step {step}/{STEPS_PER_EXP} | Loss: {avg_step_loss:.4f} | Time: {time.time() - t0:.1f}s", flush=True)

    # Post-training validation evaluation
    print(f"Evaluating {exp_name} on validation set...", flush=True)
    final_val = evaluate_on_validation(model, processor, val_loader, max_eval_batches=30)
    print(f"[{exp_name}] Final Val Acc: {final_val['accuracy']:.4f} | Balanced: {final_val['balanced_accuracy']:.4f} | F1 Yes: {final_val['f1_yes']} | F1 No: {final_val['f1_no']}")
    print(f"  Tasks: {final_val['per_task_accuracy']}")
    print(f"  Pred Distribution: {final_val['prediction_distribution']}")

    # Save checkpoint
    exp_dir = os.path.join(OUTPUT_DIR, exp_name.lower().replace(" ", "_"))
    os.makedirs(exp_dir, exist_ok=True)
    model.save_pretrained(exp_dir)

    return {
        "experiment": exp_name,
        "hyperparameters": {
            "lr": lr,
            "balanced_sampler": use_balanced_sampler,
            "lora_r": 8,
            "lora_alpha": 16,
            "steps": STEPS_PER_EXP,
            "batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS
        },
        "pre_training_val": baseline_val,
        "post_training_val": final_val,
        "checkpoint_dir": exp_dir
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = BlipProcessor.from_pretrained(MODEL_ID)

    train_ds = BigEarthNetRecoveryDataset(TRAIN_MANIFEST)
    val_ds = BigEarthNetRecoveryDataset(VAL_MANIFEST)

    results = []

    # Exp A: Baseline LoRA (lr=1e-4, standard)
    res_a = train_single_experiment("Experiment_A_Standard_LR1e-4", lr=1e-4, use_balanced_sampler=False, train_ds=train_ds, val_ds=val_ds, processor=processor)
    results.append(res_a)

    # Exp B: Low LR (lr=3e-5, standard)
    res_b = train_single_experiment("Experiment_B_Low_LR3e-5", lr=3e-5, use_balanced_sampler=False, train_ds=train_ds, val_ds=val_ds, processor=processor)
    results.append(res_b)

    # Exp C: Low LR + Task/Answer-Balanced Sampler (lr=3e-5, balanced)
    res_c = train_single_experiment("Experiment_C_Balanced_Sampler_LR3e-5", lr=3e-5, use_balanced_sampler=True, train_ds=train_ds, val_ds=val_ds, processor=processor)
    results.append(res_c)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_manifest": VAL_MANIFEST,
        "train_manifest": TRAIN_MANIFEST,
        "experiments": results
    }

    with open(ABLATION_REPORT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll 3 validation ablation experiments finished! Report saved to {ABLATION_REPORT}", flush=True)


if __name__ == "__main__":
    main()
