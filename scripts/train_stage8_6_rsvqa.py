#!/usr/bin/env python3
"""
Prompt 8.6: RS-CrossAttention-VQA Architecture Training & Evaluation.
Tests whether replacing BLIP's 30,522-token autoregressive text decoder with
a direct Remote-Sensing Cross-Attention Binary Classifier resolves the VLM bottleneck.

Architecture:
- Vision Backbone: BlipVisionModel (ViT-B/16, 86M params)
- Text Backbone: BlipTextModel (Bidirectional BERT, 110M params)
- Cross-Modal Fusion: Multi-Head Cross-Attention (dim=768, heads=8)
  Query = Question tokens, Key/Value = Vision patch tokens
- Classifier Head: LayerNorm -> Linear(768, 256) -> GELU -> Dropout(0.1) -> Linear(256, 2)
- Loss: CrossEntropyLoss directly on [No, Yes] (No tokenization or padding artifacts!)

Evaluates strictly on VALIDATION ONLY (data/manifests/bigearthnet_txt_stage8_5b_val.json).
No benchmark access.
"""

import os
import sys
import json
import time
import math
import random
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BlipProcessor, BlipVisionModel, BlipTextModel

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "Salesforce/blip-vqa-base"
TRAIN_MANIFEST = "data/manifests/bigearthnet_txt_stage8_5b_train.json"
VAL_MANIFEST = "data/manifests/bigearthnet_txt_stage8_5b_val.json"
OUTPUT_DIR = "models/satquery_rsvqa_v2"
REPORT_PATH = "docs/STAGE8_6_MODEL_UPGRADE_REPORT.json"

BATCH_SIZE = 16
STEPS = 450
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5


class RSCrossAttentionVQA(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        # Load vision and text models locally without re-downloading
        self.vision_model = BlipVisionModel.from_pretrained(model_id, local_files_only=True)
        self.text_model = BlipTextModel.from_pretrained(model_id, local_files_only=True)

        # Freeze pretrained backbones to fit comfortably within 8GB MPS (<900MB)
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False

        hidden_dim = self.vision_model.config.hidden_size  # 768

        # Cross-Attention: Question tokens attend to Vision patch tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Binary VQA classification head: 0 = "no", 1 = "yes"
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            # 1. Extract visual tokens [B, 577, 768]
            v_out = self.vision_model(pixel_values=pixel_values).last_hidden_state
            # 2. Extract question tokens [B, seq_len, 768]
            t_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 3. Cross-attention: Question queries image features (trainable)
        fused, _ = self.cross_attn(query=t_out, key=v_out, value=v_out)
        fused = self.norm(fused + t_out)

        # 4. Pool CLS token (token 0)
        cls_rep = fused[:, 0, :]

        # 5. Classify [B, 2]
        logits = self.classifier(cls_rep)
        return logits


class VQADataset(Dataset):
    def __init__(self, manifest_path: str):
        with open(manifest_path, "r") as f:
            records = json.load(f)
        self.records = [r for r in records if os.path.exists(r["local_raster_path"])]
        print(f"Loaded {len(self.records)} valid samples from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [Image.open(b["local_raster_path"]).convert("RGB") for b in batch]
        questions = [b["question"] for b in batch]
        answers = [1 if b["answer"].strip().lower() == "yes" else 0 for b in batch]
        tasks = [b["task"] for b in batch]

        # BLIP image processing
        pv = processor.image_processor(images=images, return_tensors="pt").pixel_values

        # Text tokenization
        text_inputs = processor.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")

        return {
            "pixel_values": pv,
            "input_ids": text_inputs.input_inputs if hasattr(text_inputs, "input_inputs") else text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": torch.tensor(answers, dtype=torch.long),
            "tasks": tasks
        }
    return collate_fn


def evaluate(model, val_loader, device, max_batches=20):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    task_results = {"presence": {"c": 0, "t": 0}, "area": {"c": 0, "t": 0}, "count": {"c": 0, "t": 0}, "adjacency": {"c": 0, "t": 0}}

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches and b_idx >= max_batches:
                break
            pv = batch["pixel_values"].to(device)
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            tasks = batch["tasks"]

            logits = model(pixel_values=pv, input_ids=ii, attention_mask=am)
            preds = torch.argmax(logits, dim=-1)

            for i in range(len(lbl)):
                gt = lbl[i].item()
                pr = preds[i].item()
                y_true.append(gt)
                y_pred.append(pr)

                is_corr = (gt == pr)
                if is_corr:
                    correct += 1
                total += 1

                t = tasks[i]
                if t in task_results:
                    task_results[t]["t"] += 1
                    if is_corr:
                        task_results[t]["c"] += 1

    acc = round(correct / total, 4) if total > 0 else 0.0

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

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
        "yes": sum(1 for p in y_pred if p == 1),
        "no": sum(1 for p in y_pred if p == 0)
    }

    return {
        "samples_evaluated": total,
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = BlipProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    print("Loading datasets...", flush=True)
    train_ds = VQADataset(TRAIN_MANIFEST)
    val_ds = VQADataset(VAL_MANIFEST)

    collate_fn = make_collate_fn(processor)

    # Balanced sampler across tasks and answers
    group_counts = {}
    for r in train_ds.records:
        key = (r["task"], r["answer"])
        group_counts[key] = group_counts.get(key, 0) + 1
    weights = [1.0 / group_counts[(r["task"], r["answer"])] for r in train_ds.records]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Instantiating RSCrossAttentionVQA on {DEVICE}...", flush=True)
    model = RSCrossAttentionVQA(MODEL_ID).to(DEVICE)

    # Pre-training validation baseline
    print("Evaluating pre-training validation baseline...", flush=True)
    pre_val = evaluate(model, val_loader, DEVICE, max_batches=20)
    print(f"Pre-training Val Acc: {pre_val['accuracy']:.4f} | Balanced: {pre_val['balanced_accuracy']:.4f} | Preds: {pre_val['prediction_distribution']}", flush=True)

    # Train only fusion and classifier parameters (backbones are frozen)
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
        {"params": model.cross_attn.parameters(), "lr": LR_HEAD},
        {"params": model.norm.parameters(), "lr": LR_HEAD}
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(train_loader)

    print(f"Starting training for {STEPS} steps (batch_size={BATCH_SIZE})...", flush=True)
    while step < STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pv = batch["pixel_values"].to(DEVICE)
        ii = batch["input_ids"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(pixel_values=pv, input_ids=ii, attention_mask=am)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if step % 75 == 0 or step == STEPS:
            print(f"  Step {step}/{STEPS} | Loss: {loss.item():.4f} | Elapsed: {time.time() - t0:.1f}s", flush=True)

    print("Training complete! Evaluating on validation set...", flush=True)
    post_val = evaluate(model, val_loader, DEVICE, max_batches=20)
    print(f"Final Val Acc: {post_val['accuracy']:.4f} | Balanced: {post_val['balanced_accuracy']:.4f} | F1 Yes: {post_val['f1_yes']} | F1 No: {post_val['f1_no']}", flush=True)
    print(f"  Per-Task: {post_val['per_task_accuracy']}", flush=True)
    print(f"  Predictions: {post_val['prediction_distribution']}", flush=True)

    # Save model weights
    ckpt_path = os.path.join(OUTPUT_DIR, "rs_cross_attention_vqa.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}", flush=True)

    # Save complete report
    report = {
        "model_architecture": "RS-CrossAttention-VQA",
        "vision_backbone": "BlipVisionModel (ViT-B/16, 86M params)",
        "text_backbone": "BlipTextModel (BERT, 110M params)",
        "fusion": "Multi-Head Cross-Attention (dim=768, heads=8)",
        "classifier": "Linear(768, 256) -> GELU -> Dropout -> Linear(256, 2)",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "pre_training_val": pre_val,
        "post_training_val": post_val,
        "checkpoint_path": ckpt_path
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
