#!/usr/bin/env python3
"""
Stage A: Authentic S2-RGB LoRA Adaptation for BigEarthNet.txt Remote Sensing VQA.
Adapts Salesforce/blip-vqa-base via PEFT LoRA on authentic BigEarthNet v2.0 Sentinel-2 rasters.
Supervision: binary VQA on official train split (data/manifests/bigearthnet_txt_stageA_train.json).
Model selection / early stopping: official validation split (data/manifests/bigearthnet_txt_stageA_val.json).
STRICT RULE: Official benchmark (bench) is NEVER loaded or evaluated during training.
"""

import os
import sys
import json
import time
import psutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_MANIFEST = "data/manifests/bigearthnet_txt_stageA_train.json"
VAL_MANIFEST = "data/manifests/bigearthnet_txt_stageA_val.json"
OUTPUT_DIR = "models/satquery_rs_vlm_v1"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
MODEL_ID = "Salesforce/blip-vqa-base"

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # Effective batch size = 16
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 0.01
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


class BigEarthNetStageADataset(Dataset):
    def __init__(self, manifest_path: str, processor: BlipProcessor):
        with open(manifest_path, "r") as f:
            self.records = json.load(f)
        self.processor = processor
        # Validate existence
        self.valid_records = [r for r in self.records if os.path.exists(r["local_raster_path"])]
        print(f"Dataset {manifest_path}: {len(self.valid_records)}/{len(self.records)} valid images.")

    def __len__(self):
        return len(self.valid_records)

    def __getitem__(self, idx):
        rec = self.valid_records[idx]
        img = Image.open(rec["local_raster_path"]).convert("RGB")
        question = rec["question"]
        answer = rec["answer"].lower().strip()

        # BLIP VQA training encoding: image, text=question, answer=answer
        encoding = self.processor(
            images=img,
            text=question,
            return_tensors="pt"
        )
        # Squeeze batch dim
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # Encode answer labels for decoder
        labels = self.processor(text=answer, return_tensors="pt").input_ids.squeeze(0)
        item["labels"] = labels
        item["ground_truth"] = answer
        return item


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=0
    )
    gts = [b["ground_truth"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ground_truths": gts
    }


def evaluate_val(model, val_loader, device, processor):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gts = batch["ground_truths"]

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item() * len(gts)

            # Generate predictions for binary accuracy
            gen_out = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8
            )
            for i in range(len(gts)):
                pred_str = processor.decode(gen_out[i], skip_special_tokens=True).strip().lower()
                gt_str = gts[i].strip().lower()
                # Clean binary extraction
                pred_bin = "yes" if "yes" in pred_str and "no" not in pred_str else ("no" if "no" in pred_str and "yes" not in pred_str else pred_str)
                if pred_bin == gt_str:
                    correct += 1
                total += 1

    avg_loss = val_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    print(f"Memory before load: {mem_before:.1f} MB")

    print(f"Loading processor and base model: {MODEL_ID}...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    base_model = BlipForQuestionAnswering.from_pretrained(MODEL_ID)

    # 1. Freeze vision encoder
    for param in base_model.vision_model.parameters():
        param.requires_grad = False
    print("Vision encoder frozen.")

    # 2. Configure PEFT LoRA targeting crossattention and self-attention projections
    target_modules = ["query", "value"]
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        inference_mode=False
    )

    model = get_peft_model(base_model, peft_config)
    model.to(device)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    train_ds = BigEarthNetStageADataset(TRAIN_MANIFEST, processor)
    val_ds = BigEarthNetStageADataset(VAL_MANIFEST, processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    print(f"\nBeginning Stage A training on {len(train_ds)} train samples ({EPOCHS} epochs, effective batch {BATCH_SIZE * GRAD_ACCUM_STEPS})...")
    t0_train = time.time()
    best_val_acc = 0.0
    best_epoch = -1
    training_history = []
    peak_mem_mb = mem_before

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t_epoch_start = time.time()

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            epoch_loss += outputs.loss.item()
            step_count += 1

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_mem_mb:
                peak_mem_mb = current_mem

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch {epoch+1}/{EPOCHS} Step {step+1}/{len(train_loader)} - Loss: {outputs.loss.item():.4f}")

        avg_train_loss = epoch_loss / step_count
        val_loss, val_acc = evaluate_val(model, val_loader, device, processor)
        epoch_time = time.time() - t_epoch_start

        print(f"Epoch {epoch+1} Completed in {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "epoch_time_sec": round(epoch_time, 1)
        })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"  --> Saving best checkpoint at Epoch {best_epoch} (Val Acc: {best_val_acc:.4f})...")
            model.save_pretrained(CHECKPOINT_DIR)
            processor.save_pretrained(CHECKPOINT_DIR)

    total_train_time = time.time() - t0_train

    # Save metadata files
    config_dict = {
        "experiment": "BigEarthNet.txt S2-RGB adaptation",
        "stage": "Stage A",
        "paper_id": "arXiv:2603.29630",
        "dataset_repo": "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
        "base_model": MODEL_ID,
        "peft_type": "LORA",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": target_modules,
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "trainable_percentage": round(100 * trainable_params / all_params, 4),
        "bands": ["B04", "B03", "B02"],
        "input_resolution": 120,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "device": device,
        "peak_memory_mb": round(peak_mem_mb, 1),
        "total_train_time_sec": round(total_train_time, 1),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4)
    }

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    provenance_dict = {
        "dataset": "BigEarthNet.txt (arXiv:2603.29630)",
        "imagery_source": "Copernicus Sentinel-2 L2A (Planetary Computer COG archive)",
        "benchmark_provenance": "100% disjoint human-verified bench split (1,082 image pairs, 15,029 annotations)",
        "train_manifest": TRAIN_MANIFEST,
        "val_manifest": VAL_MANIFEST,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "date_trained": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_dir": CHECKPOINT_DIR
    }

    with open(os.path.join(OUTPUT_DIR, "provenance.json"), "w") as f:
        json.dump(provenance_dict, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "training_manifest.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining finished successfully!")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}")
    print(f"Best Val Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")
    print(f"Peak memory: {peak_mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
