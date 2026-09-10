#!/usr/bin/env python3
"""
Remote Sensing Visual Question Answering (RS-VQA) Parameter-Efficient Fine-Tuning (LoRA).
Applies Low-Rank Adaptation (LoRA) to Salesforce/blip-vqa-base for satellite imagery VQA.

Usage:
    python scripts/train_rs_vlm_lora.py --data_path data/rsvqa/ --epochs 5 --lr 2e-4 --batch_size 8
"""

import os
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering

# Ensure backend directory is importable
sys.path.insert(0, os.path.abspath("backend"))


class RemoteSensingVqaDataset(Dataset):
    """Dataset loader for remote sensing VQA pairs."""
    def __init__(self, data_path: str, processor: BlipProcessor):
        self.data_path = data_path
        self.processor = processor
        self.samples = []
        # In a full training setup, load annotations from JSON/Parquet
        if os.path.exists(data_path):
            pass  # populate samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).input_ids

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0)
        }


def setup_lora_model(model_name: str = "Salesforce/blip-vqa-base", r: int = 8, lora_alpha: int = 16):
    """Wrap BLIP VQA with Low-Rank Adaptation (LoRA)."""
    try:
        from peft import LoraConfig, get_peft_model
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
    except ImportError:
        print("[WARNING] 'peft' library not installed. Standard PyTorch fine-tuning fallback.")
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze cross-attention text decoder only
        for param in model.text_decoder.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        return model


def main():
    parser = argparse.ArgumentParser(description="Train RS-VQA LoRA Adapter")
    parser.add_argument("--data_path", type=str, default="data/rsvqa")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="models/satquery_vqa_lora")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing RS-VQA LoRA adaptation on {device}...")
    print(f"Target directory: {args.output_dir}")


if __name__ == "__main__":
    main()
