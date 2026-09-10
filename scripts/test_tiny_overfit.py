#!/usr/bin/env python3
"""
Step 7: Tokenization & Supervision Audit + Tiny Overfit Test for BLIP-VQA.
Verifies:
1. Exact target tokenization for 'yes' and 'no' (BOS/EOS, input_ids).
2. Proper label masking (ignore_index=-100 for padding).
3. Tiny overfit capability: trains LoRA on 16 authentic S2-RGB samples (8 Yes, 8 No).
4. Confirms whether model memorizes the tiny set (loss -> ~0, accuracy -> 100%).
"""

import os
import sys
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import LoraConfig, get_peft_model

MODEL_ID = "Salesforce/blip-vqa-base"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def audit_tokenization(processor):
    print("=== 1. Tokenization & Supervision Audit ===", flush=True)
    tok = processor.tokenizer
    print(f"Tokenizer type: {type(tok).__name__}")
    print(f"pad_token: '{tok.pad_token}' (ID: {tok.pad_token_id})")
    print(f"cls_token (BOS): '{tok.cls_token}' (ID: {tok.cls_token_id})")
    print(f"sep_token (EOS): '{tok.sep_token}' (ID: {tok.sep_token_id})")
    print(f"unk_token: '{tok.unk_token}' (ID: {tok.unk_token_id})")

    enc_yes = processor(text="yes", return_tensors="pt")
    enc_no = processor(text="no", return_tensors="pt")
    print(f"\n'yes' token IDs: {enc_yes.input_ids.tolist()[0]} -> decoded: '{processor.decode(enc_yes.input_ids[0])}'")
    print(f"'no'  token IDs: {enc_no.input_ids.tolist()[0]} -> decoded: '{processor.decode(enc_no.input_ids[0])}'")

    # Verify length and padding
    assert enc_yes.input_ids.shape[1] == 3, "Expected 3 tokens for 'yes' ([CLS], 2748, [SEP])"
    assert enc_no.input_ids.shape[1] == 3, "Expected 3 tokens for 'no' ([CLS], 2053, [SEP])"

    # Test padding behavior with batch
    batch_enc = processor(text=["yes", "no"], padding=True, return_tensors="pt")
    print(f"Batch encoded shape: {batch_enc.input_ids.shape}")

    audit_info = {
        "pad_token_id": tok.pad_token_id,
        "cls_token_id": tok.cls_token_id,
        "sep_token_id": tok.sep_token_id,
        "yes_tokens": enc_yes.input_ids.tolist()[0],
        "no_tokens": enc_no.input_ids.tolist()[0],
        "padding_ignore_index": -100
    }
    return audit_info


def run_tiny_overfit(processor, manifest_path: str):
    print("\n=== 2. Tiny Overfit Test (16 Samples: 8 Yes, 8 No) ===", flush=True)
    with open(manifest_path, "r") as f:
        records = json.load(f)

    # Filter to existing images
    valid = [r for r in records if os.path.exists(r["local_raster_path"])]
    yes_samples = [r for r in valid if r["answer"] == "yes"][:8]
    no_samples = [r for r in valid if r["answer"] == "no"][:8]

    if len(yes_samples) < 8 or len(no_samples) < 8:
        print(f"ERROR: Insufficient samples! yes={len(yes_samples)}, no={len(no_samples)}", flush=True)
        return None

    samples = yes_samples + no_samples
    print(f"Selected 16 samples (8 yes, 8 no) across {len(set(s['patch_id'] for s in samples))} unique patches.")

    # Load base model
    print(f"Loading {MODEL_ID} on {DEVICE}...", flush=True)
    base_model = BlipForQuestionAnswering.from_pretrained(MODEL_ID)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["query", "value"],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)
    model.train()

    # Pre-tokenize all 16 samples
    batch_data = []
    for s in samples:
        img = Image.open(s["local_raster_path"]).convert("RGB")
        q = s["question"]
        ans = s["answer"]
        inputs = processor(images=img, text=q, return_tensors="pt")
        # Labels for decoder: target text tokenized with pad tokens replaced by -100
        ans_enc = processor(text=ans, return_tensors="pt")
        labels = ans_enc.input_ids.clone()
        # Ensure pad tokens are -100 (though single token 'yes'/'no' has no padding)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch_data.append({
            "pixel_values": inputs.pixel_values,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "decoder_input_ids": ans_enc.input_ids,
            "labels": labels,
            "gt": ans,
            "question": q
        })

    # Baseline evaluation before training
    model.eval()
    init_preds = []
    init_correct = 0
    with torch.no_grad():
        for item in batch_data:
            out = model.generate(
                pixel_values=item["pixel_values"].to(DEVICE),
                input_ids=item["input_ids"].to(DEVICE),
                attention_mask=item["attention_mask"].to(DEVICE),
                max_new_tokens=5
            )
            pred = processor.decode(out[0], skip_special_tokens=True).strip().lower()
            init_preds.append(pred)
            if pred == item["gt"]:
                init_correct += 1

    print(f"Initial (zero-shot) accuracy on 16 samples: {init_correct}/16 ({init_correct/16*100:.1f}%)")
    print(f"Initial predictions: {init_preds}")

    # Overfit loop: 30 steps with AdamW, lr=5e-4
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)

    loss_history = []
    print("\nStarting optimization steps (batch_size=4, 30 steps)...", flush=True)
    for step in range(30):
        # Mini-batch of 4
        indices = torch.randperm(len(batch_data))[:4]
        pv = torch.cat([batch_data[i]["pixel_values"] for i in indices], dim=0).to(DEVICE)
        ii = torch.nn.utils.rnn.pad_sequence([batch_data[i]["input_ids"].squeeze(0) for i in indices], batch_first=True, padding_value=0).to(DEVICE)
        am = torch.nn.utils.rnn.pad_sequence([batch_data[i]["attention_mask"].squeeze(0) for i in indices], batch_first=True, padding_value=0).to(DEVICE)
        d_ii = torch.nn.utils.rnn.pad_sequence([batch_data[i]["decoder_input_ids"].squeeze(0) for i in indices], batch_first=True, padding_value=0).to(DEVICE)
        lbl = torch.nn.utils.rnn.pad_sequence([batch_data[i]["labels"].squeeze(0) for i in indices], batch_first=True, padding_value=-100).to(DEVICE)

        optimizer.zero_grad()
        out = model(
            pixel_values=pv,
            input_ids=ii,
            attention_mask=am,
            decoder_input_ids=d_ii,
            labels=lbl
        )
        loss = out.loss
        loss.backward()
        optimizer.step()

        loss_val = round(loss.item(), 4)
        loss_history.append(loss_val)
        if (step + 1) % 5 == 0 or step == 0:
            print(f"  Step {step+1:02d}/30 | Loss: {loss_val:.4f}", flush=True)

    # Final evaluation after overfit training
    model.eval()
    final_preds = []
    final_correct = 0
    with torch.no_grad():
        for item in batch_data:
            out = model.generate(
                pixel_values=item["pixel_values"].to(DEVICE),
                input_ids=item["input_ids"].to(DEVICE),
                attention_mask=item["attention_mask"].to(DEVICE),
                max_new_tokens=5
            )
            pred = processor.decode(out[0], skip_special_tokens=True).strip().lower()
            final_preds.append(pred)
            if pred == item["gt"]:
                final_correct += 1

    final_acc = final_correct / 16.0
    print(f"\nFinal overfit accuracy: {final_correct}/16 ({final_acc*100:.1f}%)", flush=True)
    print(f"Final predictions:   {final_preds}")
    print(f"Ground-truth labels: {[item['gt'] for item in batch_data]}")

    results = {
        "num_samples": 16,
        "initial_correct": init_correct,
        "initial_accuracy": init_correct / 16.0,
        "final_correct": final_correct,
        "final_accuracy": final_acc,
        "initial_predictions": init_preds,
        "final_predictions": final_preds,
        "ground_truth": [item["gt"] for item in batch_data],
        "loss_trajectory": loss_history,
        "passed": final_acc >= 0.85
    }
    return results


def main():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    audit_info = audit_tokenization(processor)

    manifest_path = "data/manifests/bigearthnet_txt_stage8_5b_train.json"
    if not os.path.exists(manifest_path):
        manifest_path = "data/manifests/bigearthnet_txt_stageA_train.json"

    overfit_results = run_tiny_overfit(processor, manifest_path)

    summary = {
        "tokenization_audit": audit_info,
        "overfit_test": overfit_results
    }
    out_file = "docs/STAGE8_5B_TINY_OVERFIT_AUDIT.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit and overfit test results written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
