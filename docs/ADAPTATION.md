# Remote-Sensing Domain Adaptation Specification

**Dataset:** BigEarthNet.txt / BigEarthNet-S2  
**Task:** Remote-Sensing Multimodal Domain Adaptation  
**Status:** **PIPELINE DELIVERED — TRAINING NOT YET COMPLETED (Hardware Blocked)**

---

## 1. Dataset Acquisition & Attribution

- **Dataset**: BigEarthNet-S2 (Sentinel-2 benchmark consisting of 590,326 multi-spectral image tiles across 10 European countries).
- **Attribution**: G. Sumbul, M. Charfuelan, B. Demir, V. Markl, *"BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding"*, IGARSS 2019.
- **License**: Community Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive-1.0).
- **Split Lists**: Official `BigEarthNet.txt` split files (`train.csv`, `val.csv`, `test.csv`).

### Download Instructions
```bash
# Download official BigEarthNet split files
mkdir -p data/bigearthnet
wget https://bigearth.net/static/documents/splits.tar.gz -O data/bigearthnet/splits.tar.gz
tar -xzf data/bigearthnet/splits.tar.gz -C data/bigearthnet/
```

---

## 2. Adaptation Architecture: PEFT / LoRA

Rather than updating all model parameters (which requires >24 GB VRAM for full backpropagation), we use Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA).

### Hyperparameters
- **Base Architecture**: Vision Transformer + Text Projection
- **LoRA Rank ($r$)**: 16
- **LoRA Alpha ($\alpha$)**: 32
- **LoRA Dropout**: 0.05
- **Target Modules**: `q_proj`, `v_proj`, `vision_proj`, `text_proj`
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.01)
- **Learning Rate**: $1 \times 10^{-4}$ with cosine annealing
- **Batch Size**: 16 per device (with gradient accumulation steps = 4, effective batch size = 64)
- **Precision**: Mixed precision (fp16 / bf16)

---

## 3. Training Script

The adaptation script is located at:
`backend/app/services/models/adaptation/train_adaptation.py`

### Running the Adaptation
```bash
# Requires CUDA GPU with >= 16 GB VRAM
PYTHONPATH=backend .venv/bin/python backend/app/services/models/adaptation/train_adaptation.py \
  --data_dir data/bigearthnet \
  --output_dir checkpoints/bigearthnet_lora \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --seed 42
```

---

## 4. Evaluation Script

The evaluation script is located at:
`backend/app/services/models/adaptation/evaluate_adaptation.py`

### Evaluation Command
```bash
PYTHONPATH=backend .venv/bin/python backend/app/services/models/adaptation/evaluate_adaptation.py \
  --checkpoint checkpoints/bigearthnet_lora \
  --split val
```

---

## 5. Compliance & Integrity Statement

In strict compliance with SIH 2026 rules:
1. Benchmark test splits are **never** used during training.
2. Loading an off-the-shelf pretrained foundation model is **not** claimed as fine-tuning.
3. The training pipeline is fully reproducible and delivered in the repository. Its execution status is truthfully reported as **NOT YET COMPLETED** on the current 8 GB development machine until connected to adequate GPU compute.
