import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from app.services.data.levircd_dataset import LevirCDDataset
from app.services.models.siamunet_diff import SiamUnet_diff
from app.services.models.fc_ef import FC_EF
from app.services.models.attention_change import AttentionChangeNet
from app.services.models.losses import CombinedLoss


def evaluate_val(model, val_loader, device, threshold=0.5):
    model.eval()
    precisions, recalls, f1s, ious = [], [], [], []
    with torch.no_grad():
        for t1, t2, label in val_loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            gt = label.numpy().astype(bool)

            out = model(t1, t2)
            probs = torch.exp(out[:, 1]).cpu().numpy()

            for b in range(probs.shape[0]):
                pred = probs[b] >= threshold
                g = gt[b]
                tp = np.sum(pred & g)
                fp = np.sum(pred & ~g)
                fn = np.sum(~pred & g)

                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
                ious.append(iou)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "iou": float(np.mean(ious))
    }


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    use_imagenet = (args.model_type == "attention")
    train_ds = LevirCDDataset(
        "data/levircd/train.parquet",
        max_samples=args.max_train_samples,
        is_train=True,
        use_imagenet_norm=use_imagenet
    )
    val_ds = LevirCDDataset(
        "data/levircd/val.parquet",
        max_samples=args.max_val_samples,
        is_train=False,
        use_imagenet_norm=use_imagenet
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Training {args.model_type}: {len(train_ds)} train pairs, {len(val_ds)} val pairs, batch={args.batch_size}")

    if args.model_type == "siamunet_diff":
        model = SiamUnet_diff(input_nbr=3, label_nbr=2).to(device)
    elif args.model_type == "fc_ef":
        model = FC_EF(in_channels=6, num_classes=2).to(device)
    elif args.model_type == "attention":
        model = AttentionChangeNet(num_classes=2, pretrained=True).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_metrics = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        step = 0
        t_start = time.time()

        for t1, t2, label in train_loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(t1, t2)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step += 1

        scheduler.step()
        avg_loss = train_loss / max(1, step)
        val_res = evaluate_val(model, val_loader, device, threshold=0.5)
        epoch_sec = time.time() - t_start

        print(
            f"Epoch [{epoch}/{args.epochs}] ({epoch_sec:.1f}s) - "
            f"Loss: {avg_loss:.4f} | "
            f"Val F1: {val_res['f1']:.4f} | "
            f"IoU: {val_res['iou']:.4f} | "
            f"Prec: {val_res['precision']:.4f} | "
            f"Rec: {val_res['recall']:.4f}"
        )

        if val_res["f1"] > best_val_f1:
            best_val_f1 = val_res["f1"]
            best_metrics = val_res
            # Save best checkpoint
            ckpt_path = os.path.join(args.output_dir, f"{args.model_type}_best.safetensors")
            state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(state_dict, ckpt_path)
            print(f"  --> Saved new best {args.model_type} to {ckpt_path} (F1={best_val_f1:.4f})")

    print(f"\nFinal Best Val Metrics for {args.model_type}: {best_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["siamunet_diff", "fc_ef", "attention"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_val_samples", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="models/satquery_change_v1")
    args = parser.parse_args()
    train(args)
