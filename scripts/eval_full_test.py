import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import numpy as np
import io, os, sys, time, json
from PIL import Image
from safetensors import safe_open
from app.services.models.attention_change import AttentionChangeNet

class LevirParquetDataset(Dataset):
    def __init__(self, parquet_path):
        table = pq.read_table(parquet_path)
        self.df = table.to_pandas()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rawA = self.df['imageA'][idx]['bytes']
        rawB = self.df['imageB'][idx]['bytes']
        rawL = self.df['label'][idx]['bytes']

        imgA = np.array(Image.open(io.BytesIO(rawA))).astype(np.float32) / 255.0
        imgB = np.array(Image.open(io.BytesIO(rawB))).astype(np.float32) / 255.0
        gt = (np.array(Image.open(io.BytesIO(rawL))) > 128).astype(np.uint8)

        t1 = (imgA.transpose(2, 0, 1) - self.mean) / self.std
        t2 = (imgB.transpose(2, 0, 1) - self.mean) / self.std

        return torch.from_numpy(t1).float(), torch.from_numpy(t2).float(), torch.from_numpy(gt).long()

def evaluate():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    dataset = LevirParquetDataset("data/levircd/test.parquet")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model = AttentionChangeNet(num_classes=2, pretrained=False)
    with safe_open("models/satquery_change_v1/attention_best.safetensors", framework="pt") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    model = model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    per_img_prec = []
    per_img_rec = []
    per_img_f1 = []
    per_img_iou = []

    # Separate non-empty stats
    non_empty_f1 = []
    non_empty_iou = []
    empty_tiles_count = 0

    latencies = []
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (t1, t2, gt) in enumerate(loader):
            b_sz = t1.size(0)
            t1, t2 = t1.to(device), t2.to(device)

            t_inf_start = time.time()
            logits = model(t1, t2)
            # Logits are LogSoftmax
            p_ch = torch.exp(logits[:, 1]).cpu().numpy()  # (B, H, W)
            inf_time = (time.time() - t_inf_start) * 1000.0 / b_sz
            latencies.extend([inf_time] * b_sz)

            gt_np = gt.numpy()  # (B, H, W)

            for i in range(b_sz):
                pred = (p_ch[i] >= 0.50)
                gt_i = (gt_np[i] == 1)

                tp = np.sum(pred & gt_i)
                fp = np.sum(pred & ~gt_i)
                fn = np.sum(~pred & gt_i)
                tn = np.sum(~pred & ~gt_i)

                total_tp += int(tp)
                total_fp += int(fp)
                total_fn += int(fn)
                total_tn += int(tn)

                gt_sum = int(np.sum(gt_i))
                pred_sum = int(np.sum(pred))

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

                per_img_prec.append(prec)
                per_img_rec.append(rec)
                per_img_f1.append(f1)
                per_img_iou.append(iou)

                if gt_sum == 0:
                    empty_tiles_count += 1
                else:
                    non_empty_f1.append(f1)
                    non_empty_iou.append(iou)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.time() - t_start
                print(f"Processed {(batch_idx+1)*loader.batch_size}/{len(dataset)} samples ({elapsed:.1f}s)", flush=True)

    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    g_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
    g_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    total_pixels = total_tp + total_fp + total_fn + total_tn
    prev = (total_tp + total_fn) / total_pixels

    results = {
        "dataset": "LEVIR-CD Held-Out Test Set",
        "checkpoint": "models/satquery_change_v1/attention_best.safetensors",
        "architecture": "AttentionChangeNet (ResNet-18 + Spatial Difference Attention)",
        "device": device,
        "total_test_pairs": len(dataset),
        "empty_change_tiles": empty_tiles_count,
        "non_empty_change_tiles": len(non_empty_f1),
        "changed_pixel_prevalence": float(prev),
        "global_metrics": {
            "precision": float(g_prec),
            "recall": float(g_rec),
            "f1": float(g_f1),
            "iou": float(g_iou)
        },
        "macro_metrics": {
            "precision": float(np.mean(per_img_prec)),
            "recall": float(np.mean(per_img_rec)),
            "f1": float(np.mean(per_img_f1)),
            "iou": float(np.mean(per_img_iou))
        },
        "secondary_diagnostic_non_empty_tiles": {
            "macro_f1": float(np.mean(non_empty_f1)),
            "macro_iou": float(np.mean(non_empty_iou))
        },
        "confusion_counts": {
            "TP": int(total_tp),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "TN": int(total_tn)
        },
        "latency_ms_per_tile": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies))
        }
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/FINAL_HELDOUT_METRICS.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ FINAL HELD-OUT TEST EVALUATION ================", flush=True)
    print(f"Total Test Pairs:         {len(dataset)}", flush=True)
    print(f"Empty Change Tiles:       {empty_tiles_count} ({empty_tiles_count/len(dataset)*100:.1f}%)", flush=True)
    print(f"Changed Pixel Prevalence: {prev*100:.2f}%", flush=True)
    print(f"--- GLOBAL (PIXEL-ACCUMULATED) METRICS ---", flush=True)
    print(f"Global Precision:         {g_prec:.4f}", flush=True)
    print(f"Global Recall:            {g_rec:.4f}", flush=True)
    print(f"Global F1:                {g_f1:.4f}", flush=True)
    print(f"Global IoU:               {g_iou:.4f}", flush=True)
    print(f"--- MACRO (TILE-AVERAGED) METRICS ---", flush=True)
    print(f"Macro Precision:          {np.mean(per_img_prec):.4f}", flush=True)
    print(f"Macro Recall:             {np.mean(per_img_rec):.4f}", flush=True)
    print(f"Macro F1:                 {np.mean(per_img_f1):.4f}", flush=True)
    print(f"Macro IoU:                {np.mean(per_img_iou):.4f}", flush=True)
    print(f"--- SECONDARY NON-EMPTY TILES DIAGNOSTIC ---", flush=True)
    print(f"Non-Empty Macro F1:       {np.mean(non_empty_f1):.4f}", flush=True)
    print(f"Non-Empty Macro IoU:      {np.mean(non_empty_iou):.4f}", flush=True)
    print(f"--- LATENCY ---", flush=True)
    print(f"Mean Latency per Tile:    {np.mean(latencies):.2f} ms", flush=True)

if __name__ == "__main__":
    evaluate()
