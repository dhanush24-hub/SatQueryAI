import os
import json
import io
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class LevirCDDataset(Dataset):
    """
    In-memory / fast-reading PyArrow Parquet dataset loader for LEVIR-CD 256x256.
    Ensures deterministic ordering, no data leakage, and exact Daudt preprocessing.
    """

    def __init__(self, parquet_path: str, max_samples: int = None, is_train: bool = False, use_imagenet_norm: bool = False):
        self.is_train = is_train
        self.use_imagenet_norm = use_imagenet_norm

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        table = pq.read_table(parquet_path)
        if max_samples and max_samples < table.num_rows:
            table = table.slice(0, max_samples)

        self.df = table.to_pandas()
        self.length = len(self.df)

        # ImageNet constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract images from bytes
        raw_a = self.df["imageA"][idx]["bytes"]
        raw_b = self.df["imageB"][idx]["bytes"]
        raw_l = self.df["label"][idx]["bytes"]

        img_a = np.array(Image.open(io.BytesIO(raw_a))).astype(np.float32) / 255.0  # (H, W, 3)
        img_b = np.array(Image.open(io.BytesIO(raw_b))).astype(np.float32) / 255.0
        label = (np.array(Image.open(io.BytesIO(raw_l))) > 128).astype(np.float32)  # (H, W)

        # Transpose to (C, H, W)
        t1 = img_a.transpose(2, 0, 1)
        t2 = img_b.transpose(2, 0, 1)

        # Augmentation for training: Random horizontal and vertical flips
        if self.is_train:
            if np.random.rand() > 0.5:
                t1 = np.flip(t1, axis=2).copy()
                t2 = np.flip(t2, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.rand() > 0.5:
                t1 = np.flip(t1, axis=1).copy()
                t2 = np.flip(t2, axis=1).copy()
                label = np.flip(label, axis=0).copy()

        if self.use_imagenet_norm:
            t1 = (t1 - self.mean) / self.std
            t2 = (t2 - self.mean) / self.std

        return (
            torch.from_numpy(t1).float(),
            torch.from_numpy(t2).float(),
            torch.from_numpy(label).long()
        )
