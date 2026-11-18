"""Candidate patches from read-only memory-mapped scan arrays."""
from __future__ import annotations
import hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .geometry import crop


@lru_cache(maxsize=8)
def load_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class CandidateDataset(Dataset):
    def __init__(self, scans, candidates, patch_size=48, augment=False, seed=42):
        self.scans = {s["scan_id"]: s for s in scans}
        if len(self.scans) != len(scans):
            raise ValueError("Duplicate scan IDs")
        self.rows = list(candidates)
        self.size, self.augment, self.seed, self.epoch = patch_size, augment, seed, 0
        for row in self.rows:
            if row["scan_id"] not in self.scans:
                raise ValueError("Candidate outside scan population")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        scan = self.scans[row["scan_id"]]
        image, start = crop(load_array(scan["image_path"]), row["zyx"], self.size)
        valid, _ = crop(load_array(scan["valid_path"]), row["zyx"], self.size, False)
        image, valid = image.copy(), valid.copy()
        if self.augment:
            identity = f"{self.seed}:{self.epoch}:{row['id']}".encode()
            seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            if rng.random() < 0.5:
                image, valid = image[:, :, ::-1], valid[:, :, ::-1]
            if rng.random() < 0.5:
                k = int(rng.integers(0, 4))
                image, valid = np.rot90(image, k, (1, 2)), np.rot90(valid, k, (1, 2))
            if rng.random() < 0.2:
                image = np.clip(image + rng.normal(0, 0.01, image.shape)*valid, 0, 1)
        image = np.where(valid, image, 0).astype(np.float32)
        return {"image": torch.from_numpy(image.copy())[None],
                "label": int(row.get("label", -1)), "index": index,
                "valid": torch.from_numpy(valid.copy())[None], "start_zyx": start}
