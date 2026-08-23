"""
Dataset for chunked .npy field data with normalization.

Directory: coords.npy, chunk_XXX_data.npy, chunk_XXX_times.npy, stats.npy
Channel order: [alpha.water, Ux, Uy, Uz, p_rgh, nut]
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def compute_stats(data_dir):
    """Per-channel mean/std from all chunks → stats.npy"""
    data_dir = Path(data_dir)
    running_sum = running_sq = None
    total = 0
    for f in sorted(data_dir.glob("chunk_*_data.npy")):
        d = np.load(f).reshape(-1, 6).astype(np.float64)
        if running_sum is None:
            running_sum, running_sq = d.sum(0), (d**2).sum(0)
        else:
            running_sum += d.sum(0); running_sq += (d**2).sum(0)
        total += len(d)
    mean = (running_sum / total).astype(np.float32)
    std = np.sqrt(running_sq / total - mean.astype(np.float64)**2).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    stats = np.stack([mean, std])
    np.save(data_dir / "stats.npy", stats)
    print(f"Stats computed — mean: {mean}, std: {std}")
    return stats


class WaveDataset(Dataset):
    def __init__(self, data_dir, chunk_ids, window=1, rollout_steps=1):
        data_dir = Path(data_dir)
        self.window = window
        self.rollout_steps = rollout_steps
        self.coords = np.load(data_dir / "coords.npy")

        stats_path = data_dir / "stats.npy"
        stats = np.load(stats_path) if stats_path.exists() else compute_stats(data_dir)
        self.mean, self.std = stats[0], stats[1]

        self.frames = [np.load(data_dir / f"chunk_{cid:03d}_data.npy") for cid in chunk_ids]
        # Need window-1 past frames + current + rollout future frames
        self.samples = [(ci, fi)
                        for ci, chunk in enumerate(self.frames)
                        for fi in range(window - 1, chunk.shape[0] - rollout_steps)]

    def normalize(self, x):   return (x - self.mean) / self.std
    def denormalize(self, x):
        m, s = [(torch.tensor(v, device=x.device, dtype=x.dtype) if isinstance(x, torch.Tensor) else v)
                for v in (self.mean, self.std)]
        return x * s + m

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ci, fi = self.samples[idx]
        # Past W frames normalized and concatenated: (N, 6*W)
        fields_window = np.concatenate(
            [self.normalize(self.frames[ci][fi - w]) for w in reversed(range(self.window))],
            axis=-1)  # oldest → newest
        inp = np.concatenate([self.coords, fields_window], axis=-1)  # (N, 3 + 6*W)

        if self.rollout_steps == 1:
            tgt = self.normalize(self.frames[ci][fi + 1])
        else:
            tgt = np.stack([self.normalize(self.frames[ci][fi + s]) for s in range(1, self.rollout_steps + 1)])
        return torch.from_numpy(inp), torch.from_numpy(tgt)
