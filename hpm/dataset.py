"""
Dataset for HPM time-stepping prediction with multi-step rollout support.

Returns temporal window + R consecutive future frames for multi-step loss.
Coords are NOT returned per-sample — use load_coords() separately.
Data stored as torch.Tensor for shared memory with DataLoader workers.

Data layout on disk:
    data_dir/
        coords.npy              # (N, 3) float32
        chunk_000_data.npy      # (T_chunk, N, 6) float32
        chunk_000_times.npy     # (T_chunk,) float64
        ...
        stats.npy               # (2, 6) float32 — [mean, std] per channel
        lbo/
            lbo_eigenvectors.npy  # (N, K) float32
            lbo_eigenvalues.npy   # (K,) float32
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def compute_stats(data_dir, chunk_ids):
    """Compute per-channel mean and std from specified chunks (training set only)."""
    data_dir = Path(data_dir)
    running_sum = None
    running_sq_sum = None
    total_count = 0

    for cid in chunk_ids:
        data = np.load(data_dir / f"chunk_{cid:03d}_data.npy")  # (T, N, 6)
        T, N, C = data.shape
        n = T * N

        if running_sum is None:
            running_sum = np.zeros(C, dtype=np.float64)
            running_sq_sum = np.zeros(C, dtype=np.float64)

        flat = data.reshape(-1, C).astype(np.float64)
        running_sum += flat.sum(axis=0)
        running_sq_sum += (flat ** 2).sum(axis=0)
        total_count += n

    mean = running_sum / total_count
    std = np.sqrt(running_sq_sum / total_count - mean ** 2)
    std = np.maximum(std, 1e-8)

    stats = np.stack([mean, std], axis=0).astype(np.float32)  # (2, 6)
    np.save(data_dir / "stats.npy", stats)
    print(f"Saved stats.npy: mean={mean}, std={std}")
    return stats


def load_coords(data_dir):
    """Load and normalize coordinates. Call once, send to GPU outside loop."""
    data_dir = Path(data_dir)
    coords = np.load(data_dir / "coords.npy").astype(np.float32)
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    coord_range = np.maximum(coord_max - coord_min, 1e-8)
    coords_norm = (coords - coord_min) / coord_range
    return torch.from_numpy(coords_norm)  # (N, 3)


class WaveDataset(Dataset):
    """
    Dataset yielding (window_fields, future_frames) tuples.

    Each sample:
        window_fields: (N, W*6) float32 — W consecutive normalized frames, flattened
        future_frames: (R, N, 6) float32 — R future frames (normalized, absolute values)

    During training, the loss function handles:
      - Autoregressive rollout for R steps
      - Delta computation (residual learning) at each step
      - Accumulation of per-step losses

    Args:
        rollout_steps: number of future frames to return (default 4)
    """

    def __init__(self, data_dir, chunk_ids, window=6, stats=None, rollout_steps=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window = window
        self.rollout_steps = rollout_steps
        self.N = np.load(self.data_dir / "coords.npy").shape[0]

        # Load or compute stats
        if stats is not None:
            self.stats = stats
        else:
            stats_path = self.data_dir / "stats.npy"
            if stats_path.exists():
                self.stats = np.load(stats_path)
            else:
                self.stats = compute_stats(self.data_dir, chunk_ids)
        self.mean = self.stats[0]
        self.std = self.stats[1]

        self.chunks = []
        self.samples = []
        self._build_samples(chunk_ids)

    def _build_samples(self, chunk_ids):
        """Load chunks as torch tensors and build sample index."""
        for cid in chunk_ids:
            data = np.load(self.data_dir / f"chunk_{cid:03d}_data.npy")  # (T, N, 6)
            data_norm = ((data - self.mean) / self.std).astype(np.float32)
            # .clone() detaches from numpy memory, .share_memory_() makes
            # it safe for multi-worker DataLoader (avoids CoW on fork)
            tensor = torch.from_numpy(data_norm).clone().share_memory_()
            self.chunks.append(tensor)

            T = data.shape[0]
            # Need W frames for input + R frames for targets
            for t in range(self.window, T - self.rollout_steps + 1):
                self.samples.append((len(self.chunks) - 1, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, t = self.samples[idx]
        chunk = self.chunks[chunk_idx]

        # Window: frames [t-W, ..., t-1]
        window_frames = chunk[t - self.window:t]    # (W, N, 6)

        # Future R frames: [t, t+1, ..., t+R-1] — absolute values
        future = chunk[t:t + self.rollout_steps]     # (R, N, 6)

        # Flatten window: (W, N, 6) -> (N, W*6)
        window_flat = window_frames.permute(1, 0, 2).reshape(self.N, -1)

        return window_flat, future