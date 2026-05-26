"""
Dataset for HPM time-stepping prediction.

Loads chunked .npy files from cropped OpenFOAM data.
Returns temporal windows of W consecutive frames with normalization.

Coords are NOT returned per-sample (they're identical for every sample
on a fixed mesh). Instead, load coords once via get_coords() and send
to GPU outside the training loop.

Data are stored as torch.Tensor (not numpy) to avoid Copy-on-Write
memory explosion with num_workers > 0.

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
    Dataset yielding (window_fields, target_delta) tuples.

    Each sample:
        window_fields: (N, W*6) float32 — W consecutive normalized frames, flattened
        target_delta:  (N, 6) float32   — normalized delta (next - current)

    Coords are NOT included — use load_coords() separately.
    """

    def __init__(self, data_dir, chunk_ids, window=6, stats=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window = window
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
        self.mean = self.stats[0]  # (6,)
        self.std = self.stats[1]   # (6,)

        # Load and normalize all chunks, store as torch.Tensor for
        # shared memory with DataLoader workers (avoids numpy CoW)
        self.chunks = []
        self.samples = []
        self._build_samples(chunk_ids)

    def _build_samples(self, chunk_ids):
        """Load chunks as torch tensors and build sample index."""
        for cid in chunk_ids:
            data = np.load(self.data_dir / f"chunk_{cid:03d}_data.npy")  # (T, N, 6)
            data_norm = ((data - self.mean) / self.std).astype(np.float32)
            # Convert to torch.Tensor — uses shared memory across workers
            self.chunks.append(torch.from_numpy(data_norm))

            T = data.shape[0]
            for t in range(self.window, T):
                self.samples.append((len(self.chunks) - 1, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, t = self.samples[idx]
        chunk = self.chunks[chunk_idx]

        # Window: frames [t-W, ..., t-1], target: frame t
        window_frames = chunk[t - self.window:t]    # (W, N, 6)
        target_frame = chunk[t]                      # (N, 6)
        current_frame = chunk[t - 1]                 # (N, 6)

        # Flatten window: (W, N, 6) -> (N, W*6)
        window_flat = window_frames.permute(1, 0, 2).reshape(self.N, -1)

        # Target: delta (residual learning)
        delta = target_frame - current_frame          # (N, 6)

        return window_flat, delta