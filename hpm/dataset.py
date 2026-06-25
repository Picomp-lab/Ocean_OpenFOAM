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


def apply_alpha_weighting(data, weight_u=True, weight_nut=False):
    """Multiply velocity (and optionally nut) by alpha (water fraction) IN PLACE
    of the raw fields, in PHYSICAL space (before normalization).

    Channels: [alpha, Ux, Uy, Uz, p_rgh, nut]
      - U (1,2,3): weighted -> alpha*U if weight_u=True (momentum-density-like)
      - nut (5):   weighted -> alpha*nut only if weight_nut=True
      - alpha (0), p_rgh (4): left untouched

    NOTE: This changes the field distributions, so stats.npy MUST be recomputed
    whenever weight_nut changes. Do not reuse stats across different settings.

    Args:
        data: (..., 6) array, raw physical fields (alpha in [0,1])
    Returns:
        (..., 6) array with alpha-weighting applied (new array; input not mutated)
    """
    out = data.copy()
    alpha = data[..., 0:1]          # physical alpha in [0,1]
    if weight_u:
        out[..., 1:4] = data[..., 1:4] * alpha       # U -> alpha*U
    if weight_nut:
        out[..., 5:6] = data[..., 5:6] * alpha       # nut -> alpha*nut
    return out


def expand_range(r):
    """Chunk range spec -> explicit chunk list.
    [8]   -> [8]            (single chunk)
    [1,7] -> [1,2,...,7]    (continuous interval, inclusive)
    3+ elements -> error (forces single-or-interval, no discrete lists).
    """
    if len(r) == 1:
        return [int(r[0])]
    elif len(r) == 2:
        return list(range(int(r[0]), int(r[1]) + 1))
    else:
        raise ValueError(
            f"chunk range must be [single] or [start,end], got {list(r)} "
            f"(discrete lists not allowed; use continuous interval)")


def chunk_tag(chunk_ids):
    """Chunk list -> compact tag for filenames.
    single  -> 'c8';  interval -> 'c1-7'. Assumes continuous."""
    ids = sorted(int(c) for c in chunk_ids)
    if len(ids) == 1:
        return f"c{ids[0]}"
    return f"c{ids[0]}-{ids[-1]}"


def stats_filename(chunk_ids, weight_u=True, weight_nut=False):
    """Version-specific stats filename:
    stats_c{tag}_u{0|1}_nut{0|1}.npy  (tag = c8 or c1-7).
    Encodes train-chunk set + alpha-weighting so different settings never collide."""
    tag = chunk_tag(chunk_ids)
    return f"stats_{tag}_u{int(bool(weight_u))}_nut{int(bool(weight_nut))}.npy"


def compute_stats(data_dir, chunk_ids, weight_u=True, weight_nut=False):
    """Compute per-channel mean and std from specified chunks (training set only).

    Stats are computed on the alpha-weighted fields (U and optionally nut),
    matching what _build_samples feeds the model.
    """
    data_dir = Path(data_dir)
    running_sum = None
    running_sq_sum = None
    total_count = 0

    for cid in chunk_ids:
        data = np.load(data_dir / f"chunk_{cid:03d}_data.npy")  # (T, N, 6)
        data = apply_alpha_weighting(data, weight_u=weight_u, weight_nut=weight_nut)
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
    fname = stats_filename(chunk_ids, weight_u=weight_u, weight_nut=weight_nut)
    np.save(data_dir / fname, stats)
    print(f"Saved {fname}: mean={mean}, std={std}")
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

    def __init__(self, data_dir, chunk_ids, window=6, stats=None, rollout_steps=4,
                 weight_u_by_alpha=True, weight_nut_by_alpha=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window = window
        self.rollout_steps = rollout_steps
        self.weight_u_by_alpha = weight_u_by_alpha
        self.weight_nut_by_alpha = weight_nut_by_alpha
        self.N = np.load(self.data_dir / "coords.npy").shape[0]

        # Load or compute stats
        if stats is not None:
            self.stats = stats
        else:
            stats_path = self.data_dir / stats_filename(
                chunk_ids, weight_u=weight_u_by_alpha, weight_nut=weight_nut_by_alpha)
            if stats_path.exists():
                self.stats = np.load(stats_path)
            else:
                self.stats = compute_stats(self.data_dir, chunk_ids,
                                           weight_u=weight_u_by_alpha,
                                           weight_nut=weight_nut_by_alpha)
        self.mean = self.stats[0]
        self.std = self.stats[1]

        self.chunks = []
        self.samples = []
        self._build_samples(chunk_ids)

    def _build_samples(self, chunk_ids):
        """Load chunks as torch tensors and build sample index."""
        for cid in chunk_ids:
            data = np.load(self.data_dir / f"chunk_{cid:03d}_data.npy")  # (T, N, 6)
            data = apply_alpha_weighting(data, weight_u=self.weight_u_by_alpha,
                                         weight_nut=self.weight_nut_by_alpha)
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