"""
Dataset for HPM time-stepping prediction with multi-step rollout support.

Schema-driven: all channel selection, alpha-weighting and stats naming come
from ChannelSchema (schema.py). Disk files stay 6-channel; channels are
selected BY NAME at load time — no data regeneration for ablations.

Returns temporal window + R consecutive future frames for multi-step loss.
Coords are NOT returned per-sample — use load_coords() separately.
Data stored as torch.Tensor for shared memory with DataLoader workers.

Data layout on disk:
    data_dir/
        coords.npy              # (N, 3) float32
        chunk_000_data.npy      # (T_chunk, N, 6) float32 — DISK_CHANNELS order
        chunk_000_times.npy     # (T_chunk,) float64
        ...
        stats_{tag}_{signature}.npy   # (2, field_dim) — [mean, std] per channel
        lbo/
            lbo_eigenvectors.npy  # (N, K) float32
            lbo_eigenvalues.npy   # (K,) float32
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from schema import ChannelSchema, DISK_CHANNELS


# ============================================================
# Chunk range helpers (unchanged)
# ============================================================

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


# ============================================================
# Channel-aware loading + alpha-weighting
# ============================================================

def load_chunk(data_dir, cid, schema):
    """Load one chunk, select schema channels by name, apply alpha-weighting.

    Returns (T, N, field_dim) float array in the SAME physical space the
    model trains in (alpha-weighted where schema says so).
    """
    data_dir = Path(data_dir)
    data = np.load(data_dir / f"chunk_{cid:03d}_data.npy")   # (T, N, 6)
    assert data.shape[-1] == len(DISK_CHANNELS), (
        f"chunk_{cid:03d}_data.npy has {data.shape[-1]} channels, expected "
        f"{len(DISK_CHANNELS)} ({DISK_CHANNELS}) — disk layout mismatch")
    data = data[..., schema.disk_indices]                     # select by name
    return apply_alpha_weighting(data, schema)


def apply_alpha_weighting(data, schema):
    """Multiply alpha_weighted channels by alpha (water fraction) in PHYSICAL
    space (before normalization). alpha channel located BY NAME via schema.

    Args:
        data: (..., field_dim) array in schema channel order, alpha in [0,1]
    Returns:
        new array with alpha-weighting applied (input not mutated)
    """
    assert data.shape[-1] == schema.field_dim, (
        f"data has {data.shape[-1]} channels, schema expects "
        f"{schema.field_dim} ({schema.names})")
    out = data.copy()
    alpha = data[..., schema.alpha_idx:schema.alpha_idx + 1]  # (..., 1)
    for i, w in enumerate(schema.alpha_weighted):
        if w:
            out[..., i:i + 1] = data[..., i:i + 1] * alpha
    return out


# ============================================================
# Stats — versioned by chunk set + channel signature
# ============================================================

def stats_filename(chunk_ids, schema):
    """stats_{tag}_{signature}.npy — encodes train-chunk set + channel set +
    alpha-weighting, so different settings can never collide."""
    return f"stats_{chunk_tag(chunk_ids)}_{schema.signature()}.npy"


def _legacy_stats_filename(chunk_ids, schema):
    """Old naming (pre-schema): stats_{tag}_u{0|1}_nut{0|1}.npy.
    Only meaningful for the exact legacy 6-channel layout."""
    wu = int(schema.alpha_weighted[schema.names.index("Ux")])
    wn = int(schema.alpha_weighted[schema.names.index("nut")])
    return f"stats_{chunk_tag(chunk_ids)}_u{wu}_nut{wn}.npy"


def compute_stats(data_dir, chunk_ids, schema):
    """Compute per-channel mean/std from specified chunks (training set only),
    on the alpha-weighted, schema-selected fields — matching what the model
    is fed. Saves under the versioned filename and returns the array."""
    data_dir = Path(data_dir)
    running_sum = None
    running_sq_sum = None
    total_count = 0

    for cid in chunk_ids:
        data = load_chunk(data_dir, cid, schema)              # (T, N, C)
        T, N, C = data.shape
        flat = data.reshape(-1, C).astype(np.float64)
        if running_sum is None:
            running_sum = np.zeros(C, dtype=np.float64)
            running_sq_sum = np.zeros(C, dtype=np.float64)
        running_sum += flat.sum(axis=0)
        running_sq_sum += (flat ** 2).sum(axis=0)
        total_count += T * N

    mean = running_sum / total_count
    std = np.sqrt(running_sq_sum / total_count - mean ** 2)
    std = np.maximum(std, 1e-8)

    stats = np.stack([mean, std], axis=0).astype(np.float32)  # (2, C)
    fname = stats_filename(chunk_ids, schema)
    np.save(data_dir / fname, stats)
    print(f"Saved {fname}: mean={mean}, std={std}")
    return stats


def resolve_stats(data_dir, chunk_ids, schema, verbose=True):
    """Load stats if present, else compute. Resolution order:
      1. new versioned name  stats_{tag}_{signature}.npy
      2. legacy name         stats_{tag}_u{0|1}_nut{0|1}.npy
         (read-only fallback; only if schema is the exact legacy layout —
         numerically identical, avoids recomputation for old runs)
      3. compute from chunks
    Always validates shape against schema.field_dim (catches stale files)."""
    data_dir = Path(data_dir)

    new_path = data_dir / stats_filename(chunk_ids, schema)
    if new_path.exists():
        stats = np.load(new_path)
        if verbose:
            print(f"Loaded {new_path.name}: mean={stats[0]}, std={stats[1]}")
    elif schema.is_legacy_layout() and \
            (data_dir / _legacy_stats_filename(chunk_ids, schema)).exists():
        legacy_path = data_dir / _legacy_stats_filename(chunk_ids, schema)
        stats = np.load(legacy_path)
        if verbose:
            print(f"Loaded legacy stats {legacy_path.name} "
                  f"(identical computation, old naming)")
    else:
        if verbose:
            print(f"Computing dataset statistics -> {new_path.name}")
        stats = compute_stats(data_dir, chunk_ids, schema)

    assert stats.shape == (2, schema.field_dim), (
        f"stats shape {stats.shape} != (2, {schema.field_dim}) — stale stats "
        f"file for a different channel set? Delete and recompute.")
    return stats


# ============================================================
# Coords
# ============================================================

def load_coords(data_dir):
    """Load and normalize coordinates. Call once, send to GPU outside loop."""
    data_dir = Path(data_dir)
    coords = np.load(data_dir / "coords.npy").astype(np.float32)
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    coord_range = np.maximum(coord_max - coord_min, 1e-8)
    coords_norm = (coords - coord_min) / coord_range
    return torch.from_numpy(coords_norm)  # (N, 3)


# ============================================================
# Dataset
# ============================================================

class WaveDataset(Dataset):
    """
    Dataset yielding (window_fields, future_frames) tuples.

    Each sample:
        window_fields: (N, W*F) float32 — W consecutive normalized frames, flattened
        future_frames: (R, N, F) float32 — R future frames (normalized, absolute)

    F = schema.field_dim. The loss function handles rollout, delta computation
    and per-step accumulation (via schema.advance_window).

    Args:
        schema:        ChannelSchema — channel selection / weighting / naming
        stats:         (2, F) [mean, std]; if None, resolved via resolve_stats
        rollout_steps: number of future frames to return (default 4)
    """

    def __init__(self, data_dir, chunk_ids, window, schema, stats=None,
                 rollout_steps=4):
        super().__init__()
        assert isinstance(schema, ChannelSchema), \
            "WaveDataset now requires a ChannelSchema (see schema.py)"
        self.data_dir = Path(data_dir)
        self.window = window
        self.schema = schema
        self.rollout_steps = rollout_steps
        self.N = np.load(self.data_dir / "coords.npy").shape[0]

        if stats is None:
            stats = resolve_stats(self.data_dir, chunk_ids, schema)
        self.stats = stats
        self.mean = self.stats[0]
        self.std = self.stats[1]

        self.chunks = []
        self.samples = []
        self._build_samples(chunk_ids)

    def _build_samples(self, chunk_ids):
        """Load chunks as torch tensors and build sample index."""
        for cid in chunk_ids:
            data = load_chunk(self.data_dir, cid, self.schema)  # (T, N, F)
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
        window_frames = chunk[t - self.window:t]     # (W, N, F)

        # Future R frames: [t, t+1, ..., t+R-1] — absolute values
        future = chunk[t:t + self.rollout_steps]     # (R, N, F)

        # Flatten window: (W, N, F) -> (N, W*F)
        window_flat = window_frames.permute(1, 0, 2).reshape(self.N, -1)

        return window_flat, future