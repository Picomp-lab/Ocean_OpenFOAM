"""
dataset.py — PyTorch Dataset for FNO/PINO wave prediction.

Sliding window approach:
    - Input: n_input_steps consecutive timesteps (e.g. 5 steps = 0.25s)
    - Target: the next timestep after the input window
    - Window slides by 1 timestep each sample

    Example (n_input_steps=5):
        sample 0: input=[t(0),t(1),t(2),t(3),t(4)],  target=t(5)
        sample 1: input=[t(1),t(2),t(3),t(4),t(5)],  target=t(6)
        sample 2: input=[t(2),t(3),t(4),t(5),t(6)],  target=t(7)
        ...
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WaveDataset(Dataset):
    """
    Dataset for field-to-field prediction with multi-step input window.

    Input: n_input_steps consecutive fields stacked as channels + terrain mask
        shape: (n_input_steps * n_fields + 1, nx, nz)
    Target: field at the next timestep
        shape: (n_fields, nx, nz)
    """

    def __init__(self, data_path: str, mask_path: str,
                 n_input_steps: int = 5, rollout_steps: int = 1):
        """
        Args:
            data_path: path to .npy file, shape (T, C, nx, nz)
            mask_path: path to terrain_mask.npy, shape (nx, nz)
            n_input_steps: number of consecutive input timesteps
            rollout_steps: number of future steps per sample (for multi-step loss)
        """
        self.data = np.load(data_path).astype(np.float32)   # (T, C, nx, nz)
        self.mask = np.load(mask_path).astype(np.float32)    # (nx, nz)
        self.n_input_steps = n_input_steps
        self.n_fields = self.data.shape[1]
        self.rollout_steps = rollout_steps

        # Need n_input_steps for input + rollout_steps for targets
        self.n_samples = len(self.data) - n_input_steps - rollout_steps + 1
        assert self.n_samples > 0, (
            f"Not enough snapshots ({len(self.data)}) for "
            f"n_input_steps={n_input_steps}, rollout_steps={rollout_steps}"
        )

        # Per-channel normalization stats (over all snapshots)
        self.mean = self.data.mean(axis=(0, 2, 3))   # (C,)
        self.std = self.data.std(axis=(0, 2, 3))      # (C,)
        self.std[self.std < 1e-8] = 1.0

        print(f"Dataset: {self.n_samples} samples, "
              f"data shape {self.data.shape}, "
              f"n_input_steps={n_input_steps}, rollout={rollout_steps}")
        print(f"  Input channels: {n_input_steps} steps x {self.n_fields} fields + 1 mask "
              f"= {n_input_steps * self.n_fields + 1}")
        for i in range(self.n_fields):
            print(f"  ch{i}: mean={self.mean[i]:.4f}, std={self.std[i]:.4f}")

    def normalize(self, x):
        """Normalize field data. x shape: (C, nx, nz)"""
        mean = self.mean[:, None, None]
        std = self.std[:, None, None]
        return (x - mean) / std

    def denormalize(self, x):
        """Denormalize field data. x shape: (C, nx, nz) or (B, C, nx, nz)"""
        if x.ndim == 4:
            mean = self.mean[None, :, None, None]
            std = self.std[None, :, None, None]
        else:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
            std = torch.tensor(std, device=x.device, dtype=x.dtype)
        return x * std + mean

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Input: stack n_input_steps consecutive timesteps
        # Each timestep has n_fields channels, all normalized
        input_frames = []
        for s in range(self.n_input_steps):
            input_frames.append(self.normalize(self.data[idx + s]))  # (C, nx, nz)

        # Stack: (n_input_steps * C, nx, nz)
        fields_in = np.concatenate(input_frames, axis=0)

        # Append terrain mask
        mask = self.mask[np.newaxis, :, :]                           # (1, nx, nz)
        x = np.concatenate([fields_in, mask], axis=0)                # (n_input_steps*C + 1, nx, nz)

        # Target: timestep(s) after the input window
        target_start = idx + self.n_input_steps
        if self.rollout_steps == 1:
            target = self.normalize(self.data[target_start])         # (C, nx, nz)
        else:
            targets = []
            for s in range(self.rollout_steps):
                targets.append(self.normalize(self.data[target_start + s]))
            target = np.stack(targets, axis=0)                       # (S, C, nx, nz)

        return torch.from_numpy(x), torch.from_numpy(target)