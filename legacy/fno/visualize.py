"""
visualize.py — Visualization for FNO/PINO predictions.

Outputs:
    1. Rollout error curve: per-field relative L2 + total, vs rollout step
    2. Field comparison: GT / Pred / Error map at start, middle, end of rollout
       - 3 timesteps × 3 fields (alpha, Ux, Uz) = 9 sets of 3 images each

Usage:
    python visualize.py train.resume=outputs/.../best.pt
    python visualize.py train.resume=outputs/.../best.pt model.modes1=256 model.modes2=64
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from dataset import WaveDataset
from model import FNO2d

# ─── Settings ──────────────────────────────────────────────
DPI = 400
FIELD_FIGSIZE = (30, 4)     # wide aspect for 33:1 domain
FIELD_NAMES = ["alpha", "Ux", "Uz", "p_rgh"]
FIELD_CMAPS = ["coolwarm", "RdBu_r", "RdBu_r", "viridis"]
ERROR_CMAP = "hot_r"


# ─── Model loading ────────────────────────────────────────

def load_model(cfg, device):
    model = FNO2d(
        n_in_ch=cfg.data.n_input_ch,
        n_out_ch=cfg.model.n_out_ch,
        modes1=cfg.model.modes1,
        modes2=cfg.model.modes2,
        width=cfg.model.width,
        n_layers=cfg.model.n_layers,
    ).to(device)

    ckpt_path = cfg.train.resume or os.path.join(cfg.train.checkpoint_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {ckpt_path} (epoch {ckpt['epoch']})")
    return model


# ─── Autoregressive rollout ───────────────────────────────

def autoregressive_rollout(model, dataset, start_idx, n_steps, device, mask):
    """
    Run autoregressive rollout with sliding input window.

    Returns:
        preds: list of (C, nx, nz) denormalized predictions
        gts:   list of (C, nx, nz) denormalized ground truths
    """
    model.eval()
    mask_ch = torch.from_numpy(mask[np.newaxis]).float().to(device)
    n_input = dataset.n_input_steps

    # Initial buffer of normalized frames
    buffer = []
    for s in range(n_input):
        frame = dataset.normalize(dataset.data[start_idx + s])
        buffer.append(torch.from_numpy(frame).float().to(device))

    preds = []
    gts = []

    with torch.no_grad():
        for step in range(n_steps):
            stacked = torch.cat(buffer, dim=0)
            x = torch.cat([stacked, mask_ch], dim=0).unsqueeze(0)

            pred = model(x)

            pred_np = dataset.denormalize(pred[0].cpu()).numpy()
            gt_idx = start_idx + n_input + step
            gt_np = dataset.data[gt_idx]
            preds.append(pred_np)
            gts.append(gt_np)

            buffer.pop(0)
            buffer.append(pred[0])

    return preds, gts


# ─── Rollout error curve ─────────────────────────────────

def compute_relative_l2(pred, gt, mask):
    """Compute relative L2 error: ||pred - gt|| / ||gt|| over masked region."""
    diff = (pred - gt) ** 2 * mask
    ref = gt ** 2 * mask
    denom = ref.sum()
    if denom < 1e-12:
        return 0.0
    return np.sqrt(diff.sum() / denom)


def plot_rollout_errors(preds, gts, mask, save_path):
    """
    Plot per-field relative L2 error + total error vs rollout step.
    """
    n_steps = len(preds)
    errors = {name: [] for name in FIELD_NAMES}
    errors["total"] = []

    for p, g in zip(preds, gts):
        total_diff = 0.0
        total_ref = 0.0
        for i, name in enumerate(FIELD_NAMES):
            errors[name].append(compute_relative_l2(p[i], g[i], mask))
            total_diff += ((p[i] - g[i]) ** 2 * mask).sum()
            total_ref += (g[i] ** 2 * mask).sum()
        errors["total"].append(np.sqrt(total_diff / max(total_ref, 1e-12)))

    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, errors["alpha"], "o-", label="alpha", linewidth=2)
    ax.plot(steps, errors["Ux"], "s-", label="Ux", linewidth=2)
    ax.plot(steps, errors["Uz"], "^-", label="Uz", linewidth=2)
    ax.plot(steps, errors["total"], "D-", label="total", linewidth=2, color="black")
    ax.set_xlabel("Rollout step", fontsize=13)
    ax.set_ylabel("Relative L2 Error", fontsize=13)
    ax.set_title("Autoregressive Rollout Error", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")

    return errors


# ─── Field comparison plots ──────────────────────────────

def plot_field(data, mask, grid_x, grid_z, title, cmap, vmin, vmax, save_path):
    """Plot a single field as a high-res image."""
    fig, ax = plt.subplots(figsize=FIELD_FIGSIZE)
    xx, zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    masked_data = np.where(mask > 0.5, data, np.nan)
    im = ax.pcolormesh(xx, zz, masked_data, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("auto")
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_error_map(gt, pred, mask, grid_x, grid_z, title, save_path):
    """Plot absolute error map."""
    err = np.abs(gt - pred) * mask
    fig, ax = plt.subplots(figsize=FIELD_FIGSIZE)
    xx, zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    masked_err = np.where(mask > 0.5, err, np.nan)
    im = ax.pcolormesh(xx, zz, masked_err, cmap=ERROR_CMAP,
                       shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("auto")
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_snapshot_set(gt, pred, mask, grid_x, grid_z, step_label, vis_dir):
    """
    For one rollout timestep, produce 3 images per field (GT, Pred, Error).
    Total: 3 fields × 3 images = 9 files.
    """
    for i, (name, cmap) in enumerate(zip(FIELD_NAMES, FIELD_CMAPS)):
        gt_f = gt[i]
        pr_f = pred[i]

        # Shared colorbar range from GT
        fluid_vals = gt_f[mask > 0.5]
        if name == "alpha":
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(fluid_vals.min())
            vmax = float(fluid_vals.max())
            # Symmetric for velocity
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

        prefix = f"{step_label}_{name}"

        plot_field(gt_f, mask, grid_x, grid_z,
                   f"GT {name} — {step_label}",
                   cmap, vmin, vmax,
                   os.path.join(vis_dir, f"{prefix}_gt.png"))

        plot_field(pr_f, mask, grid_x, grid_z,
                   f"Pred {name} — {step_label}",
                   cmap, vmin, vmax,
                   os.path.join(vis_dir, f"{prefix}_pred.png"))

        plot_error_map(gt_f, pr_f, mask, grid_x, grid_z,
                       f"|Error| {name} — {step_label}",
                       os.path.join(vis_dir, f"{prefix}_error.png"))

    print(f"  Saved 9 field images for {step_label}")


# ─── Main ────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    device = cfg.hardware.device
    proc = cfg.data.processed_dir
    vis_dir = "./visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    # Load data
    test_ds = WaveDataset(
        os.path.join(proc, "test_data.npy"),
        os.path.join(proc, "terrain_mask.npy"),
        n_input_steps=cfg.data.n_input_steps,
        rollout_steps=1,
    )
    train_ds = WaveDataset(
        os.path.join(proc, "train_data.npy"),
        os.path.join(proc, "terrain_mask.npy"),
        n_input_steps=cfg.data.n_input_steps,
        rollout_steps=1,
    )
    test_ds.mean = train_ds.mean
    test_ds.std = train_ds.std

    mask = np.load(os.path.join(proc, "terrain_mask.npy"))
    grid_x = np.load(os.path.join(proc, "grid_x.npy"))
    grid_z = np.load(os.path.join(proc, "grid_z.npy"))
    test_times = np.load(os.path.join(proc, "test_times.npy"))

    # Load model
    model = load_model(cfg, device)

    # ── Autoregressive rollout ──
    print("\n--- Autoregressive rollout ---")
    n_rollout = len(test_ds.data) - cfg.data.n_input_steps
    print(f"Rolling out {n_rollout} steps from test set start")

    preds, gts = autoregressive_rollout(model, test_ds, 0, n_rollout, device, mask)

    # ── 1) Rollout error curve ──
    print("\n--- Rollout error curve ---")
    errors = plot_rollout_errors(
        preds, gts, mask,
        os.path.join(vis_dir, "rollout_error.png"),
    )

    # ── 2) Field comparisons at start / middle / end ──
    print("\n--- Field comparisons ---")
    step_indices = [0, n_rollout // 2, n_rollout - 1]

    for step_idx in step_indices:
        t_physical = test_times[cfg.data.n_input_steps + step_idx]
        step_label = f"step{step_idx+1}_t{t_physical:.2f}s"

        plot_snapshot_set(
            gts[step_idx], preds[step_idx],
            mask, grid_x, grid_z,
            step_label, vis_dir,
        )

    # ── Summary ──
    print(f"\n--- Summary ---")
    print(f"Rollout steps: {n_rollout}")
    for step_idx in step_indices:
        print(f"  Step {step_idx+1}: "
              f"alpha={errors['alpha'][step_idx]:.4f}, "
              f"Ux={errors['Ux'][step_idx]:.4f}, "
              f"Uz={errors['Uz'][step_idx]:.4f}, "
              f"total={errors['total'][step_idx]:.4f}")

    print(f"\nAll visualizations saved to {vis_dir}/")


if __name__ == "__main__":
    main()