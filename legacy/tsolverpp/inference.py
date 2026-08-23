"""
Inference & animation script for Transolver++ rollout.

Generates side-by-side mp4: ground truth (left) vs prediction (right)
on X-Z cross section at Y midplane.

Usage:
    python -u inference.py \
        --config_path /path/to/.hydra/config.yaml \
        --checkpoint /path/to/best.pt \
        --data_dir /path/to/cropped_0.05 \
        --chunk_id 6 \
        --rollout_steps 100 \
        --output rollout_chunk6.mp4
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from omegaconf import OmegaConf

from transolver_pp import TransolverPP


def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    model = TransolverPP(window=cfg.train.window, **cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_data(data_dir, chunk_id):
    data_dir = Path(data_dir)
    coords = np.load(data_dir / "coords.npy")           # (N, 3)
    data = np.load(data_dir / f"chunk_{chunk_id:03d}_data.npy")  # (T, N, 6)
    times = np.load(data_dir / f"chunk_{chunk_id:03d}_times.npy")
    stats = np.load(data_dir / "stats.npy")              # (2, 6)
    return coords, data, times, stats


def get_midplane_mask(coords):
    """Get indices of points closest to Y midplane."""
    y = coords[:, 1]
    y_mid = (y.min() + y.max()) / 2
    # Take points within a thin band around midplane
    tol = np.sort(np.abs(y - y_mid))[len(y) // 30]  # ~3% of points
    mask = np.abs(y - y_mid) <= tol
    return mask


@torch.no_grad()
def rollout(model, coords, data, stats, window, start_frame, n_steps, device):
    """Run autoregressive rollout from start_frame for n_steps."""
    mean, std = stats[0], stats[1]
    normalize = lambda x: (x - mean) / std
    denormalize = lambda x: x * std + mean

    predictions = []
    ground_truth = []

    # Build initial window: frames [start_frame - window + 1, ..., start_frame]
    fields_w = np.concatenate(
        [normalize(data[start_frame - window + 1 + w]) for w in range(window)],
        axis=-1)  # (N, 6*W)
    fields_w = torch.from_numpy(fields_w).unsqueeze(0).to(device)  # (1, N, 6W)
    coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)    # (1, N, 3)

    for step in range(n_steps):
        gt_frame = data[start_frame + 1 + step]  # ground truth
        ground_truth.append(gt_frame)

        inp = torch.cat([coords_t, fields_w], dim=-1)  # (1, N, 3+6W)
        pred = model(inp)  # (1, N, 6)

        pred_np = pred.squeeze(0).cpu().numpy()
        predictions.append(denormalize(pred_np))

        # Shift window: drop oldest, append new prediction (normalized)
        fields_w = torch.cat([fields_w[..., 6:], pred], dim=-1)

    return np.stack(predictions), np.stack(ground_truth)


def compute_per_field_rmse(preds, gts):
    """Compute RMSE per field per timestep. Returns (n_steps, 6)."""
    diff2 = (preds - gts) ** 2  # (steps, N, 6)
    return np.sqrt(diff2.mean(axis=1))  # (steps, 6)


def make_animation(preds, gts, coords, mask, times_gt, output_path, field_idx=0,
                   field_name="alpha.water", fps=20):
    """Generate side-by-side mp4 animation on X-Z midplane using scatter."""
    x = coords[mask, 0]
    z = coords[mask, 2]

    vmin = gts[:, mask, field_idx].min()
    vmax = gts[:, mask, field_idx].max()

    # 4K: 3840x2160
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(38.4, 21.6), dpi=100)
    fig.subplots_adjust(wspace=0.15, top=0.92, bottom=0.08, left=0.04, right=0.96)

    scat1 = ax1.scatter(x, z, c=gts[0, mask, field_idx], s=1.5, cmap='bwr',
                        vmin=0, vmax=1, rasterized=True)
    scat2 = ax2.scatter(x, z, c=preds[0, mask, field_idx], s=1.5, cmap='bwr',
                        vmin=0, vmax=1, rasterized=True)

    ax1.set_title("Ground Truth", fontsize=28)
    ax2.set_title("Prediction", fontsize=28)
    for ax in (ax1, ax2):
        ax.set_xlabel("X (m)", fontsize=22)
        ax.set_ylabel("Z (m)", fontsize=22)
        ax.tick_params(labelsize=18)

    fig.colorbar(scat1, ax=ax1, label=field_name, shrink=0.8).ax.tick_params(labelsize=16)
    fig.colorbar(scat2, ax=ax2, label=field_name, shrink=0.8).ax.tick_params(labelsize=16)

    title = fig.suptitle(f"t = {times_gt[0]:.2f}s | Step 0", fontsize=32)

    def update(frame):
        scat1.set_array(gts[frame, mask, field_idx])
        scat2.set_array(preds[frame, mask, field_idx])
        rmse = np.sqrt(((preds[frame, mask, field_idx] - gts[frame, mask, field_idx]) ** 2).mean())
        title.set_text(f"t = {times_gt[frame]:.2f}s | Step {frame} | RMSE = {rmse:.4f}")
        return scat1, scat2, title

    ani = animation.FuncAnimation(fig, update, frames=len(preds), interval=1000 // fps, blit=False)
    ani.save(output_path, writer="ffmpeg", fps=fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=6,
                        help="Start frame within chunk (must >= window)")
    parser.add_argument("--rollout_steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="rollout.mp4")
    parser.add_argument("--field", type=int, default=0,
                        help="Field index to visualize: 0=alpha, 1=Ux, 2=Uy, 3=Uz, 4=p_rgh, 5=nut")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load
    print("Loading model...")
    model, cfg = load_model(args.config_path, args.checkpoint, device)
    W = cfg.train.window

    print("Loading data...")
    coords, data, times, stats = load_data(args.data_dir, args.chunk_id)
    T = data.shape[0]

    start = max(args.start_frame, W - 1)
    assert start + args.rollout_steps < T, \
        f"Not enough frames: start={start}, rollout={args.rollout_steps}, chunk has {T} frames"

    # Midplane mask
    mask = get_midplane_mask(coords)
    print(f"Midplane points: {mask.sum()} / {len(coords)}")

    # Rollout
    print(f"Rolling out {args.rollout_steps} steps from frame {start}...")
    preds, gts = rollout(model, coords, data, stats, W, start, args.rollout_steps, device)

    # Per-field RMSE
    field_names = ["alpha", "Ux", "Uy", "Uz", "p_rgh", "nut"]
    rmse = compute_per_field_rmse(preds, gts)
    print(f"\nPer-field RMSE (mean over rollout):")
    for i, name in enumerate(field_names):
        print(f"  {name}: {rmse[:, i].mean():.6f}")

    # RMSE over time
    rmse_path = Path(args.output).with_suffix(".npy")
    np.save(rmse_path, rmse)
    print(f"RMSE saved: {rmse_path}")

    # Animation
    times_gt = times[start + 1: start + 1 + args.rollout_steps]
    print(f"Generating animation...")
    make_animation(preds, gts, coords, mask, times_gt, args.output,
                   field_idx=args.field, field_name=field_names[args.field])


if __name__ == "__main__":
    main()
