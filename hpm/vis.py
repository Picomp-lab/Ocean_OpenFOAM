"""
GT vs HPM Prediction animation — vertically stacked.
Top: Ground Truth, Bottom: Prediction.

Usage:
    python -u inference.py \
        --config_path /path/to/.hydra/config.yaml \
        --checkpoint /path/to/best.pt \
        --data_dir /path/to/cropped_0.05 \
        --chunk_id 6 \
        --output compare_chunk6.mp4
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from pathlib import Path
from omegaconf import OmegaConf

from hpm_model import HPM
from dataset import load_coords


@torch.no_grad()
def rollout(model, coords_norm, data, stats, window, start_frame, n_steps, device):
    mean, std = stats[0], stats[1]
    normalize = lambda x: (x - mean) / std
    denormalize = lambda x: x * std + mean

    predictions = []
    fields_w = np.concatenate(
        [normalize(data[start_frame - window + 1 + w]) for w in range(window)],
        axis=-1)  # (N, 6*W)
    fields_w = torch.from_numpy(fields_w.astype(np.float32)).unsqueeze(0).to(device)
    coords_batch = coords_norm.unsqueeze(0)  # (1, N, 3)

    for step in range(n_steps):
        delta = model(coords_batch, fields_w)  # (1, N, 6)
        pred_norm = fields_w[0, :, -6:] + delta[0]  # (N, 6)
        pred_np = denormalize(pred_norm.cpu().numpy())
        predictions.append(pred_np)
        fields_w = torch.cat([fields_w[..., 6:], pred_norm.unsqueeze(0)], dim=-1)

    return np.stack(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=6)
    parser.add_argument("--n_frames", type=int, default=93)
    parser.add_argument("--field", type=int, default=0,
                        help="0=alpha, 1=Ux, 2=Uy, 3=Uz, 4=p_rgh, 5=nut, 6=|U|")
    parser.add_argument("--output", type=str, default="compare.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fi = args.field

    # Load model
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)

    lbo_path = data_dir / "lbo" / "lbo_eigenvectors.npy"
    spectral_embedding = np.load(lbo_path)

    stats = np.load(data_dir / "stats.npy")

    model = HPM(
        space_dim=3,
        field_dim=6,
        out_dim=6,
        window=cfg.data.window,
        n_hidden=cfg.model.n_hidden,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        freq_num=cfg.model.freq_num,
        dropout=0.0,
        mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.get('spectral_pos_dim', 0),
        spectral_embedding=spectral_embedding,
        use_ckpt=False,
        use_phase_gate=cfg.model.get('use_phase_gate', False),
        gate_alpha_0=cfg.model.get('gate_alpha_0', 0.5),
        gate_k_init=cfg.model.get('gate_k_init', 10.0),
        stats=stats,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    print("Loading data...")
    coords = np.load(data_dir / "coords.npy")
    data = np.load(data_dir / f"chunk_{args.chunk_id:03d}_data.npy")
    times = np.load(data_dir / f"chunk_{args.chunk_id:03d}_times.npy")
    stats = np.load(data_dir / "stats.npy")
    coords_norm = load_coords(args.data_dir).to(device)

    W = cfg.data.window
    start = max(args.start_frame, W - 1)
    n_steps = min(args.n_frames, data.shape[0] - start - 1)

    # Rollout
    print(f"Rolling out {n_steps} steps from frame {start}...")
    preds = rollout(model, coords_norm, data, stats, W, start, n_steps, device)
    gts = data[start + 1: start + 1 + n_steps]  # (n_steps, N, 6)
    gt_times = times[start + 1: start + 1 + n_steps]

    # Y midplane slice — same method as vis_rgb_velocity.py
    y_vals = coords[:, 1]
    mid_y = y_vals.min() + (y_vals.max() - y_vals.min()) / 2.0
    mask = np.abs(y_vals - mid_y) < 0.015
    print(f"Midplane slice: {mask.sum()} points")

    x_raw = coords[mask, 0]
    z_raw = coords[mask, 2]

    # Field extraction: fi=6 is velocity magnitude
    field_names = ["alpha.water", "Ux", "Uy", "Uz", "p_rgh", "nut", "|U|"]
    fname = field_names[fi]

    if fi == 6:
        # Velocity magnitude: sqrt(Ux² + Uy² + Uz²)
        gt_slice = np.sqrt(gts[:, mask, 1]**2 + gts[:, mask, 2]**2 + gts[:, mask, 3]**2)
        pred_slice = np.sqrt(preds[:, mask, 1]**2 + preds[:, mask, 2]**2 + preds[:, mask, 3]**2)
    else:
        gt_slice = gts[:, mask, fi]
        pred_slice = preds[:, mask, fi]

    # Interpolation grid
    grid_x, grid_z = np.mgrid[
        x_raw.min():x_raw.max():2000j,
        z_raw.min():z_raw.max():400j
    ]

    # Colormap
    if fi == 0:
        cdict = {
            'red':   [[0.0, 1.0, 1.0], [1.0, 0.6, 0.6]],
            'green': [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            'blue':  [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        }
        custom_cmap = LinearSegmentedColormap('OpacityReds', cdict)
        levels = np.linspace(0, 1, 128)
    elif fi == 6:
        custom_cmap = 'magma'
        all_vals = np.concatenate([gt_slice.ravel(), pred_slice.ravel()])
        levels = np.linspace(0, np.percentile(all_vals, 99), 128)
    else:
        custom_cmap = 'coolwarm'
        all_vals = np.concatenate([gt_slice.ravel(), pred_slice.ravel()])
        levels = np.linspace(all_vals.min(), all_vals.max(), 128)

    # Figure: two 4K panels stacked vertically
    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.95, hspace=0.15)

    # Initial frame
    gt_grid = griddata((x_raw, z_raw), gt_slice[0], (grid_x, grid_z), method='linear')
    pred_grid = griddata((x_raw, z_raw), pred_slice[0], (grid_x, grid_z), method='linear')

    ax_gt.contourf(grid_x[:, 0], grid_z[0, :], gt_grid.T, levels=levels,
                   cmap=custom_cmap, extend='both')
    ax_pred.contourf(grid_x[:, 0], grid_z[0, :], pred_grid.T, levels=levels,
                     cmap=custom_cmap, extend='both')

    for ax, label in [(ax_gt, "Ground Truth"), (ax_pred, "HPM Prediction")]:
        ax.set_facecolor('white')
        ax.set_xlabel("X (m)", fontsize=20)
        ax.set_ylabel("Z (m)", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_title(label, fontsize=24)

    rmse0 = np.sqrt(((gt_slice[0] - pred_slice[0]) ** 2).mean())
    suptitle = fig.suptitle(
        f"t = {gt_times[0]:.2f}s | {fname} | Step 0 | RMSE = {rmse0:.4f}",
        fontsize=28)

    def update(frame):
        ax_gt.clear()
        ax_pred.clear()

        gt_grid = griddata((x_raw, z_raw), gt_slice[frame], (grid_x, grid_z), method='linear')
        pred_grid = griddata((x_raw, z_raw), pred_slice[frame], (grid_x, grid_z), method='linear')

        ax_gt.contourf(grid_x[:, 0], grid_z[0, :], gt_grid.T, levels=levels,
                       cmap=custom_cmap, extend='both')
        ax_pred.contourf(grid_x[:, 0], grid_z[0, :], pred_grid.T, levels=levels,
                         cmap=custom_cmap, extend='both')

        for ax, label in [(ax_gt, "Ground Truth"), (ax_pred, "HPM Prediction")]:
            ax.set_facecolor('white')
            ax.set_xlabel("X (m)", fontsize=20)
            ax.set_ylabel("Z (m)", fontsize=20)
            ax.tick_params(labelsize=16)
            ax.set_title(label, fontsize=24)

        rmse = np.sqrt(((gt_slice[frame] - pred_slice[frame]) ** 2).mean())
        suptitle.set_text(
            f"t = {gt_times[frame]:.2f}s | {fname} | Step {frame} | RMSE = {rmse:.4f}")

    ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=1000 // args.fps, blit=False)
    ani.save(args.output, writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close()

    # Print RMSE summary
    rmse_all = np.sqrt(((gt_slice - pred_slice) ** 2).mean(axis=1))
    print(f"RMSE: start={rmse_all[0]:.4f}, end={rmse_all[-1]:.4f}, mean={rmse_all.mean():.4f}")

    # Save RMSE
    rmse_full = np.sqrt(((gts - preds) ** 2).mean(axis=1))  # (n_steps, 6)
    rmse_path = Path(args.output).with_suffix(".npy")
    np.save(rmse_path, rmse_full)
    print(f"RMSE saved: {rmse_path}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()