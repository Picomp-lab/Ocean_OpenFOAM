"""
RGB Velocity Visualization for HPM rollout.

Encodes 3D velocity as color:
  R = |Ux| / max|Ux|
  G = |Uy| / max|Uy|
  B = |Uz| / max|Uz|
  Alpha = |U| / max|U|  (static water = transparent)

Top: Ground Truth, Bottom: HPM Prediction.

Usage:
    python -u vis_rgb_velocity.py \
        --config_path /path/to/.hydra/config.yaml \
        --checkpoint /path/to/best.pt \
        --data_dir /path/to/cropped_0.05 \
        --chunk_id 6 \
        --output rgb_velocity_chunk6.mp4
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        axis=-1)
    fields_w = torch.from_numpy(fields_w.astype(np.float32)).unsqueeze(0).to(device)
    coords_batch = coords_norm.unsqueeze(0)

    for step in range(n_steps):
        delta = model(coords_batch, fields_w)
        pred_norm = fields_w[0, :, -6:] + delta[0]
        pred_np = denormalize(pred_norm.cpu().numpy())
        predictions.append(pred_np)
        fields_w = torch.cat([fields_w[..., 6:], pred_norm.unsqueeze(0)], dim=-1)

    return np.stack(predictions)


def build_rgba(ux, uy, uz, ux_max, uy_max, uz_max, umag_max):
    """
    Build RGBA image from velocity components.

    Args:
        ux, uy, uz: (H, W) arrays — interpolated velocity on grid
        ux_max, uy_max, uz_max: global max for normalization
        umag_max: global max |U| for alpha normalization

    Returns:
        rgba: (H, W, 4) float array in [0, 1]
    """
    r = np.abs(ux) / max(ux_max, 1e-10)
    g = np.abs(uy) / max(uy_max, 1e-10)
    b = np.abs(uz) / max(uz_max, 1e-10)
    umag = np.sqrt(ux**2 + uy**2 + uz**2)
    a = umag / max(umag_max, 1e-10)

    # Clamp to [0, 1]
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    a = np.clip(a, 0, 1)

    rgba = np.stack([r, g, b, a], axis=-1)

    # NaN regions (outside convex hull of griddata) → fully transparent
    nan_mask = np.isnan(ux) | np.isnan(uy) | np.isnan(uz)
    rgba[nan_mask] = 0

    return rgba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=6)
    parser.add_argument("--n_frames", type=int, default=93)
    parser.add_argument("--output", type=str, default="rgb_velocity.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[0.15, 0.15, 0.15],
                        help="Background color RGB [0-1]")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)

    lbo_path = data_dir / "lbo" / "lbo_eigenvectors.npy"
    spectral_embedding = np.load(lbo_path)

    stats = np.load(data_dir / "stats.npy")

    model = HPM(
        space_dim=3, field_dim=6, out_dim=6,
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
    gts = data[start + 1: start + 1 + n_steps]
    gt_times = times[start + 1: start + 1 + n_steps]

    # Y midplane slice — same method as inference.py
    y_vals = coords[:, 1]
    mid_y = y_vals.min() + (y_vals.max() - y_vals.min()) / 2.0
    mask = np.abs(y_vals - mid_y) < 0.015
    print(f"Midplane slice: {mask.sum()} points")

    x_raw = coords[mask, 0]
    z_raw = coords[mask, 2]

    # Interpolation grid
    grid_x, grid_z = np.mgrid[
        x_raw.min():x_raw.max():2000j,
        z_raw.min():z_raw.max():400j
    ]

    # ---- Precompute global max for normalization ----
    # Use GT values for consistent normalization
    gt_ux = gts[:, mask, 1]
    gt_uy = gts[:, mask, 2]
    gt_uz = gts[:, mask, 3]
    ux_max = np.percentile(np.abs(gt_ux), 99)
    uy_max = np.percentile(np.abs(gt_uy), 99)
    uz_max = np.percentile(np.abs(gt_uz), 99)
    umag = np.sqrt(gt_ux**2 + gt_uy**2 + gt_uz**2)
    umag_max = np.percentile(umag, 99)
    print(f"Normalization: |Ux|_max={ux_max:.3f}, |Uy|_max={uy_max:.3f}, "
          f"|Uz|_max={uz_max:.3f}, |U|_max={umag_max:.3f}")

    # ---- Precompute Delaunay triangulation (one-time) ----
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    print("Precomputing Delaunay triangulation...")
    tri = Delaunay(np.column_stack([x_raw, z_raw]))
    grid_pts = np.column_stack([grid_x.ravel(), grid_z.ravel()])
    print(f"  Done. {len(x_raw)} source points, {len(grid_pts)} grid points")

    # ---- Figure setup ----
    bg = args.bg_color
    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.95, hspace=0.12)

    extent = [x_raw.min(), x_raw.max(), z_raw.min(), z_raw.max()]

    # Initial frame placeholder
    dummy = np.zeros((400, 2000, 4))
    im_gt = ax_gt.imshow(dummy, extent=extent, origin='lower', aspect='auto')
    im_pred = ax_pred.imshow(dummy, extent=extent, origin='lower', aspect='auto')

    for ax, label in [(ax_gt, "Ground Truth"), (ax_pred, "HPM Prediction")]:
        ax.set_facecolor(bg)
        ax.set_xlabel("X (m)", fontsize=20, color='white')
        ax.set_ylabel("Z (m)", fontsize=20, color='white')
        ax.tick_params(labelsize=16, colors='white')
        ax.set_title(label, fontsize=24, color='white')

    # Legend: color key
    legend_text = "R=|Ux|  G=|Uy|  B=|Uz|  Opacity=|U|  Yellow=direction error (>90°)"
    fig.text(0.5, 0.97, legend_text, ha='center', fontsize=22, color='white',
             family='monospace')

    suptitle = fig.suptitle("", fontsize=28, color='white', y=0.94)

    def interp(values):
        """Interpolate using precomputed Delaunay triangulation."""
        ip = LinearNDInterpolator(tri, values)
        return ip(grid_pts).reshape(grid_x.shape)

    def render_frame(frame):
        """Interpolate velocity, build RGBA, overlay direction error on pred."""
        grids = {}
        for label, src in [("gt", gts), ("pred", preds)]:
            ux_grid = interp(src[frame, mask, 1])
            uy_grid = interp(src[frame, mask, 2])
            uz_grid = interp(src[frame, mask, 3])
            grids[label] = (ux_grid, uy_grid, uz_grid)

        results = {}
        for label in ["gt", "pred"]:
            ux_g, uy_g, uz_g = grids[label]
            rgba = build_rgba(ux_g, uy_g, uz_g,
                              ux_max, uy_max, uz_max, umag_max)

            # Composite RGBA over background
            a = rgba[:, :, 3:4]
            rgb = rgba[:, :, :3]
            bg_arr = np.array(bg).reshape(1, 1, 3)
            composited = rgb * a + bg_arr * (1 - a)
            results[label] = np.clip(composited, 0, 1)

        # ---- Direction error overlay on pred ----
        gt_ux, gt_uy, gt_uz = grids["gt"]
        pr_ux, pr_uy, pr_uz = grids["pred"]

        # Dot product: cos(theta) = (gt · pred) / (|gt| |pred|)
        dot = gt_ux * pr_ux + gt_uy * pr_uy + gt_uz * pr_uz
        gt_mag = np.sqrt(gt_ux**2 + gt_uy**2 + gt_uz**2)
        pr_mag = np.sqrt(pr_ux**2 + pr_uy**2 + pr_uz**2)
        denom = gt_mag * pr_mag
        cos_theta = np.where(denom > 1e-8, dot / denom, 1.0)

        # Direction error: cos_theta < 0 means angle > 90°
        # Only mark where both GT and pred have significant velocity
        speed_thresh = umag_max * 0.05  # ignore static water
        dir_error = (cos_theta < 0) & (gt_mag > speed_thresh) & (pr_mag > speed_thresh)

        # Handle NaN from griddata
        dir_error = np.where(np.isnan(cos_theta), False, dir_error)

        # Overlay yellow (1, 0.9, 0, 0.5) on pred where direction is wrong
        yellow = np.array([1.0, 0.9, 0.0])
        overlay_alpha = 0.5
        pred_img = results["pred"]
        pred_img[dir_error] = pred_img[dir_error] * (1 - overlay_alpha) + yellow * overlay_alpha
        results["pred"] = np.clip(pred_img, 0, 1)

        # Count direction error percentage
        valid = (gt_mag > speed_thresh) & (~np.isnan(cos_theta))
        n_error = dir_error.sum()
        n_valid = valid.sum()
        error_pct = (n_error / max(n_valid, 1)) * 100

        # Transpose for imshow: (Nx, Nz, 3) -> (Nz, Nx, 3)
        results["gt"] = results["gt"].transpose(1, 0, 2)
        results["pred"] = results["pred"].transpose(1, 0, 2)

        return results, error_pct

    def update(frame):
        rendered, error_pct = render_frame(frame)
        im_gt.set_data(rendered["gt"])
        im_pred.set_data(rendered["pred"])

        # RMSE on velocity magnitude
        gt_umag = np.sqrt(gts[frame, mask, 1]**2 + gts[frame, mask, 2]**2 +
                          gts[frame, mask, 3]**2)
        pred_umag = np.sqrt(preds[frame, mask, 1]**2 + preds[frame, mask, 2]**2 +
                            preds[frame, mask, 3]**2)
        rmse = np.sqrt(np.mean((gt_umag - pred_umag)**2))

        suptitle.set_text(
            f"t = {gt_times[frame]:.2f}s | Step {frame} | "
            f"|U| RMSE = {rmse:.4f} | Dir err = {error_pct:.1f}%")
        return im_gt, im_pred, suptitle

    print(f"Generating animation: {n_steps} frames...")
    ani = animation.FuncAnimation(fig, update, frames=n_steps,
                                  interval=1000 // args.fps, blit=False)
    ani.save(args.output, writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close()

    # RMSE summary
    rmse_per_step = []
    for f in range(n_steps):
        gt_u = np.sqrt(gts[f, :, 1]**2 + gts[f, :, 2]**2 + gts[f, :, 3]**2)
        pred_u = np.sqrt(preds[f, :, 1]**2 + preds[f, :, 2]**2 + preds[f, :, 3]**2)
        rmse_per_step.append(np.sqrt(np.mean((gt_u - pred_u)**2)))
    rmse_arr = np.array(rmse_per_step)
    print(f"|U| RMSE: start={rmse_arr[0]:.4f}, end={rmse_arr[-1]:.4f}, mean={rmse_arr.mean():.4f}")

    rmse_path = Path(args.output).with_suffix(".npy")
    np.save(rmse_path, rmse_arr)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()