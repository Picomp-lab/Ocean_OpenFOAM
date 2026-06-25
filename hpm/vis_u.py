"""
RGB Velocity Visualization for HPM rollout — PyVista-slice based.

速度三分量编码为颜色（αU 或 U 空间，由模型 config 决定，不还原）：
  R = |Ux|/max, G = |Uy|/max, B = |Uz|/max, Opacity = |U|/max
  pred 上叠加方向误差（>90° 标黄）。
Top: GT, Bottom: HPM Prediction.

切片几何用预计算缓存 (slice_y0.30/)，运行时纯 numpy。
--style both 输出 {output}_tri.mp4（精准连续）和 {output}_scatter.mp4（快速离散）。

Usage:
    python -u vis_u.py --config_path .../config.yaml --checkpoint .../best.pt \
        --data_dir .../cropped_0.05 --chunk_id 6 --output rgb_vel_chunk6.mp4
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from pathlib import Path
from omegaconf import OmegaConf

from hpm_model import HPM
from dataset import load_coords, apply_alpha_weighting, stats_filename, expand_range

MID_Y = 0.30


@torch.no_grad()
def rollout(model, coords_norm, data_w, stats, window, start_frame, n_steps, device):
    mean, std = stats[0], stats[1]
    normalize = lambda x: (x - mean) / std
    denormalize = lambda x: x * std + mean

    predictions = []
    fields_w = np.concatenate(
        [normalize(data_w[start_frame - window + 1 + w]) for w in range(window)],
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


def build_rgba_points(ux, uy, uz, ux_max, uy_max, uz_max, umag_max):
    """逐点 RGBA。ux/uy/uz: (M,) 切片点上的速度分量。返回 (M,4)。"""
    r = np.clip(np.abs(ux) / max(ux_max, 1e-10), 0, 1)
    g = np.clip(np.abs(uy) / max(uy_max, 1e-10), 0, 1)
    b = np.clip(np.abs(uz) / max(uz_max, 1e-10), 0, 1)
    umag = np.sqrt(ux**2 + uy**2 + uz**2)
    a = np.clip(umag / max(umag_max, 1e-10), 0, 1)
    return np.stack([r, g, b, a], axis=-1)


def composite_over_bg(rgba, bg):
    """RGBA over 背景色 -> RGB。rgba:(M,4), bg:(3,)。返回 (M,3)。"""
    a = rgba[:, 3:4]
    rgb = rgba[:, :3]
    return np.clip(rgb * a + np.array(bg).reshape(1, 3) * (1 - a), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=6)
    parser.add_argument("--n_frames", type=int, default=93)
    parser.add_argument("--output", type=str, default="rgb_velocity.mp4")
    parser.add_argument("--style", type=str, default="both",
                        choices=["tri", "scatter", "both"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point_size", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[0.15, 0.15, 0.15])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bg = args.bg_color

    # ---- model + config ----
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)
    spectral_embedding = np.load(data_dir / "lbo" / "lbo_eigenvectors.npy")
    weight_u = cfg.data.get('weight_u_by_alpha', True)
    weight_nut = cfg.data.get('weight_nut_by_alpha', False)
    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = np.load(data_dir / stats_filename(train_chunks, weight_u=weight_u, weight_nut=weight_nut))

    model = HPM(
        space_dim=3, field_dim=6, out_dim=6,
        window=cfg.data.window,
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=0.0, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.get('spectral_pos_dim', 0),
        spectral_embedding=spectral_embedding, use_ckpt=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()

    # ---- data + αU ----
    print("Loading data...")
    raw = np.load(data_dir / f"chunk_{args.chunk_id:03d}_data.npy")
    times = np.load(data_dir / f"chunk_{args.chunk_id:03d}_times.npy")
    coords_norm = load_coords(args.data_dir).to(device)
    print(f"Alpha-weighting: U={weight_u}, nut={weight_nut}")
    data_w = apply_alpha_weighting(raw, weight_u=weight_u, weight_nut=weight_nut)

    W = cfg.data.window
    start = max(args.start_frame, W - 1)
    n_steps = min(args.n_frames, data_w.shape[0] - start - 1)

    print(f"Rolling out {n_steps} steps...")
    preds = rollout(model, coords_norm, data_w, stats, W, start, n_steps, device)
    gts = data_w[start + 1: start + 1 + n_steps]
    gt_times = times[start + 1: start + 1 + n_steps]

    # ---- slice cache ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    triang = mtri.Triangulation(x_s, z_s, triangles=tri_simplices)
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")

    # 速度分量在切片点上 (n_steps, M)
    gt_u = (gts[:, cell_map, 1], gts[:, cell_map, 2], gts[:, cell_map, 3])
    pr_u = (preds[:, cell_map, 1], preds[:, cell_map, 2], preds[:, cell_map, 3])

    # 归一化用 GT 的 αU/U percentile
    ux_max = np.percentile(np.abs(gt_u[0]), 99)
    uy_max = np.percentile(np.abs(gt_u[1]), 99)
    uz_max = np.percentile(np.abs(gt_u[2]), 99)
    umag_max = np.percentile(np.sqrt(gt_u[0]**2 + gt_u[1]**2 + gt_u[2]**2), 99)
    print(f"Norm: |Ux|={ux_max:.3f} |Uy|={uy_max:.3f} |Uz|={uz_max:.3f} |U|={umag_max:.3f}")

    # 动态标签：αU vs U
    vp = "αU" if weight_u else "U"   # velocity prefix
    legend_text = (f"R=|{vp}x|  G=|{vp}y|  B=|{vp}z|  "
                   f"Opacity=|{vp}|  Yellow=dir err (>90°)")

    speed_thresh = umag_max * 0.05
    yellow = np.array([1.0, 0.9, 0.0])
    xlim = (x_s.min(), x_s.max()); zlim = (z_s.min(), z_s.max())

    def frame_colors(frame):
        """返回该帧 gt_rgb(M,3), pred_rgb(M,3, 含方向误差叠加), error_pct。"""
        out = {}
        for label, src in [("gt", gt_u), ("pred", pr_u)]:
            ux, uy, uz = src[0][frame], src[1][frame], src[2][frame]
            rgba = build_rgba_points(ux, uy, uz, ux_max, uy_max, uz_max, umag_max)
            out[label] = composite_over_bg(rgba, bg)

        # 方向误差（αU 方向 = U 方向）
        gx, gy, gz = gt_u[0][frame], gt_u[1][frame], gt_u[2][frame]
        px, py, pz = pr_u[0][frame], pr_u[1][frame], pr_u[2][frame]
        dot = gx*px + gy*py + gz*pz
        gmag = np.sqrt(gx**2+gy**2+gz**2); pmag = np.sqrt(px**2+py**2+pz**2)
        denom = gmag * pmag
        cos_t = np.ones_like(denom)
        nz = denom > 1e-8
        cos_t[nz] = dot[nz] / denom[nz]      # 只在非零处除，避免 warning
        dir_err = (cos_t < 0) & (gmag > speed_thresh) & (pmag > speed_thresh)

        pred_rgb = out["pred"].copy()
        oa = 0.5
        pred_rgb[dir_err] = pred_rgb[dir_err]*(1-oa) + yellow*oa
        out["pred"] = np.clip(pred_rgb, 0, 1)

        valid = (gmag > speed_thresh)
        err_pct = dir_err.sum() / max(valid.sum(), 1) * 100
        return out["gt"], out["pred"], err_pct

    styles = ["tri", "scatter"] if args.style == "both" else [args.style]
    out_base = Path(args.output)

    for style in styles:
        out_path = out_base.with_name(f"{out_base.stem}_{style}{out_base.suffix}")
        print(f"[{style}] -> {out_path}")
        fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
        fig.patch.set_facecolor(bg)
        fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.95, hspace=0.12)
        fig.text(0.5, 0.97, legend_text, ha='center', fontsize=22, color='white',
                 family='monospace')
        suptitle = fig.suptitle("", fontsize=28, color='white', y=0.94)

        def draw(ax, rgb, label):
            ax.clear()
            if style == "tri":
                # 每三角形 RGB = 三顶点均值。PolyCollection 原生支持真彩色，
                # 不绕 tripcolor 的标量映射，跨 mpl 版本最稳。
                tri_rgb = rgb[tri_simplices].mean(axis=1)        # (T,3)
                verts = np.stack([x_s[tri_simplices], z_s[tri_simplices]], axis=-1)  # (T,3,2)
                pc = PolyCollection(verts, facecolors=tri_rgb, edgecolors='none')
                ax.add_collection(pc)
            else:
                ax.scatter(x_s, z_s, c=rgb, s=args.point_size, edgecolors='none')
            ax.set_facecolor(bg); ax.set_xlim(xlim); ax.set_ylim(zlim)
            ax.set_xlabel("X (m)", fontsize=20, color='white')
            ax.set_ylabel("Z (m)", fontsize=20, color='white')
            ax.tick_params(labelsize=16, colors='white')
            ax.set_title(label, fontsize=24, color='white')

        def update(frame):
            gt_rgb, pred_rgb, err_pct = frame_colors(frame)
            draw(ax_gt, gt_rgb, "Ground Truth")
            draw(ax_pred, pred_rgb, "HPM Prediction")
            gm = np.sqrt(gt_u[0][frame]**2 + gt_u[1][frame]**2 + gt_u[2][frame]**2)
            pm = np.sqrt(pr_u[0][frame]**2 + pr_u[1][frame]**2 + pr_u[2][frame]**2)
            rmse = np.sqrt(np.mean((gm - pm)**2))
            suptitle.set_text(f"t={gt_times[frame]:.2f}s | Step {frame} | "
                              f"|{vp}| RMSE={rmse:.4f} | Dir err={err_pct:.1f}%")

        ani = animation.FuncAnimation(fig, update, frames=n_steps,
                                      interval=1000//args.fps, blit=False)
        ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
                 extra_args=['-vcodec','libx264','-pix_fmt','yuv420p'])
        plt.close()
        print(f"[{style}] saved")

    # |U| RMSE summary (全场)
    rmse_arr = np.array([
        np.sqrt(np.mean((np.sqrt(gts[f,:,1]**2+gts[f,:,2]**2+gts[f,:,3]**2)
                       - np.sqrt(preds[f,:,1]**2+preds[f,:,2]**2+preds[f,:,3]**2))**2))
        for f in range(n_steps)])
    print(f"|{vp}| RMSE: start={rmse_arr[0]:.4f} end={rmse_arr[-1]:.4f} mean={rmse_arr.mean():.4f}")
    np.save(out_base.with_suffix(".npy"), rmse_arr)
    print("Done.")


if __name__ == "__main__":
    main()