"""
GT vs HPM Prediction animation — vertically stacked, PyVista-slice based.
Top: Ground Truth, Bottom: Prediction.

Schema-driven（见 schema.py）：
  - 通道集合 / alpha-weighting / stats 文件名全部来自 checkpoint 的 config
    （旧 config 无 channels 键 -> 自动 legacy 6 通道 fallback，仍可可视化旧模型）
  - rollout 闭合用共享 advance_window，与训练完全同一份代码
  - --field 按名传参：alpha / Ux / Uy / Uz / p_rgh / nut / Umag（|U| 特判）
    索引语义漂移问题从根上消灭。

全程 αU/U 空间（不还原），与训练一致。
切片几何来自预计算缓存 (slice_y0.30/)，运行时纯 numpy，不依赖 pyvista。
--style both 输出 {output}_scatter.mp4 和 {output}_tri.mp4。
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from omegaconf import OmegaConf

from hpm_model import HPM
from schema import ChannelSchema, advance_window
from dataset import load_coords, load_chunk, resolve_stats, expand_range

MID_Y = 0.30


@torch.no_grad()
def rollout(model, coords_norm, data_w, stats, window, start_frame, n_steps,
            device, schema):
    """data_w 已是 schema 选列 + alpha-weighted（与训练同空间）。输出同空间，不还原。
    窗口滑动 / frozen 通道处理全部由共享 advance_window 完成（与训练零漂移）。"""
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
        delta = model(coords_batch, fields_w)               # (1, N, out_dim)
        pred_frame, fields_w = advance_window(fields_w, delta, schema)
        predictions.append(denormalize(pred_frame[0].cpu().numpy()))

    return np.stack(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=6)
    parser.add_argument("--n_frames", type=int, default=93)
    parser.add_argument("--field", type=str, default="alpha",
                        help="channel NAME (e.g. alpha, Ux, p_rgh, nut) "
                             "or 'Umag' for |U|. Must exist in the model's schema.")
    parser.add_argument("--output", type=str, default="compare.mp4")
    parser.add_argument("--style", type=str, default="both",
                        choices=["scatter", "tri", "both"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load model + config ----
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)
    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    spectral_embedding = np.load(data_dir / "lbo" / "lbo_eigenvectors.npy")

    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = resolve_stats(data_dir, train_chunks, schema)

    model = HPM(
        field_dim=schema.field_dim, out_dim=schema.out_dim,
        space_dim=3,
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
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- Resolve --field against schema (name-based, no magic indices) ----
    disp = schema.display_names()
    if args.field == "Umag":
        for c in ("Ux", "Uy", "Uz"):
            assert c in schema.names, \
                f"'Umag' requires channel '{c}' in schema {schema.names}"
        iu = [schema.names.index(c) for c in ("Ux", "Uy", "Uz")]
        vp = "αU" if schema.alpha_weighted[iu[0]] else "U"
        fname = f"|{vp}|"
        fi = None
    else:
        assert args.field in schema.names, (
            f"--field '{args.field}' not in this model's channels "
            f"{list(schema.names)} (or use 'Umag')")
        fi = schema.names.index(args.field)
        fname = disp[fi]

    # ---- Load data (schema 选列 + alpha-weighting，与训练同空间) ----
    print("Loading data...")
    data_w = load_chunk(data_dir, args.chunk_id, schema)
    times = np.load(data_dir / f"chunk_{args.chunk_id:03d}_times.npy")
    coords_norm = load_coords(args.data_dir).to(device)

    W = cfg.data.window
    start = max(args.start_frame, W - 1)
    n_steps = min(args.n_frames, data_w.shape[0] - start - 1)

    print(f"Rolling out {n_steps} steps from frame {start}...")
    preds = rollout(model, coords_norm, data_w, stats, W, start, n_steps,
                    device, schema)
    gts = data_w[start + 1: start + 1 + n_steps]
    gt_times = times[start + 1: start + 1 + n_steps]

    # ---- slice cache ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")
    x_s, z_s = xz[:, 0], xz[:, 1]

    if fi is None:  # Umag
        gt_slice = np.sqrt(sum(gts[:, cell_map, i] ** 2 for i in iu))
        pred_slice = np.sqrt(sum(preds[:, cell_map, i] ** 2 for i in iu))
    else:
        gt_slice = gts[:, cell_map, fi]
        pred_slice = preds[:, cell_map, fi]

    # ---- colormap + range (按名判定，不再依赖魔法索引) ----
    if args.field == "alpha":
        cdict = {'red':[[0.,1.,1.],[1.,.6,.6]], 'green':[[0.,1.,1.],[1.,0.,0.]], 'blue':[[0.,1.,1.],[1.,0.,0.]]}
        custom_cmap = LinearSegmentedColormap('OpacityReds', cdict)
        vmin, vmax = 0.0, 1.0
    elif args.field == "Umag":
        custom_cmap = 'magma'
        av = np.concatenate([gt_slice.ravel(), pred_slice.ravel()])
        vmin, vmax = 0.0, np.percentile(av, 99)
    else:
        custom_cmap = 'coolwarm'
        av = np.concatenate([gt_slice.ravel(), pred_slice.ravel()])
        vmin, vmax = av.min(), av.max()
    levels = np.linspace(vmin, vmax, 128)

    triang = mtri.Triangulation(x_s, z_s, triangles=tri_simplices)
    xlim = (x_s.min(), x_s.max()); zlim = (z_s.min(), z_s.max())

    styles = ["scatter", "tri"] if args.style == "both" else [args.style]
    out_base = Path(args.output)

    for style in styles:
        out_path = out_base.with_name(f"{out_base.stem}_{style}{out_base.suffix}")
        print(f"[{style}] -> {out_path}")
        fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
        fig.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.95, hspace=0.15)

        def draw(ax, vals, label):
            ax.clear()
            if style == "scatter":
                ax.scatter(x_s, z_s, c=vals, s=args.point_size, vmin=vmin, vmax=vmax,
                           cmap=custom_cmap, edgecolors='none')
            else:
                ax.tricontourf(triang, vals, levels=levels, cmap=custom_cmap, extend='both')
            ax.set_facecolor('white'); ax.set_xlim(xlim); ax.set_ylim(zlim)
            ax.set_xlabel("X (m)", fontsize=20); ax.set_ylabel("Z (m)", fontsize=20)
            ax.tick_params(labelsize=16); ax.set_title(label, fontsize=24)

        draw(ax_gt, gt_slice[0], "Ground Truth")
        draw(ax_pred, pred_slice[0], "HPM Prediction")
        rmse0 = np.sqrt(((gt_slice[0]-pred_slice[0])**2).mean())
        suptitle = fig.suptitle(
            f"t={gt_times[0]:.2f}s | {fname} | Step 0 | slice-RMSE={rmse0:.4f}", fontsize=28)

        def update(frame):
            draw(ax_gt, gt_slice[frame], "Ground Truth")
            draw(ax_pred, pred_slice[frame], "HPM Prediction")
            rmse = np.sqrt(((gt_slice[frame]-pred_slice[frame])**2).mean())
            suptitle.set_text(
                f"t={gt_times[frame]:.2f}s | {fname} | Step {frame} | slice-RMSE={rmse:.4f}")

        ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=1000//args.fps, blit=False)
        ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
                 extra_args=['-vcodec','libx264','-pix_fmt','yuv420p'])
        plt.close()
        print(f"[{style}] saved")

    rmse_slice = np.sqrt(((gt_slice-pred_slice)**2).mean(axis=1))
    print(f"slice-RMSE: start={rmse_slice[0]:.4f} end={rmse_slice[-1]:.4f} mean={rmse_slice.mean():.4f}")
    rmse_full = np.sqrt(((gts-preds)**2).mean(axis=1))
    np.save(out_base.with_suffix(".npy"), rmse_full)
    print(f"full-field RMSE saved: {out_base.with_suffix('.npy')} "
          f"(columns = {list(schema.names)})")
    print("Done.")


if __name__ == "__main__":
    main()