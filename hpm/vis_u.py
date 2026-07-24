"""
RGB Velocity Visualization for HPM rollout — PyVista-slice based.

Schema-driven（见 schema.py）：
  - 通道集合 / alpha-weighting / stats 来自 checkpoint 的 config
    （旧 config 无 channels 键 -> legacy fallback，旧模型仍可可视化）
  - rollout 闭合用共享 advance_window，与训练同一份代码
  - 速度分量按名查找（Ux/Uy/Uz），不再假设索引 1/2/3

产出两类视频（同一次 rollout 复用）：

  (A) 老图：速度三分量编码为一张彩色图（αU 或 U 空间，由 schema 决定，不还原）
      R = |Ux|/max, G = |Uy|/max, B = |Uz|/max, Opacity = |U|/max
      pred 上叠加方向误差（>90° 标黄）。看整体方向性 (directionality)。

  (B) 三分量单独：Ux / Uy / Uz 各一张，暗底 + 双色相 (two-hue)
      warm 暖=正 / cool 冷=负，饱和度 saturation = |分量|。
      三分量共享一对色相 + 共享色标 V=max(P99|Ux|,P99|Uy|,P99|Uz|)，
      保留跨分量量级对比性。单分量图不叠黄块（方向诊断交给老图）。

Top: GT, Bottom: HPM Prediction。切片几何用预计算缓存 (slice_y0.30/)，运行时纯 numpy。
--style both -> 每类各出 _tri.mp4（精准连续）和 _scatter.mp4（快速离散）。

输出文件（out_base = compare_chunkN_u.mp4，chunk 目录下）：
  老图:   compare_chunkN_u_tri.mp4      compare_chunkN_u_scatter.mp4
  三分量: compare_chunkN_u_aUx_tri.mp4  compare_chunkN_u_aUx_scatter.mp4  (Uy/Uz 同)
  RMSE:   compare_chunkN_u.npy  (|U| full-field, 与老版一致)

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
from schema import ChannelSchema, advance_window
from dataset import load_coords, load_chunk, resolve_stats, expand_range

MID_Y = 0.30


@torch.no_grad()
def rollout(model, coords_norm, data_w, stats, window, start_frame, n_steps,
            device, schema):
    """窗口滑动 / frozen 通道处理由共享 advance_window 完成（与训练零漂移）。"""
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


def build_rgba_points(ux, uy, uz, ux_max, uy_max, uz_max, umag_max):
    """老图逐点 RGBA。ux/uy/uz: (M,) 切片点上的速度分量。返回 (M,4)。"""
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


def build_component_rgb(v, vmax, bg, warm, cool):
    """单分量双色相 (two-hue) 逐点 RGB。
    v: (M,) 带符号 signed 分量值；vmax: 共享色标 shared scale；
    正值走 warm 暖、负值走 cool 冷，饱和度 saturation = |v|/vmax（over 暗底）。返回 (M,3)。"""
    a = np.clip(np.abs(v) / max(vmax, 1e-10), 0, 1)[:, None]          # (M,1)
    base = np.where((v >= 0)[:, None], warm[None, :], cool[None, :])  # (M,3)
    return np.clip(base * a + np.array(bg).reshape(1, 3) * (1 - a), 0, 1)


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
    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    spectral_embedding = np.load(data_dir / "lbo" / "lbo_eigenvectors.npy")
    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = resolve_stats(data_dir, train_chunks, schema)

    # 速度分量按名查找（不再假设 1/2/3）
    for c in ("Ux", "Uy", "Uz"):
        assert c in schema.names, \
            f"vis_u.py requires velocity channel '{c}' in schema {schema.names}"
    iux, iuy, iuz = (schema.names.index(c) for c in ("Ux", "Uy", "Uz"))

    model = HPM(
        field_dim=schema.field_dim, out_dim=schema.out_dim,
        space_dim=3,
        window=cfg.data.window,
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=0.0, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.get('spectral_pos_dim', 0),
        spectral_embedding=spectral_embedding, use_ckpt=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()

    # ---- data（schema 选列 + alpha-weighting，与训练同空间）----
    print("Loading data...")
    data_w = load_chunk(data_dir, args.chunk_id, schema)
    times = np.load(data_dir / f"chunk_{args.chunk_id:03d}_times.npy")
    coords_norm = load_coords(args.data_dir).to(device)

    W = cfg.data.window
    start = max(args.start_frame, W - 1)
    n_steps = min(args.n_frames, data_w.shape[0] - start - 1)

    print(f"Rolling out {n_steps} steps...")
    preds = rollout(model, coords_norm, data_w, stats, W, start, n_steps,
                    device, schema)
    gts = data_w[start + 1: start + 1 + n_steps]
    gt_times = times[start + 1: start + 1 + n_steps]

    # ---- slice cache ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")

    # 速度分量在切片点上 (n_steps, M) — 索引来自 schema
    gt_u = (gts[:, cell_map, iux], gts[:, cell_map, iuy], gts[:, cell_map, iuz])
    pr_u = (preds[:, cell_map, iux], preds[:, cell_map, iuy], preds[:, cell_map, iuz])

    # 归一化用 GT 的 αU/U percentile
    ux_max = np.percentile(np.abs(gt_u[0]), 99)
    uy_max = np.percentile(np.abs(gt_u[1]), 99)
    uz_max = np.percentile(np.abs(gt_u[2]), 99)
    umag_max = np.percentile(np.sqrt(gt_u[0]**2 + gt_u[1]**2 + gt_u[2]**2), 99)
    print(f"Norm: |Ux|={ux_max:.3f} |Uy|={uy_max:.3f} |Uz|={uz_max:.3f} |U|={umag_max:.3f}")

    # 动态标签：αU vs U（由 schema 的 alpha_weighted 决定）
    vp = "αU" if schema.alpha_weighted[iux] else "U"   # velocity prefix
    legend_text = (f"R=|{vp}x|  G=|{vp}y|  B=|{vp}z|  "
                   f"Opacity=|{vp}|  Yellow=dir err (>90°)")

    speed_thresh = umag_max * 0.05
    yellow = np.array([1.0, 0.9, 0.0])
    xlim = (x_s.min(), x_s.max()); zlim = (z_s.min(), z_s.max())

    # ---- 共用渲染原语 (render primitive)：逐点 RGB -> tri/scatter，老图与三分量共用 ----
    def render_points(ax, rgb, style, label):
        ax.clear()
        if style == "tri":
            # 每三角形 RGB = 三顶点均值。PolyCollection 原生支持真彩色，
            # 不绕 tripcolor 的标量映射，跨 mpl 版本最稳。
            tri_rgb = rgb[tri_simplices].mean(axis=1)                        # (T,3)
            verts = np.stack([x_s[tri_simplices], z_s[tri_simplices]], axis=-1)  # (T,3,2)
            ax.add_collection(PolyCollection(verts, facecolors=tri_rgb, edgecolors='none'))
        else:
            ax.scatter(x_s, z_s, c=rgb, s=args.point_size, edgecolors='none')
        ax.set_facecolor(bg); ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_xlabel("X (m)", fontsize=20, color='white')
        ax.set_ylabel("Z (m)", fontsize=20, color='white')
        ax.tick_params(labelsize=16, colors='white')
        ax.set_title(label, fontsize=24, color='white')

    def frame_colors(frame):
        """老图：该帧 gt_rgb(M,3), pred_rgb(M,3, 含方向误差叠加), error_pct。"""
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

    # ============================================================
    # (A) 老图：RGB 合成 + 黄块方向误差
    # ============================================================
    for style in styles:
        out_path = out_base.with_name(f"{out_base.stem}_{style}{out_base.suffix}")
        print(f"[RGB/{style}] -> {out_path}")
        fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
        fig.patch.set_facecolor(bg)
        fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.95, hspace=0.12)
        fig.text(0.5, 0.97, legend_text, ha='center', fontsize=22, color='white',
                 family='monospace')
        suptitle = fig.suptitle("", fontsize=28, color='white', y=0.94)

        def update(frame):
            gt_rgb, pred_rgb, err_pct = frame_colors(frame)
            render_points(ax_gt, gt_rgb, style, "Ground Truth")
            render_points(ax_pred, pred_rgb, style, "HPM Prediction")
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
        print(f"[RGB/{style}] saved")

    # ============================================================
    # (B) 三分量单独：双色相 warm=+ / cool=−，共享色标 V
    #     V = max(P99|Ux|, P99|Uy|, P99|Uz|)  -> 保留跨分量量级对比
    # ============================================================
    V = max(ux_max, uy_max, uz_max)
    warm = np.array([1.0, 0.45, 0.12])   # 正 positive : 暖橙红 orange-red
    cool = np.array([0.12, 0.60, 1.0])   # 负 negative : 冷青蓝 cyan-blue
    print(f"Component shared scale V={V:.3f}  (warm=+, cool=-)")

    # (显示名, gt_u/pr_u 索引, +物理含义, −物理含义)
    comp_defs = [
        (f"{vp}x", 0, "shoreward +X", "seaward -X"),
        (f"{vp}y", 1, "lateral +Y",   "lateral -Y"),
        (f"{vp}z", 2, "rising +Z",     "plunging -Z"),
    ]

    for cname, cidx, pos_txt, neg_txt in comp_defs:
        g_comp = gt_u[cidx]   # (n_steps, M)
        p_comp = pr_u[cidx]
        comp_legend = (f"Warm=+ ({pos_txt})    Cool=- ({neg_txt})    "
                       f"Saturation=|{cname}|    shared max V={V:.3f}")
        cfile = cname.replace("α", "a")   # 文件名避免非 ASCII: αUx -> aUx

        for style in styles:
            out_path = out_base.with_name(f"{out_base.stem}_{cfile}_{style}{out_base.suffix}")
            print(f"[{cname}/{style}] -> {out_path}")
            fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
            fig.patch.set_facecolor(bg)
            fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.95, hspace=0.12)
            fig.text(0.5, 0.97, comp_legend, ha='center', fontsize=22, color='white',
                     family='monospace')
            suptitle = fig.suptitle("", fontsize=28, color='white', y=0.94)

            def update(frame):
                gt_rgb = build_component_rgb(g_comp[frame], V, bg, warm, cool)
                pred_rgb = build_component_rgb(p_comp[frame], V, bg, warm, cool)
                render_points(ax_gt, gt_rgb, style, "Ground Truth")
                render_points(ax_pred, pred_rgb, style, "HPM Prediction")
                rmse = np.sqrt(np.mean((g_comp[frame] - p_comp[frame])**2))
                suptitle.set_text(f"t={gt_times[frame]:.2f}s | {cname} | "
                                  f"Step {frame} | RMSE={rmse:.4f}")

            ani = animation.FuncAnimation(fig, update, frames=n_steps,
                                          interval=1000//args.fps, blit=False)
            ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
                     extra_args=['-vcodec','libx264','-pix_fmt','yuv420p'])
            plt.close()
            print(f"[{cname}/{style}] saved")

    # |U| RMSE summary (全场，与老版一致) — 索引来自 schema
    rmse_arr = np.array([
        np.sqrt(np.mean((np.sqrt(gts[f,:,iux]**2+gts[f,:,iuy]**2+gts[f,:,iuz]**2)
                       - np.sqrt(preds[f,:,iux]**2+preds[f,:,iuy]**2+preds[f,:,iuz]**2))**2))
        for f in range(n_steps)])
    print(f"|{vp}| RMSE: start={rmse_arr[0]:.4f} end={rmse_arr[-1]:.4f} mean={rmse_arr.mean():.4f}")
    np.save(out_base.with_suffix(".npy"), rmse_arr)
    print("Done.")


if __name__ == "__main__":
    main()