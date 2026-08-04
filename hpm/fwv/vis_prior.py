"""
vis_prior.py — 1b (prior-only, single step) 的三行对比可视化。

    第 1 行  prior      起点 (FUNWAVE lift 投射到 CFD 网格)
    第 2 行  pred       prior + Δ  (模型输出)
    第 3 行  GT         CFD 目标

逐帧独立推理 —— 1b 没有 rollout, 每帧只看当前 prior, 故整个 chunk 100 帧
都能可视化, 不存在误差累积、也不需要 start_frame。

切片几何复用 vis.py 的预计算缓存 (slice_y0.30/), 三者画在同一套 CFD cell、
同一个 clim 上, 逐像素可比。

标题给两个 RMSE 与增益比:
    model=0.0812  prior=0.1903  (gain 2.34x)
gain = prior_RMSE / model_RMSE, >1 表示模型确实在修正 —— 与训练日志里
"模型/基线" 的判据同源, 只是这里是切片上的逐帧值。

用法:
  python vis_prior.py --config_path outputs/hpm_prior1b_h128/<ts>/.hydra/config.yaml \\
      --checkpoint outputs/hpm_prior1b_h128/<ts>/checkpoints/best.pt \\
      --chunk_id 8 --field alpha --output vis/prior1b/c8_alpha.mp4
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf

from dataset import expand_range, load_chunk, load_coords, resolve_stats
from hpm_fwv import HPM, strip_legacy_basis
from prior_ext import assert_prior_compatible, load_prior
from schema import ChannelSchema

MID_Y = 0.30


@torch.no_grad()
def predict_all(model, coords, prior_n, device, batch_frames=1):
    """逐帧独立推理 (无 rollout)。prior_n: (T,N,F) 已归一化。返回 (T,N,F)。"""
    T = prior_n.shape[0]
    out = np.empty_like(prior_n)
    cb = coords.unsqueeze(0)
    for t in range(T):
        p = torch.from_numpy(prior_n[t]).unsqueeze(0).to(device)   # (1,N,F)
        delta = model(cb, p)                                        # (1,N,out)
        out[t] = (p + delta)[0].cpu().numpy()
        if t % 20 == 0:
            print(f"  frame {t+1}/{T}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--field", default="alpha",
                    help="通道名 (alpha/Ux/Uz/p_rgh) 或 'Umag' (|U|)")
    ap.add_argument("--n_frames", type=int, default=0, help="0 = 整个 chunk")
    ap.add_argument("--output", default="vis_prior/compare.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fig-w", type=float, default=38.4, dest="fig_w")
    ap.add_argument("--fig-h", type=float, default=24.0, dest="fig_h")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config_path)
    data_dir = Path(cfg.data.dir)
    prior_dir = Path(cfg.data.prior_dir)

    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    assert_prior_compatible(schema)
    assert cfg.data.window == 0, "vis_prior.py 只支持 1b (window=0)"

    # ---- 模型 ----
    spectral_embedding = np.load(data_dir / "lbo" / "lbo_eigenvectors.npy")
    model = HPM(
        space_dim=3, field_dim=schema.field_dim, out_dim=schema.out_dim,
        window=0,
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=0.0, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.spectral_pos_dim,
        spectral_embedding=spectral_embedding, use_ckpt=False,
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(strip_legacy_basis(ckpt["model"]), strict=True); model.eval()
    print(f"Loaded checkpoint (epoch {ck.get('epoch','?')}, "
          f"best_val {ck.get('best_val', float('nan')):.6f})")

    # ---- 数据 (与训练同一套 stats) ----
    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = resolve_stats(str(data_dir), train_chunks, schema)
    mean, std = stats[0], stats[1]

    gt = load_chunk(str(data_dir), args.chunk_id, schema)          # (T,N,F) 物理
    prior = load_prior(str(prior_dir), args.chunk_id, schema)
    times = np.load(data_dir / f"chunk_{args.chunk_id:03d}_times.npy")
    if args.n_frames > 0:
        gt, prior, times = gt[:args.n_frames], prior[:args.n_frames], times[:args.n_frames]
    print(f"chunk {args.chunk_id}: {gt.shape}  t {times[0]:.2f}..{times[-1]:.2f}")

    coords = load_coords(str(data_dir)).to(device)
    prior_n = ((prior - mean) / std).astype(np.float32)
    print("Predicting (per-frame, no rollout)...")
    pred_n = predict_all(model, coords, prior_n, device)
    pred = pred_n * std + mean                                     # 回物理空间

    # ---- 切片缓存 ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")

    disp = schema.display_names()
    if args.field == "Umag":
        iu = [schema.names.index(c) for c in ("Ux", "Uz")]   # 1b 无 Uy
        sl = lambda a: np.sqrt(sum(a[:, cell_map, i] ** 2 for i in iu))
        fname = "|U| (xz)"
    else:
        assert args.field in schema.names, \
            f"--field '{args.field}' 不在 {list(schema.names)}"
        fi = schema.names.index(args.field)
        sl = lambda a: a[:, cell_map, fi]
        fname = disp[fi]

    s_pri, s_prd, s_gt = sl(prior), sl(pred), sl(gt)

    # ---- 配色 + 共享 clim (三行必须同标, 否则不可比) ----
    if args.field == "alpha":
        cdict = {'red': [[0., 1., 1.], [1., .6, .6]],
                 'green': [[0., 1., 1.], [1., 0., 0.]],
                 'blue': [[0., 1., 1.], [1., 0., 0.]]}
        cmap = LinearSegmentedColormap('OpacityReds', cdict)
        vmin, vmax = 0.0, 1.0
    elif args.field == "Umag":
        cmap = 'magma'
        vmin, vmax = 0.0, np.percentile(np.concatenate(
            [s_gt.ravel(), s_prd.ravel()]), 99)
    else:
        cmap = 'coolwarm'
        av = np.concatenate([s_gt.ravel(), s_prd.ravel(), s_pri.ravel()])
        m = np.percentile(np.abs(av), 99)
        vmin, vmax = -m, m
    levels = np.linspace(vmin, vmax, 128)
    print(f"[clim] {fname}: {vmin:.4g} .. {vmax:.4g}")

    triang = mtri.Triangulation(x_s, z_s, triangles=tri_simplices)
    xlim = (x_s.min(), x_s.max()); zlim = (z_s.min(), z_s.max())

    # ---- 逐帧 RMSE (切片上) ----
    rmse_m = np.sqrt(((s_prd - s_gt) ** 2).mean(axis=1))
    rmse_p = np.sqrt(((s_pri - s_gt) ** 2).mean(axis=1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = s_gt.shape[0]

    fig, axes = plt.subplots(3, 1, figsize=(args.fig_w, args.fig_h), dpi=100)
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.97, hspace=0.18)

    def draw(ax, vals, label):
        ax.clear()
        ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend='both')
        ax.set_facecolor('white')
        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_xlabel("X (m)", fontsize=18)
        ax.set_ylabel("Z (m)", fontsize=18)
        ax.tick_params(labelsize=14)
        ax.set_title(label, fontsize=22)

    suptitle = fig.suptitle("", fontsize=26)

    def update(f):
        draw(axes[0], s_pri[f], "prior  (FUNWAVE lift)")
        draw(axes[1], s_prd[f], "pred  =  prior + Δ  (HPM)")
        draw(axes[2], s_gt[f], "GT  (CFD)")
        gain = rmse_p[f] / max(rmse_m[f], 1e-12)
        suptitle.set_text(
            f"chunk {args.chunk_id} | t={times[f]:.2f}s | frame {f} | {fname} | "
            f"slice-RMSE  model={rmse_m[f]:.4f}  prior={rmse_p[f]:.4f}  "
            f"(gain {gain:.2f}x)")

    update(0)
    print(f"[tri] -> {out_path}")
    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=1000 // args.fps, blit=False)
    ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)

    np.save(out_path.with_suffix(".rmse.npy"),
            np.stack([rmse_m, rmse_p], axis=0))
    g = rmse_p.mean() / max(rmse_m.mean(), 1e-12)
    print(f"[done] slice-RMSE  model={rmse_m.mean():.4f}  "
          f"prior={rmse_p.mean():.4f}  gain={g:.2f}x")
    print(f"       逐帧 RMSE 已存 {out_path.with_suffix('.rmse.npy')} "
          f"(第0行 model, 第1行 prior)")


if __name__ == "__main__":
    main()
