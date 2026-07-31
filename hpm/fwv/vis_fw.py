"""
fwv/vis_fw.py — fw (prior + self-feedback) 推理可视化。

与旧 vis.py 的根本区别
----------------------
旧 vis.py 是 window>0 管线, rollout 核心是 advance_window(滑窗自携带状态)。
fw 是 window=0 管线:
    输入  = [prior(t) | x_f * m (| m)]      (arm1: 2F 通道)
    base  = prior(t)                        feedback 是输入分支, 不是残差基座
    pred  = prior(t) + Δ                    Δ 只落在 delta_idx 通道
所以拼接 (assemble) 直接 import data_fw, 与训练逐字节一致 —— 不重写, 不漂。
渲染层 (slice cache / tri+scatter / slice-RMSE / ffmpeg) 从 vis.py 搬, 已验证。

两种推理模式 (同一循环体, mode 开关)
-------------------------------------
    tf       x_f = GT(t-1), 逐帧独立 (不递归)。teacher forcing, 自检用。
             真 t=0 恒 m=0。这一路应能复现 wandb val 曲线的量级 (跑 val chunk 时)。
    rollout  x_f = pred(t-1), 递归。真实部署条件。
             第一步 (k=0) 恒 m=0 冷启动 —— 部署时没有历史, 故意不喂 GT。

全程 normalized 空间递归; 只在算指标/出图时 denorm。
prior 与 x_f 用同一套 stats (data_fw 已保证, 这里沿用)。

输出: fwv/vis/... (约定: fwv 下的产物留在 fwv 下)

自检 (必看)
----------
本脚本一次跑同时算 tf 和 rollout, 并打印:
  1. tf 的 normalized-delta nRMSE (逐通道)  —— 跑 --chunk 8 时应≈ wandb val_nrmse
     对不上 -> 装配有 bug, 先修这个, 别看 rollout 结论。
  2. tf vs rollout 的 slice-RMSE 时间序列  —— 二者的 gap 就是 exposure bias 的量。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf

from hpm_model import HPM                                   # 父 hpm/
from schema import ChannelSchema                            # 父 hpm/
from dataset import load_coords, load_chunk, resolve_stats, expand_range  # 父 hpm/
from prior_ext import load_prior                            # fwv/
from data_fw import assemble, input_dim                     # fwv/

MID_Y = 0.30


# --------------------------------------------------------------- core -----

def reconstruct(prior_n, delta, delta_idx):
    """pred = prior + Δ, Δ 只加在 delta_idx 通道 (frozen 通道保持 = prior)。
    prior_n: (1,N,F) normalized;  delta: (1,N,out_dim);  返回 (1,N,F) normalized。"""
    pred = prior_n.clone()
    pred.index_add_(-1, delta_idx, delta)
    return pred


@torch.no_grad()
def run_infer(model, coords_b, prior_n, gt_n, delta_idx, mask_channel,
              mode, start, n_steps):
    """同一循环体, mode 切 tf / rollout。

    prior_n, gt_n: (T,N,F) normalized (torch, on device)
    返回:
      preds_n : (n_steps, N, F)   normalized 全场预测
      deltas  : (n_steps, N, out) normalized Δ (自检用)
    帧对齐: 第 k 步预测帧 t = start+k, 用 prior(t); tf 的 feedback = GT(t-1)。
    """
    N, F = prior_n.shape[1], prior_n.shape[2]
    device = prior_n.device
    x_f = torch.zeros(1, N, F, device=device)      # 冷启动初值
    preds, deltas = [], []

    for k in range(n_steps):
        t = start + k
        prior_t = prior_n[t:t + 1]                 # (1,N,F)

        if mode == "rollout":
            m_val = 0.0 if k == 0 else 1.0         # 首步冷启动; x_f 见循环末尾递归
        elif mode == "tf":
            if t == 0:
                m_val = 0.0
                x_f = torch.zeros(1, N, F, device=device)
            else:
                m_val = 1.0
                x_f = gt_n[t - 1:t]                # GT(t-1), 不递归
        else:
            raise ValueError(f"unknown mode {mode}")

        m_t = torch.full((1, 1, 1), m_val, device=device)
        x = assemble(prior_t, x_f, m_t, mask_channel)   # 与训练同一份拼接
        delta = model(coords_b, x)                      # (1,N,out)
        pred_t = reconstruct(prior_t, delta, delta_idx) # (1,N,F)

        preds.append(pred_t)
        deltas.append(delta)
        if mode == "rollout":
            x_f = pred_t                            # 递归: 喂自己上一步

    return torch.cat(preds, 0), torch.cat(deltas, 0)


# --------------------------------------------------------------- main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--prior_dir", required=True,
                    help="prior_ktuned 目录 (与训练 cfg.data.prior_dir 一致)")
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--start_frame", type=int, default=0,
                    help="rollout 起始帧; 0 = 真冷启动 (部署条件)")
    ap.add_argument("--n_frames", type=int, default=20,
                    help="rollout 步数; 先 20 帧验证链路, 通过再放全程")
    ap.add_argument("--field", default="alpha",
                    help="通道名 alpha/Ux/Uz/p_rgh 或 Umag(|U|); 必须在 schema 内")
    ap.add_argument("--mode", default="rollout",
                    choices=["rollout", "tf"],
                    help="出图/存视频用哪个模式; 自检数值两者都会打印")
    ap.add_argument("--also_tf_video", action="store_true",
                    help="额外为 tf 出一份视频 (默认只 rollout)")
    ap.add_argument("--no_save_preds", action="store_true",
                    help="默认存 rollout 预测 (float32) 以便重渲染不必重跑")
    ap.add_argument("--output", default="compare.mp4")
    ap.add_argument("--style", default="both",
                    choices=["scatter", "tri", "both"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- config / schema ----
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)
    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    assert cfg.data.window == 0, \
        f"fw 走 window=0, 当前 config.window={cfg.data.window} (加载了错的 config?)"

    F = schema.field_dim
    mask_ch = bool(cfg.model.mask_channel)
    in_dim = input_dim(F, mask_ch)
    print(f"{'arm2(+mask)' if mask_ch else 'arm1'}  "
          f"field_dim(输入宽度)={in_dim}  (F={F}, out_dim={schema.out_dim})")

    spectral_embedding = np.load(data_dir / "lbo" / "lbo_eigenvectors.npy")

    # ---- 模型: field_dim=in_dim (2F), 不是 schema.field_dim! 与 train_fw 一致 ----
    model = HPM(
        space_dim=3, field_dim=in_dim, out_dim=schema.out_dim,
        window=0,
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=0.0, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.get('spectral_pos_dim', 0),
        spectral_embedding=spectral_embedding,
        use_ckpt=False,
        max_grad_norm=cfg.train.get('max_grad_norm', 0.0),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- field 解析 (按名, 无魔法索引) ----
    disp = schema.display_names()
    if args.field == "Umag":
        for c in ("Ux", "Uy", "Uz"):
            assert c in schema.names, f"'Umag' 需要通道 {c}, schema={schema.names}"
        iu = [schema.names.index(c) for c in ("Ux", "Uy", "Uz")]
        vp = "αU" if schema.alpha_weighted[iu[0]] else "U"
        fname, fi = f"|{vp}|", None
    else:
        assert args.field in schema.names, \
            f"--field '{args.field}' 不在 {list(schema.names)} (或用 Umag)"
        fi = schema.names.index(args.field)
        fname = disp[fi]

    # ---- 数据: raw (schema 选列 + alpha-weight), 再用 stats 归一化 ----
    print("Loading data...")
    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = resolve_stats(args.data_dir, train_chunks, schema)
    mean, std = stats[0], stats[1]

    gt_raw = load_chunk(data_dir, args.chunk_id, schema)              # (T,N,F)
    pr_raw = load_prior(args.prior_dir, args.chunk_id, schema)        # (T,N,F)
    assert gt_raw.shape == pr_raw.shape, \
        f"GT {gt_raw.shape} vs prior {pr_raw.shape} 不符 (coords/裁剪不一致?)"

    gt_n = torch.from_numpy(((gt_raw - mean) / std).astype(np.float32)).to(device)
    pr_n = torch.from_numpy(((pr_raw - mean) / std).astype(np.float32)).to(device)
    coords_b = load_coords(args.data_dir).to(device).unsqueeze(0)     # (1,N,3)

    T = gt_raw.shape[0]
    start = args.start_frame
    n_steps = min(args.n_frames, T - start)
    assert n_steps > 0, f"start={start} 超出 chunk 长度 T={T}"
    delta_idx = torch.as_tensor(schema.delta_indices, device=device)
    dnames = [schema.names[i] for i in schema.delta_indices]

    times_path = data_dir / f"chunk_{args.chunk_id:03d}_times.npy"
    times = np.load(times_path) if times_path.exists() \
        else np.arange(T) * 0.05
    gt_times = times[start:start + n_steps]

    # ---- 两模式都推 (自检 + gap) ----
    print(f"Infer tf  {n_steps} 帧 (自检)...")
    preds_tf_n, delta_tf = run_infer(model, coords_b, pr_n, gt_n, delta_idx,
                                     mask_ch, "tf", start, n_steps)
    print(f"Infer rollout {n_steps} 帧 (start={start}, 冷启动)...")
    preds_ro_n, _ = run_infer(model, coords_b, pr_n, gt_n, delta_idx,
                              mask_ch, "rollout", start, n_steps)

    # ---- 自检 1: tf 的 normalized-delta nRMSE (对照 wandb val_nrmse) ----
    gt_delta_n = (gt_n - pr_n).index_select(-1, delta_idx)            # (T,N,out)
    gt_delta_win = gt_delta_n[start:start + n_steps]                  # (n,N,out)
    dd = (delta_tf - gt_delta_win)
    nrmse_tf = torch.sqrt((dd ** 2).mean(dim=(0, 1))).cpu().numpy()
    print("  [自检] tf normalized-delta nRMSE  "
          + "  ".join(f"{n}={v:.3f}" for n, v in zip(dnames, nrmse_tf))
          + "   (跑 --chunk 8 时应≈ wandb val_nrmse)")

    # ---- denorm 到 raw, 出图/指标用 ----
    to_raw = lambda a: a.cpu().numpy() * std + mean
    preds_tf = to_raw(preds_tf_n)
    preds_ro = to_raw(preds_ro_n)
    gts = gt_raw[start:start + n_steps]

    if not args.no_save_preds:
        pp = Path(args.output).with_name(Path(args.output).stem + "_preds_rollout.npy")
        pp.parent.mkdir(parents=True, exist_ok=True)
        np.save(pp, preds_ro.astype(np.float32))
        print(f"  rollout 预测已存: {pp} (float32, 重渲染免重跑)")

    # ---- slice cache ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")

    def to_slice(arr):
        if fi is None:
            return np.sqrt(sum(arr[:, cell_map, i] ** 2 for i in iu))
        return arr[:, cell_map, fi]

    gt_slice = to_slice(gts)
    tf_slice = to_slice(preds_tf)
    ro_slice = to_slice(preds_ro)
    pred_slice = ro_slice if args.mode == "rollout" else tf_slice

    # ---- 自检 2: tf vs rollout slice-RMSE (exposure bias 的量) ----
    rmse_tf = np.sqrt(((gt_slice - tf_slice) ** 2).mean(axis=1))
    rmse_ro = np.sqrt(((gt_slice - ro_slice) ** 2).mean(axis=1))
    print(f"  slice-RMSE {fname}:")
    print(f"    tf       start={rmse_tf[0]:.4f} end={rmse_tf[-1]:.4f} "
          f"mean={rmse_tf.mean():.4f}")
    print(f"    rollout  start={rmse_ro[0]:.4f} end={rmse_ro[-1]:.4f} "
          f"mean={rmse_ro.mean():.4f}")
    print(f"    gap(ro-tf) end={rmse_ro[-1]-rmse_tf[-1]:+.4f}  "
          f"(>0 = exposure bias 在累积)")

    # ---- colormap / range ----
    if args.field == "alpha":
        cdict = {'red': [[0., 1., 1.], [1., .6, .6]],
                 'green': [[0., 1., 1.], [1., 0., 0.]],
                 'blue': [[0., 1., 1.], [1., 0., 0.]]}
        cmap = LinearSegmentedColormap('OpacityReds', cdict); vmin, vmax = 0., 1.
    elif args.field == "Umag":
        cmap = 'magma'
        av = np.concatenate([gt_slice.ravel(), pred_slice.ravel()])
        vmin, vmax = 0., np.percentile(av, 99)
    else:
        cmap = 'coolwarm'
        av = np.concatenate([gt_slice.ravel(), pred_slice.ravel()])
        vmin, vmax = av.min(), av.max()
    levels = np.linspace(vmin, vmax, 128)
    triang = mtri.Triangulation(x_s, z_s, triangles=tri)
    xlim, zlim = (x_s.min(), x_s.max()), (z_s.min(), z_s.max())

    # ---- 出视频 ----
    def render(p_slice, tag, out_stem):
        styles = ["scatter", "tri"] if args.style == "both" else [args.style]
        for style in styles:
            out_path = Path(f"{out_stem}_{tag}_{style}.mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{tag}/{style}] -> {out_path}")
            fig, (ax_g, ax_p) = plt.subplots(2, 1, figsize=(38.4, 21.6), dpi=100)
            fig.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.95,
                                hspace=0.15)

            def draw(ax, vals, label):
                ax.clear()
                if style == "scatter":
                    ax.scatter(x_s, z_s, c=vals, s=args.point_size,
                               vmin=vmin, vmax=vmax, cmap=cmap, edgecolors='none')
                else:
                    ax.tricontourf(triang, vals, levels=levels, cmap=cmap,
                                   extend='both')
                ax.set_facecolor('white'); ax.set_xlim(xlim); ax.set_ylim(zlim)
                ax.set_xlabel("X (m)", fontsize=20); ax.set_ylabel("Z (m)", fontsize=20)
                ax.tick_params(labelsize=16); ax.set_title(label, fontsize=24)

            rmse = np.sqrt(((gt_slice - p_slice) ** 2).mean(axis=1))
            draw(ax_g, gt_slice[0], "Ground Truth")
            draw(ax_p, p_slice[0], f"HPM {tag}")
            sup = fig.suptitle(
                f"t={gt_times[0]:.2f}s | {fname} | {tag} | Step 0 | "
                f"slice-RMSE={rmse[0]:.4f}", fontsize=28)

            def upd(f):
                draw(ax_g, gt_slice[f], "Ground Truth")
                draw(ax_p, p_slice[f], f"HPM {tag}")
                sup.set_text(f"t={gt_times[f]:.2f}s | {fname} | {tag} | "
                             f"Step {f} | slice-RMSE={rmse[f]:.4f}")

            ani = animation.FuncAnimation(fig, upd, frames=n_steps,
                                          interval=1000 // args.fps, blit=False)
            ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            plt.close()

    out_stem = str(Path(args.output).with_suffix(""))
    render(pred_slice, args.mode, out_stem)
    if args.also_tf_video and args.mode != "tf":
        render(tf_slice, "tf", out_stem)

    # ---- 存 full-field RMSE 时间序列 (两模式) ----
    rmse_full_ro = np.sqrt(((gts - preds_ro) ** 2).mean(axis=1))
    rmse_full_tf = np.sqrt(((gts - preds_tf) ** 2).mean(axis=1))
    np.save(Path(out_stem + "_rmse_rollout.npy"), rmse_full_ro)
    np.save(Path(out_stem + "_rmse_tf.npy"), rmse_full_tf)
    print(f"full-field RMSE 已存 (columns={list(schema.names)})")
    print("Done.")


if __name__ == "__main__":
    main()
