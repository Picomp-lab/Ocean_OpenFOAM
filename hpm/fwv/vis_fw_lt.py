"""
fwv/vis_fw_lt.py — long-term rollout 可视化 (无 GT)。

与 vis_fw.py 的区别
-------------------
vis_fw.py 是"有 GT"的评估工具: 加载 GT、算自检、slice-RMSE、上下排 GT|pred。
本文件是"无 GT"的 long-term 定性工具:
  - 不加载 GT (chunk10 只有 prior, 没有 CFD 真值)
  - 单排只画 pred rollout
  - 不算 slice-RMSE (无对照)
  - 跑满 prior 全长 (1000 帧), 边 rollout 边写视频帧 (不全存内存, 防 OOM)
  - 打印每帧推理耗时 (部署参考, 对应 gen_prior 的 lift ms/帧)

装配与 vis_fw.py / 训练完全一致: assemble 复用 data_fw, pred = prior + Δ,
全程 normalized 递归, 只在渲染时 denorm。arm1 结构 (field_dim=2F)。

rollout 语义 (与部署一致)
    k=0    x_f=0, m=0    冷启动 (无历史)
    k>=1   x_f=pred(k-1) 递归 (喂自己上一步)

输出: fwv/vis/<feature>/<field>/longterm_chunk{cid}_{field}_tri.mp4
"""

import argparse
import time
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
from dataset import load_coords, resolve_stats, expand_range  # 父 hpm/
from prior_ext import load_prior                            # fwv/
from data_fw import assemble, input_dim                     # fwv/

MID_Y = 0.30


def reconstruct(prior_n, delta, delta_idx):
    """pred = prior + Δ, Δ 只加在 delta_idx 通道。normalized 空间。"""
    pred = prior_n.clone()
    pred.index_add_(-1, delta_idx, delta)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--prior_dir", required=True)
    ap.add_argument("--chunk_id", type=int, required=True,
                    help="long-term chunk (只需 prior, 无 GT)")
    ap.add_argument("--n_frames", type=int, default=0,
                    help="rollout 步数; 0 = 跑满 prior 全长")
    ap.add_argument("--field", default="alpha",
                    help="通道名; 默认 alpha (保留其它场选择, 日后用)")
    ap.add_argument("--output", default="longterm.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- config / schema / 模型 (与 vis_fw 一致) ----
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)
    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    assert cfg.data.window == 0, f"fw 走 window=0, 当前 {cfg.data.window}"

    F = schema.field_dim
    mask_ch = bool(cfg.model.mask_channel)
    in_dim = input_dim(F, mask_ch)
    print(f"{'arm2(+mask)' if mask_ch else 'arm1'}  输入宽度={in_dim} "
          f"(F={F}, out_dim={schema.out_dim})")

    spectral_embedding = np.load(data_dir / "lbo" / "lbo_eigenvectors.npy")
    model = HPM(
        space_dim=3, field_dim=in_dim, out_dim=schema.out_dim, window=0,
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=0.0, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.get('spectral_pos_dim', 0),
        spectral_embedding=spectral_embedding, use_ckpt=False,
        max_grad_norm=cfg.train.get('max_grad_norm', 0.0),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- field 解析 (无 Umag 分支, long-term 先只标量场) ----
    disp = schema.display_names()
    assert args.field in schema.names, \
        f"--field '{args.field}' 不在 {list(schema.names)}"
    fi = schema.names.index(args.field)
    fname = disp[fi]

    # ---- prior only (无 GT); 用训练 stats 归一化 ----
    print("Loading prior (no GT)...")
    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = resolve_stats(args.data_dir, train_chunks, schema)
    mean, std = stats[0], stats[1]

    pr_raw = load_prior(args.prior_dir, args.chunk_id, schema)        # (T,N,F)
    pr_n = torch.from_numpy(((pr_raw - mean) / std).astype(np.float32)).to(device)
    coords_b = load_coords(args.data_dir).to(device).unsqueeze(0)     # (1,N,3)

    T = pr_raw.shape[0]
    n_steps = T if args.n_frames <= 0 else min(args.n_frames, T)
    N = pr_raw.shape[1]
    delta_idx = torch.as_tensor(schema.delta_indices, device=device)

    times_path = data_dir / f"chunk_{args.chunk_id:03d}_times.npy"
    times = np.load(times_path) if times_path.exists() else np.arange(T) * 0.05

    # ---- slice cache ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    triang = mtri.Triangulation(x_s, z_s, triangles=tri)
    xlim, zlim = (x_s.min(), x_s.max()), (z_s.min(), z_s.max())
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")

    # ---- colormap ----
    if args.field == "alpha":
        cdict = {'red': [[0., 1., 1.], [1., .6, .6]],
                 'green': [[0., 1., 1.], [1., 0., 0.]],
                 'blue': [[0., 1., 1.], [1., 0., 0.]]}
        cmap = LinearSegmentedColormap('OpacityReds', cdict); vmin, vmax = 0., 1.
    else:
        cmap = 'coolwarm'
        # 无 GT, 用 prior 该场的分位数定色标 (raw 空间)
        pf = pr_raw[:, cell_map, fi]
        vmin, vmax = float(np.percentile(pf, 1)), float(np.percentile(pf, 99))
    levels = np.linspace(vmin, vmax, 128)

    # ---- 流式 rollout + 渲染 (不全存 pred, 防 OOM) ----
    # FuncAnimation 每帧回调里推一步 rollout, 只保留当前帧 + 递归所需的 x_f。
    out_path = Path(args.output).with_name(Path(args.output).stem + "_tri.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[longterm] rollout {n_steps} 帧, 边推边渲染 -> {out_path}")

    denorm_f = lambda col_n: col_n * std[fi] + mean[fi]   # 单通道反归一化

    fig, ax = plt.subplots(figsize=(38.4, 10.8), dpi=100)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.97)

    # rollout 状态 (闭包内递归)
    state = {"x_f": torch.zeros(1, N, F, device=device), "k": 0,
             "t_infer": 0.0}

    @torch.no_grad()
    def step_rollout(k):
        """推第 k 帧, 返回该帧 field 的 slice 值 (raw)。递归更新 x_f。"""
        prior_t = pr_n[k:k + 1]                     # (1,N,F)
        m_val = 0.0 if k == 0 else 1.0
        m_t = torch.full((1, 1, 1), m_val, device=device)
        x = assemble(prior_t, state["x_f"], m_t, mask_ch)

        t0 = time.perf_counter()
        delta = model(coords_b, x)                  # (1,N,out)
        pred_t = reconstruct(prior_t, delta, delta_idx)  # (1,N,F) normalized
        if device.type == "cuda":
            torch.cuda.synchronize()
        state["t_infer"] += time.perf_counter() - t0

        state["x_f"] = pred_t                        # 递归
        col_n = pred_t[0, cell_map, fi].cpu().numpy()   # 该场 slice (normalized)
        return denorm_f(col_n)                       # -> raw

    def draw(vals, k):
        ax.clear()
        ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend='both')
        ax.set_facecolor('white'); ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_xlabel("X (m)", fontsize=20); ax.set_ylabel("Z (m)", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_title(f"HPM rollout (long-term, no GT)  {fname}", fontsize=24)
        fig.suptitle(f"t={times[k]:.2f}s | frame {k}/{n_steps-1}", fontsize=26)

    def upd(k):
        vals = step_rollout(k)
        draw(vals, k)
        if k % 50 == 0:
            avg = state["t_infer"] / max(k + 1, 1) * 1000
            print(f"  frame {k}/{n_steps-1}  推理均值 {avg:.1f} ms/帧", flush=True)

    ani = animation.FuncAnimation(fig, upd, frames=n_steps,
                                  interval=1000 // args.fps, blit=False)
    ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close()

    avg = state["t_infer"] / max(n_steps, 1) * 1000
    print(f"[infer] rollout {n_steps} 帧, 前向累计 {state['t_infer']:.1f}s, "
          f"均值 {avg:.1f} ms/帧  ({N} cells)")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
