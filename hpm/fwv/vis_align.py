#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_align.py — 配准检查: prior 与 CFD GT 的两行对比 (训练前, 不需要 checkpoint)。

    第 1 行  prior   FUNWAVE lift + x/t-offset + 投到 CFD 网格
    第 2 行  GT      CFD

定位 —— 三个可视化工具的分工:
    vis_lift.py   FUNWAVE 原生网格, 看抬升算子本身长什么样   (只要 FUNWAVE)
    vis_align.py  CFD 网格, 看配准对不对                      (prior + GT)   <- 本文件
    vis_prior.py  CFD 网格, 看模型修正效果                    (+ checkpoint)

为什么需要它: scan_toffset.py 给的是 RMSE 曲线极小值 —— 一个数。它看不出
波高系统性偏低 (Boussinesq 浅化低估)、相位对了但波形不对、破碎区分叉形态。
而 vis_prior.py 要训练完才能跑, 配准却应该在训练之前就确认。

逐 chunk t-offset
-----------------
默认从 toffset_scan/c{cid:03d}.json 读该 chunk 的 best_k, 与 prior 落盘时用的
k_base (prior_meta*.json) 比较, 在**显示层**平移帧配对:

    GT 第 i 帧  <->  prior 第 (i + best_k - k_base) 帧

即用已有的 prior 数据, 不重新生成。代价是边缘丢 |Δk| 帧 (通常 1-2 帧)。
标题里始终写明 k_base / k_use / Δ, 不会看不出用的是哪个。

--k 可显式覆盖, 用来目视复核 scan 的结论:
    --k 2 / --k 3 / --k 4  各出一支, 看哪个真的对齐

用法
----
  python vis_align.py --chunk 9 --field alpha
  python vis_align.py --chunk 9 --field alpha --k 5 --output /tmp/c9_k5.mp4
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import lift as LC

MID_Y = 0.30
_HERE = os.path.dirname(os.path.abspath(__file__))


def read_prior_k(prior_dir, cid):
    """prior 落盘时用的 t-offset -> k_base。找不到返回 None。"""
    cands = [Path(prior_dir) / f"prior_meta_c{cid:03d}.json",
             Path(prior_dir) / "prior_meta.json"]
    for p in cands:
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            t_off, dt = m.get("t_offset"), m.get("plot_intv")
            if t_off is not None and dt:
                return int(round(t_off / dt)), str(p.name)
    return None, None


def read_scan_k(scan_dir, cid):
    """toffset_scan/c{cid}.json 里标定出的 best_k。找不到返回 None。"""
    p = Path(scan_dir) / f"c{cid:03d}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return d.get("best_k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--field", default="alpha",
                    help="通道名 (alpha/Ux/Uy/Uz/p_rgh) 或 'Umag' (sqrt(Ux^2+Uz^2))")
    ap.add_argument("--data-dir", default="../data/3d/cropped_0.05")
    ap.add_argument("--prior-dir", default="../data/3d/prior_t015")
    ap.add_argument("--scan-dir", default=os.path.join(_HERE, "toffset_scan"),
                    help="scan_toffset.py 的产物目录 (逐 chunk best_k)")
    ap.add_argument("--k", type=int, default=None,
                    help="显式指定帧移 k, 覆盖 scan 结果。用于目视复核。")
    ap.add_argument("--k-base", type=int, default=None,
                    help="prior 落盘时的 k。默认从 prior_meta*.json 读。")
    ap.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 各自的列索引")
    ap.add_argument("--n-frames", type=int, default=0, dest="n_frames",
                    help="0 = 整个 chunk")
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--fig-w", type=float, default=38.4, dest="fig_w")
    ap.add_argument("--fig-h", type=float, default=16.0, dest="fig_h")
    args = ap.parse_args()

    cid = args.chunk
    data_dir, prior_dir = Path(args.data_dir), Path(args.prior_dir)
    names = list(LC.CH_NAMES)

    # ---- 帧移: k_base (落盘时) -> k_use (本次显示) ----
    k_base, meta_src = (args.k_base, "--k-base") if args.k_base is not None \
        else read_prior_k(prior_dir, cid)
    if k_base is None:
        raise SystemExit(
            f"[err] 读不到 prior 的 k_base ({prior_dir}/prior_meta*.json 都没有),\n"
            f"      请显式给 --k-base (prior_t015 对应 --k-base 3)")

    if args.k is not None:
        k_use, k_src = args.k, "--k"
    else:
        k_use = read_scan_k(args.scan_dir, cid)
        k_src = f"{args.scan_dir}/c{cid:03d}.json"
        if k_use is None:
            print(f"[warn] {k_src} 不存在 -> 沿用落盘 k={k_base:+d} (Δ=0)。"
                  f" 先跑 scan_toffset.py, 或用 --k 指定。")
            k_use, k_src = k_base, "fallback"
    dk = k_use - k_base
    print(f"[k   ] 落盘 k_base={k_base:+d} ({meta_src})   "
          f"本次 k_use={k_use:+d} ({k_src})   位移 Δ={dk:+d} 帧")

    # ---- 数据 (mmap, 只取切片 cell) ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    print(f"[slice] {len(cell_map)} faces @ y={MID_Y}")

    gt_all = np.load(data_dir / f"chunk_{cid:03d}_data.npy", mmap_mode="r")
    pr_all = np.load(prior_dir / f"prior_{cid:03d}_data.npy", mmap_mode="r")
    times = np.load(data_dir / f"chunk_{cid:03d}_times.npy")
    T = min(gt_all.shape[0], pr_all.shape[0], len(times))

    # GT 第 i 帧 <-> prior 第 i+dk 帧; 越界的帧丢掉
    i_gt = np.arange(max(0, -dk), min(T, T - dk))
    if len(i_gt) == 0:
        raise SystemExit(f"[err] Δ={dk} 超出 chunk 长度 {T}, 无可配对帧")
    if args.n_frames > 0:
        i_gt = i_gt[:args.n_frames]
    i_pr = i_gt + dk
    if dk != 0:
        print(f"[pair] Δ={dk:+d} -> 丢弃边缘 {T - len(i_gt)} 帧, "
              f"剩 {len(i_gt)} 帧 (GT {i_gt[0]}..{i_gt[-1]})")

    gt_sl = np.asarray(gt_all[i_gt][:, cell_map, :][:, :, args.gt_channels],
                       dtype=np.float32)
    pr_sl = np.asarray(pr_all[i_pr][:, cell_map, :], dtype=np.float32)
    t_show = times[i_gt]

    # ---- 取场 ----
    if args.field == "Umag":
        iu = [names.index(c) for c in ("Ux", "Uz")]
        s_gt = np.sqrt(sum(gt_sl[..., i] ** 2 for i in iu))
        s_pr = np.sqrt(sum(pr_sl[..., i] ** 2 for i in iu))
        fname = "|U| (xz)"
    else:
        if args.field not in names:
            raise SystemExit(f"[err] --field '{args.field}' 不在 {names} / Umag")
        fi = names.index(args.field)
        s_gt, s_pr = gt_sl[..., fi], pr_sl[..., fi]
        fname = args.field

    # ---- 配色 + 共享 clim (两行必须同标, 否则不可比) ----
    if args.field == "alpha":
        cdict = {'red':   [[0., 1., 1.], [1., .6, .6]],
                 'green': [[0., 1., 1.], [1., 0., 0.]],
                 'blue':  [[0., 1., 1.], [1., 0., 0.]]}
        cmap = LinearSegmentedColormap('OpacityReds', cdict)
        vmin, vmax = 0.0, 1.0
    elif args.field == "Umag":
        cmap = 'magma'
        vmin, vmax = 0.0, float(np.percentile(s_gt, 99))
    else:
        cmap = 'coolwarm'
        m = float(np.percentile(np.abs(np.concatenate(
            [s_gt.ravel(), s_pr.ravel()])), 99))
        vmin, vmax = -m, m
    levels = np.linspace(vmin, vmax, 128)
    print(f"[clim] {fname}: {vmin:.4g} .. {vmax:.4g}")

    # ---- 逐帧指标 (切片上) ----
    rmse = np.sqrt(((s_pr - s_gt) ** 2).mean(axis=1))
    corr = np.full(len(i_gt), np.nan)
    for f in range(len(i_gt)):
        a, b = s_pr[f], s_gt[f]
        sa, sb = a.std(), b.std()
        if sa > 1e-12 and sb > 1e-12:
            corr[f] = np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)

    # ---- 渲染 ----
    triang = mtri.Triangulation(x_s, z_s, triangles=tri_simplices)
    xlim, zlim = (x_s.min(), x_s.max()), (z_s.min(), z_s.max())

    out_path = Path(args.output) if args.output else Path(
        _HERE) / "vis_align" / f"c{cid}_{args.field}_k{k_use:+d}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(args.fig_w, args.fig_h), dpi=100)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.97, hspace=0.20)

    def draw(ax, vals, label):
        ax.clear()
        ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend='both')
        ax.set_facecolor('white')
        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_xlabel("X (m)", fontsize=18)
        ax.set_ylabel("Z (m)", fontsize=18)
        ax.tick_params(labelsize=14)
        ax.set_title(label, fontsize=22)

    suptitle = fig.suptitle("", fontsize=24)
    ktag = f"k={k_use:+d} (落盘 {k_base:+d}, Δ{dk:+d})"

    def update(f):
        draw(axes[0], s_pr[f], "prior  (FUNWAVE lift -> CFD grid)")
        draw(axes[1], s_gt[f], "GT  (CFD)")
        suptitle.set_text(
            f"chunk {cid} | t={t_show[f]:.2f}s | frame {i_gt[f]} | {fname} | "
            f"{ktag} | slice-RMSE {rmse[f]:.4f}   corr {corr[f]:.3f}")

    update(0)
    print(f"[vid ] -> {out_path}")
    ani = animation.FuncAnimation(fig, update, frames=len(i_gt),
                                  interval=1000 // args.fps, blit=False)
    ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)

    print(f"[done] {fname}  k={k_use:+d}   "
          f"slice-RMSE 均值 {rmse.mean():.4f}   corr 均值 {np.nanmean(corr):.3f}")
    print("       (比较不同 --k 时看这两个数: RMSE 越低 / corr 越高 越对齐)")


if __name__ == "__main__":
    main()
