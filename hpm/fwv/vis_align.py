#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_align.py — 配准检查: prior 与 CFD GT 的两行对比。**生成 prior 之前**跑。

    第 1 行  prior   FUNWAVE lift + x/t-offset, 现算在切片 cell 上
    第 2 行  GT      CFD

定位 —— 三个可视化工具的分工:
    vis_lift.py   FUNWAVE 原生网格, 看抬升算子长什么样    (只要 FUNWAVE)
    vis_align.py  CFD 切片, 看 t-offset 对不对             (FUNWAVE + GT)   <- 本文件
    vis_prior.py  CFD 切片, 看模型修正效果                 (+ prior + checkpoint)

流程位置:
    scan_toffset  ->  vis_align  ->  gen_prior  ->  train
       测出 k        目视确认 k      按 k 生成      训练
                          ^
                    对了再往下走, 不用先花时间生成全域 prior

为什么不读 gen_prior 的产物: 那是本步骤要确认之后才该生成的东西。本文件自己
算 prior —— 但只在 y=0.30 切片的那几万个 cell 上算, 不是全域 574163, 所以比
gen_prior 便宜两个量级。用的是同一套 lift.horizontal_terms + nwogu_at_points,
数值路径与 gen_prior 完全一致。

k 的来源
--------
默认从 scan_toffset 的 toffset_scan/c{cid:03d}.json 读 best_k。
--k 可显式覆盖, 用来目视复核:  --k 2 / 3 / 4 各出一支, 看哪个真对齐。

用法
----
  python vis_align.py --fw-dir <fw>/output --chunk 9 --field alpha
  python vis_align.py --fw-dir <fw>/output --chunk 9 --field alpha --k 3
"""

import argparse
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
from fw_io import load_static
from gen_prior import Bilinear, build_frame

MID_Y = 0.30
_HERE = os.path.dirname(os.path.abspath(__file__))


def read_scan_k(scan_dir, cid):
    """toffset_scan/c{cid}.json 里标定出的 best_k。找不到返回 None。"""
    p = Path(scan_dir) / f"c{cid:03d}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("best_k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", required=True, help="FUNWAVE output 目录")
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--field", default="alpha",
                    help="通道名 (alpha/Ux/Uy/Uz/p_rgh) 或 'Umag' (sqrt(Ux^2+Uz^2))")
    ap.add_argument("--data-dir", default="../data/3d/cropped_0.05")
    ap.add_argument("--scan-dir", default=os.path.join(_HERE, "toffset_scan"))
    ap.add_argument("--k", type=int, default=None,
                    help="帧移 k, 覆盖 scan 结果。用于目视复核不同 k。")
    ap.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 各自的列索引")
    # 几何 (与 gen_prior 一致)
    ap.add_argument("--x-offset", type=float, default=15.05)
    ap.add_argument("--y-offset", type=float, default=0.0)
    ap.add_argument("--mglob", type=int, default=1575)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--no-pnh", action="store_true")
    # 输出
    ap.add_argument("--n-frames", type=int, default=0, dest="n_frames",
                    help="取多少帧; 0 = 整个 chunk")
    ap.add_argument("--start", default="mid",
                    help="起始帧: 整数, 或 'mid' 取 chunk 正中 (默认)。"
                         "配准看的是相位, 取中间段可避开 chunk 边界的暂态。")
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--fig-w", type=float, default=38.4, dest="fig_w")
    ap.add_argument("--fig-h", type=float, default=16.0, dest="fig_h")
    args = ap.parse_args()

    cid = args.chunk
    data_dir = Path(args.data_dir)
    names = list(LC.CH_NAMES)

    # ---- k ----
    if args.k is not None:
        k, k_src = args.k, "--k"
    else:
        k = read_scan_k(args.scan_dir, cid)
        k_src = f"c{cid:03d}.json"
        if k is None:
            raise SystemExit(
                f"[err] {args.scan_dir}/c{cid:03d}.json 不存在或 best_k 为 null。\n"
                f"      先跑 scan_toffset.sh, 或用 --k 显式指定。")
    print(f"[k   ] k={k:+d} ({k_src})  ->  t-offset {k*args.plot_intv:+.2f}s")

    # ---- 切片几何 (OpenFOAM 网格实际的 cell, 非均匀) ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0].astype(np.float64), xz[:, 1].astype(np.float64)
    Ns = len(cell_map)
    print(f"[slice] {Ns} cells @ y={MID_Y}   "
          f"x {x_s.min():.3f}..{x_s.max():.3f}  z {z_s.min():.3f}..{z_s.max():.3f}")

    # ---- GT ----
    gt_all = np.load(data_dir / f"chunk_{cid:03d}_data.npy", mmap_mode="r")
    times_all = np.load(data_dir / f"chunk_{cid:03d}_times.npy").astype(np.float64)
    T_all = min(gt_all.shape[0], len(times_all))

    if 0 < args.n_frames < T_all:
        T = args.n_frames
        i0 = ((T_all - T) // 2 if str(args.start).lower() == "mid"
              else int(args.start))
        i0 = max(0, min(i0, T_all - T))
    else:
        T, i0 = T_all, 0
    win = slice(i0, i0 + T)
    fidx = np.arange(i0, i0 + T)                 # chunk 内的绝对帧号

    gt = np.asarray(gt_all[win][:, cell_map, :][:, :, args.gt_channels],
                    dtype=np.float32)
    times = times_all[win]
    print(f"[gt  ] {gt.shape}  frame {i0}..{i0+T-1}/{T_all}  "
          f"t {times[0]:.2f}..{times[-1]:.2f}")

    # ---- prior: 只在切片点上现算 ----
    h_grid, avail, cache = load_static(args.fw_dir, args.mglob, args.nglob)

    # 切面固定 y=0.30 -> 水平去重后只剩不同的 x, 省一个量级的插值
    xy = np.round(np.stack([x_s + args.x_offset,
                            np.full(Ns, MID_Y + args.y_offset)], axis=1), 9)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    bil = Bilinear(uniq[:, 0], uniq[:, 1], args.mglob, args.nglob,
                   args.dx, args.dy)
    n_out = int((~bil.inside).sum())
    print(f"[uniq] 水平唯一点 {len(uniq)} / {Ns} cells "
          f"({Ns/max(len(uniq),1):.1f} cells per column)"
          + (f"   [warn] 域外 {n_out}" if n_out else ""))

    n_fw = np.round(times / args.plot_intv).astype(np.int64) + k
    missing = [int(n) for n in n_fw if n not in avail]
    if missing:
        print(f"[warn] {len(missing)} 帧 FUNWAVE 不存在 -> 该帧留空 "
              f"(示例 {missing[:5]})")

    pr = np.zeros((T, Ns, 5), dtype=np.float32)
    n_bad = 0
    for i, n in enumerate(n_fw):
        n = int(n)
        if n not in avail:
            continue
        out = build_frame(n, cache, avail, h_grid, bil, z_s, inv,
                          args.dx, args.dy, args.plot_intv,
                          use_pnh=not args.no_pnh)
        fin = np.isfinite(out).all(axis=1)
        out[~fin, :] = 0.0                 # 与 gen_prior 同一约定: 无定义处填 0
        n_bad += int((~fin).sum())
        pr[i] = out.astype(np.float32)
        if i % 20 == 0:
            print(f"  prior {i+1}/{T}  fw#{n}", flush=True)
    print(f"[prior] 无定义 cell·帧 {n_bad} / {T*Ns} "
          f"({n_bad/max(T*Ns,1)*100:.2f}%, 已填 0)")

    # ---- 取场 ----
    if args.field == "Umag":
        iu = [names.index(c) for c in ("Ux", "Uz")]
        s_gt = np.sqrt(sum(gt[..., i] ** 2 for i in iu))
        s_pr = np.sqrt(sum(pr[..., i] ** 2 for i in iu))
        fname = "|U| (xz)"
    else:
        if args.field not in names:
            raise SystemExit(f"[err] --field '{args.field}' 不在 {names} / Umag")
        fi = names.index(args.field)
        s_gt, s_pr, fname = gt[..., fi], pr[..., fi], args.field

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
    corr = np.full(T, np.nan)
    for f in range(T):
        a, b = s_pr[f], s_gt[f]
        sa, sb = a.std(), b.std()
        if sa > 1e-12 and sb > 1e-12:
            corr[f] = np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)

    # ---- 渲染 ----
    triang = mtri.Triangulation(x_s, z_s, triangles=tri_simplices)
    xlim, zlim = (x_s.min(), x_s.max()), (z_s.min(), z_s.max())

    wtag = "" if T == T_all else f"_f{i0}-{i0+T-1}"
    out_path = Path(args.output) if args.output else \
        Path(_HERE) / "vis_align" / f"c{cid}_{args.field}_k{k:+d}{wtag}.mp4"
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

    def update(f):
        draw(axes[0], s_pr[f], "prior  (FUNWAVE lift, on CFD slice cells)")
        draw(axes[1], s_gt[f], "GT  (CFD)")
        suptitle.set_text(
            f"chunk {cid} | t={times[f]:.2f}s | frame {fidx[f]} | fw#{n_fw[f]} | "
            f"{fname} | k={k:+d} | slice-RMSE {rmse[f]:.4f}   corr {corr[f]:.3f}")

    update(0)
    print(f"[vid ] -> {out_path}")
    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=1000 // args.fps, blit=False)
    ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)

    print(f"[done] {fname}  k={k:+d}   slice-RMSE 均值 {rmse.mean():.4f}   "
          f"corr 均值 {np.nanmean(corr):.3f}")
    print("       (扫不同 --k 时看这两个数: RMSE 越低 / corr 越高 越对齐)")


if __name__ == "__main__":
    main()