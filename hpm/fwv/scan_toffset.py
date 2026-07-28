#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_toffset.py — 扫描 t-offset, 找 prior 与 CFD GT 最一致的帧移。

原理
----
t_offset 的唯一作用是"哪一帧 FUNWAVE 配哪一帧 CFD", 分辨率就是 plot_intv (0.05s)。
故直接把候选帧移 k ∈ [k_min, k_max] 逐个试过去, 取 prior 与 GT 最一致的 k。
比 wave gauge 互相关更贴合目的: 直接优化"训练数据上的一致性", 且自动吸收
x-offset 残差与相速度偏差 (找的是给定 x-offset 下的最优 t)。

两个指标 (应给出同一个 k, 不一致本身就是信号):
  RMSE  —— 受波高系统偏差影响 (Boussinesq 低估浅化), 但那是不随 k 变的常数偏置
  corr  —— 尺度无关, 纯看相位

比较区域默认限制在破碎区之前 (--x-win)。破碎区 prior 与 GT 本就该不同,
纳入只会给扫描加噪声。

用法
----
  python scan_toffset.py \\
      --fw-dir <fw>/output \\
      --coords <data>/coords.npy \\
      --gt-data <data>/chunk_006_data.npy \\
      --gt-times <data>/chunk_006_times.npy \\
      --x-offset 15.05 --k-range -40 40 --x-win -2.5 7.0
"""

import argparse
import os
import sys

import numpy as np

import lift as LC
from gen_prior import read2d, FrameCache, Bilinear, build_frame, load_static


def parabolic_min(ks, ys):
    """在离散最小值附近做抛物线拟合, 给出亚帧精度的极值位置 (仅诊断)。"""
    i = int(np.argmin(ys))
    if i == 0 or i == len(ys) - 1:
        return float(ks[i]), False
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-30:
        return float(ks[i]), False
    delta = 0.5 * (y0 - y2) / denom
    return float(ks[i] + delta), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--gt-data", required=True, help="chunk_{cid}_data.npy")
    ap.add_argument("--gt-times", required=True)
    ap.add_argument("--x-offset", type=float, required=True)
    ap.add_argument("--y-offset", type=float, default=0.0)
    ap.add_argument("--k-range", type=int, nargs=2, default=[-40, 40],
                    metavar=("KMIN", "KMAX"),
                    help="候选帧移范围 (帧数; t_offset = k * plot_intv)")
    ap.add_argument("--x-win", type=float, nargs=2, default=[-1e9, 7.0],
                    metavar=("XMIN", "XMAX"),
                    help="参与比较的 cell 的 x_cfd 窗口。默认排除破碎区 (x>7)。")
    ap.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 各自的列索引")
    ap.add_argument("--mglob", type=int, default=1575)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--no-pnh", action="store_true")
    ap.add_argument("--min-contrast", type=float, default=0.05,
                    dest="min_contrast",
                    help="RMSE 曲线相对起伏低于此值视为不含相位信息, "
                         "不参与各通道一致性判断 (默认 5%%)")
    ap.add_argument("--out", default="toffset_scan")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    h_grid, avail, cache = load_static(args)

    # ---- CFD 坐标 + 比较窗口 ----
    coords = np.load(args.coords).astype(np.float64)
    keep = (coords[:, 0] >= args.x_win[0]) & (coords[:, 0] <= args.x_win[1])
    n_keep = int(keep.sum())
    if n_keep == 0:
        sys.exit("[err] --x-win 选中 0 个 cell")
    print(f"[cell] 比较窗口 x_cfd ∈ [{args.x_win[0]}, {args.x_win[1]}]  "
          f"-> {n_keep}/{len(coords)} cells ({n_keep/len(coords)*100:.1f}%)")

    c = coords[keep]
    x_fw = c[:, 0] + args.x_offset
    y_fw = c[:, 1] + args.y_offset
    z_c = c[:, 2]

    xy = np.round(np.stack([x_fw, y_fw], axis=1), 9)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    bil = Bilinear(uniq[:, 0], uniq[:, 1], args.mglob, args.nglob,
                   args.dx, args.dy)
    print(f"[uniq] 水平唯一点 {len(uniq)}  "
          f"(域外 {int((~bil.inside).sum())})")

    # ---- GT ----
    t_cfd = np.load(args.gt_times).astype(np.float64)
    T = len(t_cfd)
    gt_all = np.load(args.gt_data, mmap_mode="r")
    gt = np.asarray(gt_all[:, keep, :][:, :, args.gt_channels], dtype=np.float32)
    print(f"[gt  ] {gt.shape}  t {t_cfd[0]:.2f}..{t_cfd[-1]:.2f}")

    # ---- 需要计算的 FUNWAVE 帧范围 ----
    n0 = np.round(t_cfd / args.plot_intv).astype(np.int64)   # k=0 时的对应帧
    kmin, kmax = args.k_range
    ks = np.arange(kmin, kmax + 1)
    need = np.arange(n0[0] + kmin, n0[-1] + kmax + 1)
    have = np.array([n in avail for n in need])
    print(f"[fw  ] 需要帧 {need[0]}..{need[-1]} ({len(need)} 帧), "
          f"缺失 {int((~have).sum())}")

    # ---- 逐帧算 prior (只在比较窗口的 cell 上) ----
    P = np.full((len(need), n_keep, 5), np.nan, dtype=np.float32)
    for idx, n in enumerate(need):
        if not have[idx]:
            continue
        out = build_frame(int(n), cache, avail, h_grid, bil, z_c, inv,
                          args.dx, args.dy, args.plot_intv,
                          use_pnh=not args.no_pnh)
        P[idx] = out.astype(np.float32)
        if idx % 20 == 0:
            print(f"  prior {idx+1}/{len(need)}  fw#{n}", flush=True)
    base = int(need[0])

    # ---- 扫描 ----
    names = list(LC.CH_NAMES)
    rmse = np.full((len(ks), 5), np.nan)
    corr = np.full((len(ks), 5), np.nan)

    for a, k in enumerate(ks):
        rows = n0 + k - base                      # (T,) 在 P 中的行号
        p = P[rows]                                # (T, n_keep, 5)
        for c_i in range(5):
            pv, gv = p[..., c_i].ravel(), gt[..., c_i].ravel()
            m = np.isfinite(pv) & np.isfinite(gv)
            if m.sum() < 100:
                continue
            pv, gv = pv[m].astype(np.float64), gv[m].astype(np.float64)
            rmse[a, c_i] = np.sqrt(np.mean((pv - gv) ** 2))
            sp, sg = pv.std(), gv.std()
            if sp > 1e-12 and sg > 1e-12:
                corr[a, c_i] = np.mean((pv - pv.mean()) * (gv - gv.mean())) / (sp * sg)

    # ---- 报告 ----
    print()
    print("=" * 78)
    print(f"{'k':>5s} {'t_off':>7s} | " +
          " ".join(f"{n[:5]:>8s}" for n in names) + " | " +
          " ".join(f"{'r_'+n[:4]:>7s}" for n in names))
    print("-" * 78)
    best_r = {}
    for c_i in range(5):
        if np.isfinite(rmse[:, c_i]).any():
            best_r[c_i] = int(ks[np.nanargmin(rmse[:, c_i])])
    for a, k in enumerate(ks):
        mark = ""
        if any(k == v for v in best_r.values()):
            mark = "  <-- min " + ",".join(names[i][:5] for i, v in best_r.items()
                                           if v == k)
        print(f"{k:5d} {k*args.plot_intv:7.2f} | " +
              " ".join(f"{rmse[a,i]:8.4f}" if np.isfinite(rmse[a, i]) else f"{'--':>8s}"
                       for i in range(5)) + " | " +
              " ".join(f"{corr[a,i]:7.4f}" if np.isfinite(corr[a, i]) else f"{'--':>7s}"
                       for i in range(5)) + mark)
    print("=" * 78)

    print("\n各通道最优 k:")
    agree = []
    for c_i in range(5):
        col = rmse[:, c_i]
        if not np.isfinite(col).any():
            print(f"  {names[c_i]:6s} 无有效数据")
            continue
        # 判据: RMSE 曲线随 k 是否有实质起伏。曲线平坦 = 该通道不含相位信息
        # (例如准二维工况下的 Uy), 不参与一致性判断。
        lo, hi, mu = np.nanmin(col), np.nanmax(col), np.nanmean(col)
        contrast = (hi - lo) / mu if mu > 1e-30 else 0.0
        flat = contrast < args.min_contrast

        kr = int(ks[np.nanargmin(col)])
        kc = (int(ks[np.nanargmax(corr[:, c_i])])
              if np.isfinite(corr[:, c_i]).any() else None)
        sub, okp = parabolic_min(ks, col)
        if not flat:
            agree.append(kr)
        kc_s = f"{kc:+3d}" if kc is not None else " -- "
        tag = f"   [曲线平坦 {contrast*100:.1f}%, 不计入]" if flat else \
              f"   [对比度 {contrast*100:.0f}%]"
        print(f"  {names[c_i]:6s} RMSE最小 k={kr:+3d} (t={kr*args.plot_intv:+.2f}s)"
              f"   corr最大 k={kc_s}"
              f"   亚帧 k={sub:+.2f}"
              + ("" if okp else "(边界)") + tag)

    if agree:
        kk = int(np.median(agree))
        print(f"\n  各通道中位数 k = {kk:+d}  ->  --t-offset {kk*args.plot_intv:+.2f}")
        spread = max(agree) - min(agree)
        if spread == 0:
            print("  五通道完全一致 ✓")
        elif spread <= 2:
            print(f"  通道间分散 {spread} 帧, 可接受")
        else:
            print(f"  [warn] 通道间分散 {spread} 帧 —— 可能 x-offset 有残差, "
                  f"或相速度偏差沿程累积")
        if kk in (ks[0], ks[-1]):
            print(f"  [warn] 最优 k 落在扫描边界, 请扩大 --k-range")

    np.savez(os.path.join(args.out, "scan.npz"),
             ks=ks, rmse=rmse, corr=corr, names=np.array(names),
             plot_intv=args.plot_intv, x_offset=args.x_offset,
             x_win=np.array(args.x_win))
    print(f"\n[saved] {os.path.join(args.out, 'scan.npz')}")


if __name__ == "__main__":
    main()
