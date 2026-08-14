#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_toffset.py — 标定 t-offset: 哪一帧 FUNWAVE 配哪一帧 CFD。

原理
----
t_offset 的唯一作用是帧配对, 分辨率就是 plot_intv (0.05s)。故直接把候选帧移
k ∈ [k_min, k_max] 逐个试过去, 取 prior 与 GT 最一致的 k。比 wave gauge 互相关
更贴合目的: 直接优化"训练数据上的一致性", 且自动吸收 x-offset 残差与相速度偏差。

两个指标 (应给出同一个 k, 不一致本身就是信号):
  RMSE  —— 受波高系统偏差影响 (Boussinesq 低估浅化), 但那是不随 k 变的常数偏置
  corr  —— 尺度无关, 纯看相位

逐通道扫, 取"曲线不平坦"通道的中位数 —— 曲线平坦 = 该通道不含相位信息
(准二维工况下的 Uy 会被自动排除)。比较区域默认限制在破碎区之前 (--x-win):
破碎区 prior 与 GT 本就该不同, 纳入只给扫描加噪声。

多 chunk
--------
静态量 (coords / 去重 / Bilinear 权重 / 水深) 与 chunk 无关, 只准备一次,
之后串行扫各 chunk, 末尾直接给跨 chunk 汇总 —— 判断单一 t-offset 是否成立。

用法
----
  # 全量标定 + 汇总 (chunk 0 静水 OOD, chunk 10 GT 损坏, 均不扫)
  python scan_toffset.py --fw-dir <fw>/output --data-dir <data> --chunks 1-9

  # 单个 chunk
  python scan_toffset.py --fw-dir <fw>/output --data-dir <data> --chunks 9

  # 只把已有结果重新汇总 (不重扫)
  python scan_toffset.py --summary

输出
----
  <out>/c{cid:03d}.json    自描述: 结论 + 逐通道明细 + 完整曲线
"""

import argparse
import json
import os
import sys

import numpy as np

import lift as LC                              # 只用 CH_NAMES
from fw_io import load_static                  # FUNWAVE 文件读取
from gen_prior import Bilinear, build_frame    # 投射层

# 默认产物目录跟着脚本自己走, 不依赖 cwd, 也不写死目录名
_HERE = os.path.dirname(os.path.abspath(__file__))
# --data-dir 默认: repo/data/... (锚 __file__, 与 config/vis 同源, 不随 cwd 飘)。
_DATA = os.path.join(os.path.dirname(_HERE), "data", "3d", "cropped_0.05")


def expand(spec):
    """'1-9' / '2,6' / '1-3,9' -> [1,2,...]"""
    out = []
    for part in str(spec).split(","):
        if "-" in part[1:]:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


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


# ------------------------------------------------------------ 单 chunk ------

def scan_chunk(cid, args, ks, names, static):
    """扫一个 chunk, 落盘 scan.json, 返回结论 dict (缺数据则 None)。"""
    h_grid, avail, cache, bil, z_c, inv, n_keep, keep = static

    gt_data = os.path.join(args.data_dir, f"chunk_{cid:03d}_data.npy")
    gt_times = os.path.join(args.data_dir, f"chunk_{cid:03d}_times.npy")
    for p in (gt_data, gt_times):
        if not os.path.exists(p):
            print(f"[skip] chunk {cid}: 缺 {os.path.basename(p)}")
            return None

    t_cfd = np.load(gt_times).astype(np.float64)
    T = len(t_cfd)
    gt_all = np.load(gt_data, mmap_mode="r")
    gt = np.asarray(gt_all[:, keep, :][:, :, args.gt_channels], dtype=np.float32)
    print(f"[gt  ] {gt.shape}  t {t_cfd[0]:.2f}..{t_cfd[-1]:.2f}")

    # ---- 需要计算的 FUNWAVE 帧范围 ----
    n0 = np.round(t_cfd / args.plot_intv).astype(np.int64)   # k=0 时的对应帧
    need = np.arange(n0[0] + ks[0], n0[-1] + ks[-1] + 1)
    have = np.array([n in avail for n in need])
    print(f"[fw  ] 需要帧 {need[0]}..{need[-1]} ({len(need)} 帧), "
          f"缺失 {int((~have).sum())}")

    # ---- 逐帧算 prior (只在比较窗口的 cell 上) ----
    P = np.full((len(need), n_keep, 5), np.nan, dtype=np.float32)
    for idx, n in enumerate(need):
        if not have[idx]:
            continue
        P[idx] = build_frame(int(n), cache, avail, h_grid, bil, z_c, inv,
                             args.dx, args.dy, args.plot_intv,
                             use_pnh=not args.no_pnh).astype(np.float32)
        if idx % 20 == 0:
            print(f"  prior {idx+1}/{len(need)}  fw#{n}", flush=True)
    base = int(need[0])

    # ---- 扫描 ----
    rmse = np.full((len(ks), 5), np.nan)
    corr = np.full((len(ks), 5), np.nan)
    for a, k in enumerate(ks):
        p = P[n0 + k - base]                       # (T, n_keep, 5)
        for c_i in range(5):
            pv, gv = p[..., c_i].ravel(), gt[..., c_i].ravel()
            m = np.isfinite(pv) & np.isfinite(gv)
            if m.sum() < 100:
                continue
            pv, gv = pv[m].astype(np.float64), gv[m].astype(np.float64)
            rmse[a, c_i] = np.sqrt(np.mean((pv - gv) ** 2))
            sp, sg = pv.std(), gv.std()
            if sp > 1e-12 and sg > 1e-12:
                corr[a, c_i] = (np.mean((pv - pv.mean()) * (gv - gv.mean()))
                                / (sp * sg))
    del P

    # ---- 报告: 扫描表 ----
    print()
    print("=" * 78)
    print(f"{'k':>5s} {'t_off':>7s} | " +
          " ".join(f"{n[:5]:>8s}" for n in names) + " | " +
          " ".join(f"{'r_'+n[:4]:>7s}" for n in names))
    print("-" * 78)
    best_r = {i: int(ks[np.nanargmin(rmse[:, i])])
              for i in range(5) if np.isfinite(rmse[:, i]).any()}
    for a, k in enumerate(ks):
        mark = ""
        if any(k == v for v in best_r.values()):
            mark = "  <-- min " + ",".join(names[i][:5]
                                           for i, v in best_r.items() if v == k)
        print(f"{k:5d} {k*args.plot_intv:7.2f} | " +
              " ".join(f"{rmse[a,i]:8.4f}" if np.isfinite(rmse[a, i])
                       else f"{'--':>8s}" for i in range(5)) + " | " +
              " ".join(f"{corr[a,i]:7.4f}" if np.isfinite(corr[a, i])
                       else f"{'--':>7s}" for i in range(5)) + mark)
    print("=" * 78)

    # ---- 报告: 逐通道最优 k ----
    print("\n各通道最优 k:")
    agree, ch_result = [], {}
    for c_i in range(5):
        col = rmse[:, c_i]
        if not np.isfinite(col).any():
            print(f"  {names[c_i]:6s} 无有效数据")
            ch_result[names[c_i]] = None
            continue
        # 曲线平坦 = 该通道不含相位信息 (如准二维下的 Uy), 不参与投票
        lo, hi, mu = np.nanmin(col), np.nanmax(col), np.nanmean(col)
        contrast = (hi - lo) / mu if mu > 1e-30 else 0.0
        flat = contrast < args.min_contrast

        kr = int(ks[np.nanargmin(col)])
        kc = (int(ks[np.nanargmax(corr[:, c_i])])
              if np.isfinite(corr[:, c_i]).any() else None)
        sub, okp = parabolic_min(ks, col)
        if not flat:
            agree.append(kr)
        ch_result[names[c_i]] = dict(
            k_rmse=kr, k_corr=kc, k_subframe=round(float(sub), 3),
            subframe_ok=bool(okp), contrast=round(float(contrast), 5),
            flat=bool(flat), rmse_min=round(float(np.nanmin(col)), 6))

        kc_s = f"{kc:+3d}" if kc is not None else " -- "
        tag = (f"   [曲线平坦 {contrast*100:.1f}%, 不计入]" if flat
               else f"   [对比度 {contrast*100:.0f}%]")
        print(f"  {names[c_i]:6s} RMSE最小 k={kr:+3d} "
              f"(t={kr*args.plot_intv:+.2f}s)   corr最大 k={kc_s}"
              f"   亚帧 k={sub:+.2f}" + ("" if okp else "(边界)") + tag)

    kk = spread = None
    if agree:
        kk = int(np.median(agree))
        spread = max(agree) - min(agree)
        print(f"\n  各通道中位数 k = {kk:+d}  ->  t-offset {kk*args.plot_intv:+.2f}")
        if spread == 0:
            print("  参与投票的通道完全一致 ✓")
        elif spread <= 2:
            print(f"  通道间分散 {spread} 帧, 可接受")
        else:
            print(f"  [warn] 通道间分散 {spread} 帧 —— 可能 x-offset 有残差, "
                  f"或相速度偏差沿程累积")
        if kk in (int(ks[0]), int(ks[-1])):
            print("  [warn] 最优 k 落在扫描边界, 请扩大 --k-range")

    # ---- 落盘 JSON (体量小, 可 cat / grep / 进版本控制) ----
    def clean(a):
        """NaN 不是合法 JSON -> null。"""
        return [None if not np.isfinite(x) else round(float(x), 6) for x in a]

    out = dict(
        chunk=cid,
        best_k=kk,
        t_offset=(round(kk * args.plot_intv, 4) if kk is not None else None),
        channel_spread=spread,
        at_scan_boundary=(kk is not None and kk in (int(ks[0]), int(ks[-1]))),
        channels=ch_result,
        setup=dict(x_offset=args.x_offset, y_offset=args.y_offset,
                   plot_intv=args.plot_intv, x_win=list(args.x_win),
                   k_range=[int(ks[0]), int(ks[-1])],
                   min_contrast=args.min_contrast,
                   n_cells=int(n_keep), n_frames=int(T),
                   pnh=not args.no_pnh),
        curves=dict(ks=[int(k) for k in ks],
                    rmse={n: clean(rmse[:, i]) for i, n in enumerate(names)},
                    corr={n: clean(corr[:, i]) for i, n in enumerate(names)}),
    )
    jp = os.path.join(args.out, f"c{cid:03d}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {jp}")
    return out


# ---------------------------------------------------------- 跨 chunk 汇总 ---

def summarize(results, names):
    """判断'跨 chunk 单一 t-offset'是否成立。results: list[dict]。"""
    rows = [r for r in results if r and r["best_k"] is not None]
    if not rows:
        print("[err] 没有有效结果")
        return
    dt = rows[0]["setup"]["plot_intv"]

    print()
    print("=" * 74)
    print("跨 chunk 汇总")
    print("=" * 74)
    print(f"{'chunk':>6} {'k':>5} {'t_off':>7} {'通道内分散':>10}  | " +
          " ".join(f"{n[:5]:>6}" for n in names))
    print("-" * 74)
    for r in rows:
        cells = " ".join(
            f"{r['channels'][n]['k_rmse']:+6d}" if r["channels"].get(n)
            else f"{'--':>6}" for n in names)
        flag = "  <- 落在扫描边界, 需扩大 --k-range" if r["at_scan_boundary"] else ""
        print(f"{r['chunk']:>6} {r['best_k']:+5d} {r['t_offset']:+7.2f} "
              f"{r['channel_spread']:>10}  | {cells}{flag}")
    print("=" * 74)

    ks = [r["best_k"] for r in rows]
    cids = [r["chunk"] for r in rows]
    print(f"\nk ∈ [{min(ks)}, {max(ks)}]   极差 {max(ks)-min(ks)} 帧 "
          f"({(max(ks)-min(ks))*dt:.2f} s)")

    if max(ks) == min(ks):
        print(f"  → 全部一致 k={ks[0]:+d}。'跨 chunk 单一 t-offset' 成立, "
              f"gen_prior 用 --t-offset {ks[0]*dt:+.2f} 即可。")
        return
    if len(ks) < 3:
        print("  → 有效 chunk 少于 3 个, 不足以区分漂移与个别异常。")
        return

    # 线性拟合只用于判断"是否单调漂移" —— 生成仍用逐 chunk 整数 k (帧号本就离散)
    slope, icpt = np.polyfit(cids, ks, 1)
    resid = np.array(ks) - (slope * np.array(cids) + icpt)
    r2 = 1.0 - resid.var() / max(np.var(ks), 1e-30)
    print(f"  线性拟合 k ≈ {slope:+.3f}·chunk {icpt:+.2f}   R² = {r2:.3f}")

    if r2 > 0.8 and abs(slope) > 0.1:
        print(f"  → 单调漂移。相速度系统偏差: 每 chunk (5s) 累积 "
              f"{abs(slope)*dt*1000:.0f} ms  (~{abs(slope)*dt/5.0*100:.2f}%)。")
        print("     建议: 逐 chunk 用各自的 k 重新生成 prior。")
    else:
        med = float(np.median(ks))
        off = [c for c, k in zip(cids, ks) if k != med]
        print(f"  → 非单调。偏离中位数 ({med:+.0f}) 的 chunk: {off}")
        print("     这些 chunk 本身可能有问题, 不是全局漂移。")


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", help="FUNWAVE output 目录 (扫描模式必给)")
    ap.add_argument("--data-dir", default=_DATA,
                    help="CFD 数据目录 (含 coords.npy 与 chunk_*_{data,times}.npy)")
    ap.add_argument("--chunks", default="1-9",
                    help="要扫的 chunk, 如 '1-9' / '9' / '2,6'")
    ap.add_argument("--summary", action="store_true",
                    help="只读 --out 下已有的 c*.json 重新汇总, 不重扫")
    ap.add_argument("--out", default=None,
                    help="产物目录, 默认 <data-dir>/toffset_scan/ (随 --data-dir 走)")
    # 几何
    ap.add_argument("--x-offset", type=float, default=15.05,
                    help="x_fw = x_cfd + x_offset (给定, 不扫)")
    ap.add_argument("--y-offset", type=float, default=0.0)
    ap.add_argument("--k-range", type=int, nargs=2, default=[-10, 10],
                    metavar=("KMIN", "KMAX"),
                    help="候选帧移范围 (帧; t_offset = k * plot_intv)")
    ap.add_argument("--x-win", type=float, nargs=2, default=[-1e9, 7.0],
                    metavar=("XMIN", "XMAX"),
                    help="参与比较的 cell 的 x_cfd 窗口。默认排除破碎区 (x>7)。")
    ap.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 各自的列索引")
    # FUNWAVE 网格
    ap.add_argument("--mglob", type=int, default=1575)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--no-pnh", action="store_true")
    ap.add_argument("--min-contrast", type=float, default=0.05,
                    dest="min_contrast",
                    help="RMSE 曲线相对起伏低于此值视为不含相位信息, 不参与投票")
    args = ap.parse_args()

    # toffset_scan 产物随 data 目录走 (与 vis.py align 的读取默认同源, 不硬编码
    # 一条平行路径 —— 改 --data-dir 时读写自动跟着走, 不会漂)
    if args.out is None:
        args.out = os.path.join(args.data_dir, "toffset_scan")

    names = list(LC.CH_NAMES)
    chunks = expand(args.chunks)

    # ---- 只汇总模式 ----
    if args.summary:
        results = []
        for cid in chunks:
            p = os.path.join(args.out, f"c{cid:03d}.json")
            if os.path.exists(p):
                with open(p) as f:
                    results.append(json.load(f))
        if not results:
            sys.exit(f"[err] {args.out}/c*.json 一个都没找到")
        summarize(results, names)
        return

    if not args.fw_dir:
        sys.exit("[err] 扫描模式必须给 --fw-dir")
    os.makedirs(args.out, exist_ok=True)
    ks = np.arange(args.k_range[0], args.k_range[1] + 1)

    # ---- 静态量: 与 chunk 无关, 只准备一次 ----
    h_grid, avail, cache = load_static(args.fw_dir, args.mglob, args.nglob)

    coords = np.load(os.path.join(args.data_dir, "coords.npy")).astype(np.float64)
    keep = (coords[:, 0] >= args.x_win[0]) & (coords[:, 0] <= args.x_win[1])
    n_keep = int(keep.sum())
    if n_keep == 0:
        sys.exit("[err] --x-win 选中 0 个 cell")
    print(f"[cell] 比较窗口 x_cfd ∈ [{args.x_win[0]}, {args.x_win[1]}]  "
          f"-> {n_keep}/{len(coords)} cells ({n_keep/len(coords)*100:.1f}%)")

    c = coords[keep]
    xy = np.round(np.stack([c[:, 0] + args.x_offset,
                            c[:, 1] + args.y_offset], axis=1), 9)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    bil = Bilinear(uniq[:, 0], uniq[:, 1], args.mglob, args.nglob,
                   args.dx, args.dy)
    print(f"[uniq] 水平唯一点 {len(uniq)}  (域外 {int((~bil.inside).sum())})")
    print(f"[scan] k ∈ [{ks[0]}, {ks[-1]}]  chunks {chunks}")

    static = (h_grid, avail, cache, bil, c[:, 2], inv, n_keep, keep)

    # ---- 逐 chunk 串行扫 ----
    results = []
    for cid in chunks:
        print(f"\n{'#'*78}\n#  chunk {cid}\n{'#'*78}")
        results.append(scan_chunk(cid, args, ks, names, static))

    if len([r for r in results if r]) > 1:
        summarize(results, names)


if __name__ == "__main__":
    main()