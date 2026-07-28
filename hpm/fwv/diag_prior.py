#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_prior.py — prior vs GT 残差统计诊断。

回答三个问题:
  1. prior 值不值得当 base?
     判据 nRMSE = RMSE(prior, GT) / std(GT)
       < 1  prior 优于常数基线 -> `X̂ = prior + Δ` 有意义
       > 1  prior 不如直接预测均值 -> 该通道不该走 P 架构
  2. 残差该怎么归一化?
     报告 std(GT) 与 std(GT - prior) 的比值。若残差方差远小于场方差,
     直接套现有 stats 会让学习信号被压扁, 需要给 Δ 单独算 stats。
  3. 各 chunk 的 prior 质量是否随时间变化? (启动暂态假设)

空间歧义: prior 的 alpha 是 sharp 0/1, 故其 U 天然等于 αU。GT 若存的是裸 U,
两边不同空间。脚本同时报 raw-U 与 αU 两种口径, 由数值判断哪个是对的。

用法:
  python diag_prior.py --gt-dir ../data/3d/cropped_0.05 \\
      --prior-dir ../data/3d/prior_t015 --chunks 1-10
"""

import argparse
import os
import sys

import numpy as np

CH = ["alpha", "Ux", "Uy", "Uz", "p_rgh"]


def expand(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def stats_pair(p, g):
    """p, g: (n,) 有限值。返回 (std_g, rmse, nrmse, corr, r2)。"""
    sg = g.std()
    d = p - g
    rmse = np.sqrt(np.mean(d ** 2))
    nrmse = rmse / sg if sg > 1e-30 else np.nan
    sp = p.std()
    corr = (np.mean((p - p.mean()) * (g - g.mean())) / (sp * sg)
            if sp > 1e-30 and sg > 1e-30 else np.nan)
    r2 = 1.0 - (rmse ** 2) / (sg ** 2) if sg > 1e-30 else np.nan
    return sg, rmse, nrmse, corr, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--prior-dir", required=True)
    ap.add_argument("--chunks", default="1-10")
    ap.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 的列索引")
    ap.add_argument("--x-win", type=float, nargs=2, default=None,
                    metavar=("XMIN", "XMAX"),
                    help="可选: 只统计该 x_cfd 窗口 (如 -2.5 7 排除破碎区)")
    ap.add_argument("--coords", default=None, help="--x-win 需要")
    ap.add_argument("--space", choices=["raw", "alphaU", "both"], default="both")
    args = ap.parse_args()

    chunks = expand(args.chunks)
    keep = None
    if args.x_win:
        if not args.coords:
            sys.exit("[err] --x-win 需要 --coords")
        c = np.load(args.coords)
        keep = (c[:, 0] >= args.x_win[0]) & (c[:, 0] <= args.x_win[1])
        print(f"[win] x_cfd ∈ {args.x_win} -> {keep.sum()}/{len(c)} cells")

    spaces = ["raw", "alphaU"] if args.space == "both" else [args.space]
    acc = {s: {c: [] for c in range(5)} for s in spaces}

    for cid in chunks:
        gp = os.path.join(args.gt_dir, f"chunk_{cid:03d}_data.npy")
        pp = os.path.join(args.prior_dir, f"prior_{cid:03d}_data.npy")
        vp = os.path.join(args.prior_dir, f"prior_{cid:03d}_valid.npy")
        if not (os.path.exists(gp) and os.path.exists(pp)):
            print(f"[skip] chunk {cid}: 文件缺失")
            continue

        gt_raw = np.load(gp, mmap_mode="r")
        prior = np.load(pp)
        valid = np.load(vp) if os.path.exists(vp) else None
        gt = np.asarray(gt_raw[:, :, args.gt_channels], dtype=np.float32)

        if keep is not None:
            gt, prior = gt[:, keep], prior[:, keep]
            valid = valid[:, keep] if valid is not None else None

        print(f"\n{'='*76}")
        print(f"chunk {cid}   GT {gt.shape}   prior {prior.shape}"
              + (f"   valid {valid.mean()*100:.2f}%" if valid is not None else ""))
        for sp in spaces:
            if sp == "alphaU":
                a = gt[..., 0:1]
                g = gt.copy(); g[..., 1:5] = gt[..., 1:5] * a     # GT -> αU
                p = prior                                         # prior 本就是 αU
                tag = "αU 空间"
            else:
                g, p, tag = gt, prior, "raw 空间"
            print(f"  --- {tag} ---")
            print(f"  {'ch':8s} {'std(GT)':>11s} | {'RMSE_pri':>10s} {'nRMSE':>7s} "
                  f"{'corr':>7s} | {'RMSE_per':>10s} {'nRMSE':>7s} |  最优 base")
            # 只用 t=1..T-1, 使 prior 与 persistence 在同一批帧上可比
            for ci, name in enumerate(CH):
                gt_t = g[1:, :, ci].ravel()          # target
                pr_t = p[1:, :, ci].ravel()          # prior 在同帧
                pe_t = g[:-1, :, ci].ravel()         # persistence: 前一帧 GT
                m = np.isfinite(pr_t) & np.isfinite(gt_t) & np.isfinite(pe_t)
                if valid is not None:
                    m &= valid[1:].ravel()           # 同一批 cell, 苹果对苹果
                if m.sum() < 100:
                    continue
                gv = gt_t[m].astype(np.float64)
                sp_ = stats_pair(pr_t[m].astype(np.float64), gv)
                pe_ = stats_pair(pe_t[m].astype(np.float64), gv)
                acc[sp][ci].append((sp_, pe_))

                cands = {"prior": sp_[2], "前一帧": pe_[2], "常数": 1.0}
                best = min(cands, key=cands.get)
                print(f"  {name:8s} {sp_[0]:11.5g} | {sp_[1]:10.4g} {sp_[2]:7.3f} "
                      f"{sp_[3]:7.3f} | {pe_[1]:10.4g} {pe_[2]:7.3f} |  {best}")
        del gt, prior, gt_raw

    # ---- 汇总 ----
    for sp in spaces:
        print(f"\n{'='*76}")
        print(f"跨 chunk 汇总 ({'αU 空间' if sp=='alphaU' else 'raw 空间'})")
        print(f"  {'ch':8s} {'nRMSE_pri':>10s} {'范围':>16s} {'corr':>7s} | "
              f"{'nRMSE_per':>10s} |  建议")
        for ci, name in enumerate(CH):
            rows = acc[sp][ci]
            if not rows:
                continue
            nr = np.array([r[0][2] for r in rows])       # prior
            co = np.array([r[0][3] for r in rows])
            npe = np.array([r[1][2] for r in rows])      # persistence
            mp, mq = np.nanmean(nr), np.nanmean(npe)

            if mp >= 1.0:
                sug = "prior 无效 -> 走自回归 (同 nut)"
            elif mp < mq:
                sug = "prior 优于前一帧 -> X̂ = prior + Δ"
            else:
                sug = f"prior 有效但逊于前一帧 ({mp:.2f} vs {mq:.2f}) -> 见下"
            print(f"  {name:8s} {mp:10.3f} {np.nanmin(nr):7.3f}..{np.nanmax(nr):-7.3f} "
                  f"{np.nanmean(co):7.3f} | {mq:10.3f} |  {sug}")

    print(f"\n{'='*76}")
    print("读法:")
    print("  三个 base 的 nRMSE 直接可比 (常数基线恒为 1.000):")
    print("    nRMSE_pri  = RMSE(prior(t), GT(t)) / std(GT)")
    print("    nRMSE_per  = RMSE(GT(t-1), GT(t)) / std(GT)      persistence")
    print()
    print("  nRMSE_pri > 1  -> prior 在加噪, 不该当 base; 改走自回归 (同 nut)")
    print("  nRMSE_pri < nRMSE_per -> prior 单步就更准, P 架构直接成立")
    print("  nRMSE_pri > nRMSE_per -> 前一帧单步更准, 但它在 rollout 中会漂移;")
    print("        prior 每步独立生成、不漂 (report 里的 persistent anchor)。")
    print("        单步吃亏、长程占优, 这正是 P 架构要买的东西 —— 但要清楚代价。")
    print()
    print("  归一化: nRMSE_pri 远小于 1 时残差方差 << 场方差, 套现有 stats 会压扁")
    print("          学习信号, 建议给 Δ 单独算 stats。")
    print("  空间: 若 raw 与 αU 口径下速度通道差异巨大, 说明 GT 存的是裸 U,")
    print("        残差统计应以 αU 口径为准 (那才是训练空间)。")


if __name__ == "__main__":
    main()