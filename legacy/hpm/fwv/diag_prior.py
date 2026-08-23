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
     报告 std(GT) 与 RMSE 的关系。若残差方差远小于场方差, 直接套现有 stats
     会让学习信号被压扁, 需要给 Δ 单独算 stats。
  3. 各 chunk 的 prior 质量是否随时间变化?

空间歧义: prior 的 alpha 是 sharp 0/1, 故其 U 天然等于 αU。GT 若存的是裸 U,
两边不同空间。脚本同时报 raw-U 与 αU 两种口径, 由数值判断哪个是对的。

内存: 逐帧流式累加 (float64 一阶/二阶矩), 峰值 ~几十 MB —— 不再把整个 chunk
的 GT+prior 载进内存 (那是 2.3 GB/chunk, 登录节点会 OOM)。

用法:
  python diag_prior.py --gt-dir ../data/3d/cropped_0.05 \\
      --prior-dir ../data/3d/cropped_0.05/prior_ktuned --chunks 1-9
"""

import argparse
import os
import sys

import numpy as np

import lift as LC

CH = list(LC.CH_NAMES)
NCH = len(CH)

# 累加器列布局: [n, Σg, Σg², Σp, Σp², Σpg, Σ(p-g)²]
_N, _SG, _SG2, _SP, _SP2, _SPG, _SD2 = range(7)


def expand(spec):
    out = []
    for part in str(spec).split(","):
        if "-" in part[1:]:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def accumulate(acc, p, g):
    """把一帧一个通道的 (p, g) 累进 acc (7,)。p/g 已是 float64 一维有限值。"""
    acc[_N] += p.size
    acc[_SG] += g.sum()
    acc[_SG2] += (g * g).sum()
    acc[_SP] += p.sum()
    acc[_SP2] += (p * p).sum()
    acc[_SPG] += (p * g).sum()
    d = p - g
    acc[_SD2] += (d * d).sum()


def finish(acc):
    """(std_g, rmse, nrmse, corr, r2); 数据不足返回 NaN。"""
    n = acc[_N]
    if n < 100:
        return (np.nan,) * 5
    mg, mp = acc[_SG] / n, acc[_SP] / n
    vg = max(acc[_SG2] / n - mg * mg, 0.0)      # 抵消误差可能给出微负值
    vp = max(acc[_SP2] / n - mp * mp, 0.0)
    sg, sp = np.sqrt(vg), np.sqrt(vp)
    rmse = np.sqrt(acc[_SD2] / n)
    nrmse = rmse / sg if sg > 1e-30 else np.nan
    cov = acc[_SPG] / n - mg * mp
    corr = cov / (sp * sg) if (sp > 1e-30 and sg > 1e-30) else np.nan
    r2 = 1.0 - (rmse ** 2) / vg if vg > 1e-30 else np.nan
    return sg, rmse, nrmse, corr, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--prior-dir", required=True)
    ap.add_argument("--chunks", default="1-9")
    ap.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 的列索引")
    ap.add_argument("--x-win", type=float, nargs=2, default=None,
                    metavar=("XMIN", "XMAX"),
                    help="可选: 只统计该 x_cfd 窗口 (如 -2.5 7 排除破碎区)")
    ap.add_argument("--coords", default=None, help="--x-win 需要")
    ap.add_argument("--space", choices=["raw", "alphaU", "both"], default="both")
    ap.add_argument("--use-valid", action="store_true",
                    help="只统计 prior valid 的 cell (默认统计全部, 与训练一致)")
    args = ap.parse_args()

    chunks = expand(args.chunks)
    ch_idx = list(args.gt_channels)

    keep = None
    if args.x_win:
        if not args.coords:
            sys.exit("[err] --x-win 需要 --coords")
        c = np.load(args.coords)
        keep = (c[:, 0] >= args.x_win[0]) & (c[:, 0] <= args.x_win[1])
        print(f"[win] x_cfd ∈ {args.x_win} -> {keep.sum()}/{len(c)} cells")

    spaces = ["raw", "alphaU"] if args.space == "both" else [args.space]
    summary = {s: {ci: [] for ci in range(NCH)} for s in spaces}

    for cid in chunks:
        gp = os.path.join(args.gt_dir, f"chunk_{cid:03d}_data.npy")
        pp = os.path.join(args.prior_dir, f"prior_{cid:03d}_data.npy")
        vp = os.path.join(args.prior_dir, f"prior_{cid:03d}_valid.npy")
        if not (os.path.exists(gp) and os.path.exists(pp)):
            print(f"[skip] chunk {cid}: 文件缺失")
            continue

        gt_m = np.load(gp, mmap_mode="r")
        pr_m = np.load(pp, mmap_mode="r")
        vd_m = np.load(vp, mmap_mode="r") if os.path.exists(vp) else None
        T = min(gt_m.shape[0], pr_m.shape[0])

        # 每 (space, channel) 两套累加器: prior 与 persistence
        A = {s: np.zeros((NCH, 7)) for s in spaces}
        B = {s: np.zeros((NCH, 7)) for s in spaces}

        def load_gt(t):
            a = np.asarray(gt_m[t], dtype=np.float64)[:, ch_idx]
            return a[keep] if keep is not None else a

        prev = load_gt(0)
        for t in range(1, T):                   # t=1.. 使 prior 与 persistence 可比
            cur = load_gt(t)
            pri = np.asarray(pr_m[t], dtype=np.float64)
            if keep is not None:
                pri = pri[keep]
            vmask = None
            if args.use_valid and vd_m is not None:
                vmask = np.asarray(vd_m[t], dtype=bool)
                if keep is not None:
                    vmask = vmask[keep]

            for sp in spaces:
                if sp == "alphaU":
                    g1 = cur.copy(); g1[:, 1:5] *= cur[:, 0:1]
                    g0 = prev.copy(); g0[:, 1:5] *= prev[:, 0:1]
                    p1 = pri                    # prior 本就是 αU
                else:
                    g1, g0, p1 = cur, prev, pri
                for ci in range(NCH):
                    gv, pv, ev = g1[:, ci], p1[:, ci], g0[:, ci]
                    m = np.isfinite(gv) & np.isfinite(pv) & np.isfinite(ev)
                    if vmask is not None:
                        m &= vmask
                    if m.sum() < 1:
                        continue
                    accumulate(A[sp][ci], pv[m], gv[m])
                    accumulate(B[sp][ci], ev[m], gv[m])
            prev = cur

        vr = f"   valid {vd_m[:].mean()*100:.2f}%" if vd_m is not None else ""
        print(f"\n{'='*76}")
        print(f"chunk {cid}   GT {tuple(gt_m.shape)}   prior {tuple(pr_m.shape)}{vr}")
        for sp in spaces:
            tag = "αU 空间" if sp == "alphaU" else "raw 空间"
            print(f"  --- {tag} ---")
            print(f"  {'ch':8s} {'std(GT)':>11s} | {'RMSE_pri':>10s} {'nRMSE':>7s} "
                  f"{'corr':>7s} | {'RMSE_per':>10s} {'nRMSE':>7s} |  最优 base")
            for ci, name in enumerate(CH):
                s_ = finish(A[sp][ci])
                e_ = finish(B[sp][ci])
                if not np.isfinite(s_[2]):
                    continue
                summary[sp][ci].append((s_, e_))
                cands = {"prior": s_[2], "前一帧": e_[2], "常数": 1.0}
                best = min(cands, key=lambda k: (cands[k] if np.isfinite(cands[k])
                                                 else np.inf))
                print(f"  {name:8s} {s_[0]:11.5g} | {s_[1]:10.4g} {s_[2]:7.3f} "
                      f"{s_[3]:7.3f} | {e_[1]:10.4g} {e_[2]:7.3f} |  {best}")
        del gt_m, pr_m, vd_m

    # ---- 汇总 ----
    for sp in spaces:
        print(f"\n{'='*76}")
        print(f"跨 chunk 汇总 ({'αU 空间' if sp=='alphaU' else 'raw 空间'})")
        print(f"  {'ch':8s} {'nRMSE_pri':>10s} {'范围':>16s} {'corr':>7s} | "
              f"{'nRMSE_per':>10s} |  建议")
        for ci, name in enumerate(CH):
            rows = summary[sp][ci]
            if not rows:
                continue
            nr = np.array([r[0][2] for r in rows])
            co = np.array([r[0][3] for r in rows])
            npe = np.array([r[1][2] for r in rows])
            mp_, mq = np.nanmean(nr), np.nanmean(npe)
            if mp_ >= 1.0:
                sug = "prior 无效 -> 走自回归 (同 nut)"
            elif mp_ < mq:
                sug = "prior 优于前一帧 -> X̂ = prior + Δ"
            else:
                sug = f"prior 有效但逊于前一帧 ({mp_:.2f} vs {mq:.2f}) -> 见下"
            print(f"  {name:8s} {mp_:10.3f} {np.nanmin(nr):7.3f}..{np.nanmax(nr):-7.3f} "
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