#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_prior.py — 把 FUNWAVE lift 投射到 CFD 不规则网格 (cell 中心) 上。

产物与 GT (chunk_{cid:03d}_data.npy) 同形同序, 可直接配对喂 HPM:
    prior_{cid:03d}_data.npy    (T, N_cells, 5)  float32
    prior_{cid:03d}_valid.npy   (T, N_cells)     bool   [诊断用, 非训练输入]
    prior_{cid:03d}_times.npy   (T,)  t_cfd [s]
    prior_{cid:03d}_meta.json   offset / 参数 / valid 统计 (自描述)
                                逐 chunk 一份 —— array 并行下共用一个文件会串。
通道序: [alpha, Ux, Uy, Uz, p_rgh]   (与 lift.CH_NAMES 一致; 无 nut)

关键设计
--------
1. z 方向不插值: Nwogu 剖面在 z 上是解析的, 每个 cell 用它自己的 z_c 代入求值。
   全流程唯一的插值 = 水平面双线性 (bilinear), 且只在去重后的 (x,y) 上算一次。
2. 分层: 剖面公式唯一实现在 lift.nwogu_at_points, 本文件只做工程层 ——
   坐标 offset、水平双线性插值、(x,y) 去重、时间映射、chunk 循环、落盘。
   FUNWAVE 文件读取在 fw_io.py。
3. NaN (干单元 / 梯度扩散 / 域外 / 床下) -> 填 0, 同时记 valid=False。
   填 0 与 P 架构 (X̂ = prior + Δ) 天然兼容: prior 无效处退化为 X̂ = Δ。

用法
----
  python gen_prior.py --fw-dir <fw>/output --coords <data>/coords.npy \\
      --gt-times <data>/chunk_006_times.npy --chunk 6 \\
      --x-offset 15.05 --t-offset 0.15 --out <data>/prior_t015
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import lift as LC              # 剖面公式与常数
from fw_io import load_static  # FUNWAVE 文件读取


# --------------------------------------------------- 水平双线性插值 (唯一插值) --

class Bilinear:
    """FUNWAVE 规则网格 -> 任意水平散点的双线性插值器 (权重预计算一次)。

    数组布局 (nglob, mglob) = (ny, nx); 网格点 x=i*dx, y=j*dy。
    任一 stencil 点为 NaN -> 结果 NaN (诚实传播, 与 lift.py 一致)。
    落在网格外 -> 结果 NaN (不外推)。
    """

    def __init__(self, xq, yq, mglob, nglob, dx, dy, snap_tol=1e-9):
        fi = xq / dx
        fj = yq / dy

        # 网格吸附 (grid snapping): (i*dx)/dx 在浮点下未必精确等于 i, 实测 1575 个
        # 节点中有 159 个偏差 ~1e-15。这会让本该为 0 的权重变成 1e-15, 从而把
        # 邻居的 NaN 传过来。落在节点 snap_tol 以内 (默认 2e-11 m) 即视为在节点上
        # —— 远低于任何物理尺度, 远高于 float64 噪声。
        ri, rj = np.round(fi), np.round(fj)
        fi = np.where(np.abs(fi - ri) < snap_tol, ri, fi)
        fj = np.where(np.abs(fj - rj) < snap_tol, rj, fj)

        i0 = np.floor(fi).astype(np.int64)
        j0 = np.floor(fj).astype(np.int64)

        # 需要 i0, i0+1 都在网格内 (右端点 i=mglob-1 退化为它自己, 权重 0)
        self.inside = (i0 >= 0) & (i0 <= mglob - 1) & \
                      (j0 >= 0) & (j0 <= nglob - 1)
        # 恰好落在最后一个节点上时允许 (wx=0), 越过则视为域外
        self.inside &= (fi <= mglob - 1 + 1e-9) & (fj <= nglob - 1 + 1e-9)

        i0c = np.clip(i0, 0, mglob - 2) if mglob >= 2 else np.zeros_like(i0)
        j0c = np.clip(j0, 0, nglob - 2) if nglob >= 2 else np.zeros_like(j0)
        wx = np.clip(fi - i0c, 0.0, 1.0)
        wy = np.clip(fj - j0c, 0.0, 1.0)

        self.i0, self.j0 = i0c, j0c
        self.i1 = np.minimum(i0c + 1, mglob - 1)
        self.j1 = np.minimum(j0c + 1, nglob - 1)
        self.w00 = (1 - wx) * (1 - wy)
        self.w10 = wx * (1 - wy)
        self.w01 = (1 - wx) * wy
        self.w11 = wx * wy

    def __call__(self, F):
        """F: (nglob, mglob) -> (n_query,)

        注: 权重恰为 0 的 stencil 点不参与 —— 因为 0.0*np.nan = nan, 若不特判,
        查询点正好落在节点上时, 零权重的邻居 (可能是干单元 NaN) 会污染结果。
        数学上零权重意味着该点值不进入插值, 故置 0 是精确而非近似。
        """
        def term(w, vals):
            return np.where(w == 0.0, 0.0, w * vals)

        out = (term(self.w00, F[self.j0, self.i0]) +
               term(self.w10, F[self.j0, self.i1]) +
               term(self.w01, F[self.j1, self.i0]) +
               term(self.w11, F[self.j1, self.i1]))
        return np.where(self.inside, out, np.nan)


# ------------------------------------------------------------ 单帧驱动 ------

def build_frame(n_fw, cache, avail, h_grid, bil, z_c, inv, dx, dy, plot_intv,
                use_pnh=True):
    """算 FUNWAVE 第 n_fw 帧在 CFD cell 上的 prior。返回 (N,5) float64。"""
    eta_g, u_g, v_g = cache(n_fw)
    T = LC.horizontal_terms(eta_g, u_g, v_g, h_grid, dx, dy)

    A_dot_g = B_dot_g = None
    if use_pnh and (n_fw - 1) in avail and (n_fw + 1) in avail:
        em, um, vm = cache(n_fw - 1)
        ep, up, vp = cache(n_fw + 1)
        Tm = LC.horizontal_terms(em, um, vm, h_grid, dx, dy)
        Tp = LC.horizontal_terms(ep, up, vp, h_grid, dx, dy)
        dt2 = 2.0 * plot_intv
        A_dot_g = (Tp["A"] - Tm["A"]) / dt2
        B_dot_g = (Tp["B"] - Tm["B"]) / dt2

    # ---- 唯一的插值: 在去重后的水平点上做一次双线性 ----
    q = {k: bil(F) for k, F in dict(
        eta=eta_g, h=h_grid, u=u_g, v=v_g,
        dAdx=T["dAdx"], dAdy=T["dAdy"], dBdx=T["dBdx"], dBdy=T["dBdy"],
        A=T["A"], B=T["B"]).items()}
    qA_dot = bil(A_dot_g) if A_dot_g is not None else None
    qB_dot = bil(B_dot_g) if B_dot_g is not None else None

    # ---- 展开到每个 cell, 用各自的 z_c 解析求值 (无 z 插值) ----
    e = {k: val[inv] for k, val in q.items()}
    ad = qA_dot[inv] if qA_dot is not None else None
    bd = qB_dot[inv] if qB_dot is not None else None

    return LC.nwogu_at_points(z_c, e["eta"], e["h"], e["u"], e["v"],
                              e["dAdx"], e["dAdy"], e["dBdx"], e["dBdy"],
                              e["A"], e["B"], A_dot=ad, B_dot=bd)


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", required=True)
    # CFD 侧
    ap.add_argument("--coords", help="CFD cell 中心坐标 coords.npy (N,3) 物理坐标")
    ap.add_argument("--gt-times", help="对应 chunk 的 GT times npy (决定 t_cfd)")
    ap.add_argument("--chunk", type=int, help="chunk id (仅用于输出命名)")
    ap.add_argument("--out", default="prior")
    # offset
    ap.add_argument("--x-offset", type=float, default=None,
                    help="x_fw = x_cfd + x_offset  (本 case: 15.05)")
    ap.add_argument("--y-offset", type=float, default=0.0,
                    help="y_fw = y_cfd + y_offset  (未验证, 默认 0)")
    ap.add_argument("--t-offset", type=float, default=None,
                    help="t_fw = t_cfd + t_offset  [s] (scan_toffset.py 标定)")
    # FUNWAVE 网格
    ap.add_argument("--mglob", type=int, default=1575)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--no-pnh", action="store_true", help="关闭非静水 p_rgh 修正")
    args = ap.parse_args()

    for req in ("coords", "gt_times", "chunk", "x_offset", "t_offset"):
        if getattr(args, req) is None:
            sys.exit(f"[err] 必须显式给 --{req.replace('_', '-')}")

    os.makedirs(args.out, exist_ok=True)
    h_grid, avail, cache = load_static(args.fw_dir, args.mglob, args.nglob)

    # ---- CFD 坐标 -> FUNWAVE 坐标 ----
    coords = np.load(args.coords).astype(np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        sys.exit(f"[err] coords 形状 {coords.shape}, 期望 (N,3)")
    N = coords.shape[0]
    x_fw = coords[:, 0] + args.x_offset
    y_fw = coords[:, 1] + args.y_offset
    z_c = coords[:, 2]
    print(f"[cfd ] {N} cells   x_cfd {coords[:,0].min():.4f}..{coords[:,0].max():.4f}"
          f"  -> x_fw {x_fw.min():.4f}..{x_fw.max():.4f}")
    print(f"[fw  ] 网格 x 0..{(args.mglob-1)*args.dx:.4f}  "
          f"y 0..{(args.nglob-1)*args.dy:.4f}")

    # ---- 水平位置去重 (同一柱上多个 z 共享 (x,y), 大幅省插值) ----
    xy = np.round(np.stack([x_fw, y_fw], axis=1), 9)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    print(f"[uniq] 水平唯一点 {len(uniq)} / {N} cells "
          f"({N/max(len(uniq),1):.1f} cells per column)")

    bil = Bilinear(uniq[:, 0], uniq[:, 1], args.mglob, args.nglob,
                   args.dx, args.dy)
    n_out = int((~bil.inside).sum())
    if n_out:
        print(f"[warn] {n_out}/{len(uniq)} 个水平点落在 FUNWAVE 网格外 "
              f"-> 该列全部 invalid")

    # ---- 时间映射 ----
    t_cfd = np.load(args.gt_times).astype(np.float64)
    T = len(t_cfd)
    t_fw = t_cfd + args.t_offset
    n_fw_f = t_fw / args.plot_intv
    n_fw = np.round(n_fw_f).astype(np.int64)
    resid = np.abs(n_fw_f - n_fw).max() * args.plot_intv
    print(f"[time] {T} 帧  t_cfd {t_cfd[0]:.3f}..{t_cfd[-1]:.3f}  "
          f"t_offset={args.t_offset}  -> fw#{n_fw[0]}..{n_fw[-1]}")
    if resid > 1e-9:
        print(f"[warn] t-offset 非 plot_intv 整数倍, 取整残差最大 {resid*1000:.2f} ms "
              f"(未做时间插值)")
    missing = [int(k) for k in n_fw if k not in avail]
    if missing:
        print(f"[warn] {len(missing)} 帧 FUNWAVE 不存在 -> 该帧全 invalid "
              f"(示例 {missing[:5]})")

    # ---- invalid 归因预备: 静态类别 (域外 / 床下) ----
    # 域外: 水平点落在 FUNWAVE 网格外 (逐 cell 展开)
    cell_outside = ~bil.inside[inv]
    # 床下: 需要每个 cell 处的 h (只用于归因, 不参与数值)
    h_at_cell = bil(h_grid)[inv]
    cell_below_bed = np.isfinite(h_at_cell) & (z_c < -h_at_cell)
    print(f"[geom] 域外 cell {cell_outside.sum()}  "
          f"床下 cell {cell_below_bed.sum()}  (静态, 与时间无关)")

    # ---- 逐帧生成 ----
    data = np.zeros((T, N, 5), dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)

    t_lift = 0.0          # build_frame 累计耗时 (纯 lift 计算, 不含读盘/落盘)
    n_lift = 0            # 实际算过 build_frame 的帧数 (跳过 missing)

    for i, k in enumerate(n_fw):
        k = int(k)
        if k not in avail:
            continue                       # data 保持 0, valid 保持 False
        t0 = time.perf_counter()
        out = build_frame(k, cache, avail, h_grid, bil, z_c, inv,
                          args.dx, args.dy, args.plot_intv,
                          use_pnh=not args.no_pnh)
        t_lift += time.perf_counter() - t0
        n_lift += 1
        fin = np.isfinite(out).all(axis=1)
        out[~fin, :] = 0.0                 # NaN -> 0 (与 P 架构兼容)
        data[i] = out.astype(np.float32)
        valid[i] = fin
        if i % 10 == 0:
            print(f"  {i+1}/{T}  fw#{k}  valid {fin.mean()*100:.2f}%", flush=True)

    print(f"[lift] build_frame {n_lift} 帧累计 {t_lift:.1f}s, "
          f"均值 {t_lift/max(n_lift,1)*1000:.1f} ms/帧  ({N} cells)")

    # ---- 落盘 ----
    cid = f"{args.chunk:03d}"
    dp = os.path.join(args.out, f"prior_{cid}_data.npy")
    np.save(dp, data)
    np.save(os.path.join(args.out, f"prior_{cid}_valid.npy"), valid)
    np.save(os.path.join(args.out, f"prior_{cid}_times.npy"), t_cfd)

    vr = valid.mean()
    per_cell = valid.all(axis=0)

    # ---- invalid 归因 (逐帧平均) ----
    iv = ~valid                                       # (T,N)
    n_iv = iv.sum()
    n_out_c = (iv & cell_outside[None, :]).sum()
    n_bed = (iv & ~cell_outside[None, :] & cell_below_bed[None, :]).sum()
    n_dry = n_iv - n_out_c - n_bed                    # 干单元 + 梯度扩散
    # 干区的空间范围 (决定要不要 mask)
    dry_cells = (iv & ~cell_outside[None, :] & ~cell_below_bed[None, :]).any(axis=0)
    dry_x = coords[dry_cells, 0]

    meta = dict(chunk=args.chunk, n_cells=int(N), n_frames=int(T),
                channels=list(LC.CH_NAMES),
                x_offset=args.x_offset, y_offset=args.y_offset,
                t_offset=args.t_offset,
                mglob=args.mglob, nglob=args.nglob, dx=args.dx, dy=args.dy,
                plot_intv=args.plot_intv, pnh=not args.no_pnh,
                beta=LC.BETA, alpha="sharp_heaviside",
                valid_ratio=float(vr),
                cells_always_valid=int(per_cell.sum()),
                cells_never_valid=int((~valid.any(axis=0)).sum()),
                invalid_outside=int(n_out_c), invalid_below_bed=int(n_bed),
                invalid_dry=int(n_dry),
                dry_cells_total=int(dry_cells.sum()),
                dry_x_min=float(dry_x.min()) if dry_cells.any() else None,
                dry_x_max=float(dry_x.max()) if dry_cells.any() else None,
                horiz_points_outside_grid=n_out,
                frames_missing=len(missing))
    with open(os.path.join(args.out, f"prior_{cid}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"[done] {dp}  shape={data.shape} float32 "
          f"({os.path.getsize(dp)/1e9:.2f} GB)")
    print(f"[valid] 总体 {vr*100:.3f}%   "
          f"全程有效 cell {per_cell.sum()}/{N} ({per_cell.mean()*100:.2f}%)   "
          f"从不有效 {(~valid.any(axis=0)).sum()}")
    print(f"[归因] invalid 共 {n_iv} 个 (cell·帧):")
    print(f"       域外          {n_out_c:>12} ({n_out_c/max(n_iv,1)*100:5.1f}%)")
    print(f"       床下          {n_bed:>12} ({n_bed/max(n_iv,1)*100:5.1f}%)")
    print(f"       干单元+梯度扩散 {n_dry:>12} ({n_dry/max(n_iv,1)*100:5.1f}%)  <- 决定要不要 mask")
    if dry_cells.any():
        print(f"       干区 cell {dry_cells.sum()} 个, x_cfd 范围 "
              f"{dry_x.min():.3f} .. {dry_x.max():.3f}")
    print("=" * 60)
    if n_dry > 0:
        print("提示: 干单元 invalid 非零 —— 这些 cell 上 CFD 有数据但 prior 无定义,")
        print("      填 0 会被 HPM 误读成 '物理上真的没水'。建议加 validity mask 通道。")
    elif vr == 1.0:
        print("提示: valid 100%, 无需 validity mask。")


if __name__ == "__main__":
    main()