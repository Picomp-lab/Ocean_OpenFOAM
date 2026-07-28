#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_prior.py — 把 FUNWAVE lift 投射到 CFD 不规则网格 (cell 中心) 上。

产物与 GT (chunk_{cid:03d}_data.npy) 同形同序, 可直接配对喂 HPM:
    prior_{cid:03d}_data.npy    (T, N_cells, 5)  float32
    prior_{cid:03d}_valid.npy   (T, N_cells)     bool   [诊断用, 非训练输入]
    prior_{cid:03d}_times.npy   (T,)  t_cfd [s]
    prior_meta.json             offset / 参数 / valid 统计 (自描述)
通道序: [alpha, Ux, Uy, Uz, p_rgh]   (与 lift.CH_NAMES 一致; 无 nut)

关键设计
--------
1. z 方向不插值: Nwogu 剖面在 z 上是解析的, 每个 cell 用它自己的 z_c 代入求值。
   全流程唯一的插值 = 水平面双线性 (bilinear), 且只在去重后的 (x,y) 上算一次。
2. lift.py 一字不动 (纯度契约)。散点求值需要重写剖面公式, 故提供 --self-check
   模式: 在 FUNWAVE 原生网格点上与 LC.lift_frame 逐点比对, 锁死重实现漂移。
3. NaN (干单元 / 梯度扩散 / 域外 / 床下) -> 填 0, 同时记 valid=False。
   填 0 与 P 架构 (X̂ = prior + Δ) 天然兼容: prior 无效处退化为 X̂ = Δ。

用法
----
  # 一致性自检 (不需要 CFD 坐标, 建议先跑)
  python gen_prior.py --fw-dir <fw>/output --self-check

  # 生成 chunk 6 的 prior
  python gen_prior.py --fw-dir <fw>/output --coords <data>/coords.npy \\
      --gt-times <data>/chunk_006_times.npy --chunk 6 \\
      --x-offset 15.05 --t-offset 0.0 --out <data>/prior
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import lift as LC          # 只用 horizontal_terms / 常数; 不改动


# ------------------------------------------------------------------ IO ------

def read2d(path, mglob, nglob):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape == (nglob, mglob):
        return arr
    if arr.shape == (mglob, nglob):
        return arr.T
    raise ValueError(f"{path}: shape {arr.shape} (期望 {(nglob, mglob)})")


class FrameCache:
    """(eta,u,v) 帧缓存 —— p_nh 需要 n±1, 顺序推进时邻帧可复用。"""

    def __init__(self, fw_dir, mglob, nglob, cap=6):
        self.fw_dir, self.m, self.n, self.cap = fw_dir, mglob, nglob, cap
        self._c = {}

    def __call__(self, k):
        if k in self._c:
            return self._c[k]
        fp = lambda v: os.path.join(self.fw_dir, f"{v}_{k:05d}")
        eta = read2d(fp("eta"), self.m, self.n)
        u = read2d(fp("u"), self.m, self.n)
        v = read2d(fp("v"), self.m, self.n)
        mp = fp("mask")
        if os.path.exists(mp):
            dry = read2d(mp, self.m, self.n) < 0.5
            for f in (eta, u, v):
                f[dry] = np.nan
        while len(self._c) >= self.cap:
            self._c.pop(next(iter(self._c)))
        self._c[k] = (eta, u, v)
        return self._c[k]


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


# ------------------------------------------- Nwogu 剖面 (散点版, 与 lift.py 同式) --

def nwogu_at_points(z_c, eta, h, u, v, dAdx, dAdy, dBdx, dBdy, A, B,
                    A_dot=None, B_dot=None):
    """在散点上求值 3D 场。所有入参形状相同 (N,), z_c 为各点自己的高程。

    公式与 lift.lift_frame 逐项一致 (由 --self-check 锁死):
      Ux = u + (za - z)dAdx + 0.5(za^2 - z^2)dBdx        Nwogu 二次剖面
      Uz = -(A + z B)                                     连续性
      p  = rho g eta - rho[Ȧ(eta - z) + 0.5 Ḃ(eta^2 - z^2)]
    返回 (N, 5), 通道序 lift.CH_NAMES; 空气区 U=p=0; 无定义处 NaN。
    """
    za = LC.BETA * h
    water = z_c <= eta                      # eta 为 NaN 时 -> False, 后面统一置 NaN

    Ux = u + (za - z_c) * dAdx + 0.5 * (za ** 2 - z_c ** 2) * dBdx
    Uy = v + (za - z_c) * dAdy + 0.5 * (za ** 2 - z_c ** 2) * dBdy
    Uz = -(A + z_c * B)

    p = LC.RHO * LC.G * eta * np.ones_like(Ux)
    if A_dot is not None and B_dot is not None:
        p = p - LC.RHO * (A_dot * (eta - z_c)
                          + 0.5 * B_dot * (eta ** 2 - z_c ** 2))

    alpha = water.astype(np.float64)
    air = ~water
    for F in (Ux, Uy, Uz, p):
        F[air] = 0.0

    out = np.stack([alpha, Ux, Uy, Uz, p], axis=-1)

    # 无定义: eta 为 NaN (干单元/域外/梯度扩散) -> 整点 NaN
    bad = ~np.isfinite(eta)
    # 床底以下同样无定义 (CFD 网格通常不含, 但仍显式处理)
    bad |= np.isfinite(h) & (z_c < -h)
    out[bad, :] = np.nan
    return out


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

    return nwogu_at_points(z_c, e["eta"], e["h"], e["u"], e["v"],
                           e["dAdx"], e["dAdy"], e["dBdx"], e["dBdy"],
                           e["A"], e["B"], A_dot=ad, B_dot=bd)


# --------------------------------------------------------- 一致性自检 ------

def check_bilinear(args):
    """双线性插值器单测。原 self-check 的查询点落在网格节点上, 权重退化为
    (1,0,0,0), 不经过插值路径 —— 故此处单独测。"""
    print("-" * 60)
    print("双线性插值器 (Bilinear) 单测")
    print("-" * 60)
    m, n, dx, dy = 40, 10, args.dx, args.dy
    X, Y = np.meshgrid(np.arange(m) * dx, np.arange(n) * dy)
    rng = np.random.default_rng(1)
    xq = rng.uniform(0, (m - 1) * dx, 5000)
    yq = rng.uniform(0, (n - 1) * dy, 5000)
    bil = Bilinear(xq, yq, m, n, dx, dy)
    ok = True

    # 1) 线性场: 双线性应精确复现
    F = 3.0 * X - 2.0 * Y + 1.5
    e = np.abs(bil(F) - (3.0 * xq - 2.0 * yq + 1.5)).max()
    ok &= e < 1e-12
    print(f"  {'OK ' if e < 1e-12 else '!! '}线性场精确性     max|Δ|={e:.3e}")

    # 2) 二次场: 截断误差应 <= dx^2/4
    e2 = np.abs(bil(X ** 2) - xq ** 2).max()
    bound = dx ** 2 / 4
    ok &= e2 <= bound * 1.01
    print(f"  {'OK ' if e2 <= bound*1.01 else '!! '}二次场截断误差   "
          f"max|Δ|={e2:.3e}  理论上界={bound:.3e}")

    # 3) NaN 传播: stencil 任一点 NaN -> 结果 NaN
    F3 = F.copy(); F3[5, 20] = np.nan
    q = Bilinear(np.array([0.395, 0.30]), np.array([0.10, 0.10]),
                 m, n, dx, dy)(F3)
    p3 = (not np.isfinite(q[0])) and np.isfinite(q[1])
    ok &= p3
    print(f"  {'OK ' if p3 else '!! '}NaN 传播          "
          f"含NaN stencil -> {q[0]}, 远离处 -> {q[1]:.4f}")

    # 3b) 零权重不传播 NaN (回归测试: 0.0*nan=nan 曾导致节点上查询被误伤)
    #     查询点落在节点 19 上, 右邻节点 20 是 NaN 但权重为 0 -> 应取节点 19 的值
    qz = Bilinear(np.array([19 * dx]), np.array([5 * dy]), m, n, dx, dy)(F3)
    exp = F3[5, 19]
    p3b = np.isfinite(qz[0]) and abs(qz[0] - exp) < 1e-12
    ok &= p3b
    print(f"  {'OK ' if p3b else '!! '}零权重不传播 NaN  "
          f"节点上查询 -> {qz[0]} (期望 {exp})")

    # 3c) 网格吸附 (回归测试): (i*dx)/dx 浮点误差曾使零权重变成 1e-15,
    #     导致节点查询被右邻的 NaN 误伤。用真实网格规模复现。
    M = 1575
    Fs = np.zeros((n, M)); Fs[:, 1000:] = np.nan          # index 1000 起为 NaN
    rs = Bilinear(np.arange(M) * dx, np.full(M, 5 * dy), M, n, dx, dy)(Fs)
    first_nan = np.flatnonzero(~np.isfinite(rs))
    p3c = len(first_nan) > 0 and first_nan.min() == 1000
    ok &= p3c
    print(f"  {'OK ' if p3c else '!! '}浮点吸附          "
          f"NaN 起始 index={first_nan.min() if len(first_nan) else '无'} (期望 1000)")

    # 4) 域外不外推
    q2 = Bilinear(np.array([-0.01, (m - 1) * dx + 0.01]),
                  np.array([0.05, 0.05]), m, n, dx, dy)(F)
    p4 = not np.isfinite(q2).any()
    ok &= p4
    print(f"  {'OK ' if p4 else '!! '}域外不外推        {q2}")
    return ok


def self_check(args):
    """在 FUNWAVE 原生网格点上, 把散点重实现与 LC.lift_frame 逐点比对。

    构造 "假 CFD cell" = 原生网格点 (i,j) × 若干 z, 走完整散点路径,
    再与 lift.lift_frame 的同位置输出作差。二者应完全一致 (浮点误差内)。
    注意: 查询点落在节点上 -> 不经过插值路径, 故另有 check_bilinear 单测。
    """
    print("=" * 60)
    print("一致性自检 (self-check): 散点重实现 vs lift.lift_frame")
    print("=" * 60)

    h_grid, avail, cache = load_static(args)

    # 取一帧 (优先中间, 且需 n±1 可用)
    ns = sorted(avail)
    n = ns[len(ns) // 2]
    if (n - 1) not in avail or (n + 1) not in avail:
        n = ns[1]
    print(f"[frame] fw#{n}")

    # 参考: lift.lift_frame 在 (Nz, Ny, Nx) 上
    z = np.linspace(args.z_min, args.z_max, args.check_nz)
    eta_g, u_g, v_g = cache(n)
    em, um, vm = cache(n - 1)
    ep, up, vp = cache(n + 1)
    Tm = LC.horizontal_terms(em, um, vm, h_grid, args.dx, args.dy)
    Tp = LC.horizontal_terms(ep, up, vp, h_grid, args.dx, args.dy)
    dt2 = 2.0 * args.plot_intv
    A_dot_g = (Tp["A"] - Tm["A"]) / dt2
    B_dot_g = (Tp["B"] - Tm["B"]) / dt2
    ref = LC.lift_frame(eta_g, u_g, v_g, h_grid, args.dx, args.dy, z,
                        A_dot=A_dot_g, B_dot=B_dot_g)      # (Nz,Ny,Nx,5)

    # 散点路径: 把同样的 (i,j,z) 当成 CFD cell
    ny, nx = args.nglob, args.mglob
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x_fw = (ii * args.dx).ravel()
    y_fw = (jj * args.dy).ravel()
    n_xy = x_fw.size
    z_rep = np.repeat(z, n_xy)                     # (Nz*n_xy,)
    x_rep = np.tile(x_fw, len(z))
    y_rep = np.tile(y_fw, len(z))

    bil = Bilinear(x_rep, y_rep, args.mglob, args.nglob, args.dx, args.dy)
    inv = np.arange(x_rep.size)                    # 不去重, 直接一一对应
    got = build_frame(n, cache, avail, h_grid, bil, z_rep, inv,
                      args.dx, args.dy, args.plot_intv, use_pnh=True)
    got = got.reshape(len(z), ny, nx, 5)

    # 比对 (NaN 位置也必须一致)
    # 注: LC.lift_frame 末尾 .astype(np.float32), 故把散点结果也转 float32 再比,
    #     否则会把 float32 舍入 (~1e-7 相对) 误判成公式漂移。
    got = got.astype(np.float32)
    TOL = 1e-6                                   # 数个 float32 ULP
    ok = True
    for c, name in enumerate(LC.CH_NAMES):
        a, b = ref[..., c], got[..., c]
        na, nb = ~np.isfinite(a), ~np.isfinite(b)
        if not np.array_equal(na, nb):
            print(f"  [{name}] NaN 位置不一致: ref {na.sum()} vs got {nb.sum()}")
            ok = False
            continue
        m = ~na
        if m.sum() == 0:
            print(f"  [{name}] 全 NaN, 跳过")
            continue
        d = np.abs(a[m].astype(np.float64) - b[m].astype(np.float64))
        scale = max(np.abs(a[m]).max(), 1e-30)
        rel = d.max() / scale
        flag = "OK " if rel < TOL else "!! "
        if rel >= TOL:
            ok = False
        print(f"  {flag}[{name:6s}] max|Δ|={d.max():.3e}  "
              f"rel={rel:.3e}  (n={m.sum()})")

    print("=" * 60)
    print("剖面公式: 通过 ✓" if ok else "剖面公式: 失败 ✗ —— 与 lift.py 有漂移")
    ok_bil = check_bilinear(args)
    print("=" * 60)
    all_ok = ok and ok_bil
    print("全部自检通过 ✓" if all_ok else "自检失败 ✗")
    print("=" * 60)
    return 0 if all_ok else 1


# ------------------------------------------------------------ 静态加载 ------

def load_static(args):
    dep_path = os.path.join(args.fw_dir, "dep.out")
    if not os.path.exists(dep_path):
        cand = sorted(glob.glob(os.path.join(args.fw_dir, "dep*")))
        if not cand:
            sys.exit(f"[err] 找不到 dep 文件于 {args.fw_dir}")
        dep_path = cand[0]
    h_grid = read2d(dep_path, args.mglob, args.nglob)

    avail = set(int(m.group(1)) for p in
                glob.glob(os.path.join(args.fw_dir, "eta_*"))
                if (m := re.search(r"eta_(\d+)$", os.path.basename(p))))
    if not avail:
        sys.exit(f"[err] {args.fw_dir} 下无 eta_* 帧")

    cache = FrameCache(args.fw_dir, args.mglob, args.nglob)
    return h_grid, avail, cache


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", required=True)
    ap.add_argument("--self-check", action="store_true",
                    help="只跑与 lift.lift_frame 的一致性自检, 不生成数据")
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
                    help="t_fw = t_cfd + t_offset  [s] (需标定; 先用 0 做几何诊断)")
    # FUNWAVE 网格
    ap.add_argument("--mglob", type=int, default=1575)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--no-pnh", action="store_true", help="关闭非静水 p_rgh 修正")
    # 自检用
    ap.add_argument("--z-min", type=float, default=-0.399772)
    ap.add_argument("--z-max", type=float, default=0.147884)
    ap.add_argument("--check-nz", type=int, default=12, dest="check_nz")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(self_check(args))

    for req in ("coords", "gt_times", "chunk", "x_offset", "t_offset"):
        if getattr(args, req) is None:
            sys.exit(f"[err] 生成模式必须显式给 --{req.replace('_', '-')}")

    os.makedirs(args.out, exist_ok=True)
    h_grid, avail, cache = load_static(args)

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

    for i, k in enumerate(n_fw):
        k = int(k)
        if k not in avail:
            continue                       # data 保持 0, valid 保持 False
        out = build_frame(k, cache, avail, h_grid, bil, z_c, inv,
                          args.dx, args.dy, args.plot_intv,
                          use_pnh=not args.no_pnh)
        fin = np.isfinite(out).all(axis=1)
        out[~fin, :] = 0.0                 # NaN -> 0 (与 P 架构兼容)
        data[i] = out.astype(np.float32)
        valid[i] = fin
        if i % 10 == 0:
            print(f"  {i+1}/{T}  fw#{k}  valid {fin.mean()*100:.2f}%", flush=True)

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
    with open(os.path.join(args.out, "prior_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

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