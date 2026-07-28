#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_lift.py — pure lifting 的可视化视频, 按 chunk 组织 (y=0.30 行 x-z 切片)。

渲染风格对齐 vis.py / vis_u.py（matplotlib + FuncAnimation + ffmpeg）:
  - 每个场单独一支 mp4 (single panel; lift 无 GT/pred 之分)
  - pcolormesh 连续填充场 (对齐 vis.py 的 tri 观感; 规则网格无需三角化)
  - aspect='auto' + set_xlim/set_ylim —— 与 GT 同一套 z 拉伸机制 (无显式 z_scale)
  - clim = 本 chunk 数据实际 min/max (场内比, 不跨 chunk); alpha 固定 0..1
  - 配色: alpha -> OpacityReds(白→红), 其余 -> coolwarm

纯度契约 (不变):
  - 场数值 100% 来自 lift.py (无插值/无平滑/无 offset)
  - chunk 定义对齐 GT times 约定: chunk N = FW 帧 [N*100+1, N*100+100]
    (使 t_fw 首值 = GT chunk_{N}_times.npy[0]; 逐帧标签对齐, 无物理 t-offset)
    (沿用 GT 切分约定; lift chunk N 与 GT chunk N 不是同一物理时间窗 ——
     对应需要 t-offset, 本流程不引入; chunk 在此仅是文件组织约定。)

依赖: matplotlib + ffmpeg (imageio-ffmpeg 或系统 ffmpeg)

用法:
  python vis_lift.py \
      --fw-dir /nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling/output \
      --chunk 6
输出:
  vis_lift/lift_006_alpha_tri.mp4   (Ux / Uz / p_rgh 同构命名)
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

import lift as LC


# ------------------------------------------------------------------ IO ------

def read2d(path, mglob, nglob):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape == (nglob, mglob):
        return arr
    if arr.shape == (mglob, nglob):
        return arr.T
    raise ValueError(f"{path}: shape {arr.shape}")


class FrameCache:
    """(eta,u,v) 帧缓存 —— p_nh 需要 n±1, 顺序播放时邻帧可复用。"""

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


# ---- alpha 配色: 与 vis.py 完全同款 OpacityReds (白→红) ----
_ALPHA_CDICT = {
    'red':   [[0., 1., 1.], [1., .6, .6]],
    'green': [[0., 1., 1.], [1., 0., 0.]],
    'blue':  [[0., 1., 1.], [1., 0., 0.]],
}
ALPHA_CMAP = LinearSegmentedColormap('OpacityReds', _ALPHA_CDICT)


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", required=True)
    ap.add_argument("--chunk", type=int, required=True,
                    help="chunk N = FW 帧 [N*chunk_len+1, N*chunk_len+chunk_len] (对齐 GT times)")
    ap.add_argument("--chunk-len", type=int, default=100)
    ap.add_argument("--mglob", type=int, default=1258)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--y-cfd", type=float, default=0.30,
                    help="切片 y; 两侧最近行线性插值, 恰在行上则直取该行")
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=1e9)
    ap.add_argument("--x-shift", type=float, default=0.0,
                    help="显示层坐标平移: x_disp = x_fw - x_shift。"
                         "对齐 CFD 坐标系用 15.05 (x_fw = x_cfd + 15.05)。"
                         "纯显示, 不改任何数值; --x-min/--x-max 仍按 FUNWAVE 原生坐标选窗。")
    ap.add_argument("--x-lim", type=float, nargs=2, default=None,
                    help="显示 x 轴范围 (在平移后的坐标系中)。不给则用数据边界。")
    ap.add_argument("--z-lim", type=float, nargs=2, default=None,
                    help="显示 z 轴范围。不给则用数据边界。")
    ap.add_argument("--z-min", type=float, default=-0.40)
    ap.add_argument("--z-max", type=float, default=0.15)
    ap.add_argument("--nz", type=int, default=240)
    ap.add_argument("--no-pnh", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", default="vis_lift")
    # 每个 panel 的 figure 尺寸: 宽:高 与 vis.py 单个子图 (~38.4x10.5, 3.6:1) 同量级,
    # 使 aspect='auto' 下 z 的隐式拉伸倍率与 GT 落在同一量级。
    # 默认 3840x1080 px, 与 GT mp4 (gt_alpha_*.mp4) 尺寸完全一致 ->
    # stitch 时无需 scale, 直接 vstack, 且两 panel 的 aspect='auto' 拉伸倍率相同。
    ap.add_argument("--fig-w", type=float, default=38.4)
    ap.add_argument("--fig-h", type=float, default=10.8)
    ap.add_argument("--margins", type=float, nargs=4,
                    default=[0.05, 0.13, 0.95, 0.90],
                    metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                    help="subplots_adjust 边距 (figure 比例)。左右默认取 vis.py 的 "
                         "0.05/0.95; 若与 GT 仍有偏差, 按 vis_gt.py 的实际值调。")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # y 切片: 两侧最近行线性插值; y 恰在行上时权重退化为 (1,0) = 直接取行
    yf = args.y_cfd / args.dy
    j0 = int(np.clip(np.floor(yf), 0, args.nglob - 1))
    j1 = int(np.clip(j0 + 1, 0, args.nglob - 1))
    w1 = float(yf - j0) if j1 != j0 else 0.0
    w0 = 1.0 - w1
    # 窗口 +1 帧对齐 GT: GT chunk N 的 times[0] = (N*chunk_len+1)*plot_intv
    # (选项1 纯 bookkeeping, 逐帧标签对齐, 不引入物理 t-offset)
    n0 = args.chunk * args.chunk_len + 1
    n1 = n0 + args.chunk_len - 1
    print(f"[cfg] chunk {args.chunk}: fw#{n0}..{n1} "
          f"(t_fw {n0*args.plot_intv:.2f}..{n1*args.plot_intv:.2f}s)  "
          f"y={args.y_cfd} -> rows {j0}/{j1} w=({w0:.2f},{w1:.2f})")

    # 静态量
    dep_path = os.path.join(args.fw_dir, "dep.out")
    if not os.path.exists(dep_path):
        dep_path = sorted(glob.glob(os.path.join(args.fw_dir, "dep*")))[0]
    h = read2d(dep_path, args.mglob, args.nglob)
    x_fw = np.arange(args.mglob) * args.dx
    sel = (x_fw >= args.x_min) & (x_fw <= args.x_max)     # 选窗: FUNWAVE 原生坐标
    xs = x_fw[sel] - args.x_shift    # 显示坐标 (纯平移, 不改数值; 0 时为原生坐标)
    z = np.linspace(args.z_min, args.z_max, args.nz)

    avail = set(int(m.group(1)) for p in
                glob.glob(os.path.join(args.fw_dir, "eta_*"))
                if (m := re.search(r"eta_(\d+)$", os.path.basename(p))))
    frames = [n for n in range(n0, n1 + 1) if n in avail]
    if not frames:
        raise SystemExit(f"[err] chunk {args.chunk} 无可用 FW 帧")
    if len(frames) < args.chunk_len:
        print(f"[warn] chunk {args.chunk}: 仅 {len(frames)}/{args.chunk_len} 帧可用")

    cache = FrameCache(args.fw_dir, args.mglob, args.nglob)

    # pcolormesh 需要 cell 边界 (edges): x 用 FW 网格边界, z 用均匀边界。
    dz = z[1] - z[0]
    x_edges = np.concatenate([xs - args.dx / 2, xs[-1:] + args.dx / 2])
    z_edges = np.concatenate([z - dz / 2, z[-1:] + dz / 2])

    # 床底行 (bed line) — 静态, 一次算好
    bed_row = w0 * (-h[j0, sel]) + w1 * (-h[j1, sel])

    # 场定义: (显示名, lift 通道索引, cmap, 文件名 field, 是否固定 0..1)
    panels = [
        ("alpha",       0, ALPHA_CMAP, "alpha",  True),
        ("Ux [m/s]",    1, "coolwarm", "Ux",     False),
        ("Uz [m/s]",    3, "coolwarm", "Uz",     False),
        ("p_rgh [Pa]",  4, "coolwarm", "p_rgh",  False),
    ]

    # ---- 预先跑完整个 chunk 的 lift, 缓存切片 (Nz, Nx_sel) per frame per channel。
    #      理由: clim 需要全 chunk 数据的实际 min/max (场内定标), 必须先扫一遍;
    #      顺带避免渲染循环里重复 lift。内存: 4 场 × T × Nz × Nx_sel × f32。----
    print("[lift] 预计算全 chunk 切片 (用于 clim 场内定标)...")
    T = len(frames)
    Nz, Nx = args.nz, len(xs)
    ch_idx = [p[1] for p in panels]
    field_slices = {p[3]: np.full((T, Nz, Nx), np.nan, np.float32) for p in panels}
    eta_lines = np.full((T, Nx), np.nan, np.float32)

    for fi, n in enumerate(frames):
        eta, u, v = cache(n)
        A_dot = B_dot = None
        if not args.no_pnh and (n - 1) in avail and (n + 1) in avail:
            em, um, vm = cache(n - 1)
            ep, up, vp = cache(n + 1)
            Tm = LC.horizontal_terms(em, um, vm, h, args.dx, args.dy)
            Tp = LC.horizontal_terms(ep, up, vp, h, args.dx, args.dy)
            dt2 = 2.0 * args.plot_intv
            A_dot, B_dot = (Tp["A"] - Tm["A"]) / dt2, (Tp["B"] - Tm["B"]) / dt2

        out = LC.lift_frame(eta, u, v, h, args.dx, args.dy, z,
                            A_dot=A_dot, B_dot=B_dot)
        sl = (w0 * out[:, j0, :, :] + w1 * out[:, j1, :, :])[:, sel, :]  # (Nz, Nx, 5)
        for (_, ch, _, field, _) in panels:
            field_slices[field][fi] = sl[..., ch]
        eta_lines[fi] = w0 * eta[j0, sel] + w1 * eta[j1, sel]

        if fi % 20 == 0:
            print(f"  lift {fi+1}/{T}  fw#{n}", flush=True)

    # 每场 clim: alpha 固定 0..1; 其余用本 chunk 数据实际 min/max (nan 安全)。
    clims = {}
    for (_, _, _, field, fixed01) in panels:
        if fixed01:
            clims[field] = (0.0, 1.0)
        else:
            arr = field_slices[field]
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            # coolwarm 发散场对称化 (0 居中), 与物理零点对齐; 全同号则退回实际范围。
            if vmin < 0 < vmax:
                m = max(abs(vmin), abs(vmax))
                clims[field] = (-m, m)
            else:
                clims[field] = (vmin, vmax)
    print("[clim] " + "  ".join(f"{k}=[{v[0]:.3g},{v[1]:.3g}]"
                                 for k, v in clims.items()))

    xlim = tuple(args.x_lim) if args.x_lim else (x_edges[0], x_edges[-1])
    zlim = tuple(args.z_lim) if args.z_lim else (z_edges[0], z_edges[-1])
    print(f"[axes] xlim={xlim[0]:.4f}..{xlim[1]:.4f}  "
          f"zlim={zlim[0]:.4f}..{zlim[1]:.4f}  (x_shift={args.x_shift})")

    # ---- 逐场渲染 mp4 ----
    for (disp, ch, cmap, field, _) in panels:
        out_path = os.path.join(args.out_dir,
                                f"lift_{args.chunk:03d}_{field}_tri.mp4")
        print(f"[{field}] -> {out_path}")
        vmin, vmax = clims[field]
        data = field_slices[field]                     # (T, Nz, Nx)

        fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h), dpi=100)
        ml, mb, mr, mt = args.margins
        fig.subplots_adjust(left=ml, bottom=mb, right=mr, top=mt)

        # 初始帧 pcolormesh (shading='flat' 需 edges; nan 显示为底色)
        qm = ax.pcolormesh(x_edges, z_edges, data[0],
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        qm.cmap.set_bad('lightgray')
        # 无 colorbar (对齐 GT/vis.py); clim 已场内定标, 场名在 title 中标注

        bed_ln, = ax.plot(xs, bed_row, color='black', lw=2)      # 床底 (静态)
        eta_ln, = ax.plot(xs, eta_lines[0], color='black', lw=1)  # 自由面 (逐帧)

        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_aspect('auto')                       # 与 GT 同机制: z 由框比隐式拉伸
        ax.set_xlabel("X (m)", fontsize=16)
        ax.set_ylabel("Z (m)", fontsize=16)
        ax.tick_params(labelsize=12)
        n_init = frames[0]
        title = ax.set_title(
            f"pure lift | {disp} | chunk {args.chunk} | fw#{n_init} | "
            f"t_fw={n_init*args.plot_intv:.2f}s | Step 0", fontsize=18)

        def update(fi, ch=ch, field=field, disp=disp, data=data,
                   qm=qm, eta_ln=eta_ln, title=title):
            n = frames[fi]
            # pcolormesh(flat) 的 set_array 期望展平的 cell 值 (Nz*Nx)
            qm.set_array(data[fi].ravel())
            eta_ln.set_ydata(eta_lines[fi])
            title.set_text(
                f"pure lift | {disp} | chunk {args.chunk} | fw#{n} | "
                f"t_fw={n*args.plot_intv:.2f}s | Step {fi}")
            return qm, eta_ln, title

        ani = animation.FuncAnimation(fig, update, frames=T,
                                      interval=1000 // args.fps, blit=False)
        ani.save(out_path, writer="ffmpeg", fps=args.fps,
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        plt.close(fig)
        print(f"[{field}] saved")

    print(f"[done] {len(panels)} 支 mp4 -> {args.out_dir}/")


if __name__ == "__main__":
    main()