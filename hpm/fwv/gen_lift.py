#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_lift.py — 把 pure lifting 的完整 3D 产物按 chunk 落盘 (阶段资产)。

纯度契约: 数值 100% 来自 lift.py, 无插值/无平滑/无 offset;
chunk 定义在 FUNWAVE 自己的时间轴: chunk N = FW 帧 [N*100, N*100+99]。

输出 (镜像 GT chunk 约定):
  <out>/lift_{cid:03d}_data.npy    (T, Nz, Ny, Nx, 5)  默认 float16 (~9GB/chunk)
  <out>/lift_{cid:03d}_times.npy   (T,)  t_fw [s]
  <out>/lift_coords.npz            x, y, z 轴 + 参数 (一次性, 自描述)
通道序: [alpha, Ux, Uy, Uz, p_rgh]

用法:
  python gen_lift.py --fw-dir .../TingKirby1994_3D_spilling/output \
      --chunk 6 --out ../data/3d/lift
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from numpy.lib.format import open_memmap

import lift as LC


# ---------------------------------------------------- IO (与 vis_lift 同构) --

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


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw-dir", required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--chunk-len", type=int, default=100)
    ap.add_argument("--out", default="lift_data")
    ap.add_argument("--mglob", type=int, default=1258)
    ap.add_argument("--nglob", type=int, default=30)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    ap.add_argument("--z-min", type=float, default=-0.40)
    ap.add_argument("--z-max", type=float, default=0.15)
    ap.add_argument("--nz", type=int, default=240)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--no-pnh", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dep_path = os.path.join(args.fw_dir, "dep.out")
    if not os.path.exists(dep_path):
        dep_path = sorted(glob.glob(os.path.join(args.fw_dir, "dep*")))[0]
    h = read2d(dep_path, args.mglob, args.nglob)
    x = np.arange(args.mglob) * args.dx
    y = np.arange(args.nglob) * args.dy
    z = np.linspace(args.z_min, args.z_max, args.nz)
    avail = set(int(m.group(1)) for p in
                glob.glob(os.path.join(args.fw_dir, "eta_*"))
                if (m := re.search(r"eta_(\d+)$", os.path.basename(p))))

    n0 = args.chunk * args.chunk_len
    frames = list(range(n0, n0 + args.chunk_len))
    missing = [n for n in frames if n not in avail]
    if len(missing) == len(frames):
        raise SystemExit(f"[err] chunk {args.chunk} 无可用 FW 帧")
    if missing:
        print(f"[warn] chunk {args.chunk}: {len(missing)} 帧缺失, 对应帧全 NaN")

    # 一次性坐标/参数文件 (幂等覆盖)
    np.savez(os.path.join(args.out, "lift_coords.npz"),
             x=x, y=y, z=z, ch=np.array(LC.CH_NAMES),
             meta=json.dumps(dict(beta=LC.BETA, dx=args.dx, dy=args.dy,
                                  plot_intv=args.plot_intv,
                                  chunk_len=args.chunk_len,
                                  pnh=not args.no_pnh)))

    T = len(frames)
    dpath = os.path.join(args.out, f"lift_{args.chunk:03d}_data.npy")
    data = open_memmap(dpath, mode="w+", dtype=args.dtype,
                       shape=(T, args.nz, args.nglob, args.mglob, 5))
    cache = FrameCache(args.fw_dir, args.mglob, args.nglob)

    for i, n in enumerate(frames):
        if n not in avail:
            data[i] = np.nan
            continue
        eta, u, v = cache(n)
        A_dot = B_dot = None
        if not args.no_pnh and (n - 1) in avail and (n + 1) in avail:
            em, um, vm = cache(n - 1)
            ep, up, vp = cache(n + 1)
            Tm = LC.horizontal_terms(em, um, vm, h, args.dx, args.dy)
            Tp = LC.horizontal_terms(ep, up, vp, h, args.dx, args.dy)
            dt2 = 2.0 * args.plot_intv
            A_dot, B_dot = (Tp["A"] - Tm["A"]) / dt2, (Tp["B"] - Tm["B"]) / dt2
        data[i] = LC.lift_frame(eta, u, v, h, args.dx, args.dy, z,
                                A_dot=A_dot, B_dot=B_dot)
        if i % 10 == 0:
            print(f"  {i+1}/{T}  fw#{n}  t_fw={n*args.plot_intv:.2f}s",
                  flush=True)
    data.flush()

    np.save(os.path.join(args.out, f"lift_{args.chunk:03d}_times.npy"),
            np.array(frames, float) * args.plot_intv)
    gb = os.path.getsize(dpath) / 1e9
    print(f"[done] {dpath}  shape={data.shape} dtype={args.dtype}  {gb:.1f}GB")


if __name__ == "__main__":
    main()