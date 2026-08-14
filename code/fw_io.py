#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fw_io.py — FUNWAVE 原生输出的读取层。

只做 IO: 二维场文件读取、帧缓存、静态量 (水深 / 可用帧) 加载。
不含任何物理公式 —— 那些在 lift.py。

被 gen_prior.py / vis.py(align 子命令) / scan_toffset.py 共用。
"""

import glob
import os
import re
import sys

import numpy as np


def read2d(path, mglob, nglob):
    """读一个 FUNWAVE 二维场文件, 统一返回 (nglob, mglob) = (ny, nx)。"""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape == (nglob, mglob):
        return arr
    if arr.shape == (mglob, nglob):
        return arr.T
    raise ValueError(f"{path}: shape {arr.shape} (期望 {(nglob, mglob)})")


class FrameCache:
    """(eta,u,v) 帧缓存 —— p_nh 需要 n±1, 顺序推进时邻帧可复用。

    干单元 (MASK<0.5) 置 NaN, 与 lift.py 的"无定义处保持无定义"一致。
    """

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


def load_static(fw_dir, mglob, nglob):
    """加载与时间无关的量。

    Returns:
        h_grid : (nglob, mglob) 静水深
        avail  : set[int]       磁盘上实际存在的帧号
        cache  : FrameCache
    """
    dep_path = os.path.join(fw_dir, "dep.out")
    if not os.path.exists(dep_path):
        cand = sorted(glob.glob(os.path.join(fw_dir, "dep*")))
        if not cand:
            sys.exit(f"[err] 找不到 dep 文件于 {fw_dir}")
        dep_path = cand[0]
    h_grid = read2d(dep_path, mglob, nglob)

    avail = set(int(m.group(1)) for p in
                glob.glob(os.path.join(fw_dir, "eta_*"))
                if (m := re.search(r"eta_(\d+)$", os.path.basename(p))))
    if not avail:
        sys.exit(f"[err] {fw_dir} 下无 eta_* 帧")

    cache = FrameCache(fw_dir, mglob, nglob)
    return h_grid, avail, cache