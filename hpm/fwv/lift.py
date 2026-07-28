#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lift.py — 最 pure 的 lifting 算子本体。

契约:
  - 只做数学: Nwogu 二次剖面 + 连续性 W + Heaviside alpha + 静水/非静水 p_rgh
  - 在 FUNWAVE 原生网格点上求值, 无插值、无平滑、无坐标偏移
  - 唯一的离散操作: ∇A/∇B 的中心差分 (np.gradient) —— 算子定义内在需要
  - 干单元 (MASK=0) -> NaN, 不置零、不外推 —— 理论无定义处保持无定义
  - 零自由参数; alpha 恒为锐利 Heaviside

若将来 generate 模式需要在 CFD 点取值 / 平滑 / offset, 那些属于工程层,
写在调用方, 不进入本文件。
"""

import numpy as np

G = 9.81
RHO = 1000.0
BETA = -0.531          # z_alpha = BETA * h, Nwogu 色散最优参考深度
CH_NAMES = ["alpha", "Ux", "Uy", "Uz", "p_rgh"]


def horizontal_terms(eta, u, v, h, dx, dy):
    """在原生网格上计算 A, B 及其水平梯度。

    A = div(h u_a)   B = div(u_a)
    干单元为 NaN 时, np.gradient 会让 NaN 沿差分模板自然扩散一格 ——
    这是"理论无定义"的诚实传播, 不做任何修补。
    """
    A = np.gradient(h * u, dx, axis=1) + np.gradient(h * v, dy, axis=0)
    B = np.gradient(u, dx, axis=1) + np.gradient(v, dy, axis=0)
    return dict(
        A=A, B=B,
        dAdx=np.gradient(A, dx, axis=1), dAdy=np.gradient(A, dy, axis=0),
        dBdx=np.gradient(B, dx, axis=1), dBdy=np.gradient(B, dy, axis=0),
    )


def lift_frame(eta, u, v, h, dx, dy, z,
               A_dot=None, B_dot=None):
    """把一帧 2D 状态提升为 3D 场 (在原生 x-y 网格 × 给定 z 轴上)。

    参数
      eta, u, v : (Ny, Nx)  自由面 / u_alpha / v_alpha, 干单元应为 NaN
      h         : (Ny, Nx)  静水深 (正值)
      z         : (Nz,)     求值高程
      A_dot, B_dot : (Ny, Nx) 或 None —— A、B 的时间导数 (由调用方用相邻帧
                     中心差分提供)。None 时 p_rgh 退化为纯静水项。

    返回 (Nz, Ny, Nx, 5), 通道序 CH_NAMES。空气区 U=V=W=0 (Design 2 αU 约定),
    p_rgh 空气区为 0。干单元整柱 NaN。
    """
    T = horizontal_terms(eta, u, v, h, dx, dy)
    za = BETA * h                                   # (Ny, Nx)
    zc = z[:, None, None]                           # (Nz, 1, 1)

    water = zc <= eta[None]                         # (Nz, Ny, Nx)
    alpha = water.astype(np.float32)

    # Nwogu 二次剖面: u(z) = u_a + (za - z) ∇A + ½(za² - z²) ∇B
    Ux = u[None] + (za[None] - zc) * T["dAdx"][None] \
        + 0.5 * (za[None] ** 2 - zc ** 2) * T["dBdx"][None]
    Uy = v[None] + (za[None] - zc) * T["dAdy"][None] \
        + 0.5 * (za[None] ** 2 - zc ** 2) * T["dBdy"][None]
    # 连续性: W(z) = -[A + z B]
    Uz = -(T["A"][None] + zc * T["B"][None])

    # p_rgh: 静水扰动 ρgη + 非静水修正 (仅当提供 Ȧ, Ḃ)
    p = RHO * G * eta[None] * np.ones_like(Ux)
    if A_dot is not None and B_dot is not None:
        p = p - RHO * (A_dot[None] * (eta[None] - zc)
                       + 0.5 * B_dot[None] * (eta[None] ** 2 - zc ** 2))

    air = ~water
    for F in (Ux, Uy, Uz, p):
        F[air] = 0.0

    out = np.stack([alpha, Ux, Uy, Uz, p], axis=-1).astype(np.float32)

    # 干单元: eta 为 NaN -> 整柱 NaN (含 alpha), 诚实标注无定义区
    dry = ~np.isfinite(eta)
    out[:, dry, :] = np.nan
    # 床底以下同样无定义
    below_bed = zc < (-h)[None]
    out[below_bed] = np.nan
    return out
