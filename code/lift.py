#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lift.py — 抬升算子 (lifting operator) 本体。

把 FUNWAVE 的 2D 状态 (eta, u_a, v_a) 按 Nwogu (1993) 抬升为 3D 场。

契约:
  - 只做数学: 水平项 A/B + Nwogu 二次剖面 + 连续性 W + Heaviside alpha
    + 静水/非静水 p_rgh
  - 无插值、无平滑、无坐标偏移 —— 这些属于工程层, 写在调用方
  - 唯一的离散操作: ∇A/∇B 的中心差分 (np.gradient) —— 算子定义内在需要
  - 干单元 (MASK=0) -> NaN, 不置零、不外推 —— 理论无定义处保持无定义
  - 零自由参数; alpha 恒为锐利 Heaviside

分工:
  horizontal_terms  在 FUNWAVE 原生网格上算 A、B 及其水平梯度
  nwogu_at_points   在任意散点上求剖面 —— 每点用自己的 z, z 方向不插值

调用方负责把 horizontal_terms 的输出插到目标点, 再喂给 nwogu_at_points。
剖面公式在本仓库中只有这一份实现。
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


def nwogu_at_points(z_c, eta, h, u, v, dAdx, dAdy, dBdx, dBdy, A, B,
                    A_dot=None, B_dot=None):
    """在散点上求值 3D 场。所有入参形状相同 (N,), z_c 为各点自己的高程。

      Ux = u + (za - z)dAdx + 0.5(za^2 - z^2)dBdx      Nwogu 二次剖面
      Uy = v + (za - z)dAdy + 0.5(za^2 - z^2)dBdy
      Uz = -(A + z B)                                   连续性
      p  = rho g eta - rho[Adot(eta - z) + 0.5 Bdot(eta^2 - z^2)]
           A_dot / B_dot 为 None 时退化为纯静水 rho g eta

    剖面对 z 是解析的, 故直接代入各点自己的 z —— 无论目标网格规则与否,
    全程不做 z 方向插值。规则网格的用法是把它摊平成散点后 reshape 回去。

    返回 (N, 5), 通道序 CH_NAMES; 空气区 U=p=0; 无定义处整点 NaN。
    """
    za = BETA * h
    water = z_c <= eta                      # eta 为 NaN 时 -> False, 后面统一置 NaN

    Ux = u + (za - z_c) * dAdx + 0.5 * (za ** 2 - z_c ** 2) * dBdx
    Uy = v + (za - z_c) * dAdy + 0.5 * (za ** 2 - z_c ** 2) * dBdy
    Uz = -(A + z_c * B)

    p = RHO * G * eta * np.ones_like(Ux)
    if A_dot is not None and B_dot is not None:
        p = p - RHO * (A_dot * (eta - z_c)
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