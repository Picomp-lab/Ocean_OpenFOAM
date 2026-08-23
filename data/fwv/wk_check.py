#!/usr/bin/env python3
"""
FUNWAVE-TVD  WK_REG 内造波参数体检
用法:  python3 wk_check.py  H  T   [h_wavemaker]
       H = 目标波高(m)   T = 周期(s)   h = 造波处水深(默认 0.4)

复刻 src/wavemaker.F:2188 WK_WAVEMAKER_REGULAR_WAVE 的色散求解
(alpha = -0.39 的改进型 Boussinesq 色散关系)。
"""
import sys, math

g = 9.81

# 本算例的固定几何(取自 input.txt)
DX = DY = 0.02
XC_WK = 6.35          # 造波源中心
SPONGE_W = 3.175      # 西侧海绵层宽度
X_SLP = 14.35         # 坡脚
SLP = 0.028571429     # 1:35
DEPTH_FLAT = 0.4
MGLOB = 1575
DELTA_WK = 3.0

def wavenumber(T, h):
    """源码同款:alpha=-0.39 改进色散关系解 k"""
    alpha = -0.39
    alpha1 = alpha + 1.0/3.0
    omgn = 2*math.pi/T
    tb = omgn*omgn*h/g
    tc = 1.0 + tb*alpha
    k = math.sqrt((tc - math.sqrt(tc*tc - 4.0*alpha1*tb)) / (2.0*alpha1)) / h
    return k, omgn

def airy_k(T, h):
    """线性 Airy 色散,用于对照"""
    omg = 2*math.pi/T
    k = omg**2/g
    for _ in range(200):
        k = omg**2/(g*math.tanh(k*h))
    return k

def check(H, T, h=DEPTH_FLAT):
    a = H/2.0
    k, omg = wavenumber(T, h)
    L = 2*math.pi/k
    C = L/T
    kh = k*h
    width = DELTA_WK*L/2.0          # wavemaker.F:2215  Width_WK,方框截断半宽
    beta = 80.0/DELTA_WK**2/L**2    # wavemaker.F:2216  高斯系数
    # sources.F:102  源项 ∝ exp(-beta*(x-Xc)^2)，等价高斯标准差：
    sigma = 1.0/math.sqrt(2*beta)   # = Delta_WK*L/sqrt(160)
    ka = airy_k(T, h); L_airy = 2*math.pi/ka

    print("="*66)
    print("  输入:  H = %.4f m   T = %.3f s   造波处水深 h = %.3f m" % (H, T, h))
    print("="*66)
    print("\n  【必须写进 input.txt 的值】")
    print("    AMP_WK   = %.5f        <- 振幅 a = H/2,不是波高!" % a)
    print("    Tperiod  = %.3f" % T)
    print("    DEP_WK   = %.3f" % h)

    print("\n  【波浪特征】")
    print("    波长 L        = %8.4f m   (线性 Airy 对照 %.4f m)" % (L, L_airy))
    print("    相速 C        = %8.4f m/s" % C)
    print("    kh            = %8.4f      %s" % (kh, ok(kh < 3.0, "Boussinesq 适用 (kh<3)", "kh 过大,超出 Boussinesq 适用范围")))
    print("    H/h (造波处)  = %8.4f      %s" % (H/h, ok(H/h < 0.5, "非线性适中", "波陡过大,造波处可能提前破碎")))
    print("    Ursell 数     = %8.2f" % (H*L*L/h**3))

    print("\n  【网格分辨率】")
    ppw = L/DX
    print("    每波长网格数  = %8.1f      %s" % (ppw, ok(ppw >= 40, "充足 (>=40)", "偏少,建议 >=40,考虑减小 DX")))

    print("\n  【造波源几何】 源项 ∝ exp(-Beta_gen*(x-Xc_WK)^2),sources.F:102")
    print("    高斯标准差 σ  = %8.4f m   (= Delta_WK*L/sqrt(160),Delta_WK = %.1f)" % (sigma, DELTA_WK))
    x_lo, x_hi = XC_WK - 3*sigma, XC_WK + 3*sigma
    print("    有效源区 ±3σ  = %.3f ~ %.3f m   (含 99.7%% 源强)" % (x_lo, x_hi))
    print("    方框截断 ±W   = %.3f ~ %.3f m   (Width_WK = %.3f,仅作截断,不决定源宽)"
          % (XC_WK - width, XC_WK + width, width))
    # 海绵层外沿处源强还剩多少
    r_sp = math.exp(-beta*(XC_WK - SPONGE_W)**2)
    r_sl = math.exp(-beta*(X_SLP - XC_WK)**2)
    print("    海绵层外沿 x=%.3f 处源强 = %.2e 倍峰值   %s"
          % (SPONGE_W, r_sp, ok(r_sp < 0.01, "可忽略 (<1%)", "源被海绵层吃掉一部分,造波偏弱!")))
    print("    坡脚     x=%.3f 处源强 = %.2e 倍峰值   %s"
          % (X_SLP, r_sl, ok(r_sl < 0.01, "可忽略 (<1%)", "源伸到斜坡上,违反常水深假设!")))
    print("    海绵层宽 %.3f m = %.2f L    %s" % (SPONGE_W, SPONGE_W/L,
          ok(SPONGE_W/L >= 0.8, "足够吸收 (>=0.8L)", "偏窄,建议 >=1 个波长,否则反射")))

    print("\n  【破波位置估计】")
    # 线性浅化 + 经验破波判据 H/h = 0.78
    hb = shoal_break(H, T, h)
    if hb:
        xb = X_SLP + (DEPTH_FLAT - hb)/SLP
        print("    破波水深 h_b  ≈ %6.4f m" % hb)
        print("    破波位置 x_b  ≈ %6.2f m   (岸线 %.2f m)" % (xb, X_SLP + DEPTH_FLAT/SLP))
        print("    → 测点 gauges.txt 应重新布置在 x_b 附近及其岸侧")
        print("      当前测点覆盖 x = 13.78 ~ 24.78 m  %s"
              % ok(13.0 < xb < 25.0, "破波点在测点范围内", "破波点已移出当前测点范围,需改 gauges.txt"))
    print("\n  【时间设置建议】")
    t_travel = (X_SLP - XC_WK)/C + 8.0   # 粗估传播+爬坡
    print("    起动段建议    >= %5.1f s   (约 %.0f 个周期)" % (max(20*T, t_travel), max(20, t_travel/T)))
    print("    TOTAL_TIME    建议 %5.1f s  (起动 + 50 个周期用于统计)" % (max(20*T, t_travel) + 50*T))
    print("    PLOT_INTV     建议 %.4f s  (T/40,与当前 T/%.0f 一致性)" % (T/40, 2.0/0.05))
    print()

def shoal_break(H0, T, h0):
    """线性浅化 + Green 定律,配合 H/h=0.78 判据估破波水深"""
    k0, _ = wavenumber(T, h0)
    n0 = 0.5*(1 + 2*k0*h0/math.sinh(2*k0*h0))
    Cg0 = n0*(2*math.pi/k0)/T
    h = h0
    while h > 0.005:
        k = airy_k(T, h)
        n = 0.5*(1 + 2*k*h/math.sinh(2*k*h))
        Cg = n*(2*math.pi/k)/T
        H = H0*math.sqrt(Cg0/Cg)
        if H/h >= 0.78:
            return h
        h -= 0.001
    return None

def ok(cond, yes, no):
    return ("[OK]   " + yes) if cond else ("[注意] " + no)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        print("当前算例(Ting & Kirby 1994 溢出型)体检:")
        check(0.127, 2.0)
    else:
        H = float(sys.argv[1]); T = float(sys.argv[2])
        h = float(sys.argv[3]) if len(sys.argv) > 3 else DEPTH_FLAT
        check(H, T, h)
