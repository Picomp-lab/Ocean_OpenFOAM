"""
Courant-number 诊断 (COARSE, streaming) — cropped_0.05 subdomain
================================================================
判断离散 alpha-advection 残差  R = dalpha/dt + div(alpha*U)  作为训练 target
在各区是否有效 (~0 on true data)。依据: 逐格 Courant C = |U|*dt/dx, 按 alpha-band 分层。
  C <~0.5 残差忠实; C >1 残差会惩罚正确预测 (target 本身错)。
粗版: C_i = |U_i| * dt / V_i^(1/3),  用 GT 数据 (raw U + true alpha)。

recon 已确认 (probe_alignment):
  - chunk_data: (T=100, N=574163, C=6) float32, 通道 [alpha,Ux,Uy,Uz,p_rgh,nut]
  - ch1-3 是 RAW 物理 U (m/s), 非归一化;  dt=0.05 (times 5.05->5.10...)
  - 对齐索引 full_cell_ids.npy (574163,) int64 存在
  - lbo 是 distance-weighted graph Laplacian, 无 mass matrix (故不能 diag(M) 取 V)

流式: 逐 chunk 累计直方图, 不 concat -> 峰值内存 ~单 chunk, 登录节点可跑。
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
DATA_DIR   = "../data/3d/cropped_0.05"     # 相对 models/hpm/; 按需改
DT_ML      = 0.05                          # snapshot 间隔 [s]
CH_ALPHA   = 0
CH_U       = (1, 2, 3)                     # Ux,Uy,Uz
SKIP_CHUNKS = ("chunk_000",)              # still-water OOD, 会虚高 low-C, 剔除

ALPHA_BANDS = [                            # (label, lo, hi)
    ("pure_air  (a<0.01)",   0.00, 0.01),
    ("near_air  (0.01-0.1)", 0.01, 0.10),
    ("interface (0.1-0.9)",  0.10, 0.90),
    ("water     (a>0.9)",    0.90, 1.0001),
]
C_THRESHOLDS = [0.5, 1.0]
BINS = np.logspace(-4, 2, 240)            # 共享 C 分箱 (1e-4 .. 1e2)
OUT_PNG = "courant_hist.png"

# --- V 来源: writeCellVolumes 的全网格 V + 对齐索引 --------------------------
# 先在 case 上写体积场 (静止网格, 一次即可):
#   cd /nfs/stak/users/baoh/hpc-share/ocean_project/case
#   postProcess -func writeCellVolumes -time 0
# 会在 0/ 里生成 volScalarField 'V' (ascii)。下面二选一填路径:
FULL_CELL_IDS = os.path.join(DATA_DIR, "full_cell_ids.npy")   # (574163,) int64
V_FULL_OF     = "/nfs/stak/users/baoh/hpc-share/ocean_project/case/0/V"  # OF ascii 场
V_FULL_NPY    = ""    # 或: 若你已把全网格 V 存成 .npy, 填这里, 优先用它
# ===============================================================


def read_of_scalar_internalfield(path):
    """读 OpenFOAM ascii volScalarField 的 internalField (nonuniform List<scalar>)。"""
    txt = open(path, "r", errors="ignore").read()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\(",
                  txt)
    if not m:
        mu = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", txt)
        raise ValueError("V 是 uniform 或无法解析 internalField"
                         + (f" (uniform={mu.group(1)})" if mu else ""))
    n = int(m.group(1))
    start = txt.index("(", m.end() - 1) + 1
    end = txt.index(")", start)
    vals = np.array(txt[start:end].split(), dtype=np.float64)
    assert vals.size == n, f"解析到 {vals.size} 个值, header 说 {n}"
    return vals


def load_cell_volumes(n_expect):
    idx = np.load(FULL_CELL_IDS)
    assert idx.shape[0] == n_expect, f"full_cell_ids {idx.shape[0]} != N {n_expect}"
    if V_FULL_NPY and os.path.exists(V_FULL_NPY):
        V_full = np.load(V_FULL_NPY)
    else:
        assert os.path.exists(V_FULL_OF), \
            f"没找到 {V_FULL_OF} — 先跑 postProcess -func writeCellVolumes"
        V_full = read_of_scalar_internalfield(V_FULL_OF)
    assert idx.max() < V_full.size, \
        f"索引越界: max idx {idx.max()} >= N_full {V_full.size} (全网格 V 对不上?)"
    V = V_full[idx].astype(np.float64)
    assert np.all(V > 0), "V 出现 <=0"
    print(f"[V] N_full={V_full.size}  N_sub={V.size}  "
          f"range=[{V.min():.3e},{V.max():.3e}] m^3  "
          f"max/min={V.max()/V.min():.1f}x (非均匀确认)")
    return V


def iter_chunks():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*_data.npy")))
    files = [f for f in files if not any(s in f for s in SKIP_CHUNKS)]
    assert files, "没找到 chunk 文件 (或全被 SKIP 了)"
    print(f"[chunks] {len(files)} 个 (已剔 {SKIP_CHUNKS}):",
          [os.path.basename(f) for f in files])
    for f in files:
        d = np.load(f)                     # (T, N, C) float32
        # 若布局不是 (T,N,C), 在此 transpose
        yield os.path.basename(f), d


def main():
    nb = len(ALPHA_BANDS)
    hist  = np.zeros((nb, BINS.size - 1), dtype=np.int64)
    n_tot = np.zeros(nb, dtype=np.int64)
    n_lt  = {thr: np.zeros(nb, dtype=np.int64) for thr in C_THRESHOLDS}

    V = None
    for ci, (name, d) in enumerate(iter_chunks()):
        T, N, C = d.shape
        if V is None:
            V = load_cell_volumes(N)
            dx = np.cbrt(V)                 # (N,)
        alpha = d[..., CH_ALPHA]                                   # (T,N) view
        speed = np.sqrt(d[..., CH_U[0]]**2 + d[..., CH_U[1]]**2
                        + d[..., CH_U[2]]**2)                      # (T,N)
        Cc = speed * DT_ML / dx[None, :]                          # (T,N)

        if ci == 0:   # 首 chunk 自检
            print(f"[check] alpha range=[{alpha.min():.4f},{alpha.max():.4f}] (应~[0,1])")
            print(f"[check] |U| p50/p95/max="
                  f"{np.percentile(speed,50):.3f}/{np.percentile(speed,95):.3f}"
                  f"/{speed.max():.3f} m/s (应 O(0.1-3), 否则不是 raw U)")

        af, Cf = alpha.ravel(), Cc.ravel()
        for b, (_, lo, hi) in enumerate(ALPHA_BANDS):
            cb = Cf[(af >= lo) & (af < hi)]
            if cb.size == 0:
                continue
            n_tot[b] += cb.size
            hist[b] += np.histogram(cb, bins=BINS)[0]
            for thr in C_THRESHOLDS:
                n_lt[thr][b] += int(np.count_nonzero(cb < thr))
        del d, alpha, speed, Cc, af, Cf
        print(f"  [{name}] done")

    # ---- 报告 (fractions 精确; median/p95 由分箱 CDF 近似) ----
    centers = np.sqrt(BINS[:-1] * BINS[1:])
    def approx_pct(h, q):
        c = np.cumsum(h); 
        return centers[np.searchsorted(c, q * c[-1])] if c[-1] > 0 else float("nan")
    hdr = f"{'band':<22}{'cells':>12}{'~medC':>9}{'~p95':>9}"
    for thr in C_THRESHOLDS: hdr += f"{'%<'+str(thr):>9}"
    print("\n" + hdr); print("-" * len(hdr))
    for b, (label, _, _) in enumerate(ALPHA_BANDS):
        if n_tot[b] == 0:
            print(f"{label:<22}{0:>12}"); continue
        row = (f"{label:<22}{n_tot[b]:>12}"
               f"{approx_pct(hist[b],0.5):>9.3f}{approx_pct(hist[b],0.95):>9.3f}")
        for thr in C_THRESHOLDS:
            row += f"{100*n_lt[thr][b]/n_tot[b]:>8.1f}%"
        print(row)

    # ---- 图 ----
    plt.figure(figsize=(9, 5.5))
    for b, (label, _, _) in enumerate(ALPHA_BANDS):
        if hist[b].sum() == 0: continue
        y = hist[b] / hist[b].sum()
        plt.plot(centers, y, drawstyle="steps-mid", lw=1.8, label=label)
    for thr, ls in zip(C_THRESHOLDS, ["--", "-"]):
        plt.axvline(thr, color="k", ls=ls, lw=1, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("per-cell Courant  C = |U|·dt / V^(1/3)")
    plt.ylabel("normalized count"); plt.legend(fontsize=9)
    plt.title(f"Courant by alpha-band (cropped_0.05, dt={DT_ML}s)")
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=140)
    print(f"\n[saved] {os.path.abspath(OUT_PNG)}")

# 读法: 只看 pure_air / near_air 两行的 '%<0.5'。
#   高 -> Cat B 散度残差 scoped 到 (near-)air 可行, gate 用 C<0.5 mask。
#   低 -> Cat B 也救不了 Problem 1, 回纯 Cat A。
#   interface 行 C 大是正常的 (Problem 2/capacity 地盘, 不靠守恒)。
if __name__ == "__main__":
    main()