"""
只读侦查 (read-only recon) —— 不改任何文件。
目的: 搞清楚 (a) chunk_data 的布局与通道, (b) full->sub 索引存没存,
      (c) 从哪条路能拿到与 chunk_data 同序的 cell 体积 V。
把全部输出贴回来即可。
"""
import os
# 登录节点 RLIMIT_NPROC 很小, OpenBLAS 默认开满线程会 pthread_create failed。
# 必须在 import numpy 之前把线程摁到 1 (纯 I/O 脚本单线程足够)。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import numpy as np

# OpenFOAM 算例目录在仓库外（23 G 的 CFD 数据，没进版本库）。
OF_CASE = os.environ.get("OCEAN_CASE",
                         os.path.expanduser("~/hpc-share/ocean_project/case"))

DATA_DIR = "../data/3d/cropped_0.05"          # 需要时改成你的实际路径
CASE     = OF_CASE

def sec(t): print("\n" + "="*8 + " " + t + " " + "="*8)

# --- 1. chunk_data 布局 + 通道 range ---
sec("1. chunk_data 布局")
fs = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*_data.npy")))
print("chunks found:", len(fs))
if fs:
    d = np.load(fs[1] if len(fs) > 1 else fs[0])   # 避开 chunk_000 (still water)
    print("single-chunk shape:", d.shape, d.dtype)
    ax = int(np.argmax(d.shape))                    # 最大轴多半是 N=574163
    print("guessed cell-axis (largest dim):", ax, "-> N?=", d.shape[ax])
    C = d.shape[-1] if d.shape[-1] <= 16 else d.shape[1]
    print("guessed channel count:", C)
    # 逐通道 range (按最后一轴是通道假设; 若不是, 下面数字会很怪)
    flat = d.reshape(-1, d.shape[-1]) if d.shape[-1] <= 16 else None
    if flat is not None:
        for c in range(d.shape[-1]):
            col = flat[:, c]
            print(f"  ch{c}: min={col.min():.4g}  max={col.max():.4g}  mean={col.mean():.4g}")
    print("^ 找 range~[0,1] 的通道 = alpha; 找 ~O(0.1-3) 对称的 = raw U(m/s);"
          " 若全是 ~[-3,3] 标准化值 = 不是 raw, 要另找源")

# --- 2. times: 确认 dt=0.05 ---
sec("2. times / dt")
ts = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*_times.npy")))
if ts:
    t = np.load(ts[1] if len(ts) > 1 else ts[0]).ravel()
    print("times[:5]:", t[:5])
    if t.size > 1: print("diff[:5]:", np.diff(t)[:5])

# --- 3. lbo 参照 (已知与 chunk_data 同序) ---
sec("3. lbo eigenvectors (对齐参照)")
ev = os.path.join(DATA_DIR, "lbo", "lbo_eigenvectors.npy")
if os.path.exists(ev):
    e = np.load(ev, mmap_mode="r")
    print("eigenvectors shape:", e.shape, "-> N =", e.shape[0])

# --- 4. 有没有存过 full->sub 索引 ? 扫一切可疑 .npy ---
sec("4. 搜 full->sub 索引 / cell id 文件")
cands = []
for root, _, files in os.walk(DATA_DIR):
    for fn in files:
        if fn.endswith(".npy") and any(k in fn.lower()
               for k in ["idx","index","id","cell","map","sub","crop","mask"]):
            p = os.path.join(root, fn)
            try:
                a = np.load(p, mmap_mode="r")
                cands.append((p, a.shape, str(a.dtype)))
            except Exception as ex:
                cands.append((p, "?", str(ex)))
if cands:
    for p, s, dt in cands: print(f"  {p}  shape={s} dtype={dt}")
    print("^ 找 shape=(574163,) 且 dtype 是整数的 = 很可能就是 full->sub 索引")
else:
    print("  没找到候选 —— 索引大概没单独存, 得从 cellSet 重建 (需保证同序)")

# --- 5. OpenFOAM case 可达性 + 是否已有 V 场 ---
sec("5. OpenFOAM case")
print("case exists:", os.path.isdir(CASE))
if os.path.isdir(CASE):
    cset = os.path.join(CASE, "constant/polyMesh/sets/subdomainCells")
    print("cellSet subdomainCells exists:", os.path.exists(cset))
    # 扫时间目录里有没有现成的 V 场
    vfound = glob.glob(os.path.join(CASE, "*/V"))
    print("existing 'V' fields:", vfound[:3], "..." if len(vfound) > 3 else "")