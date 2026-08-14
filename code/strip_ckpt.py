#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strip_ckpt.py — 剥掉旧 ckpt 里的 LBO basis 副本 (one-off 维护工具)。

parent-era 的 checkpoint 把 spectral_basis / spectral_pos_emb 当 persistent
buffer 存了进去 —— 每层一份 (8, N, freq_num), N=574163。结果单个 ckpt 6.6~13.2
GiB, 其中 99.9% 是这份副本, 真正的权重 + 优化器状态不到 20 MB。

剥掉是**零信息损失**的:
  1. 两个 buffer 现在都是 persistent=False (hpm_model.py 的 SpectralBasis 与
     HPM.__init__), 由 data/<...>/lbo/lbo_eigenvectors.npy 确定性派生
     (F.normalize / 取前 spectral_pos_dim 列), 没有任何学习成分;
  2. train.py 的 resume 与 vis.py 的 load 都是
     load_state_dict(strip_legacy_basis(ck['model']), strict=True) ——
     现有代码每次加载**已经**把这些键丢掉、从 npy 重建。文件里那份从来没被读过。
  3. 本脚本直接 import hpm_model.strip_legacy_basis, 与加载路径同一份判据,
     不会漂。

用法
----
  python strip_ckpt.py                      # 干跑: 只列表, 不动任何文件
  python strip_ckpt.py --max-gb 10          # 干跑, 只看 <10 GB 的
  python strip_ckpt.py --apply --max-gb 10  # 真剥离 (登录节点 ulimit -v 15 GB,
                                            #   只能处理小于 ~10 GB 的)
  sbatch strip_ckpt.sh                      # 全部, 含 13.2 GiB 那几个

干跑不加载张量数据 (只解 zip 里的 data.pkl), 所以再大的文件也扫得动。

安全性: 先写同目录下的 .tmp, 校验通过 (键集合/epoch/best_val/权重求和一致)
才 os.replace 原子替换; 任何一步失败都删 tmp、保留原文件。原文件 mtime 保留,
免得 run 的时间线被改乱。
"""

import argparse
import io
import os
import pickle
import zipfile
from pathlib import Path

# 纯搬运活, 一次矩阵乘都不做; 登录节点 RLIMIT_NPROC 只有 400, OpenBLAS 默认开
# 64 线程会直接 pthread_create 失败并把 numpy 带崩。必须在 import torch 之前设。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch                                                # noqa: E402

GiB = 2 ** 30
_ESZ = {"Float": 4, "Double": 8, "Half": 2, "BFloat16": 2, "Long": 8, "Int": 4,
        "Short": 2, "Char": 1, "Byte": 1, "Bool": 1,
        "ComplexFloat": 8, "ComplexDouble": 16}


# ============================================================
# 干跑: 只解 data.pkl, 一个张量字节都不读
# ============================================================

class _Stor:
    def __init__(self, esz): self.esz = esz


class _Stub:
    def __init__(self, *a, **k): pass


def _rebuild(storage, offset, size, stride, *a):
    n = 1
    for s in size:
        n *= s
    return {"__t__": True, "shape": tuple(size), "bytes": n * getattr(storage, "esz", 4)}


class _Peek(pickle.Unpickler):
    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild
        if name == "_rebuild_parameter":
            return lambda data, rg, hooks: data
        if mod.startswith("torch"):
            return _Stub
        try:
            return super().find_class(mod, name)
        except Exception:
            return _Stub

    def persistent_load(self, pid):
        st = pid[1]
        nm = getattr(st, "__name__", "") or st.__class__.__name__
        for k, v in _ESZ.items():
            if nm.startswith(k):
                return _Stor(v)
        return _Stor(4)


def _walk(o, pre=""):
    if isinstance(o, dict) and o.get("__t__"):
        yield pre, o["bytes"]
        return
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _walk(v, f"{pre}[{i}]")


def peek(path):
    """返回 (总张量字节, legacy 字节) —— 不加载数据。"""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        ck = _Peek(io.BytesIO(z.read(name))).load()
    tot = leg = 0
    for name, nbytes in _walk(ck):
        tot += nbytes
        if name.endswith("spectral_basis") or name.endswith("spectral_pos_emb"):
            leg += nbytes
    return tot, leg


# ============================================================
# 真剥离
# ============================================================

def strip_legacy_basis(state_dict, verbose=True):
    """与 hpm_model.strip_legacy_basis 逐字一致 —— 改一处要同步另一处。

    这里不 import hpm_model: 那条链会拉进 timm/einops, 而本脚本连模型都不建,
    没必要为了两个键名扛整个依赖 (登录节点上 timm 的 import 还时好时坏)。
    """
    drop = [k for k in state_dict
            if k.endswith('spectral_basis') or k.endswith('spectral_pos_emb')]
    if verbose and drop:
        freed = sum(state_dict[k].numel() * state_dict[k].element_size() for k in drop)
        print(f"[strip_legacy_basis] dropped {len(drop)} keys ({freed/1e9:.2f} GB)")
    return {k: v for k, v in state_dict.items() if k not in drop}


def _clone(o):
    """把 mmap 上的视图拷成独立张量 —— 剥完只剩几 MB, 拷贝成本可忽略。"""
    if torch.is_tensor(o):
        return o.clone()
    if isinstance(o, dict):
        return {k: _clone(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clone(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_clone(v) for v in o)
    return o


def _sig(sd):
    """权重指纹: (键数, 各张量元素和之和) —— 用来确认剥离没动到真权重。"""
    tot = 0.0
    for v in sd.values():
        if torch.is_tensor(v) and v.is_floating_point():
            s = v.double().sum().item()
            tot += 0.0 if s != s else s          # NaN 跳过, 不然签名不可比
    return len(sd), round(tot, 6)


def strip_one(path, verbose=True):
    """剥离单个 ckpt。返回 (省下的字节, 说明)。原文件出问题一律不替换。"""
    before = os.path.getsize(path)
    st = os.stat(path)

    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(ck, dict) or "model" not in ck:
        return 0, "跳过: 不是 {'model': ...} 结构"

    kept = strip_legacy_basis(ck["model"], verbose=verbose)
    if len(kept) == len(ck["model"]):
        return 0, "跳过: 已经是干净的"

    ref_keys, ref_sig = set(kept), _sig(kept)
    ref_epoch, ref_best = ck.get("epoch"), ck.get("best_val")

    out = {k: (_clone(kept) if k == "model" else _clone(v)) for k, v in ck.items()}
    del ck                                        # 释放 mmap, 免得 replace 撞上

    tmp = Path(str(path) + ".stripped.tmp")
    try:
        torch.save(out, tmp)
        chk = torch.load(tmp, map_location="cpu", weights_only=False)
        assert set(chk["model"]) == ref_keys, "键集合对不上"
        assert _sig(chk["model"]) == ref_sig, "权重指纹对不上"
        assert chk.get("epoch") == ref_epoch and chk.get("best_val") == ref_best, \
            "epoch / best_val 对不上"
        for k in ("optimizer", "scheduler"):
            assert (k in chk) == (k in out), f"{k} 丢了"
        del chk
        os.replace(tmp, path)
        os.utime(path, (st.st_atime, st.st_mtime))          # 保留原时间线
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return 0, f"失败, 原文件未动: {type(e).__name__}: {e}"

    after = os.path.getsize(path)
    return before - after, f"{before/GiB:.2f} GiB -> {after/2**20:.1f} MiB"


# ============================================================

def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="剥掉 ckpt 里的 legacy LBO basis 副本")
    ap.add_argument("roots", nargs="*", default=[str(here.parent / "hpm" / "outputs")],
                    help="要扫的目录或文件 (默认 ../hpm/outputs)")
    ap.add_argument("--apply", action="store_true",
                    help="真改文件; 不给就只是干跑列表")
    ap.add_argument("--max-gb", type=float, default=float("inf"),
                    help="跳过大于此值的文件 (登录节点 ulimit -v 15 GB, 用 10)")
    a = ap.parse_args()

    files = []
    for r in a.roots:
        p = Path(r)
        files += [p] if p.is_file() else sorted(p.rglob("*.pt"))
    if not files:
        print("没找到 .pt")
        return

    print(f"{'磁盘':>9} {'legacy':>9} {'剥离后':>9}  文件")
    todo, tot_disk, tot_leg, skipped = [], 0, 0, 0
    for f in files:
        sz = f.stat().st_size
        try:
            _, leg = peek(f)
        except Exception as e:
            print(f"{sz/GiB:8.2f}G {'?':>9} {'?':>9}  {f}  [解析失败 {e}]")
            continue
        tot_disk += sz
        tot_leg += leg
        big = sz / GiB > a.max_gb
        if leg > 0 and not big:
            todo.append(f)
        elif big:
            skipped += 1
        print(f"{sz/GiB:8.2f}G {leg/GiB:8.2f}G {(sz-leg)/2**20:8.1f}M  {f}"
              f"{'   [超 --max-gb, 跳过]' if big else '' if leg else '   [已干净]'}")

    print(f"\n合计 {len(files)} 个 | 磁盘 {tot_disk/GiB:.1f} GB | "
          f"legacy {tot_leg/GiB:.1f} GB | 剥离后约 {(tot_disk-tot_leg)/GiB:.2f} GB")
    if skipped:
        print(f"其中 {skipped} 个超过 --max-gb={a.max_gb}, 本次跳过 (用 sbatch strip_ckpt.sh 跑)")

    if not a.apply:
        print(f"\n干跑, 没动任何文件。要真剥离: python strip_ckpt.py --apply"
              f"{'' if a.max_gb == float('inf') else f' --max-gb {a.max_gb}'}")
        return

    print(f"\n开始剥离 {len(todo)} 个 ...")
    freed = 0
    for i, f in enumerate(todo, 1):
        saved, msg = strip_one(f)
        freed += saved
        print(f"[{i}/{len(todo)}] {msg}\n          {f}", flush=True)
    print(f"\n完成, 释放 {freed/GiB:.1f} GB")


if __name__ == "__main__":
    main()
