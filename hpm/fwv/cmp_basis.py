"""
cmp_basis.py — 跨 checkpoint 比对 spectral_basis 指纹。

对 outputs/ 下所有 best.pt / latest.pt:
  - 取每个 blocks.*.mixer.spectral_basis, 各算一个 xxh64
  - 6 个拼成组合指纹 (ckpt 身份); 按组合指纹分配编号 A/B/C...
  - 6 个全同 -> 指纹列显 1 个组合指纹, 占 1 行
    6 个不全同 -> 展开 6 行, 指纹列显各 block 自己的 hash (看出哪个不一样)
  - 缺 basis 的 ckpt -> 指纹 <no-basis>, 也给编号

输出表格: 编号 | 指纹 | 相对路径 (从 outputs/ 后截断), 按编号分组。

xxhash 硬依赖 (没有直接退出)。mmap 读 + fallback。跑在 CPU 节点即可:
    srun --partition=eecs --mem=16G --time=00:30:00 --pty bash
    python fwv/cmp_basis.py
"""

import gc
import sys
from pathlib import Path

import torch

try:
    import xxhash
except ImportError:
    sys.exit("需要 xxhash: pip install xxhash --break-system-packages")

OUTPUTS = Path("outputs")
BASIS_SUFFIX = "mixer.spectral_basis"


def load_sd(path):
    """mmap 读 (省内存, 不全载 7GB); 失败退普通读。"""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def basis_hashes(sd):
    """返回 [(block_key, xxh64_hex), ...], 按 block 序号排序。缺则空 list。"""
    model = sd.get("model", sd)          # ckpt 可能是 {"model":...} 或裸 state_dict
    keys = sorted(k for k in model.keys() if k.endswith(BASIS_SUFFIX))
    out = []
    for k in keys:
        t = model[k].contiguous()
        h = xxhash.xxh64(t.numpy().tobytes()).hexdigest()
        out.append((k, h))
        del t
    return out


def combined(hashes):
    """组合指纹 = 各 block hash 顺序拼接再 hash。空 -> <no-basis>。"""
    if not hashes:
        return "<no-basis>"
    joined = "|".join(h for _, h in hashes)
    return xxhash.xxh64(joined.encode()).hexdigest()


def block_label(key):
    """blocks.3.mixer.spectral_basis -> 'blocks.3'"""
    parts = key.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else key


def main():
    files = sorted(OUTPUTS.rglob("best.pt")) + sorted(OUTPUTS.rglob("latest.pt"))
    if not files:
        sys.exit(f"{OUTPUTS}/ 下没找到 best.pt / latest.pt")

    print(f"扫描 {len(files)} 个 checkpoint...\n")

    # 收集: 每个文件 -> (组合指纹, block_hashes)
    records = []          # (rel_path, combined_fp, block_hashes)
    for f in files:
        rel = str(f.relative_to(OUTPUTS))
        try:
            sd = load_sd(f)
            bh = basis_hashes(sd)
            fp = combined(bh)
            records.append((rel, fp, bh))
            print(f"  ✓ {rel}  ({len(bh)} basis)")
        except Exception as e:
            print(f"  ✗ {rel}  读取失败: {e}")
            records.append((rel, "<load-error>", []))
        finally:
            del sd
            gc.collect()

    # 组合指纹 -> 编号 (首次出现顺序 A,B,C,...)
    label_of, order = {}, []
    for _, fp, _ in records:
        if fp not in label_of:
            label_of[fp] = chr(ord("A") + len(label_of)) if len(label_of) < 26 \
                else f"A{len(label_of)}"
            order.append(fp)

    # 按编号分组输出
    print("\n" + "=" * 72)
    print(f"{'编号':<6}{'指纹':<20}相对路径")
    print("-" * 72)
    for fp in order:
        lab = label_of[fp]
        group = [(rel, bh) for rel, f2, bh in records if f2 == fp]
        for rel, bh in group:
            uniform = len({h for _, h in bh}) <= 1
            if fp == "<no-basis>" or fp == "<load-error>" or uniform:
                # 1 行: 组合指纹 (或 6 个全同时显示那一个值)
                shown = fp if fp.startswith("<") else bh[0][1]
                print(f"{lab:<6}{shown:<20}{rel}")
            else:
                # 展开: 6 个 block 各显各自 hash
                print(f"{lab:<6}{'(6 blocks 不一致):':<20}{rel}")
                for k, h in bh:
                    print(f"{'':<6}{h:<20}    {block_label(k)}")
    print("=" * 72)

    # 摘要
    print(f"\n共 {len(order)} 种 (编号 A..{label_of[order[-1]]}), "
          f"{len(files)} 个文件。")
    nonuniform = [rel for rel, fp, bh in records
                  if bh and len({h for _, h in bh}) > 1]
    if nonuniform:
        print(f"⚠️  {len(nonuniform)} 个 ckpt 内部 6 blocks 不一致 (见展开行):")
        for rel in nonuniform:
            print(f"     {rel}")
    else:
        print("所有 ckpt 内部 6 blocks 均一致 (删除后统一重建安全)。")


if __name__ == "__main__":
    main()
