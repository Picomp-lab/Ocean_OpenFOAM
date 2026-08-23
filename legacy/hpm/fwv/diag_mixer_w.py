"""
diag_mixer_w.py — mlp_trans_weights 的结构诊断 (CPU, 秒级, 不需要 GPU)。

问题: mixer 谱域里唯一的通道混合是
    spectral = einsum("bhgi,io->bhgo", spectral, mlp_trans_weights)
一个 (dim_head, dim_head) 的自由矩阵。Clifford 化就是把它换成受几何积
约束的映射 —— 参数更少、结构更强。

这个诊断回答: 训出来的 W 到底用满了多少自由度?
    有效秩 << dim_head  -> 自由矩阵没用满, 加结构约束有空间, Clifford 值得试
    接近满秩            -> 网络需要这些自由度, 几何积约束是净损失

用法:
    python fwv/diag_mixer_w.py outputs/hpm_fw_ss_R4/*/checkpoints/best.pt
    python fwv/diag_mixer_w.py <ckpt> --init      # 附带随机初始化对照
"""

import argparse
import glob
import math

import torch


def stable_rank(s):
    """‖W‖_F² / ‖W‖_2² — 连续版的秩, 对阈值不敏感。"""
    return float((s ** 2).sum() / (s[0] ** 2))


def participation(s):
    """奇异值能量的参与比: 1 = 全在一个方向, dim = 完全均匀。"""
    p = (s ** 2) / (s ** 2).sum()
    return float(1.0 / (p ** 2).sum())


def describe(W, tag):
    d = W.shape[0]
    s = torch.linalg.svdvals(W.float())
    fro = W.norm().item()

    # 对称 / 反对称分解: W = S + A
    S = 0.5 * (W + W.T)
    A = 0.5 * (W - W.T)
    sym_frac = float((S.norm() / W.norm()) ** 2)      # 能量占比, 两者相加 = 1

    # 距离最近的正交矩阵 (旋转样): 若 W = U diag(s) V^T, 最近正交阵是 U V^T
    # 相对残差小 -> W 本身接近一个缩放的旋转
    ortho_res = float(((s - s.mean()) ** 2).sum().sqrt() / s.norm())

    ranks = {t: int((s > t * s[0]).sum()) for t in (0.1, 0.05, 0.01)}

    print(f"{tag:>10s}  d={d:3d}  ‖W‖_F={fro:7.3f}   "
          f"rank@0.1={ranks[0.1]:3d} @0.05={ranks[0.05]:3d} @0.01={ranks[0.01]:3d}"
          f"   stable={stable_rank(s):5.2f}  particip={participation(s):5.2f}"
          f"   sym={sym_frac:.2f} anti={1-sym_frac:.2f}"
          f"   iso_res={ortho_res:.3f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--init", action="store_true",
                    help="附带同尺寸 kaiming_uniform 随机阵作对照")
    ap.add_argument("--spectrum", action="store_true",
                    help="打印每个 block 的完整奇异值谱")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.ckpt))
    assert paths, f"没有匹配的 checkpoint: {args.ckpt}"
    path = paths[-1]
    print(f"checkpoint: {path}\n")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))

    keys = sorted(k for k in sd if k.endswith("mlp_trans_weights"))
    assert keys, ("没找到 mlp_trans_weights —— key 名对不上?\n"
                  f"可用 key 示例: {list(sd)[:5]}")

    print("rank@t  = 奇异值 > t·s_max 的个数 (阈值版有效秩)")
    print("stable  = ‖W‖_F²/‖W‖_2², 连续版秩, 不依赖阈值")
    print("sym/anti= 对称/反对称分量的能量占比")
    print("iso_res = 奇异值离均匀的相对偏差; ->0 表示 W 接近缩放正交阵(旋转样)")
    print("-" * 104)

    all_s = []
    for k in keys:
        s = describe(sd[k], k.split(".")[1] if "." in k else k)
        all_s.append(s)

    if args.init:
        d = sd[keys[0]].shape[0]
        print("-" * 104)
        for i in range(3):
            W0 = torch.empty(d, d)
            torch.nn.init.kaiming_uniform_(W0, a=math.sqrt(5))
            describe(W0, f"init{i}")

    if args.spectrum:
        print("-" * 104)
        for k, s in zip(keys, all_s):
            norm = (s / s[0]).tolist()
            print(k)
            print("   " + " ".join(f"{v:.3f}" for v in norm))

    d = sd[keys[0]].shape[0]
    print("-" * 104)
    print(f"判读: dim_head={d}。若 rank@0.01 明显小于 {d} 且 stable rank 远低于 {d},")
    print(f"      说明自由矩阵没用满 -> 结构约束(几何积)有空间。")
    print(f"      若接近 {d} 且与 init 对照无差异, 说明网络需要全部自由度。")
    print(f"      注意 d={d} 本身很小, 阈值版秩的分辨率有限, 以 stable rank 为准。")


if __name__ == "__main__":
    main()
