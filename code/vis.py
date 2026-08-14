#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis.py — 唯一可视化入口 (single visualization entry point)。

合并了原先的 vis_gt / vis_align / vis (纯 HPM) / vis_u / vis_prior / vis_fw /
vis_fw_lt 七个脚本。它们的渲染层几乎逐字相同 (slice 缓存、OpacityReds 配色、
draw、ffmpeg 导出), 拆开写就是七份拷贝, 改一处要同步七遍。

五个子命令 —— 按流水线位置排列
------------------------------
    gt      纯数据探查。1 行 GT alpha, 可一次跑多个 chunk。不加载模型。
    align   配准检查。2 行: prior | GT。**生成 prior 之前**跑, 不要 checkpoint。
            prior 由 FUNWAVE 现算, 只算 y=0.30 切片上那几万个 cell (非全域
            574163), 比 gen_prior 便宜两个量级。用来目视确认 t-offset k。
    pred    主力对比。2 行: GT | pred。**两条线通用**。
            速度看分量也走这里: --field Ux / Uz (或 Umag 看模长), 一次一个分量。
    nofb    无反馈臂专用。3 行: prior | pred | GT, 逐帧独立无 rollout。
    lt      长期 rollout, 无 GT。1 行, 边推边写视频帧 (防 OOM)。

两条模型线由 checkpoint config 的 data.window 自动区分 —— 与 train.py 同一判据:
    window >= 3   纯 HPM        rollout 走 schema.advance_window (滑窗自携带状态)
    window == 0   HPM+FUNWAVE   base=prior, 输入 [prior | x_f*m]

流水线位置
    scan_toffset -> vis.py align -> gen_prior -> train -> vis.py {pred,nofb,lt}
        测出 k      目视确认 k     按 k 生成    训练       看模型效果

装配一致性: assemble / reconstruct / advance_window 一律 import dataset 与
schema —— 与训练逐字节一致, 不重写, 不漂。全程 normalized 空间递归, 只在算
指标/出图时 denorm。

用法
----
  python vis.py gt    --data_dir <d> --chunks 0-10
  python vis.py align --fw-dir <fw>/output --chunk 9
  python vis.py pred  --config_path <run>/.hydra/config.yaml \\
                      --checkpoint <run>/checkpoints/best.pt \\
                      --data_dir <d> --chunk_id 8 --field alpha
  python vis.py lt    --config_path ... --checkpoint ... --data_dir ... \\
                      --prior_dir <p> --chunk_id 10
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf

MID_Y = 0.30
_HERE = os.path.dirname(os.path.abspath(__file__))
# repo 根 = code/ 的上一级 (vis.py 在 code/ 下)。锚 __file__, 与 train.py 的 ${repo:}
# 同源 —— 路径随代码位置走, 不随 cwd 变、不写死绝对路径 (前提: code/ data/ results/ 同级)。
_REPO = os.path.dirname(_HERE)
_VIS_OUT = os.path.join(_REPO, "results", "vis")           # vis 输出 (子: gt_alpha/align/pred/nofb/lt)
_DATA = os.path.join(_REPO, "data", "3d", "cropped_0.05")  # --data_dir 默认 (CFD 数据目录)

# vis 会加载 checkpoint 的存档 config, 其中 data.prior_dir 用了 ${repo:} 插值。
# vis 不走 Hydra, 故在此注册同一个 repo resolver, 否则解析 ${repo:} 会报错。
# 解析结果 = 当前 vis.py 所在 repo, 所以存档 config 搬到哪都指向当前 repo (可移植)。
if not OmegaConf.has_resolver("repo"):
    OmegaConf.register_new_resolver("repo", lambda: _REPO)


# ============================================================
# 共用: 切片几何
# ============================================================

def load_slice(data_dir):
    """y=MID_Y 上的真实拓扑切片 (OpenFOAM 网格实际的 cell, 非均匀)。

    三个 .npy 是 PyVista 预计算的**可再生产物**, 不入库 —— 丢了重算即可。
    Returns: dict(cell_map, x, z, tri, triang, xlim, zlim)
    """
    sdir = Path(data_dir) / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri = np.load(sdir / "slice_tri.npy")
    x, z = xz[:, 0].astype(np.float64), xz[:, 1].astype(np.float64)
    print(f"[slice] {len(cell_map)} cells @ y={MID_Y}   "
          f"x {x.min():.3f}..{x.max():.3f}  z {z.min():.3f}..{z.max():.3f}")
    return dict(cell_map=cell_map, x=x, z=z, tri=tri,
                triang=mtri.Triangulation(x, z, triangles=tri),
                xlim=(x.min(), x.max()), zlim=(z.min(), z.max()))


# ============================================================
# 共用: 标量画笔 (色标 + tricontourf/scatter)
# ============================================================

def make_cmap(field, arrays, pct=99):
    """配色与 clim。同一次渲染的所有行**必须共享 clim**, 否则逐像素不可比。

    约定 (项目统一):
      alpha  -> 白->红 顺序色标, 固定 [0,1]      (无符号, 物理上有界)
      Umag   -> magma 顺序色标, [0, p_pct]       (无符号)
      其余   -> coolwarm 发散色标, **对称** [-m, +m]  (有符号: Ux/Uz/p_rgh)

    对称是硬要求: 发散色标的中点必须是 0, 否则零值不落在白色上, 正负不可比。
    (合并前的纯 HPM vis.py 用 [min, max] 非对称范围, 那是错的, 不恢复。)

    pct<100 (默认 99) -> m = |v| 的该分位数。超出部分被 extend='both' 压成满色。
    pct>=100          -> m = max|v|, 不裁剪任何数据 —— fwv/vis_fw.py 的老行为。

    默认为什么是 p99 而不是 max: 色标同时吃 GT 与 pred, 所以**预测的离群 cell
    也会撑开色标**。撑开一次全图就整幅泛白, 结构看不见了。实测 chunk 9 GT 的
    max/p99 就已经是 3.8x (αUx) / 2.6x (αUz), 换成 max 后波峰与回流几乎不可辨;
    被 p99 压成饱和色的只有 1% 的 cell, 这个代价小得多。
    要逐值复现 vis_fw 的老图就 --clim-pct 100。
    """
    if field == "alpha":
        cdict = {'red':   [[0., 1., 1.], [1., .6, .6]],
                 'green': [[0., 1., 1.], [1., 0., 0.]],
                 'blue':  [[0., 1., 1.], [1., 0., 0.]]}
        cmap = LinearSegmentedColormap('OpacityReds', cdict)
        vmin, vmax = 0.0, 1.0
    elif field == "Umag":
        cmap = 'magma'
        av = np.concatenate([a.ravel() for a in arrays])
        vmin = 0.0
        vmax = float(av.max() if pct >= 100 else np.percentile(av, pct))
    else:
        cmap = 'coolwarm'
        av = np.abs(np.concatenate([a.ravel() for a in arrays]))
        m = float(av.max() if pct >= 100 else np.percentile(av, pct))
        vmin, vmax = -m, m
    note = ("  (物理上界, pct 不参与)" if field == "alpha" else
            "  (max, 不裁剪)" if pct >= 100 else
            f"  (p{pct:g}, 超出部分压成满色)")
    print(f"[clim] {field}: {vmin:.4g} .. {vmax:.4g}{note}")
    return cmap, np.linspace(vmin, vmax, 128), vmin, vmax


def scalar_painter(sl, cmap, levels, vmin, vmax, style, point_size):
    """标量场画笔。返回 painter(ax, vals, label)。"""
    def paint(ax, vals, label):
        ax.clear()
        if style == "scatter":
            ax.scatter(sl["x"], sl["z"], c=vals, s=point_size,
                       vmin=vmin, vmax=vmax, cmap=cmap, edgecolors='none')
        else:
            ax.tricontourf(sl["triang"], vals, levels=levels, cmap=cmap,
                           extend='both')
        ax.set_facecolor('white')
        ax.set_xlim(sl["xlim"]); ax.set_ylim(sl["zlim"])
        ax.set_xlabel("X (m)", fontsize=20)
        ax.set_ylabel("Z (m)", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.set_title(label, fontsize=24)
    return paint


# ============================================================
# 共用: 通用多行动画
# ============================================================

def render(out_path, n_frames, labels, frame_fn, title_fn, painter,
           fps=20, fig_w=38.4, row_h=10.8):
    """五个子命令共用这一个渲染器。

    Args:
        labels:   每行标题, len(labels) == 行数
        frame_fn: k -> [row0, row1, ...]。**可以有副作用** (lt 在这里推一步
                  rollout), 故保证每帧只调一次
        title_fn: k -> suptitle
        painter:  scalar_painter 的返回值
    """
    rows = len(labels)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vid ] {rows} 行 x {n_frames} 帧 -> {out_path}")

    fig, axes = plt.subplots(rows, 1, figsize=(fig_w, row_h * rows), dpi=100)
    if rows == 1:
        axes = [axes]
    # 顶部按绝对英寸 (absolute-inch) 预留标题带, 不用图坐标分数 (figure-fraction)。
    # 根因: top/y 是分数, 文字高度是点 (points); 单行图 (lt/gt, rows==1) 太矮,
    # 同一条分数带换算成绝对高度塞不下 suptitle + 行标题 -> 重合。这里固定留
    # ~1.2 in 给两段标题, 与行数无关, 单行多行同治。多行几乎不动 (top 0.93->~0.94)。
    fig_h = row_h * rows
    fig.subplots_adjust(top=1 - 1.2 / fig_h, bottom=0.06,
                        left=0.05, right=0.96, hspace=0.15)
    sup = fig.suptitle("", fontsize=26, color='black', y=1 - 0.45 / fig_h)

    def upd(k):
        for ax, v, lab in zip(axes, frame_fn(k), labels):
            painter(ax, v, lab)
        sup.set_text(title_fn(k))

    ani = animation.FuncAnimation(fig, upd, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    ani.save(str(out_path), writer="ffmpeg", fps=fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close(fig)
    return out_path


# ============================================================
# 共用: 模型加载 / 两条线的 rollout
# ============================================================

def load_model(config_path, checkpoint, data_dir, device, expect_feedback=None):
    """载入 config + schema + 模型。torch 相关 import 延后到这里 —— gt / align
    两个子命令不需要模型栈。

    strip_legacy_basis 无条件调用: 新 ckpt 没有那些键 (no-op), 旧 ckpt 有
    persistent 的 spectral_basis (会让 strict=True 报缺键)。一处处理, 新旧通吃。
    (合并前 vis.py / vis_u.py 缺这一步, 换到共享 basis 的模型后会直接报缺键。)

    Returns: (cfg, schema, model, line)   line ∈ {'pure', 'fwv'}
    """
    import torch
    from hpm_model import HPM, strip_legacy_basis
    from schema import ChannelSchema
    from dataset import assert_prior_compatible, input_dim

    cfg = OmegaConf.load(config_path)
    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())

    W = int(cfg.data.window)
    line = "pure" if W > 0 else "fwv"
    if line == "fwv":
        assert_prior_compatible(schema)
        fb = str(cfg.get('rollout', {}).get('feedback', 'self'))
        if expect_feedback is not None:
            assert fb == expect_feedback, (
                f"本子命令要求 feedback={expect_feedback}, 但 config 是 {fb}。"
                f"  none -> `vis.py nofb`,  self -> `vis.py pred` / `lt`")
        in_dim = schema.field_dim if fb == "none" else input_dim(schema.field_dim)
        print(f"[line] HPM+FUNWAVE  feedback={fb}  输入宽度={in_dim}")
    else:
        assert expect_feedback in (None, "none"), \
            "纯 HPM 线没有自反馈槽 (窗口本身即状态), 本子命令仅适用于 fwv 线"
        in_dim = schema.field_dim
        print(f"[line] 纯 HPM  window={W}  输入宽度={in_dim}")

    spectral_embedding = np.load(Path(data_dir) / "lbo" / "lbo_eigenvectors.npy")
    model = HPM(
        space_dim=3, field_dim=in_dim, out_dim=schema.out_dim, window=W,
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=0.0, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.get('spectral_pos_dim', 0),
        spectral_embedding=spectral_embedding, use_ckpt=False,
        max_grad_norm=cfg.train.get('max_grad_norm', 0.0),
    ).to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(strip_legacy_basis(ck["model"]), strict=True)
    model.eval()
    print(f"[ckpt] epoch {ck.get('epoch', '?')}, "
          f"best_val {ck.get('best_val', float('nan')):.6f}")
    return cfg, schema, model, line


def rollout_pure(model, coords, data_w, stats, window, start, n_steps,
                 device, schema):
    """纯 HPM: 滑窗自回归。窗口移位 / frozen 通道由共享 advance_window 完成
    (与训练零漂移)。data_w 已是 schema 选列 + alpha 加权; 输出同空间, 不还原。"""
    import torch
    from schema import advance_window

    mean, std = stats[0], stats[1]
    fields_w = np.concatenate(
        [(data_w[start - window + 1 + w] - mean) / std for w in range(window)],
        axis=-1)
    fields_w = torch.from_numpy(fields_w.astype(np.float32)).unsqueeze(0).to(device)
    cb = coords.unsqueeze(0)

    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            delta = model(cb, fields_w)                     # (1,N,out_dim)
            pred_frame, fields_w = advance_window(fields_w, delta, schema)
            preds.append(pred_frame[0].cpu().numpy() * std + mean)
    return np.stack(preds)


def rollout_fwv(model, coords_b, prior_n, gt_n, delta_idx, mode, start, n_steps):
    """HPM+FUNWAVE: 同一循环体, mode 切 tf / rollout。

        tf       x_f = GT(t-1), 逐帧独立 (不递归)。teacher forcing, 自检用。
                 真 t=0 恒 m=0。跑 val chunk 时应复现 wandb val 曲线量级。
        rollout  x_f = pred(t-1), 递归。真实部署条件。
                 首步 (k=0) 恒 m=0 冷启动 —— 部署时没有历史, 故意不喂 GT。

    帧对齐: 第 k 步预测帧 t = start+k, 用 prior(t)。
    Returns: preds_n (n,N,F), deltas (n,N,out), 均 normalized。
    """
    import torch
    from dataset import assemble, reconstruct

    N, F = prior_n.shape[1], prior_n.shape[2]
    device = prior_n.device
    x_f = torch.zeros(1, N, F, device=device)      # 冷启动初值
    preds, deltas = [], []

    with torch.no_grad():
        for k in range(n_steps):
            t = start + k
            prior_t = prior_n[t:t + 1]
            if mode == "rollout":
                m_val = 0.0 if k == 0 else 1.0     # x_f 见循环末尾递归
            elif mode == "tf":
                if t == 0:
                    m_val, x_f = 0.0, torch.zeros(1, N, F, device=device)
                else:
                    m_val, x_f = 1.0, gt_n[t - 1:t]     # GT(t-1), 不递归
            else:
                raise ValueError(f"unknown mode {mode}")
            m_t = torch.full((1, 1, 1), m_val, device=device)
            delta = model(coords_b, assemble(prior_t, x_f, m_t))
            pred_t = reconstruct(prior_t, delta, delta_idx)
            preds.append(pred_t)
            deltas.append(delta)
            if mode == "rollout":
                x_f = pred_t                        # 递归: 喂自己上一步
    return torch.cat(preds, 0), torch.cat(deltas, 0)


def field_slicer(field, schema, cell_map):
    """--field -> (取切片的函数, 显示名)。按名查找, 无魔法索引。"""
    disp = schema.display_names()
    if field == "Umag":
        comps = [c for c in ("Ux", "Uy", "Uz") if c in schema.names]
        assert comps, f"'Umag' 需要至少一个速度通道, schema={schema.names}"
        iu = [schema.names.index(c) for c in comps]
        vp = "αU" if schema.alpha_weighted[iu[0]] else "U"
        return (lambda a: np.sqrt(sum(a[:, cell_map, i] ** 2 for i in iu)),
                f"|{vp}| ({''.join(c[-1] for c in comps)})")
    assert field in schema.names, \
        f"--field '{field}' 不在 {list(schema.names)} (或用 Umag)"
    fi = schema.names.index(field)
    return (lambda a: a[:, cell_map, fi]), disp[fi]


def load_times(data_dir, chunk_id, T):
    p = Path(data_dir) / f"chunk_{chunk_id:03d}_times.npy"
    return np.load(p) if p.exists() else np.arange(T) * 0.05


def predict(a, device):
    """两条线通用的 "载数据 + 推理" 前端。pred / nofb / lt 共用。

    Returns dict:
        gts, preds     (n,N,F) raw 空间, 帧已对齐
        preds_tf       仅 fwv 线 (teacher forcing 自检); 纯 HPM 为 None
        times, schema, cfg, line, n_steps
    """
    import torch
    from dataset import (expand_range, load_chunk, load_coords, load_prior,
                         resolve_stats)

    data_dir = Path(a.data_dir)
    cfg, schema, model, line = load_model(a.config_path, a.checkpoint,
                                          data_dir, device)
    stats = resolve_stats(str(data_dir), expand_range(cfg.data.train_chunk_range),
                          schema)
    mean, std = stats[0], stats[1]
    data_w = load_chunk(str(data_dir), a.chunk_id, schema)          # (T,N,F)
    T = data_w.shape[0]
    times = load_times(data_dir, a.chunk_id, T)
    coords = load_coords(str(data_dir)).to(device)

    if line == "pure":
        W = int(cfg.data.window)
        start = max(a.start_frame, W - 1)
        n_steps = min(a.n_frames if a.n_frames > 0 else T, T - start - 1)
        assert n_steps > 0, f"start={start} 超出 chunk 长度 T={T}"
        print(f"Rolling out {n_steps} steps from frame {start}...")
        preds = rollout_pure(model, coords, data_w, stats, W, start, n_steps,
                             device, schema)
        return dict(gts=data_w[start + 1: start + 1 + n_steps], preds=preds,
                    preds_tf=None, times=times[start + 1: start + 1 + n_steps],
                    schema=schema, cfg=cfg, line=line, n_steps=n_steps)

    # ---- fwv ----
    prior_dir = a.prior_dir or cfg.data.prior_dir
    pr_raw = load_prior(str(prior_dir), a.chunk_id, schema)
    assert pr_raw.shape == data_w.shape, \
        f"GT {data_w.shape} vs prior {pr_raw.shape} 不符 (coords/裁剪不一致?)"
    to_t = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device)
    gt_n, pr_n = to_t(data_w), to_t(pr_raw)
    coords_b = coords.unsqueeze(0)
    delta_idx = torch.as_tensor(schema.delta_indices, device=device)

    start = a.start_frame
    n_steps = (T - start) if a.n_frames <= 0 else min(a.n_frames, T - start)
    assert n_steps > 0, f"start={start} 超出 chunk 长度 T={T}"
    print(f"Infer tf {n_steps} 帧 (自检)...")
    tf_n, delta_tf = rollout_fwv(model, coords_b, pr_n, gt_n, delta_idx,
                                 "tf", start, n_steps)
    print(f"Infer rollout {n_steps} 帧 (start={start}, 冷启动)...")
    ro_n, _ = rollout_fwv(model, coords_b, pr_n, gt_n, delta_idx,
                          "rollout", start, n_steps)

    # 自检: tf 的 normalized-delta nRMSE, 跑 val chunk 时应 ≈ wandb val_nrmse。
    # 对不上 = 装配有 bug, 先修这个, 别看 rollout 结论。
    gd = (gt_n - pr_n).index_select(-1, delta_idx)[start:start + n_steps]
    nrmse_tf = torch.sqrt(((delta_tf - gd) ** 2).mean(dim=(0, 1))).cpu().numpy()
    dn = [schema.names[i] for i in schema.delta_indices]
    print("  [自检] tf normalized-delta nRMSE  "
          + "  ".join(f"{n}={v:.3f}" for n, v in zip(dn, nrmse_tf))
          + "   (跑 val chunk 时应≈ wandb val_nrmse)")

    to_raw = lambda x: x.cpu().numpy() * std + mean
    return dict(gts=data_w[start:start + n_steps], preds=to_raw(ro_n),
                preds_tf=to_raw(tf_n), times=times[start:start + n_steps],
                schema=schema, cfg=cfg, line=line, n_steps=n_steps)


# ============================================================
# 子命令 1: gt —— 纯数据探查 (不加载模型)
# ============================================================

def parse_chunks(s):
    """'0-10' -> [0..10]; '6,9' -> [6,9]; '6' -> [6]."""
    out = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def cmd_gt(a):
    """GT alpha 动画, 每 chunk 一个文件。不加载模型/stats/rollout。

    只看 alpha (磁盘通道 0), 不受 alpha 加权影响 —— alpha 永远是原始 [0,1]。
    """
    data_dir = Path(a.data_dir)
    sl = load_slice(data_dir)
    cmap, levels, vmin, vmax = make_cmap("alpha", [], a.clim_pct)
    styles = ["scatter", "tri"] if a.style == "both" else [a.style]

    for cid in parse_chunks(a.chunks):
        dpath = data_dir / f"chunk_{cid:03d}_data.npy"
        if not dpath.exists():
            print(f"chunk {cid}: 缺 {dpath.name}, 跳过")
            continue
        data = np.load(dpath)                              # (T,N,6)
        T = data.shape[0]
        times = load_times(data_dir, cid, T)
        alpha_slice = data[:, sl["cell_map"], 0]           # 通道 0 = alpha
        print(f"chunk {cid}: {T} frames")
        for st in styles:
            render(Path(a.out_dir) / f"gt_alpha_chunk{cid:03d}_{st}.mp4", T,
                   [f"GT alpha | chunk {cid}"],
                   lambda f, _s=alpha_slice: [_s[f]],
                   lambda f, _t=times: f"t={_t[f]:.2f}s | frame {f}",
                   scalar_painter(sl, cmap, levels, vmin, vmax, st, a.point_size),
                   fps=a.fps, fig_w=a.fig_w, row_h=a.row_h)
    print("Done.")


# ============================================================
# 子命令 2: align —— 配准检查 (生成 prior 之前跑)
# ============================================================

def read_scan_k(scan_dir, cid):
    """toffset_scan/c{cid}.json 里标定出的 best_k。找不到返回 None。"""
    p = Path(scan_dir) / f"c{cid:03d}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("best_k")


def cmd_align(a):
    """prior 与 CFD GT 的两行对比。prior 在切片点上**现算**。

    为什么不读 gen_prior 的产物: 那是本步骤确认之后才该生成的东西。用的是同一套
    lift.horizontal_terms + nwogu_at_points, 数值路径与 gen_prior 完全一致。

    k 的来源: 默认读 scan_toffset 的 best_k; --k 显式覆盖用于目视复核 (扫 2/3/4
    各出一支, 看 RMSE 最低 / corr 最高的那个)。
    """
    import lift as LC
    from fw_io import load_static
    from gen_prior import Bilinear, build_frame

    cid = a.chunk
    data_dir = Path(a.data_dir)
    # scan-dir 默认随 data 目录走 (与 scan_toffset.py --out 同源)
    scan_dir = a.scan_dir or os.path.join(a.data_dir, "toffset_scan")
    names = list(LC.CH_NAMES)

    if a.k is not None:
        k, k_src = a.k, "--k"
    else:
        k = read_scan_k(scan_dir, cid)
        k_src = f"c{cid:03d}.json"
        if k is None:
            raise SystemExit(
                f"[err] {scan_dir}/c{cid:03d}.json 不存在或 best_k 为 null。\n"
                f"      先跑 scan_toffset.sh, 或用 --k 显式指定。")
    print(f"[k   ] k={k:+d} ({k_src})  ->  t-offset {k*a.plot_intv:+.2f}s")

    sl = load_slice(data_dir)
    cell_map, x_s, z_s = sl["cell_map"], sl["x"], sl["z"]
    Ns = len(cell_map)

    gt_all = np.load(data_dir / f"chunk_{cid:03d}_data.npy", mmap_mode="r")
    times_all = np.load(data_dir / f"chunk_{cid:03d}_times.npy").astype(np.float64)
    T_all = min(gt_all.shape[0], len(times_all))

    if 0 < a.n_frames < T_all:
        T = a.n_frames
        # 配准看的是相位, 取中间段可避开 chunk 边界的暂态
        i0 = ((T_all - T) // 2 if str(a.start).lower() == "mid" else int(a.start))
        i0 = max(0, min(i0, T_all - T))
    else:
        T, i0 = T_all, 0
    win = slice(i0, i0 + T)
    fidx = np.arange(i0, i0 + T)

    gt = np.asarray(gt_all[win][:, cell_map, :][:, :, a.gt_channels],
                    dtype=np.float32)
    times = times_all[win]
    print(f"[gt  ] {gt.shape}  frame {i0}..{i0+T-1}/{T_all}  "
          f"t {times[0]:.2f}..{times[-1]:.2f}")

    h_grid, avail, cache = load_static(a.fw_dir, a.mglob, a.nglob)
    # 切面固定 y=0.30 -> 水平去重后只剩不同的 x, 省一个量级的插值
    xy = np.round(np.stack([x_s + a.x_offset,
                            np.full(Ns, MID_Y + a.y_offset)], axis=1), 9)
    uniq, inv = np.unique(xy, axis=0, return_inverse=True)
    bil = Bilinear(uniq[:, 0], uniq[:, 1], a.mglob, a.nglob, a.dx, a.dy)
    n_out = int((~bil.inside).sum())
    print(f"[uniq] 水平唯一点 {len(uniq)} / {Ns} cells "
          f"({Ns/max(len(uniq),1):.1f} cells per column)"
          + (f"   [warn] 域外 {n_out}" if n_out else ""))

    n_fw = np.round(times / a.plot_intv).astype(np.int64) + k
    missing = [int(n) for n in n_fw if n not in avail]
    if missing:
        print(f"[warn] {len(missing)} 帧 FUNWAVE 不存在 -> 该帧留空 "
              f"(示例 {missing[:5]})")

    pr = np.zeros((T, Ns, 5), dtype=np.float32)
    n_bad = 0
    for i, n in enumerate(n_fw):
        n = int(n)
        if n not in avail:
            continue
        out = build_frame(n, cache, avail, h_grid, bil, z_s, inv,
                          a.dx, a.dy, a.plot_intv, use_pnh=not a.no_pnh)
        fin = np.isfinite(out).all(axis=1)
        out[~fin, :] = 0.0                 # 与 gen_prior 同一约定: 无定义处填 0
        n_bad += int((~fin).sum())
        pr[i] = out.astype(np.float32)
        if i % 20 == 0:
            print(f"  prior {i+1}/{T}  fw#{n}", flush=True)
    print(f"[prior] 无定义 cell·帧 {n_bad} / {T*Ns} "
          f"({n_bad/max(T*Ns,1)*100:.2f}%, 已填 0)")

    if a.field == "Umag":
        iu = [names.index(c) for c in ("Ux", "Uz")]
        s_gt = np.sqrt(sum(gt[..., i] ** 2 for i in iu))
        s_pr = np.sqrt(sum(pr[..., i] ** 2 for i in iu))
        fname = "|U| (xz)"
    else:
        if a.field not in names:
            raise SystemExit(f"[err] --field '{a.field}' 不在 {names} / Umag")
        fi = names.index(a.field)
        s_gt, s_pr, fname = gt[..., fi], pr[..., fi], a.field

    cmap, levels, vmin, vmax = make_cmap(a.field, [s_gt, s_pr], a.clim_pct)

    rmse = np.sqrt(((s_pr - s_gt) ** 2).mean(axis=1))
    corr = np.full(T, np.nan)
    for f in range(T):
        p, g = s_pr[f], s_gt[f]
        sp, sg = p.std(), g.std()
        if sp > 1e-12 and sg > 1e-12:
            corr[f] = np.mean((p - p.mean()) * (g - g.mean())) / (sp * sg)

    wtag = "" if T == T_all else f"_f{i0}-{i0+T-1}"
    out_path = Path(a.output) if a.output else \
        Path(_VIS_OUT) / "align" / f"c{cid}_{a.field}_k{k:+d}{wtag}.mp4"

    render(out_path, T,
           ["prior  (FUNWAVE lift, on CFD slice cells)", "GT  (CFD)"],
           lambda f: [s_pr[f], s_gt[f]],
           lambda f: (f"chunk {cid} | t={times[f]:.2f}s | frame {fidx[f]} | "
                      f"fw#{n_fw[f]} | {fname} | k={k:+d} | "
                      f"slice-RMSE {rmse[f]:.4f}   corr {corr[f]:.3f}"),
           scalar_painter(sl, cmap, levels, vmin, vmax, a.style, a.point_size),
           fps=a.fps, fig_w=a.fig_w, row_h=a.row_h)

    print(f"[done] {fname}  k={k:+d}   slice-RMSE 均值 {rmse.mean():.4f}   "
          f"corr 均值 {np.nanmean(corr):.3f}")
    print("       (扫不同 --k 时看这两个数: RMSE 越低 / corr 越高 越对齐)")


# ============================================================
# 子命令 3: pred —— GT | pred 两行 (两条线通用)
# ============================================================

def cmd_pred(a):
    """主力对比。fwv 线额外打印 tf vs rollout 的 slice-RMSE gap。

    那个 gap 就是 exposure bias 的量, 对任意 checkpoint 都测得到 —— 不需要
    专门训一个 tf 模型来量它。

    速度分量也走这个子命令: --field Ux / Uz, 一次一个分量。
    """
    import torch
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    d = predict(a, device)
    schema, gts, preds, times = d["schema"], d["gts"], d["preds"], d["times"]

    sl = load_slice(Path(a.data_dir))
    take, fname = field_slicer(a.field, schema, sl["cell_map"])
    gt_slice, pred_slice = take(gts), take(preds)

    out_stem = str(Path(a.output).with_suffix(""))
    # 目录必须在**第一次落盘之前**建好。--save_rmse 时 _rmse_tf.npy 早于
    # render(), 而 render() 才是原本唯一建目录的地方 —— 合并前 vis_fw.sh 在
    # shell 里 mkdir -p, 合并后没有 vis.sh 接手, 于是目录不存在就直接崩在这一步。
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    rows = [("pred", pred_slice)]
    if d["preds_tf"] is not None:
        tf_slice = take(d["preds_tf"])
        rmse_tf = np.sqrt(((gt_slice - tf_slice) ** 2).mean(axis=1))
        rmse_ro = np.sqrt(((gt_slice - pred_slice) ** 2).mean(axis=1))
        print(f"  slice-RMSE {fname}:")
        print(f"    tf       start={rmse_tf[0]:.4f} end={rmse_tf[-1]:.4f} "
              f"mean={rmse_tf.mean():.4f}")
        print(f"    rollout  start={rmse_ro[0]:.4f} end={rmse_ro[-1]:.4f} "
              f"mean={rmse_ro.mean():.4f}")
        print(f"    gap(ro-tf) end={rmse_ro[-1]-rmse_tf[-1]:+.4f}  "
              f"(>0 = exposure bias 在累积)")
        if a.also_tf_video:
            rows.append(("tf", tf_slice))
        if a.save_rmse:
            np.save(Path(out_stem + "_rmse_tf.npy"),
                    np.sqrt(((gts - d["preds_tf"]) ** 2).mean(axis=1)))

    if a.save_preds:
        pp = Path(out_stem + "_preds.npy")
        np.save(pp, preds.astype(np.float32))
        print(f"  预测已存: {pp} (float32, ~0.9 GB)")

    cmap, levels, vmin, vmax = make_cmap(a.field, [gt_slice, pred_slice], a.clim_pct)
    styles = ["scatter", "tri"] if a.style == "both" else [a.style]
    for st in styles:
        for tag, ps in rows:
            r = np.sqrt(((gt_slice - ps) ** 2).mean(axis=1))
            render(f"{out_stem}_{tag}_{st}.mp4", d["n_steps"],
                   ["Ground Truth", f"HPM {tag}"],
                   lambda f, _p=ps: [gt_slice[f], _p[f]],
                   lambda f, _r=r, _t=tag: (f"t={times[f]:.2f}s | {fname} | {_t} | "
                                            f"Step {f} | slice-RMSE={_r[f]:.4f}"),
                   scalar_painter(sl, cmap, levels, vmin, vmax, st, a.point_size),
                   fps=a.fps, fig_w=a.fig_w, row_h=a.row_h)

    # 数值本身无条件算+打印 (成本与原来一致), 只有落盘才看 --save_rmse ——
    # 不存文件时终端日志仍是这次 run 的量化留档。
    rmse_full = np.sqrt(((gts - preds) ** 2).mean(axis=1))          # (T, C)
    print("full-field RMSE 均值  " + "  ".join(
        f"{n}={v:.4f}" for n, v in zip(schema.names, rmse_full.mean(axis=0))))
    if a.save_rmse:
        np.save(Path(out_stem + "_rmse.npy"), rmse_full)
        print(f"  逐帧 RMSE 已存 {out_stem}_rmse.npy "
              f"(columns={list(schema.names)})")
    print("Done.")


# ============================================================
# 子命令 4: nofb —— 无反馈臂, prior | pred | GT 三行
# ============================================================

def cmd_nofb(a):
    """逐帧独立推理 (无 rollout, 无冷启动), 整个 chunk 都能画。

    标题给两个 RMSE 与增益比 gain = prior_RMSE / model_RMSE, >1 表示模型确实在
    修正 —— 与训练日志"模型/基线"的判据同源, 只是这里是切片上的逐帧值。
    """
    import torch
    from dataset import (expand_range, load_chunk, load_coords, load_prior,
                         reconstruct, resolve_stats)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(a.data_dir)
    cfg, schema, model, line = load_model(a.config_path, a.checkpoint,
                                          data_dir, device, "none")
    assert line == "fwv", "nofb 是 HPM+FUNWAVE 线的子命令 (需要 prior)"
    prior_dir = a.prior_dir or cfg.data.prior_dir

    stats = resolve_stats(str(data_dir), expand_range(cfg.data.train_chunk_range),
                          schema)
    mean, std = stats[0], stats[1]
    gt = load_chunk(str(data_dir), a.chunk_id, schema)
    prior = load_prior(str(prior_dir), a.chunk_id, schema)
    times = load_times(data_dir, a.chunk_id, gt.shape[0])
    if a.n_frames > 0:
        gt, prior, times = gt[:a.n_frames], prior[:a.n_frames], times[:a.n_frames]
    print(f"chunk {a.chunk_id}: {gt.shape}  t {times[0]:.2f}..{times[-1]:.2f}")

    coords = load_coords(str(data_dir)).to(device).unsqueeze(0)
    delta_idx = torch.as_tensor(schema.delta_indices, device=device)
    prior_n = ((prior - mean) / std).astype(np.float32)

    print("Predicting (per-frame, no rollout)...")
    pred_n = np.empty_like(prior_n)
    with torch.no_grad():
        for t in range(prior_n.shape[0]):
            p = torch.from_numpy(prior_n[t]).unsqueeze(0).to(device)
            # 必须经 reconstruct 只落在 delta_idx —— 直接 p+delta 在
            # out_dim == field_dim 时恰好等价, 有 frozen 通道就会加错。
            pred_n[t] = reconstruct(p, model(coords, p), delta_idx)[0].cpu().numpy()
            if t % 20 == 0:
                print(f"  frame {t+1}/{prior_n.shape[0]}", flush=True)
    pred = pred_n * std + mean

    sl = load_slice(data_dir)
    take, fname = field_slicer(a.field, schema, sl["cell_map"])
    s_pri, s_prd, s_gt = take(prior), take(pred), take(gt)
    cmap, levels, vmin, vmax = make_cmap(a.field, [s_gt, s_prd, s_pri], a.clim_pct)

    rmse_m = np.sqrt(((s_prd - s_gt) ** 2).mean(axis=1))
    rmse_p = np.sqrt(((s_pri - s_gt) ** 2).mean(axis=1))

    out_path = render(
        a.output, s_gt.shape[0],
        ["prior  (FUNWAVE lift)", "pred  =  prior + Δ  (HPM)", "GT  (CFD)"],
        lambda f: [s_pri[f], s_prd[f], s_gt[f]],
        lambda f: (f"chunk {a.chunk_id} | t={times[f]:.2f}s | frame {f} | {fname} | "
                   f"slice-RMSE  model={rmse_m[f]:.4f}  prior={rmse_p[f]:.4f}  "
                   f"(gain {rmse_p[f]/max(rmse_m[f],1e-12):.2f}x)"),
        scalar_painter(sl, cmap, levels, vmin, vmax, a.style, a.point_size),
        fps=a.fps, fig_w=a.fig_w, row_h=a.row_h)

    print(f"[done] slice-RMSE  model={rmse_m.mean():.4f}  prior={rmse_p.mean():.4f}"
          f"  gain={rmse_p.mean()/max(rmse_m.mean(),1e-12):.2f}x")
    if a.save_rmse:
        np.save(out_path.with_suffix(".rmse.npy"),
                np.stack([rmse_m, rmse_p], axis=0))
        print(f"       逐帧 RMSE 已存 {out_path.with_suffix('.rmse.npy')} "
              f"(第0行 model, 第1行 prior)")


# ============================================================
# 子命令 5: lt —— 长期 rollout, 无 GT
# ============================================================

def cmd_lt(a):
    """单行 pred, 跑满 prior 全长, **边推边写视频帧** (不全存内存, 防 OOM)。

    与 pred 的区别: 不加载 GT (long-term chunk 只有 prior), 不算 slice-RMSE
    (无对照), 打印每帧推理耗时 (部署参考, 对应 gen_prior 的 lift ms/帧)。

    rollout 语义与部署一致: k=0 冷启动 (x_f=0, m=0); k>=1 喂自己上一步。
    """
    import torch
    from dataset import (assemble, expand_range, load_coords, load_prior,
                         reconstruct, resolve_stats)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(a.data_dir)
    cfg, schema, model, line = load_model(a.config_path, a.checkpoint,
                                          data_dir, device, "self")
    assert line == "fwv", "lt 是 HPM+FUNWAVE 线的子命令 (需要 prior)"
    prior_dir = a.prior_dir or cfg.data.prior_dir

    stats = resolve_stats(str(data_dir), expand_range(cfg.data.train_chunk_range),
                          schema)
    mean, std = stats[0], stats[1]
    pr_raw = load_prior(str(prior_dir), a.chunk_id, schema)
    pr_n = torch.from_numpy(((pr_raw - mean) / std).astype(np.float32)).to(device)
    coords_b = load_coords(str(data_dir)).to(device).unsqueeze(0)
    delta_idx = torch.as_tensor(schema.delta_indices, device=device)

    T, N, F = pr_raw.shape
    n_steps = T if a.n_frames <= 0 else min(a.n_frames, T)
    times = load_times(data_dir, a.chunk_id, T)

    sl = load_slice(data_dir)
    take, fname = field_slicer(a.field, schema, sl["cell_map"])
    # 无 GT, 用 prior 该场的分位数定色标 (raw 空间)
    cmap, levels, vmin, vmax = make_cmap(a.field, [take(pr_raw)], a.clim_pct)

    assert a.style != "both", \
        "lt 是流式 rollout (frame_fn 有副作用), 不能渲染两遍。请指定 tri 或 scatter。"

    state = {"x_f": torch.zeros(1, N, F, device=device), "t": 0.0}

    @torch.no_grad()
    def step(k):
        prior_t = pr_n[k:k + 1]
        m_t = torch.full((1, 1, 1), 0.0 if k == 0 else 1.0, device=device)
        x = assemble(prior_t, state["x_f"], m_t)
        t0 = time.perf_counter()
        delta = model(coords_b, x)
        pred_t = reconstruct(prior_t, delta, delta_idx)     # (1,N,F) normalized
        if device.type == "cuda":
            torch.cuda.synchronize()
        state["t"] += time.perf_counter() - t0
        state["x_f"] = pred_t                                # 递归
        raw = pred_t[0].cpu().numpy() * std + mean           # (N,F) raw
        if k % 50 == 0:
            print(f"  frame {k}/{n_steps-1}  推理均值 "
                  f"{state['t']/max(k+1,1)*1000:.1f} ms/帧", flush=True)
        return [take(raw[None])[0]]

    print(f"[longterm] rollout {n_steps} 帧, 边推边渲染")
    out_base = Path(a.output)
    out_path = out_base.with_name(
        f"{out_base.with_suffix('').name}_{a.style}{out_base.suffix or '.mp4'}")
    render(out_path, n_steps,
           [f"HPM rollout (long-term, no GT)  {fname}"], step,
           lambda k: f"t={times[k]:.2f}s | frame {k}/{n_steps-1}",
           scalar_painter(sl, cmap, levels, vmin, vmax, a.style, a.point_size),
           fps=a.fps, fig_w=a.fig_w, row_h=a.row_h)

    print(f"[infer] rollout {n_steps} 帧, 前向累计 {state['t']:.1f}s, "
          f"均值 {state['t']/max(n_steps,1)*1000:.1f} ms/帧  ({N} cells)")
    print(f"[done] {out_path}")


# ============================================================
# CLI —— 子命令, 每个 mode 只暴露自己的参数
# ============================================================

def _render_args(p, row_h=10.8, style="tri"):
    p.add_argument("--style", default=style, choices=["tri", "scatter", "both"])
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--point_size", type=float, default=4.0)
    p.add_argument("--fig-w", type=float, default=38.4, dest="fig_w")
    p.add_argument("--row-h", type=float, default=row_h, dest="row_h",
                   help="每行高度 (总高 = row_h * 行数)")
    p.add_argument("--clim-pct", type=float, default=99.0, dest="clim_pct",
                   help="色标取 |v| 的这个分位数 (默认 99, 抗离群)。100 = 真正的 "
                        "max, 不裁剪任何数据 —— 即 vis_fw 的老行为, 但预测出一个"
                        "离群 cell 就会把全图冲淡。")


def _model_args(p):
    p.add_argument("--config_path", required=True,
                   help="训练 run 的 .hydra/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--chunk_id", type=int, required=True)
    p.add_argument("--prior_dir", default=None,
                   help="仅 fwv 线; 默认取 config 的 data.prior_dir")
    p.add_argument("--device", default="cuda")


def main():
    ap = argparse.ArgumentParser(
        description="HPM 可视化。五个子命令按流水线位置排列: "
                    "gt / align (训练前) -> pred / nofb / lt (训练后)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- gt ----
    p = sub.add_parser("gt", help="纯数据探查: GT alpha, 可多 chunk, 不加载模型")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--chunks", default="0-10", help="如 '0-10' 或 '6,9'")
    p.add_argument("--out_dir", default=os.path.join(_VIS_OUT, "gt_alpha"))
    _render_args(p, row_h=10.8, style="both")
    p.set_defaults(fn=cmd_gt)

    # ---- align ----
    p = sub.add_parser("align", help="配准检查: prior | GT 两行 (不要 checkpoint)")
    p.add_argument("--fw-dir", required=True, dest="fw_dir",
                   help="FUNWAVE output 目录")
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--data-dir", default=_DATA, dest="data_dir")
    p.add_argument("--scan-dir", default=None, dest="scan_dir",
                   help="best_k 的 c*.json 目录; 默认 <data-dir>/toffset_scan/")
    p.add_argument("--k", type=int, default=None,
                   help="帧移 k, 覆盖 scan 结果。用于目视复核不同 k。")
    p.add_argument("--field", default="alpha")
    p.add_argument("--gt-channels", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                   dest="gt_channels",
                   help="GT 中 [alpha,Ux,Uy,Uz,p_rgh] 各自的列索引")
    p.add_argument("--x-offset", type=float, default=15.05, dest="x_offset")
    p.add_argument("--y-offset", type=float, default=0.0, dest="y_offset")
    p.add_argument("--mglob", type=int, default=1575)
    p.add_argument("--nglob", type=int, default=30)
    p.add_argument("--dx", type=float, default=0.02)
    p.add_argument("--dy", type=float, default=0.02)
    p.add_argument("--plot-intv", type=float, default=0.05, dest="plot_intv")
    p.add_argument("--no-pnh", action="store_true", dest="no_pnh")
    p.add_argument("--n-frames", type=int, default=0, dest="n_frames",
                   help="取多少帧; 0 = 整个 chunk")
    p.add_argument("--start", default="mid",
                   help="起始帧: 整数, 或 'mid' 取 chunk 正中 (默认)")
    p.add_argument("--output", default=None)
    _render_args(p, row_h=8.0)
    p.set_defaults(fn=cmd_align)

    # ---- pred ----
    p = sub.add_parser("pred", help="GT | pred 两行 (两条线通用)")
    _model_args(p)
    p.add_argument("--field", default="alpha", help="通道名 或 'Umag' (速度模)")
    p.add_argument("--start_frame", type=int, default=0,
                   help="rollout 起始帧; fwv 线 0 = 真冷启动 (部署条件)")
    p.add_argument("--n_frames", type=int, default=0, help="0 = 跑到 chunk 末尾")
    p.add_argument("--also_tf_video", action="store_true",
                   help="仅 fwv 线: 额外为 tf 出一份视频")
    # 三个 npy 一律默认**不**存: 代码里没有任何地方读它们 (pred 每次都重跑
    # predict(), 没有 --load_preds 这条路), 要拿去离线分析时再显式打开。
    # RMSE 的数值不受 --save_rmse 影响, 无论如何都打印到 stdout。
    p.add_argument("--save_preds", action="store_true",
                   help="额外落盘全场预测 _preds.npy (~0.9 GB/场)")
    p.add_argument("--save_rmse", action="store_true",
                   help="额外落盘逐帧全场 RMSE _rmse{,_tf}.npy (各 ~1.7 KB)")
    p.add_argument("--output", default=os.path.join(_VIS_OUT, "pred", "compare.mp4"))
    _render_args(p, row_h=10.8)
    p.set_defaults(fn=cmd_pred)

    # ---- nofb ----
    p = sub.add_parser("nofb", help="无反馈臂: prior | pred | GT 三行")
    _model_args(p)
    p.add_argument("--field", default="alpha")
    p.add_argument("--n_frames", type=int, default=0, help="0 = 整个 chunk")
    p.add_argument("--save_rmse", action="store_true",
                   help="额外落盘逐帧 slice-RMSE .rmse.npy ((2,T): 0=model 1=prior)")
    p.add_argument("--output", default=os.path.join(_VIS_OUT, "nofb", "compare.mp4"))
    _render_args(p, row_h=8.0)
    p.set_defaults(fn=cmd_nofb)

    # ---- lt ----
    p = sub.add_parser("lt", help="长期 rollout (无 GT): 单行 pred, 流式渲染")
    _model_args(p)
    p.add_argument("--field", default="alpha")
    p.add_argument("--n_frames", type=int, default=0,
                   help="rollout 步数; 0 = 跑满 prior 全长")
    p.add_argument("--output", default=os.path.join(_VIS_OUT, "lt", "longterm.mp4"))
    _render_args(p, row_h=10.8)
    p.set_defaults(fn=cmd_lt)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()