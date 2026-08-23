"""
GT-only alpha animation, per chunk. 纯数据探查：不加载模型/stats/rollout。
只看 alpha（通道0），不受 alpha-weighting 影响（alpha 永远原始 [0,1]）。
切片几何用预计算缓存。每个 chunk 一个文件，scatter + tri 双输出。

Usage:
    python vis_gt.py --data_dir /path/to/cropped_0.05 \
        --chunks 0-10 --out_dir vis/gt_alpha
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

MID_Y = 0.30


def parse_chunks(s):
    """'0-10' -> [0..10]; '6,9' -> [6,9]; '6' -> [6]."""
    out = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--chunks", type=str, default="0-10",
                    help="如 '0-10' 或 '6,9'")
    ap.add_argument("--out_dir", type=str, default="vis/gt_alpha")
    ap.add_argument("--style", type=str, default="both",
                    choices=["scatter", "tri", "both"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--point_size", type=float, default=4.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = parse_chunks(args.chunks)

    # ---- slice cache (纯 numpy) ----
    sdir = data_dir / f"slice_y{MID_Y:.2f}"
    cell_map = np.load(sdir / "slice_cell_map.npy")
    xz = np.load(sdir / "slice_xz.npy")
    tri_simplices = np.load(sdir / "slice_tri.npy")
    x_s, z_s = xz[:, 0], xz[:, 1]
    triang = mtri.Triangulation(x_s, z_s, triangles=tri_simplices)
    xlim = (x_s.min(), x_s.max()); zlim = (z_s.min(), z_s.max())
    print(f"Slice cache: {len(cell_map)} faces @ y={MID_Y}")

    # alpha colormap (透明白->红) + 固定 [0,1]
    cdict = {'red':[[0.,1.,1.],[1.,.6,.6]], 'green':[[0.,1.,1.],[1.,0.,0.]], 'blue':[[0.,1.,1.],[1.,0.,0.]]}
    cmap = LinearSegmentedColormap('OpacityReds', cdict)
    vmin, vmax = 0.0, 1.0
    levels = np.linspace(0, 1, 128)

    styles = ["scatter", "tri"] if args.style == "both" else [args.style]

    for cid in chunk_ids:
        dpath = data_dir / f"chunk_{cid:03d}_data.npy"
        tpath = data_dir / f"chunk_{cid:03d}_times.npy"
        if not dpath.exists():
            print(f"chunk {cid}: 缺 {dpath.name}, 跳过")
            continue
        data = np.load(dpath)                      # (T, N, 6)
        times = np.load(tpath) if tpath.exists() else np.arange(data.shape[0])
        alpha_slice = data[:, cell_map, 0]          # (T, M) — 通道0=alpha
        T = data.shape[0]
        print(f"chunk {cid}: {T} frames")

        for style in styles:
            out_path = out_dir / f"gt_alpha_chunk{cid:03d}_{style}.mp4"
            fig, ax = plt.subplots(figsize=(38.4, 10.8), dpi=100)
            fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.97)

            def draw(frame):
                ax.clear()
                vals = alpha_slice[frame]
                if style == "scatter":
                    ax.scatter(x_s, z_s, c=vals, s=args.point_size,
                               vmin=vmin, vmax=vmax, cmap=cmap, edgecolors='none')
                else:
                    ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend='both')
                ax.set_facecolor('white'); ax.set_xlim(xlim); ax.set_ylim(zlim)
                ax.set_xlabel("X (m)", fontsize=20); ax.set_ylabel("Z (m)", fontsize=20)
                ax.tick_params(labelsize=16)
                ax.set_title(f"GT alpha | chunk {cid} | t={times[frame]:.2f}s | "
                             f"frame {frame}", fontsize=24)

            draw(0)
            ani = animation.FuncAnimation(fig, draw, frames=T,
                                          interval=1000 // args.fps, blit=False)
            ani.save(str(out_path), writer="ffmpeg", fps=args.fps,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            plt.close()
            print(f"  [{style}] saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
