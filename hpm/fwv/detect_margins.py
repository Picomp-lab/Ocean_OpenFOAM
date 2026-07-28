#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_margins.py — 从 GT mp4 抽一帧, 自动检测 matplotlib axes box (黑色 spine)
的像素位置, 换算成 subplots_adjust 比例, 直接给出 vis_lift.py 的 --margins 参数。

原理: axes 的四条 spine 是纯黑 (0,0,0) 的长直线。数据是暗红 (R 高 G/B 低),
      标题/刻度文字是黑但短。按 "近黑 (R,G,B 全低)" 筛出 spine 像素, 再找
      占据画幅大部分宽/高的那几行/列 —— 即上下左右四条边。

用法:
  python detect_margins.py --mp4 vis/gt_alpha/gt_alpha_chunk006_tri.mp4
  python detect_margins.py --mp4 ... --frame 10 --save-debug box.png
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def extract_frame(mp4, frame_idx, out_png):
    """用 ffmpeg 抽第 frame_idx 帧 (0-based) 存成 png。"""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", mp4,
           "-vf", f"select=eq(n\\,{frame_idx})", "-vframes", "1", out_png]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_png):
        sys.exit(f"[err] ffmpeg 抽帧失败:\n{r.stderr}")
    return out_png


def find_axes_box(img, black_thr=80, line_frac=0.5):
    """返回 axes box 的像素边界 (x0, y0, x1, y1), 含 spine 本身。

    img       : (H, W, 3) uint8
    black_thr : 三通道都低于此值算 "近黑" (排除暗红数据 R~139)
    line_frac : 一行/列中近黑像素占比超过此值, 视为 spine 线
    """
    H, W = img.shape[:2]
    black = (img[:, :, 0] < black_thr) & \
            (img[:, :, 1] < black_thr) & \
            (img[:, :, 2] < black_thr)

    row_cnt = black.sum(axis=1)          # 每行的近黑像素数
    col_cnt = black.sum(axis=0)          # 每列的近黑像素数

    # 候选 spine: 覆盖画幅相当比例的行/列
    rows = np.flatnonzero(row_cnt > line_frac * W)
    cols = np.flatnonzero(col_cnt > line_frac * H)

    if len(rows) < 2 or len(cols) < 2:
        sys.exit(f"[err] 未检出完整 axes box "
                 f"(找到 {len(rows)} 条横线 / {len(cols)} 条竖线)。\n"
                 f"     试着调 --black-thr 或 --line-frac。")

    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True, help="GT mp4 路径")
    ap.add_argument("--frame", type=int, default=0, help="抽第几帧 (0-based)")
    ap.add_argument("--black-thr", type=int, default=80, dest="black_thr")
    ap.add_argument("--line-frac", type=float, default=0.5, dest="line_frac")
    ap.add_argument("--save-debug", default=None,
                    help="存一张标出检测框的 png, 用于目视核对")
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("[err] 需要 Pillow: pip install Pillow")

    with tempfile.TemporaryDirectory() as td:
        png = extract_frame(args.mp4, args.frame, os.path.join(td, "f.png"))
        img = np.array(Image.open(png).convert("RGB"))

    H, W = img.shape[:2]
    x0, y0, x1, y1 = find_axes_box(img, args.black_thr, args.line_frac)

    # matplotlib 的 figure 坐标: 原点在左下, y 向上
    left = x0 / W
    right = (x1 + 1) / W
    top = 1.0 - y0 / H
    bottom = 1.0 - (y1 + 1) / H

    print(f"[img]  {os.path.basename(args.mp4)}  frame {args.frame}  {W}x{H}")
    print(f"[box]  像素 x:{x0}..{x1}  y:{y0}..{y1}  "
          f"(宽 {x1-x0+1}px, 高 {y1-y0+1}px)")
    print()
    print(f"[fig]  left={left:.4f}  bottom={bottom:.4f}  "
          f"right={right:.4f}  top={top:.4f}")
    print()
    print("直接填进 vis_lift.py:")
    print(f"    --margins {left:.4f} {bottom:.4f} {right:.4f} {top:.4f}")
    print()
    print(f"提示: lift 的 figsize 也要是 {W/100:.1f}x{H/100:.1f} (dpi=100), "
          f"即 --fig-w {W/100:.1f} --fig-h {H/100:.1f}")

    if args.save_debug:
        from PIL import Image as I, ImageDraw
        im = I.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=5)
        im.save(args.save_debug)
        print(f"[dbg]  {args.save_debug}")


if __name__ == "__main__":
    main()
