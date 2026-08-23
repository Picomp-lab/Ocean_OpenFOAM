import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--field", type=int, default=0, help="0=alpha")
    parser.add_argument("--output", type=str, default="openfoam_gt.mp4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    # 1. 加载数据
    data_dir = Path(args.data_dir)
    coords = np.load(data_dir / "coords.npy")
    raw_data = np.load(data_dir / f"chunk_{args.chunk_id:03d}_data.npy")
    times = np.load(data_dir / f"chunk_{args.chunk_id:03d}_times.npy")
    fi = args.field

    # 2. 宽带捞点：把槽道中心线（Y=0.025）附近、包含局部加密层的所有多面体点全部抓出来
    y_vals = coords[:, 1]
    mid_y = y_vals.min() + (y_vals.max() - y_vals.min()) / 2.0
    # 捞取中心线前后 1.5 厘米范围内的点（可根据网格最大尺寸微调，确保捞满）
    mask = np.abs(y_vals - mid_y) < 0.015 
    
    slice_coords = coords[mask]
    x_raw = slice_coords[:, 0]
    z_raw = slice_coords[:, 2]
    slice_data = raw_data[:, mask, fi]

    end = min(args.start_frame + args.n_frames, slice_data.shape[0])
    n = end - args.start_frame

    # 3. 【核心创新】：在 X-Z 平面上构建一幅规整的“虚拟画布”（Resampling Grid）
    # 针对你 10 多米长、零点几米高的长条形槽道，建立高精细度的等间距像素网格
    grid_x, grid_z = np.mgrid[
        x_raw.min():x_raw.max():2000j,  # X 方向划分 2000 个格子
        z_raw.min():z_raw.max():400j    # Z 方向划分 400 个格子
    ]

    # 4. 2D 动态透明度色板配置
    if fi == 0:
        cdict = {
            'red':   [[0.0, 1.0, 1.0], [1.0, 0.6, 0.6]],
            'green': [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            'blue':  [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            'alpha': [[0.0, 0.0, 0.0],  # 纯空气完全透明
                      [0.05, 0.0, 0.0], # 过滤微小噪声
                      [0.5, 0.5, 0.5],  # 过渡带半透明展示状态演变
                      [0.9, 1.0, 1.0], 
                      [1.0, 1.0, 1.0]]  # 纯水完全不透明
        }
        custom_cmap = LinearSegmentedColormap('OpacityReds', cdict)
        levels = np.linspace(0, 1, 128)
    else:
        custom_cmap = 'coolwarm'
        levels = np.linspace(slice_data.min(), slice_data.max(), 128)

    # 5. 画布初始化 (长幅槽道电影级比例)
    fig, ax = plt.subplots(figsize=(38.4, 10.8), dpi=100)
    fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.95)
    ax.set_facecolor('white')  # 背景纯白，以便透明部分透出

    # 6. 插值绘制初始帧
    # 用线性插值（linear）把第一帧多面体散点上的数据映射到我们的规整网格上
    # griddata 会自动把网格外部（比如波浪上方的空气、槽道边界外）插值为 NaN (空值)
    grid_data_init = griddata((x_raw, z_raw), slice_data[args.start_frame], (grid_x, grid_z), method='linear')
    
    # 规整网格用常规的 contourf 绘制，没有了复杂的三角拓扑，速度极快且绝无爆图乱线！
    cf = ax.contourf(grid_x[:, 0], grid_z[0, :], grid_data_init.T, levels=levels,
                     cmap=custom_cmap, extend='both')

    ax.set_xlabel("X (m)", fontsize=22)
    ax.set_ylabel("Z (m)", fontsize=22)
    ax.tick_params(labelsize=18)
    
    cbar = fig.colorbar(cf, ax=ax, label="alpha.water", shrink=0.7)
    cbar.ax.tick_params(labelsize=16)
    title_text = ax.set_title(f"t = {times[args.start_frame]:.2f}s | OpenFOAM Poly Mesh Resampled", fontsize=28)

    # 7. 动画更新函数
    def update(frame):
        idx = args.start_frame + frame
        ax.clear()
        grid_data_frame = griddata((x_raw, z_raw), slice_data[idx], (grid_x, grid_z), method='linear')
        ax.contourf(grid_x[:, 0], grid_z[0, :], grid_data_frame.T, levels=levels,
                    cmap=custom_cmap, extend='both')
        ax.set_facecolor('white')
        ax.set_xlabel("X (m)", fontsize=22)
        ax.set_ylabel("Z (m)", fontsize=22)
        ax.tick_params(labelsize=18)
        ax.set_title(f"t = {times[idx]:.2f}s | alpha.water", fontsize=28)

    # 开启 blit 加速
    ani = animation.FuncAnimation(fig, update, frames=n, interval=1000 // args.fps, blit=False)
    ani.save(args.output, writer="ffmpeg", fps=args.fps,
             extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close()
    print(f"OpenFOAM Post-Process Animation Saved: {args.output}")

if __name__ == "__main__":
    main()