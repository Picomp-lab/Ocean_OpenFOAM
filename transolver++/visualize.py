"""
Visualize Transolver++ predictions vs ground truth (v2)
========================================================
Each timestep generates a separate figure with GT / Pred / Error stacked.
Optimized for narrow flume geometry (25m x 0.55m).

Usage:
    python visualize.py --results_dir ./results --data_dir ./data
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize


def make_triangulation(coords, max_edge=0.3):
    """Create filtered triangulation for the flume mesh."""
    x, z = coords[:, 0], coords[:, 1]
    triang = tri.Triangulation(x, z)
    triangles = triang.triangles
    x_tri = x[triangles]
    z_tri = z[triangles]
    mask = (
        (np.abs(x_tri[:, 0] - x_tri[:, 1]) > max_edge) |
        (np.abs(x_tri[:, 0] - x_tri[:, 2]) > max_edge) |
        (np.abs(x_tri[:, 1] - x_tri[:, 2]) > max_edge) |
        (np.abs(z_tri[:, 0] - z_tri[:, 1]) > max_edge) |
        (np.abs(z_tri[:, 0] - z_tri[:, 2]) > max_edge) |
        (np.abs(z_tri[:, 1] - z_tri[:, 2]) > max_edge)
    )
    triang.set_mask(mask)
    return triang


def plot_field_row(axes, triang, gt_vals, pred_vals, field_name,
                   vmin, vmax, cmap, coords):
    """Plot GT, Pred, Error in a row of 3 axes."""
    for ax in axes:
        ax.set_xlim(coords[:, 0].min() - 0.1, coords[:, 0].max() + 0.1)
        ax.set_ylim(coords[:, 1].min() - 0.02, coords[:, 1].max() + 0.02)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # GT
    axes[0].tripcolor(triang, gt_vals, shading='flat',
                      vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_ylabel(f'{field_name}\nGT', fontsize=9)

    # Pred
    axes[1].tripcolor(triang, pred_vals, shading='flat',
                      vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_ylabel('Pred', fontsize=9)

    # Error
    err = np.abs(pred_vals - gt_vals)
    err_max = max(np.percentile(err, 99), 0.01)
    axes[2].tripcolor(triang, err, shading='flat',
                      vmin=0, vmax=err_max, cmap='hot')
    axes[2].set_ylabel('|Error|', fontsize=9)

    # Colorbar for GT/Pred
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
    plt.colorbar(sm, ax=axes[:2].tolist(), shrink=0.8, pad=0.02)
    # Colorbar for Error
    sm_err = plt.cm.ScalarMappable(cmap='hot', norm=Normalize(0, err_max))
    plt.colorbar(sm_err, ax=[axes[2]], shrink=0.8, pad=0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--steps', type=str, default='1,5,10,20,50',
                        help='Rollout steps to visualize')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output or os.path.join(args.results_dir, 'figs')
    os.makedirs(out_dir, exist_ok=True)

    coords = np.load(os.path.join(args.data_dir, 'coords_2d.npy'))
    pred = np.load(os.path.join(args.results_dir, 'rollout_pred.npy'))
    gt = np.load(os.path.join(args.results_dir, 'rollout_gt.npy'))

    print(f"Coords: {coords.shape}, Pred: {pred.shape}, GT: {gt.shape}")

    steps = [int(s) for s in args.steps.split(',')]
    steps = [s for s in steps if s < min(pred.shape[0], gt.shape[0])]

    triang = make_triangulation(coords)

    # Global ranges
    ux_min, ux_max = gt[:, :, 1].min(), gt[:, :, 1].max()
    uz_min, uz_max = gt[:, :, 2].min(), gt[:, :, 2].max()

    for step in steps:
        dt = step * 0.05
        err_total = np.linalg.norm(pred[step] - gt[step]) / \
                    (np.linalg.norm(gt[step]) + 1e-8)

        print(f"\nStep {step} (t+{dt:.2f}s), rel error={err_total:.4f}")

        fig, axes = plt.subplots(9, 1, figsize=(20, 18))

        plot_field_row(axes[0:3], triang,
                       gt[step, :, 0], pred[step, :, 0],
                       'alpha', 0, 1, 'RdYlBu_r', coords)

        plot_field_row(axes[3:6], triang,
                       gt[step, :, 1], pred[step, :, 1],
                       'Ux', ux_min, ux_max, 'coolwarm', coords)

        plot_field_row(axes[6:9], triang,
                       gt[step, :, 2], pred[step, :, 2],
                       'Uz', uz_min, uz_max, 'coolwarm', coords)

        fig.suptitle(f'Rollout step {step} (t + {dt:.2f}s)  |  '
                     f'Relative L2 error: {err_total:.4f} ({err_total*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fname = f'step_{step:03d}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        print(f"  Saved: {out_dir}/{fname}")
        plt.close()

    # Rollout error curve
    print("\nPlotting rollout error curve...")
    n_eval = min(pred.shape[0], gt.shape[0]) - 1
    times_arr = np.arange(1, n_eval + 1) * 0.05
    errors = {n: [] for n in ['alpha', 'Ux', 'Uz', 'total']}

    for s in range(1, n_eval + 1):
        for j, name in enumerate(['alpha', 'Ux', 'Uz']):
            err = np.linalg.norm(pred[s, :, j] - gt[s, :, j]) / \
                  (np.linalg.norm(gt[s, :, j]) + 1e-8)
            errors[name].append(err)
        err_t = np.linalg.norm(pred[s] - gt[s]) / (np.linalg.norm(gt[s]) + 1e-8)
        errors['total'].append(err_t)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times_arr, errors['alpha'], label='alpha', linewidth=2)
    ax.plot(times_arr, errors['Ux'], label='Ux', linewidth=2)
    ax.plot(times_arr, errors['Uz'], label='Uz', linewidth=2)
    ax.plot(times_arr, errors['total'], label='Total', linewidth=2.5,
            color='black', linestyle='--')
    ax.set_xlabel('Rollout time (s)', fontsize=12)
    ax.set_ylabel('Relative L2 error', fontsize=12)
    ax.set_title('Autoregressive rollout error', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rollout_error.png'), dpi=150)
    print(f"  Saved: {out_dir}/rollout_error.png")
    plt.close()

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == '__main__':
    main()