#!/usr/bin/env python3
"""
Reconstruct physical flow fields from POD-LSTM predicted coefficients.

Takes LSTM predictions in POD coefficient space, reconstructs full flow fields
via POD basis vectors, and compares against ground truth (original CFD data
projected onto POD basis then reconstructed).

Usage:
    python reconstruct_fields.py \
        --pod_dir $OCEAN_DATA/pod_results \
        --lstm_dir ./lstm_results_v6_noprgh \
        --output ./reconstruction_results

Outputs:
    - Flow field snapshot comparisons (True vs Predicted vs Error)
    - Per-step relative L2 error curves (per variable + overall)
    - Error statistics JSON
    - Reconstructed field .npy files
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# Configuration (must match LSTM training)
# ============================================================

# v6_noprgh: no p_rgh, 3 variables
VARIABLES = ['alpha_water', 'Ux', 'Uz']
MODE_COUNTS = {
    'alpha_water': 21,
    'Ux': 27,
    'Uz': 32,
}
TOTAL_MODES = sum(MODE_COUNTS.values())  # 80

TRANSIENT_STEPS = 200     # v6 uses 200
WINDOW_SIZE = 40          # v6 uses 40
TRAIN_STEPS = 480
TEST_STEPS = 120

# Colormaps per variable
CMAPS = {
    'alpha_water': 'Blues',
    'Ux': 'RdBu_r',
    'Uz': 'RdBu_r',
}

# Display names
DISPLAY_NAMES = {
    'alpha_water': r'$\alpha_{water}$',
    'Ux': r'$U_x$',
    'Uz': r'$U_z$',
}


# ============================================================
# Load data
# ============================================================

def load_pod_basis(pod_dir):
    """Load POD modes, means, and coordinates."""
    coords = np.load(os.path.join(pod_dir, 'coords.npy'))  # (N_points, 2)
    print(f"Coordinates: {coords.shape}")

    basis = {}
    for var in VARIABLES:
        modes = np.load(os.path.join(pod_dir, f'{var}_modes.npy'))   # (100, N_points)
        mean = np.load(os.path.join(pod_dir, f'{var}_mean.npy'))     # (N_points,)
        coeffs = np.load(os.path.join(pod_dir, f'{var}_coeffs.npy')) # (1002, 100)
        n_modes = MODE_COUNTS[var]
        basis[var] = {
            'modes': modes[:n_modes],      # (n_modes, N_points)
            'mean': mean,                  # (N_points,)
            'coeffs_full': coeffs,         # (1002, 100) — for ground truth
        }
        print(f"  {var}: modes {modes[:n_modes].shape}, mean {mean.shape}")

    return coords, basis


def load_lstm_results(lstm_dir):
    """Load LSTM predictions, targets, scaler, and var_info."""
    pred_ss = np.load(os.path.join(lstm_dir, 'pred_singlestep.npy'))
    pred_ar = np.load(os.path.join(lstm_dir, 'pred_autoregressive.npy'))
    targets = np.load(os.path.join(lstm_dir, 'targets.npy'))
    scaler = joblib.load(os.path.join(lstm_dir, 'scaler.pkl'))

    with open(os.path.join(lstm_dir, 'var_info.json'), 'r') as f:
        var_info = json.load(f)

    print(f"  Single-step predictions: {pred_ss.shape}")
    print(f"  Autoregressive predictions: {pred_ar.shape}")
    print(f"  Targets: {targets.shape}")

    return pred_ss, pred_ar, targets, scaler, var_info


# ============================================================
# Reconstruction
# ============================================================

def inverse_transform_coeffs(predictions, scaler):
    """Inverse StandardScaler to get original-scale POD coefficients."""
    return scaler.inverse_transform(predictions)


def reconstruct_field(coeffs_var, modes_var, mean_var):
    """
    Reconstruct physical field from POD coefficients.

    Args:
        coeffs_var: (T, n_modes) POD coefficients for one variable
        modes_var:  (n_modes, N_points) POD spatial modes
        mean_var:   (N_points,) temporal mean

    Returns:
        fields: (T, N_points) reconstructed physical field
    """
    return coeffs_var @ modes_var + mean_var[np.newaxis, :]


def reconstruct_all(coeffs_orig, var_info, basis):
    """
    Reconstruct all variables from combined POD coefficient array.

    Args:
        coeffs_orig: (T, TOTAL_MODES) inverse-transformed coefficients
        var_info: dict with start_col and n_modes per variable
        basis: dict with modes and mean per variable

    Returns:
        fields: dict of var_name -> (T, N_points) arrays
    """
    fields = {}
    for var in VARIABLES:
        start = var_info[var]['start_col']
        n = var_info[var]['n_modes']
        coeffs_var = coeffs_orig[:, start:start + n]
        fields[var] = reconstruct_field(
            coeffs_var, basis[var]['modes'], basis[var]['mean']
        )
    return fields


def get_ground_truth_fields(basis, test_step_indices):
    """
    Reconstruct ground truth fields from original POD coefficients
    (truncated to same number of modes as LSTM uses).

    This is the fair comparison: both predicted and ground truth
    are reconstructed from the same truncated POD basis.
    """
    fields = {}
    for var in VARIABLES:
        n_modes = MODE_COUNTS[var]
        coeffs_full = basis[var]['coeffs_full']  # (1002, 100)
        coeffs_trunc = coeffs_full[test_step_indices, :n_modes]  # (T_test, n_modes)
        fields[var] = reconstruct_field(
            coeffs_trunc, basis[var]['modes'], basis[var]['mean']
        )
    return fields


# ============================================================
# Error metrics
# ============================================================

def relative_l2_error(pred, true):
    """Per-step relative L2 error: ||pred - true||_2 / ||true||_2"""
    diff_norm = np.linalg.norm(pred - true, axis=1)
    true_norm = np.linalg.norm(true, axis=1)
    return diff_norm / (true_norm + 1e-10)


def compute_errors(pred_fields, true_fields):
    """Compute per-variable and overall errors."""
    errors = {}
    all_pred = []
    all_true = []

    for var in VARIABLES:
        pred = pred_fields[var]
        true = true_fields[var]
        rel_l2 = relative_l2_error(pred, true)
        rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
        max_err = np.max(np.abs(pred - true), axis=1)

        errors[var] = {
            'rel_l2_per_step': rel_l2,
            'rmse_per_step': rmse,
            'max_err_per_step': max_err,
            'rel_l2_mean': float(np.mean(rel_l2)),
            'rel_l2_std': float(np.std(rel_l2)),
            'rmse_mean': float(np.mean(rmse)),
            'max_err_mean': float(np.mean(max_err)),
        }
        all_pred.append(pred)
        all_true.append(true)

    # Overall: concatenate all variables along feature axis
    all_pred = np.concatenate(all_pred, axis=1)
    all_true = np.concatenate(all_true, axis=1)
    overall_rel_l2 = relative_l2_error(all_pred, all_true)
    overall_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2, axis=1))

    errors['overall'] = {
        'rel_l2_per_step': overall_rel_l2,
        'rmse_per_step': overall_rmse,
        'rel_l2_mean': float(np.mean(overall_rel_l2)),
        'rel_l2_std': float(np.std(overall_rel_l2)),
        'rmse_mean': float(np.mean(overall_rmse)),
    }

    return errors


# ============================================================
# Plotting
# ============================================================

def create_triangulation(coords):
    """Create Delaunay triangulation for contour plots."""
    x = coords[:, 0]
    z = coords[:, 1]
    return tri.Triangulation(x, z)


def plot_snapshots(coords, pred_fields, true_fields, times_test,
                   snapshot_indices, output_dir, prefix='singlestep'):
    """
    Plot flow field snapshots: True vs Predicted vs Absolute Error.

    Args:
        snapshot_indices: list of indices into test arrays to plot
    """
    triang = create_triangulation(coords)
    n_vars = len(VARIABLES)

    for si in snapshot_indices:
        t_val = times_test[si]
        fig, axes = plt.subplots(n_vars, 3, figsize=(18, 4 * n_vars))

        for row, var in enumerate(VARIABLES):
            true_snap = true_fields[var][si]
            pred_snap = pred_fields[var][si]
            err_snap = np.abs(pred_snap - true_snap)

            # Shared color limits for true and predicted
            vmin = min(true_snap.min(), pred_snap.min())
            vmax = max(true_snap.max(), pred_snap.max())

            cmap = CMAPS[var]
            disp = DISPLAY_NAMES[var]

            # True
            ax = axes[row, 0]
            im = ax.tricontourf(triang, true_snap, levels=50,
                                cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'{disp} — True')
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')

            # Predicted
            ax = axes[row, 1]
            im = ax.tricontourf(triang, pred_snap, levels=50,
                                cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'{disp} — Predicted')
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')

            # Error
            ax = axes[row, 2]
            im = ax.tricontourf(triang, err_snap, levels=50, cmap='hot_r')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'{disp} — |Error|')
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')

        fig.suptitle(f'{prefix} reconstruction — t = {t_val:.2f}s',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{prefix}_snapshot_t{t_val:.2f}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Snapshot saved: {path}")


def plot_error_curves(errors_ss, errors_ar, times_test, output_dir):
    """
    Plot per-step relative L2 error curves.
    One subplot per variable + one for overall.
    Single-step and autoregressive on the same plot.
    """
    plot_vars = VARIABLES + ['overall']
    n_plots = len(plot_vars)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots),
                             sharex=True)

    for i, var in enumerate(plot_vars):
        ax = axes[i]
        n_ss = len(errors_ss[var]['rel_l2_per_step'])
        n_ar = len(errors_ar[var]['rel_l2_per_step'])

        t_ss = times_test[:n_ss]
        t_ar = times_test[:n_ar]

        ax.plot(t_ss, errors_ss[var]['rel_l2_per_step'],
                'b-', alpha=0.8, linewidth=1.2, label='Single-step')
        ax.plot(t_ar, errors_ar[var]['rel_l2_per_step'],
                'r--', alpha=0.8, linewidth=1.2, label='Autoregressive')

        disp = DISPLAY_NAMES.get(var, 'Overall')
        mean_ss = errors_ss[var]['rel_l2_mean']
        mean_ar = errors_ar[var]['rel_l2_mean']
        ax.set_title(f'{disp}  (SS mean: {mean_ss:.4f}, AR mean: {mean_ar:.4f})')
        ax.set_ylabel('Relative L2 Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Per-step Relative L2 Error in Physical Space', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'error_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error curves saved: {path}")


# ============================================================
# Save results
# ============================================================

def save_error_summary(errors_ss, errors_ar, output_dir):
    """Save error statistics to JSON (without per-step arrays)."""
    summary = {'single_step': {}, 'autoregressive': {}}

    for var in VARIABLES + ['overall']:
        for label, errors in [('single_step', errors_ss),
                              ('autoregressive', errors_ar)]:
            summary[label][var] = {
                k: v for k, v in errors[var].items()
                if not k.endswith('_per_step')
            }

    path = os.path.join(output_dir, 'reconstruction_errors.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Error summary saved: {path}")
    return summary


def save_reconstructed_fields(pred_fields, true_fields, output_dir, prefix):
    """Save reconstructed fields as .npy for further analysis."""
    for var in VARIABLES:
        np.save(os.path.join(output_dir, f'{prefix}_pred_{var}.npy'),
                pred_fields[var])
    # Save true fields once
    if prefix == 'singlestep':
        for var in VARIABLES:
            np.save(os.path.join(output_dir, f'true_{var}.npy'),
                    true_fields[var])


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct flow fields from POD-LSTM predictions')
    parser.add_argument('--pod_dir', type=str, required=True,
                        help='Directory containing POD results (.npy files)')
    parser.add_argument('--lstm_dir', type=str, required=True,
                        help='Directory containing LSTM results (v6_noprgh)')
    parser.add_argument('--output', type=str, default='./reconstruction_results',
                        help='Output directory')
    parser.add_argument('--snapshot_count', type=int, default=3,
                        help='Number of snapshots to plot (default: 3)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("Flow Field Reconstruction from POD-LSTM")
    print("=" * 60)

    # ---- 1. Load POD basis ----
    print("\n[1] Loading POD basis...")
    coords, basis = load_pod_basis(args.pod_dir)

    # ---- 2. Load LSTM results ----
    print("\n[2] Loading LSTM results...")
    pred_ss, pred_ar, targets, scaler, var_info = load_lstm_results(args.lstm_dir)

    # ---- 3. Inverse transform to original POD coefficient scale ----
    print("\n[3] Inverse transforming coefficients...")
    pred_ss_orig = inverse_transform_coeffs(pred_ss, scaler)
    pred_ar_orig = inverse_transform_coeffs(pred_ar, scaler)
    targets_orig = inverse_transform_coeffs(targets, scaler)

    # ---- 4. Determine test time indices ----
    # Test data starts at TRANSIENT_STEPS + TRAIN_STEPS in original timestep array
    # But predictions start at TRANSIENT_STEPS + TRAIN_STEPS + WINDOW_SIZE
    # (because the first WINDOW_SIZE steps of test data are used as input)
    times = np.load(os.path.join(args.pod_dir, 'times.npy'))  # (1002,)
    test_start_global = TRANSIENT_STEPS + TRAIN_STEPS + WINDOW_SIZE
    n_test_samples = len(targets)
    test_indices_global = np.arange(test_start_global,
                                    test_start_global + n_test_samples)
    times_test = times[test_indices_global]

    print(f"  Test samples: {n_test_samples}")
    print(f"  Time range: [{times_test[0]:.2f}, {times_test[-1]:.2f}]s")

    # ---- 5. Reconstruct predicted fields ----
    print("\n[4] Reconstructing predicted flow fields...")
    pred_fields_ss = reconstruct_all(pred_ss_orig, var_info, basis)
    pred_fields_ar = reconstruct_all(pred_ar_orig[:n_test_samples], var_info, basis)
    print("  Done.")

    # ---- 6. Reconstruct ground truth fields (from truncated POD) ----
    print("\n[5] Reconstructing ground truth fields...")
    true_fields = get_ground_truth_fields(basis, test_indices_global)
    print("  Done.")

    for var in VARIABLES:
        print(f"  {var}: pred {pred_fields_ss[var].shape}, "
              f"true {true_fields[var].shape}")

    # ---- 7. Compute errors ----
    print("\n[6] Computing reconstruction errors...")
    errors_ss = compute_errors(pred_fields_ss, true_fields)
    errors_ar = compute_errors(pred_fields_ar, true_fields)

    print("\n  Single-step errors:")
    for var in VARIABLES + ['overall']:
        disp = DISPLAY_NAMES.get(var, 'Overall')
        print(f"    {disp}: rel_L2 = {errors_ss[var]['rel_l2_mean']:.6f}")

    print("\n  Autoregressive errors:")
    for var in VARIABLES + ['overall']:
        disp = DISPLAY_NAMES.get(var, 'Overall')
        print(f"    {disp}: rel_L2 = {errors_ar[var]['rel_l2_mean']:.6f}")

    # ---- 8. Plot snapshots ----
    print("\n[7] Plotting flow field snapshots...")
    # Pick snapshot indices: start, middle, end of test range
    n = n_test_samples
    snapshot_indices = [0, n // 2, n - 1]
    if args.snapshot_count > 3:
        snapshot_indices = np.linspace(0, n - 1, args.snapshot_count,
                                       dtype=int).tolist()

    plot_snapshots(coords, pred_fields_ss, true_fields, times_test,
                   snapshot_indices, args.output, prefix='singlestep')
    plot_snapshots(coords, pred_fields_ar, true_fields, times_test,
                   snapshot_indices, args.output, prefix='autoregressive')

    # ---- 9. Plot error curves ----
    print("\n[8] Plotting error curves...")
    plot_error_curves(errors_ss, errors_ar, times_test, args.output)

    # ---- 10. Save results ----
    print("\n[9] Saving results...")
    summary = save_error_summary(errors_ss, errors_ar, args.output)
    save_reconstructed_fields(pred_fields_ss, true_fields, args.output,
                              'singlestep')
    save_reconstructed_fields(pred_fields_ar, true_fields, args.output,
                              'autoregressive')

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()