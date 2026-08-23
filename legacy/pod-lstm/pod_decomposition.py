"""
POD Decomposition for 2D Wave Slice Data
=========================================
Reads OpenFOAM postProcessing/sample raw files,
builds snapshot matrices, performs SVD, saves results.

Usage:
    python pod_decomposition.py --data_dir /path/to/postProcessing/sample --output ./pod_results

Output:
    - coords.npy          : (N_points, 2) array of (x, z) coordinates
    - {var}_modes.npy      : (N_modes, N_points) spatial POD modes
    - {var}_coeffs.npy     : (N_timesteps, N_modes) temporal coefficients
    - {var}_singular.npy   : (N_modes,) singular values
    - {var}_mean.npy       : (N_points,) temporal mean field
    - times.npy            : (N_timesteps,) time values
    - energy_spectrum.png  : cumulative energy plot
    - mode_summary.txt     : truncation summary for each variable
"""

import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Data loading
# ============================================================

SCALAR_VARS = ['alpha.water', 'p_rgh', 'nut']
ALL_VARS = ['alpha.water', 'p_rgh', 'nut', 'Ux', 'Uz']


def get_sorted_timesteps(data_dir):
    dirs = []
    for p in Path(data_dir).expanduser().iterdir():
        if p.is_dir():
            try:
                dirs.append((float(p.name), p.name))
            except ValueError:
                continue
    dirs.sort(key=lambda x: x[0])
    return dirs


def read_raw_scalar(filepath):
    df = pl.read_csv(
        filepath,
        skip_rows=2,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,
        infer_schema_length=10000
    )
    coords_xz = df.select([df.columns[0], df.columns[2]]).to_numpy()
    values = df[df.columns[3]].to_numpy()
    return coords_xz, values


def read_raw_vector(filepath):
    df = pl.read_csv(
        filepath,
        skip_rows=2,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,
        infer_schema_length=10000
    )
    coords_xz = df.select([df.columns[0], df.columns[2]]).to_numpy()
    Ux = df[df.columns[3]].to_numpy()
    Uy = df[df.columns[4]].to_numpy()
    Uz = df[df.columns[5]].to_numpy()
    return coords_xz, Ux, Uy, Uz


def _read_one_timestep(args):
    """Read all variables for a single timestep. Used by ThreadPoolExecutor."""
    idx, t_val, t_name, data_dir = args
    t_dir = os.path.join(data_dir, t_name)
    result = {}
    coords_xz = None

    for var in SCALAR_VARS:
        fpath = os.path.join(t_dir, f"{var}_ySlice.raw")
        xz, vals = read_raw_scalar(fpath)
        result[var] = vals
        if coords_xz is None:
            coords_xz = xz

    fpath = os.path.join(t_dir, "U_ySlice.raw")
    _, ux, _, uz = read_raw_vector(fpath)
    result['Ux'] = ux
    result['Uz'] = uz

    return idx, t_val, coords_xz, result


def build_snapshot_matrices(data_dir, timesteps, n_workers=8):
    """Build snapshot matrices for all variables with parallel I/O.

    Returns:
        coords   : (N_points, 2) - (x, z) from first timestep
        snapshots: dict of variable_name -> (N_timesteps, N_points) array
        times    : (N_timesteps,) array of time values
    """
    n_steps = len(timesteps)
    print(f"Reading {n_steps} timesteps with {n_workers} workers...")
    t_start = time.time()

    # Read first timestep to get n_points for pre-allocation
    _, _, coords, first_result = _read_one_timestep(
        (0, timesteps[0][0], timesteps[0][1], data_dir)
    )
    n_points = coords.shape[0]

    # Pre-allocate arrays
    snapshots = {var: np.empty((n_steps, n_points), dtype=np.float64) for var in ALL_VARS}
    times = np.empty(n_steps, dtype=np.float64)

    # Fill first timestep
    for var in ALL_VARS:
        snapshots[var][0, :] = first_result[var]
    times[0] = timesteps[0][0]

    # Parallel read remaining timesteps
    tasks = [(i, t_val, t_name, data_dir)
             for i, (t_val, t_name) in enumerate(timesteps) if i > 0]
    completed = 1

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_read_one_timestep, task): task for task in tasks}
        for future in as_completed(futures):
            idx, t_val, _, result = future.result()
            for var in ALL_VARS:
                snapshots[var][idx, :] = result[var]
            times[idx] = t_val
            completed += 1

            if completed % 100 == 0 or completed == n_steps:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                remaining = (n_steps - completed) / rate
                print(f"  [{completed}/{n_steps}] "
                      f"({rate:.1f} steps/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t_start
    print(f"Data loading complete: {elapsed:.1f}s")
    print(f"  Points per timestep: {n_points}")
    print(f"  Timesteps loaded: {n_steps}")
    print(f"  Snapshot matrix shape: ({n_steps}, {n_points})")

    return coords, snapshots, times


# ============================================================
# POD decomposition
# ============================================================

def pod_decompose(snapshot_matrix, var_name, n_components=100):
    """Perform POD on a snapshot matrix using sklearn PCA.

    Args:
        snapshot_matrix: (N_timesteps, N_points) array
        var_name: string for logging
        n_components: number of modes to keep (default 100)

    Returns:
        mean_field : (N_points,)
        modes      : (N_modes, N_points) - spatial POD modes
        coeffs     : (N_timesteps, N_modes) - temporal coefficients
        singular   : (N_modes,) - singular values
        energy     : (N_modes,) - cumulative energy fraction
    """
    N_t, N_x = snapshot_matrix.shape
    n_components = min(n_components, N_t, N_x)
    print(f"\nPOD for '{var_name}': matrix shape ({N_t}, {N_x}), "
          f"keeping {n_components} components")

    t0 = time.time()
    pca = PCA(n_components=n_components)
    coeffs = pca.fit_transform(snapshot_matrix)     # (N_t, n_components)
    t_pca = time.time() - t0

    mean_field = pca.mean_                           # (N_x,)
    modes = pca.components_                          # (n_components, N_x)
    singular = pca.singular_values_                  # (n_components,)
    energy = np.cumsum(pca.explained_variance_ratio_)  # (n_components,)

    print(f"  PCA completed in {t_pca:.1f}s")
    print(f"  Number of modes: {len(singular)}")

    for threshold in [0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(energy, threshold)
        n_modes = idx + 1 if idx < len(energy) else len(energy)
        marker = "" if energy[min(idx, len(energy)-1)] >= threshold else " (not reached)"
        print(f"  {threshold*100:.1f}% energy: {n_modes} modes{marker}")

    return mean_field, modes, coeffs, singular, energy


# ============================================================
# Saving & plotting
# ============================================================

def save_results(output_dir, coords, times, pod_results):
    """Save all POD results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'coords.npy'), coords)
    np.save(os.path.join(output_dir, 'times.npy'), times)

    for var_name, (mean_f, modes, coeffs, singular, energy) in pod_results.items():
        prefix = var_name.replace('.', '_')
        np.save(os.path.join(output_dir, f'{prefix}_mean.npy'), mean_f)
        np.save(os.path.join(output_dir, f'{prefix}_modes.npy'), modes)
        np.save(os.path.join(output_dir, f'{prefix}_coeffs.npy'), coeffs)
        np.save(os.path.join(output_dir, f'{prefix}_singular.npy'), singular)
        np.save(os.path.join(output_dir, f'{prefix}_energy.npy'), energy)

    print(f"\nResults saved to {output_dir}")


def plot_energy_spectrum(pod_results, output_dir):
    """Plot cumulative energy spectrum for all variables."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for var_name, (_, _, _, singular, energy) in pod_results.items():
        ax.plot(range(1, len(energy)+1), energy, label=var_name, linewidth=1.5)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label='99%')
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Cumulative energy fraction')
    ax.set_title('POD Cumulative Energy Spectrum')
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(len(e) for _, _, _, _, e in pod_results.values()))

    ax.set_ylim(0.5, 1.005)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for var_name, (_, _, _, singular, _) in pod_results.items():
        normalized = singular / singular[0]
        ax.semilogy(range(1, len(normalized)+1), normalized, label=var_name, linewidth=1.5)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title('Singular Value Decay')
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(len(e) for _, _, _, _, e in pod_results.values()))

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'energy_spectrum.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Energy spectrum plot saved: {fig_path}")


def plot_first_modes(coords, pod_results, output_dir, n_modes=4):
    """Plot the first few spatial POD modes for each variable."""
    var_names = list(pod_results.keys())
    n_vars = len(var_names)

    fig, axes = plt.subplots(n_vars, n_modes, figsize=(4*n_modes, 3*n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    x = coords[:, 0]
    z = coords[:, 1]

    for row, var_name in enumerate(var_names):
        mean_f, modes, coeffs, singular, energy = pod_results[var_name]
        for col in range(n_modes):
            ax = axes[row, col]
            mode = modes[col, :]
            sc = ax.tricontourf(x, z, mode, levels=50, cmap='RdBu_r')
            plt.colorbar(sc, ax=ax, shrink=0.8)
            pct = (singular[col]**2 / np.sum(singular**2)) * 100
            ax.set_title(f'{var_name} mode {col+1}\n({pct:.1f}% energy)', fontsize=8)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)', fontsize=7)
            ax.set_ylabel('z (m)', fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'spatial_modes.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Spatial modes plot saved: {fig_path}")


def write_summary(pod_results, output_dir):
    """Write a text summary of truncation for each variable."""
    summary_path = os.path.join(output_dir, 'mode_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("POD Mode Summary\n")
        f.write("=" * 60 + "\n\n")
        for var_name, (_, _, _, singular, energy) in pod_results.items():
            f.write(f"Variable: {var_name}\n")
            f.write(f"  Total modes: {len(singular)}\n")
            for threshold in [0.90, 0.95, 0.99, 0.999]:
                n = np.searchsorted(energy, threshold) + 1
                f.write(f"  {threshold*100:.1f}% energy: {n} modes\n")
            f.write(f"  Top 5 singular values: {singular[:5]}\n")
            f.write(f"  Top 5 energy fractions: "
                    f"{(singular[:5]**2 / np.sum(singular**2) * 100)}\n\n")
    print(f"Summary saved: {summary_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='POD decomposition of 2D wave slices')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to postProcessing/sample directory')
    parser.add_argument('--output', type=str, default='./pod_results',
                        help='Output directory for results')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Limit number of timesteps (for testing)')
    parser.add_argument('--n_components', type=int, default=100,
                        help='Number of POD modes to keep (default: 100)')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel I/O threads (default: 8)')
    args = parser.parse_args()

    print("=" * 60)
    print("POD Decomposition for 2D Wave Slice")
    print("=" * 60)

    # Get timesteps
    timesteps = get_sorted_timesteps(args.data_dir)
    print(f"Found {len(timesteps)} timesteps: "
          f"t=[{timesteps[0][0]:.2f}, {timesteps[-1][0]:.2f}]")

    if args.max_steps:
        timesteps = timesteps[:args.max_steps]
        print(f"Limited to first {args.max_steps} timesteps")

    # Load data
    coords, snapshots, times = build_snapshot_matrices(
        args.data_dir, timesteps, n_workers=args.n_workers
    )

    # POD for each variable
    pod_results = {}
    for var in ALL_VARS:
        if var in snapshots:
            result = pod_decompose(snapshots[var], var, n_components=args.n_components)
            pod_results[var] = result

    # Save
    save_results(args.output, coords, times, pod_results)

    # Plot
    plot_energy_spectrum(pod_results, args.output)
    plot_first_modes(coords, pod_results, args.output)
    write_summary(pod_results, args.output)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()