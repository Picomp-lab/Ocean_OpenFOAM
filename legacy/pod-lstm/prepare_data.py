"""
Prepare wave data for Transolver training.
============================================
Reads OpenFOAM postProcessing/sample raw files and packages them
into numpy arrays that the Transolver training script expects.

Usage:
    python prepare_data.py \
        --raw_dir $OCEAN_DATA/case/postProcessing/sample \
        --output  $OCEAN_DATA/transolver_data

Outputs:
    coords.npy        — (N, 2) float32, (x, z) for each point
    fields.npy        — (T, N, 3) float32, (alpha, Ux, Uz) per timestep
    times.npy         — (T,) float64, time values
    data_summary.txt  — human-readable summary

Notes:
    - Skips t=0 (initial condition, often uniform)
    - Verifies point consistency across all timesteps
    - Only keeps alpha.water, Ux, Uz (no p_rgh, consistent with POD-LSTM)
"""

import os
import argparse
import numpy as np
from glob import glob
import time as timer


def parse_raw_scalar(filepath):
    """Parse a scalar .raw file (alpha.water, nut, p_rgh).
    Returns coords (N,3) and values (N,)."""
    data = np.loadtxt(filepath, skiprows=2)
    coords = data[:, :3]   # x, y, z
    values = data[:, 3]    # scalar value
    return coords, values


def parse_raw_vector(filepath):
    """Parse a vector .raw file (U).
    Returns coords (N,3) and values (N,3) for Ux, Uy, Uz."""
    data = np.loadtxt(filepath, skiprows=2)
    coords = data[:, :3]   # x, y, z
    values = data[:, 3:6]  # Ux, Uy, Uz
    return coords, values


def get_timestep_dirs(raw_dir):
    """Get all timestep directories sorted by time value."""
    dirs = []
    for name in os.listdir(raw_dir):
        path = os.path.join(raw_dir, name)
        if os.path.isdir(path):
            try:
                t = float(name)
                dirs.append((t, name, path))
            except ValueError:
                continue
    dirs.sort(key=lambda x: x[0])
    return dirs


def main():
    parser = argparse.ArgumentParser(description='Prepare wave data for Transolver')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to postProcessing/sample directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--skip_zero', action='store_true', default=True,
                        help='Skip t=0 timestep')
    parser.add_argument('--dt_filter', type=float, default=None,
                        help='Only keep timesteps at this interval (e.g., 0.1 to halve data)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Discover timestep directories
    print("Scanning timestep directories...")
    timestep_dirs = get_timestep_dirs(args.raw_dir)
    print(f"  Found {len(timestep_dirs)} timestep directories")

    if args.skip_zero:
        timestep_dirs = [(t, n, p) for t, n, p in timestep_dirs if t > 0]
        print(f"  After skipping t=0: {len(timestep_dirs)} timesteps")

    if args.dt_filter is not None:
        filtered = []
        for t, n, p in timestep_dirs:
            if abs(t % args.dt_filter) < 1e-6 or abs(t % args.dt_filter - args.dt_filter) < 1e-6:
                filtered.append((t, n, p))
        timestep_dirs = filtered
        print(f"  After dt_filter={args.dt_filter}: {len(timestep_dirs)} timesteps")

    n_timesteps = len(timestep_dirs)

    # Read first timestep to establish reference coordinates
    t0, name0, path0 = timestep_dirs[0]
    print(f"\nReading reference timestep t={t0} ({name0})...")

    alpha_file = os.path.join(path0, 'alpha.water_ySlice.raw')
    U_file = os.path.join(path0, 'U_ySlice.raw')

    coords_alpha, alpha_vals = parse_raw_scalar(alpha_file)
    coords_U, U_vals = parse_raw_vector(U_file)

    # Verify coordinates match between alpha and U files
    assert np.allclose(coords_alpha, coords_U, atol=1e-6), \
        "Coordinate mismatch between alpha.water and U files!"

    n_points = coords_alpha.shape[0]
    ref_coords_xz = coords_alpha[:, [0, 2]]  # keep only (x, z), drop y=0.3

    print(f"  N_points = {n_points}")
    print(f"  x range: [{ref_coords_xz[:, 0].min():.3f}, {ref_coords_xz[:, 0].max():.3f}]")
    print(f"  z range: [{ref_coords_xz[:, 1].min():.3f}, {ref_coords_xz[:, 1].max():.3f}]")

    # Pre-allocate arrays
    fields = np.zeros((n_timesteps, n_points, 3), dtype=np.float32)  # alpha, Ux, Uz
    times = np.zeros(n_timesteps, dtype=np.float64)

    # Fill first timestep
    fields[0, :, 0] = alpha_vals
    fields[0, :, 1] = U_vals[:, 0]  # Ux
    fields[0, :, 2] = U_vals[:, 2]  # Uz (skip Uy)
    times[0] = t0

    # Process remaining timesteps
    start = timer.time()
    mismatches = 0

    for i, (t, name, path) in enumerate(timestep_dirs[1:], start=1):
        if i % 100 == 0:
            elapsed = timer.time() - start
            rate = i / elapsed
            eta = (n_timesteps - i) / rate
            print(f"  Processing t={t:.2f} ({i+1}/{n_timesteps}), "
                  f"rate={rate:.1f} steps/s, ETA={eta:.0f}s")

        alpha_file = os.path.join(path, 'alpha.water_ySlice.raw')
        U_file = os.path.join(path, 'U_ySlice.raw')

        try:
            coords_a, alpha_v = parse_raw_scalar(alpha_file)
            coords_u, U_v = parse_raw_vector(U_file)
        except Exception as e:
            print(f"  WARNING: Failed to read t={t}: {e}")
            mismatches += 1
            continue

        # Verify point count
        if coords_a.shape[0] != n_points:
            print(f"  WARNING: Point count mismatch at t={t}: "
                  f"expected {n_points}, got {coords_a.shape[0]}")
            mismatches += 1
            continue

        # Verify coordinates haven't changed (spot check first & last 100 points)
        if not np.allclose(coords_a[:100, [0, 2]], ref_coords_xz[:100], atol=1e-4):
            print(f"  WARNING: Coordinate mismatch at t={t}")
            mismatches += 1

        fields[i, :, 0] = alpha_v
        fields[i, :, 1] = U_v[:, 0]  # Ux
        fields[i, :, 2] = U_v[:, 2]  # Uz
        times[i] = t

    elapsed = timer.time() - start
    print(f"\nProcessed {n_timesteps} timesteps in {elapsed:.1f}s")
    if mismatches > 0:
        print(f"  WARNING: {mismatches} timesteps had issues")

    # Save
    coords_path = os.path.join(args.output, 'coords.npy')
    fields_path = os.path.join(args.output, 'fields.npy')
    times_path = os.path.join(args.output, 'times.npy')

    np.save(coords_path, ref_coords_xz.astype(np.float32))
    np.save(fields_path, fields)
    np.save(times_path, times)

    print(f"\nSaved:")
    print(f"  coords.npy  : {ref_coords_xz.shape} ({os.path.getsize(coords_path)/1e6:.1f} MB)")
    print(f"  fields.npy  : {fields.shape} ({os.path.getsize(fields_path)/1e6:.1f} MB)")
    print(f"  times.npy   : {times.shape}")

    # Field statistics
    print(f"\nField statistics:")
    var_names = ['alpha.water', 'Ux', 'Uz']
    for j, vname in enumerate(var_names):
        vals = fields[:, :, j]
        print(f"  {vname:15s}: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Write summary
    summary_path = os.path.join(args.output, 'data_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Wave Data for Transolver\n")
        f.write(f"========================\n")
        f.write(f"Source: {args.raw_dir}\n")
        f.write(f"N_points: {n_points}\n")
        f.write(f"N_timesteps: {n_timesteps}\n")
        f.write(f"Time range: [{times[0]:.2f}, {times[-1]:.2f}]\n")
        f.write(f"dt: {times[1]-times[0]:.4f}\n")
        f.write(f"x range: [{ref_coords_xz[:, 0].min():.3f}, {ref_coords_xz[:, 0].max():.3f}]\n")
        f.write(f"z range: [{ref_coords_xz[:, 1].min():.3f}, {ref_coords_xz[:, 1].max():.3f}]\n")
        f.write(f"Fields: alpha.water, Ux, Uz\n")
        f.write(f"coords.npy shape: {ref_coords_xz.shape}\n")
        f.write(f"fields.npy shape: {fields.shape}\n")
        for j, vname in enumerate(var_names):
            vals = fields[:, :, j]
            f.write(f"{vname}: min={vals.min():.4f}, max={vals.max():.4f}, "
                    f"mean={vals.mean():.4f}, std={vals.std():.4f}\n")

    print(f"\nDone! Data ready for Transolver training.")
    print(f"  Total disk usage: {sum(os.path.getsize(os.path.join(args.output, f)) for f in ['coords.npy','fields.npy','times.npy'])/1e9:.2f} GB")


if __name__ == '__main__':
    main()