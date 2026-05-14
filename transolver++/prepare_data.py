"""
Data Preparation for Transolver++ (Polars-optimized)
=====================================================
Reads OpenFOAM postProcessing .raw files using Polars for fast I/O,
extracts 2D (x,z) coordinates and (alpha.water, Ux, Uz) fields.

Usage:
    python prepare_data.py \
        --raw_dir ~/ocean/data/y03 \
        --output  ~/ocean/models/transolver_pp/data
"""

import os
import argparse
import numpy as np
import polars as pl
from time import time


def read_scalar_raw(filepath):
    """Read scalar .raw file with Polars. Returns DataFrame."""
    return pl.read_csv(
        filepath,
        skip_rows=2,
        has_header=False,
        separator=' ',
        columns=[0, 1, 2, 3],
        new_columns=['x', 'y', 'z', 'val'],
        dtypes=[pl.Float64] * 4,
    )


def read_vector_raw(filepath):
    """Read vector .raw file with Polars. Returns DataFrame."""
    return pl.read_csv(
        filepath,
        skip_rows=2,
        has_header=False,
        separator=' ',
        columns=[0, 1, 2, 3, 4, 5],
        new_columns=['x', 'y', 'z', 'Ux', 'Uy', 'Uz'],
        dtypes=[pl.Float64] * 6,
    )


def get_sorted_timesteps(raw_dir, t_start, t_end):
    """Get sorted list of (time, path) tuples in range."""
    steps = []
    for name in os.listdir(raw_dir):
        try:
            t = float(name)
        except ValueError:
            continue
        if t_start <= t <= t_end:
            steps.append((t, os.path.join(raw_dir, name)))
    steps.sort(key=lambda x: x[0])
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--t_start', type=float, default=0.0)
    parser.add_argument('--t_end', type=float, default=50.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Get timesteps
    steps = get_sorted_timesteps(args.raw_dir, args.t_start, args.t_end)
    print(f"Found {len(steps)} timesteps in [{args.t_start}, {args.t_end}]s")

    if len(steps) == 0:
        print("ERROR: No timesteps found!")
        return

    # Read first timestep to get coords
    t0_start = time()
    t0, dir0 = steps[0]
    df_alpha0 = read_scalar_raw(os.path.join(dir0, 'alpha.water_ySlice.raw'))
    coords_2d = df_alpha0.select(['x', 'z']).to_numpy()
    N = coords_2d.shape[0]
    print(f"Points: {N}")
    print(f"x: [{coords_2d[:,0].min():.2f}, {coords_2d[:,0].max():.2f}]")
    print(f"z: [{coords_2d[:,1].min():.2f}, {coords_2d[:,1].max():.2f}]")
    print(f"First file read: {time()-t0_start:.2f}s")

    # Pre-allocate
    n_steps = len(steps)
    fields = np.zeros((n_steps, N, 5), dtype=np.float64)
    times = np.zeros(n_steps, dtype=np.float64)

    # Read all timesteps
    t_loop = time()
    for i, (t, d) in enumerate(steps):
        # Alpha
        df_a = read_scalar_raw(os.path.join(d, 'alpha.water_ySlice.raw'))
        fields[i, :, 0] = df_a['val'].to_numpy()

        # Velocity
        df_u = read_vector_raw(os.path.join(d, 'U_ySlice.raw'))
        fields[i, :, 1] = df_u['Ux'].to_numpy()
        fields[i, :, 2] = df_u['Uz'].to_numpy()

        # p_rgh
        df_p = read_scalar_raw(os.path.join(d, 'p_rgh_ySlice.raw'))
        fields[i, :, 3] = df_p['val'].to_numpy()

        # nut
        df_nut = read_scalar_raw(os.path.join(d, 'nut_ySlice.raw'))
        fields[i, :, 4] = df_nut['val'].to_numpy()


        times[i] = t

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time() - t_loop
            rate = (i + 1) / elapsed
            eta = (n_steps - i - 1) / rate
            print(f"  [{i+1}/{n_steps}] {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")

    total = time() - t_loop
    print(f"\nAll files read in {total:.1f}s ({total/n_steps:.3f}s per step)")

    # Save
    np.save(os.path.join(args.output, 'coords_2d.npy'), coords_2d)
    np.save(os.path.join(args.output, 'fields.npy'), fields)
    np.save(os.path.join(args.output, 'times.npy'), times)

    print(f"\nSaved to {args.output}/")
    print(f"  coords_2d.npy: {coords_2d.shape}")
    print(f"  fields.npy:    {fields.shape} ({fields.nbytes/1e9:.2f} GB)")
    print(f"  times.npy:     {times.shape}")

    for j, name in enumerate(['alpha', 'Ux', 'Uz', 'p_rgh', 'nut']):
        v = fields[:, :, j]
        print(f"  {name}: mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")


if __name__ == '__main__':
    main()