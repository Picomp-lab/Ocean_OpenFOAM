"""
prepare_data.py — Crop, interpolate to regular grid, generate terrain mask.

Usage:
    python prepare_data.py                          # use defaults from configs/default.yaml
    python prepare_data.py data.nx=1024 data.nz=64  # override from CLI
"""

import os
import numpy as np
from scipy.spatial import Delaunay
import hydra
from omegaconf import DictConfig, OmegaConf


def load_raw(raw_dir: str):
    coords = np.load(os.path.join(raw_dir, "coords_2d.npy"))   # (N, 2)
    fields = np.load(os.path.join(raw_dir, "fields.npy"))       # (T, N, 5)
    times  = np.load(os.path.join(raw_dir, "times.npy"))        # (T,)
    print(f"Raw data: coords {coords.shape}, fields {fields.shape}, times {times.shape}")
    return coords, fields, times


def select_time_indices(times, t_start, t_end):
    """Return indices for ALL timesteps in [t_start, t_end]."""
    mask = (times >= t_start - 1e-6) & (times <= t_end + 1e-6)
    selected = np.where(mask)[0]
    dt = times[selected[1]] - times[selected[0]]
    print(f"Time selection: {len(selected)} snapshots "
          f"from t={times[selected[0]]:.2f}s to t={times[selected[-1]]:.2f}s, "
          f"dt={dt:.3f}s")
    return selected


def crop_domain(coords, x_min, x_max, z_min, z_max):
    """Return boolean mask for points inside the cropped domain."""
    mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= z_min) & (coords[:, 1] <= z_max)
    )
    print(f"Cropped domain: {mask.sum()} / {len(coords)} points "
          f"in x=[{x_min}, {x_max}], z=[{z_min}, {z_max}]")
    return mask


def build_regular_grid(x_min, x_max, z_min, z_max, nx, nz):
    """Create regular meshgrid for interpolation."""
    x_lin = np.linspace(x_min, x_max, nx)
    z_lin = np.linspace(z_min, z_max, nz)
    xx, zz = np.meshgrid(x_lin, z_lin, indexing="ij")  # (nx, nz)
    print(f"Regular grid: {nx} x {nz} = {nx * nz} points, "
          f"dx={x_lin[1]-x_lin[0]:.4f}m, dz={z_lin[1]-z_lin[0]:.4f}m")
    return xx, zz, x_lin, z_lin


def compute_terrain_mask(coords_cropped, xx, zz):
    """
    Build a binary mask: 1 = inside fluid domain, 0 = outside (below seabed).

    Uses Delaunay triangulation of the original mesh points to determine
    which regular grid points fall inside the computational domain.
    """
    tri = Delaunay(coords_cropped)
    grid_pts = np.column_stack([xx.ravel(), zz.ravel()])
    inside = tri.find_simplex(grid_pts) >= 0
    mask = inside.reshape(xx.shape).astype(np.float32)
    n_inside = mask.sum()
    n_total = mask.size
    print(f"Terrain mask: {int(n_inside)} / {n_total} grid points inside domain "
          f"({n_inside/n_total*100:.1f}%)")
    return mask


def build_interpolator(coords_cropped, xx, zz):
    """
    Pre-compute Delaunay triangulation and simplex indices for fast interpolation.
    This is the expensive step — done once, then reused for all snapshots/channels.

    Returns:
        tri: Delaunay triangulation
        simplex_indices: (N_grid,) which simplex each grid point falls in (-1 = outside)
        vertices: (N_grid, 3) vertex indices of the containing simplex
        weights: (N_grid, 3) barycentric weights
    """
    print("Building Delaunay triangulation (one-time cost) ...")
    tri = Delaunay(coords_cropped)

    grid_pts = np.column_stack([xx.ravel(), zz.ravel()])
    simplex_indices = tri.find_simplex(grid_pts)

    # Pre-compute barycentric coordinates for all grid points
    N = len(grid_pts)
    vertices = np.zeros((N, 3), dtype=np.int32)
    weights = np.zeros((N, 3), dtype=np.float64)

    inside = simplex_indices >= 0
    simplices = tri.simplices[simplex_indices[inside]]     # (M, 3) vertex indices
    transforms = tri.transform[simplex_indices[inside]]    # (M, 3, 3)
    pts = grid_pts[inside]

    # Barycentric: b = T @ (pt - r3)
    delta = pts - transforms[:, 2]                          # (M, 2)
    bary = np.einsum('ijk,ik->ij', transforms[:, :2], delta)  # (M, 2)
    bary3 = np.column_stack([bary, 1.0 - bary.sum(axis=1)])   # (M, 3)

    vertices[inside] = simplices
    weights[inside] = bary3

    n_inside = inside.sum()
    print(f"  {n_inside} / {N} grid points inside triangulation "
          f"({n_inside/N*100:.1f}%)")

    return simplex_indices, vertices, weights


def interpolate_fast(field_vals, simplex_indices, vertices, weights, shape):
    """
    Fast interpolation using pre-computed barycentric weights.

    Args:
        field_vals: (M,) field values at source points
        simplex_indices: (N_grid,) simplex index per grid point
        vertices: (N_grid, 3) vertex indices
        weights: (N_grid, 3) barycentric weights
        shape: (nx, nz) output shape

    Returns:
        (nx, nz) interpolated field
    """
    inside = simplex_indices >= 0
    result = np.zeros(len(simplex_indices), dtype=np.float32)

    # Weighted sum of vertex values
    v_vals = field_vals[vertices[inside]]  # (M, 3)
    result[inside] = (v_vals * weights[inside]).sum(axis=1)

    return result.reshape(shape)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg.data))

    # --- Load raw data ---
    coords, fields, times = load_raw(cfg.data.raw_dir)

    # --- Time selection ---
    time_idx = select_time_indices(times, cfg.data.t_start, cfg.data.t_end)
    selected_times = times[time_idx]

    # --- Spatial crop ---
    crop_mask = crop_domain(coords, cfg.data.x_min, cfg.data.x_max, cfg.data.z_min, cfg.data.z_max)
    coords_crop = coords[crop_mask]

    # --- Build regular grid ---
    xx, zz, x_lin, z_lin = build_regular_grid(
        cfg.data.x_min, cfg.data.x_max, cfg.data.z_min, cfg.data.z_max,
        cfg.data.nx, cfg.data.nz
    )

    # --- Terrain mask ---
    terrain_mask = compute_terrain_mask(coords_crop, xx, zz)

    # --- Pre-compute interpolation weights (one-time) ---
    simplex_indices, vertices, weights = build_interpolator(coords_crop, xx, zz)

    # --- Interpolate fields ---
    channels = list(cfg.data.input_channels)  # [0, 1, 2] = alpha, Ux, Uz
    n_times = len(time_idx)
    n_ch = len(channels)
    nx, nz = cfg.data.nx, cfg.data.nz
    grid_shape = (nx, nz)

    # Output array: (T, C, nx, nz)
    data_grid = np.zeros((n_times, n_ch, nx, nz), dtype=np.float32)

    print(f"\nInterpolating {n_times} snapshots x {n_ch} channels ...")
    import time as timer
    t0 = timer.time()
    for i, tidx in enumerate(time_idx):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = timer.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_times - i - 1) / rate if rate > 0 else 0
            print(f"  snapshot {i+1}/{n_times} (t={times[tidx]:.2f}s) "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
        for j, ch in enumerate(channels):
            vals = fields[tidx, crop_mask, ch]
            data_grid[i, j] = interpolate_fast(
                vals, simplex_indices, vertices, weights, grid_shape
            )
        # Apply terrain mask: set outside-domain values to 0
        data_grid[i] *= terrain_mask[np.newaxis, :, :]

    # Clip alpha to [0, 1] (interpolation overshoot)
    data_grid[:, 0] = np.clip(data_grid[:, 0], 0.0, 1.0)

    # --- Split train / test ---
    train_mask = selected_times < cfg.data.t_split
    test_mask = selected_times >= cfg.data.t_split
    train_data = data_grid[train_mask]
    test_data = data_grid[test_mask]
    train_times = selected_times[train_mask]
    test_times = selected_times[test_mask]

    print(f"\nTrain: {len(train_data)} snapshots [{train_times[0]:.1f}s - {train_times[-1]:.1f}s]")
    print(f"Test:  {len(test_data)} snapshots [{test_times[0]:.1f}s - {test_times[-1]:.1f}s]")

    # --- Save ---
    out_dir = cfg.data.processed_dir
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "train_data.npy"), train_data)
    np.save(os.path.join(out_dir, "test_data.npy"), test_data)
    np.save(os.path.join(out_dir, "train_times.npy"), train_times)
    np.save(os.path.join(out_dir, "test_times.npy"), test_times)
    np.save(os.path.join(out_dir, "terrain_mask.npy"), terrain_mask)
    np.save(os.path.join(out_dir, "grid_x.npy"), x_lin)
    np.save(os.path.join(out_dir, "grid_z.npy"), z_lin)

    print(f"\nSaved to {out_dir}/")
    print(f"  train_data.npy:   {train_data.shape}  (T, C, nx, nz)")
    print(f"  test_data.npy:    {test_data.shape}")
    print(f"  terrain_mask.npy: {terrain_mask.shape}  (nx, nz)")
    print(f"  grid_x.npy:       {x_lin.shape}")
    print(f"  grid_z.npy:       {z_lin.shape}")


if __name__ == "__main__":
    main()