"""
Crop OpenFOAM field data using a cellSet and save as chunked .npy files.

Output structure:
  output_dir/
    coords.npy              # (N_crop, 3) float32 - cell centers
    chunk_000_data.npy      # (100, N_crop, 6) float32
    chunk_000_times.npy     # (100,) float64
    chunk_001_data.npy
    ...

Channel order: [alpha.water, Ux, Uy, Uz, p_rgh, nut]
"""

import numpy as np
from pathlib import Path
import time
import argparse


def read_cell_set(path):
    """Read OpenFOAM cellSet file -> sorted integer array of cell indices."""
    with open(path, "r") as f:
        text = f.read()
    start = text.index("(") + 1
    end = text.rindex(")")
    indices = np.array([int(x) for x in text[start:end].split()], dtype=np.int64)
    indices.sort()
    return indices


def read_scalar_field(filepath, n_cells_total):
    """Read OpenFOAM volScalarField -> (n_cells_total,) float64 array."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find internalField
    for i, line in enumerate(lines):
        if "internalField" in line:
            break
    else:
        raise ValueError(f"No internalField found in {filepath}")

    field_line = lines[i].strip()

    # uniform case
    if "uniform" in field_line and "nonuniform" not in field_line:
        val = float(field_line.split()[-1].rstrip(";"))
        return np.full(n_cells_total, val, dtype=np.float64)

    # nonuniform List<scalar>
    # Next line(s): count, then '('
    # Count might be on the same line or the next
    j = i + 1
    # Skip until we find the count
    while j < len(lines):
        stripped = lines[j].strip()
        if stripped.isdigit():
            count = int(stripped)
            j += 1
            break
        elif stripped == "":
            j += 1
        else:
            # Count might be at the end of the internalField line
            # Try parsing from field_line
            parts = field_line.split()
            count = int(parts[-1])
            break
    else:
        raise ValueError(f"Could not find field count in {filepath}")

    # Skip '('
    while lines[j].strip() != "(":
        j += 1
    j += 1

    # Read values in bulk
    data_text = "\n".join(lines[j : j + count])
    data = np.fromstring(data_text, sep="\n", dtype=np.float64)

    return data


def read_vector_field(filepath, n_cells_total):
    """Read OpenFOAM volVectorField -> (n_cells_total, 3) float64 array."""
    with open(filepath, "r") as f:
        text = f.read()

    # Find internalField
    idx = text.index("internalField")
    field_section = text[idx:]

    # uniform case
    if "uniform" in field_section.split("\n")[0] and "nonuniform" not in field_section.split("\n")[0]:
        first_line = field_section.split("\n")[0]
        # Extract (x y z)
        paren_start = first_line.index("(")
        paren_end = first_line.index(")")
        vals = [float(v) for v in first_line[paren_start + 1 : paren_end].split()]
        return np.tile(vals, (n_cells_total, 1))

    # nonuniform List<vector>
    # Find count before the big '('
    lines = field_section.split("\n")
    j = 1
    count = None
    while j < len(lines):
        stripped = lines[j].strip()
        if stripped.isdigit():
            count = int(stripped)
            j += 1
            break
        elif "nonuniform" in lines[0]:
            # count might be on the next line after 'nonuniform List<vector>'
            parts = lines[0].split()
            for p in parts:
                if p.isdigit():
                    count = int(p)
                    break
            if count is not None:
                break
        j += 1

    if count is None:
        raise ValueError(f"Could not find vector field count in {filepath}")

    # Find the opening '(' for data block
    while lines[j].strip() != "(":
        j += 1
    j += 1

    # Read (x y z) lines in bulk
    block = "\n".join(lines[j : j + count])
    block = block.replace("(", "").replace(")", "")
    data = np.fromstring(block, sep=" ", dtype=np.float64).reshape(-1, 3)

    return data


def get_time_dirs(case_dir):
    """Get sorted list of time step directories (excluding 0)."""
    time_dirs = []
    for d in case_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        try:
            t = float(name)
            if t > 0:  # skip initial condition
                time_dirs.append((t, d))
        except ValueError:
            continue
    time_dirs.sort(key=lambda x: x[0])
    return time_dirs


def main():
    parser = argparse.ArgumentParser(description="Crop OpenFOAM fields using cellSet")
    parser.add_argument("--case", type=str, required=True, help="Path to OpenFOAM case directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .npz files")
    parser.add_argument("--cellset", type=str, default="subdomainCells",
                        help="Name of cellSet in constant/polyMesh/sets/")
    parser.add_argument("--chunk-size", type=int, default=100, help="Time steps per chunk file")
    parser.add_argument("--t-start", type=float, default=None, help="Start time (inclusive)")
    parser.add_argument("--t-end", type=float, default=None, help="End time (inclusive)")
    args = parser.parse_args()

    case_dir = Path(args.case)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cells_total = 9398667  # from your mesh

    # --- 1. Read cellSet indices ---
    print("Reading cellSet...")
    cellset_path = case_dir / "constant/polyMesh/sets" / args.cellset
    indices = read_cell_set(cellset_path)
    n_crop = len(indices)
    print(f"  Cropped cells: {n_crop}")

    # --- 2. Read and crop coordinates ---
    print("Reading cell centres (C)...")
    c_path = case_dir / "0" / "C"
    coords_full = read_vector_field(str(c_path), n_cells_total)
    coords_crop = coords_full[indices].astype(np.float32)
    del coords_full
    np.save(output_dir / "coords.npy", coords_crop)
    print(f"  Saved coords.npy: {coords_crop.shape}")
    del coords_crop

    # --- 3. Get time directories ---
    time_dirs = get_time_dirs(case_dir)
    if args.t_start is not None:
        time_dirs = [(t, d) for t, d in time_dirs if t >= args.t_start]
    if args.t_end is not None:
        time_dirs = [(t, d) for t, d in time_dirs if t <= args.t_end]
    n_steps = len(time_dirs)
    print(f"  Time steps to process: {n_steps}")

    # --- 4. Process in chunks ---
    fields_scalar = ["alpha.water", "p_rgh", "nut"]
    fields_vector = ["U"]
    # Channel order: alpha.water, Ux, Uy, Uz, p_rgh, nut
    n_channels = 6

    chunk_idx = 0
    chunk_data = []
    chunk_times = []

    for step_i, (t, t_dir) in enumerate(time_dirs):
        t0 = time.time()

        frame = np.empty((n_crop, n_channels), dtype=np.float32)

        # alpha.water -> channel 0
        alpha = read_scalar_field(str(t_dir / "alpha.water"), n_cells_total)
        frame[:, 0] = alpha[indices].astype(np.float32)
        del alpha

        # U -> channels 1,2,3
        U = read_vector_field(str(t_dir / "U"), n_cells_total)
        frame[:, 1:4] = U[indices].astype(np.float32)
        del U

        # p_rgh -> channel 4
        p_rgh = read_scalar_field(str(t_dir / "p_rgh"), n_cells_total)
        frame[:, 4] = p_rgh[indices].astype(np.float32)
        del p_rgh

        # nut -> channel 5
        nut = read_scalar_field(str(t_dir / "nut"), n_cells_total)
        frame[:, 5] = nut[indices].astype(np.float32)
        del nut

        chunk_data.append(frame)
        chunk_times.append(t)

        elapsed = time.time() - t0
        print(f"  [{step_i+1}/{n_steps}] t={t:.4f} ({elapsed:.1f}s)")

        # Save chunk
        if len(chunk_data) == args.chunk_size or step_i == n_steps - 1:
            chunk_arr = np.stack(chunk_data, axis=0)  # (chunk_len, N_crop, 6)
            times_arr = np.array(chunk_times, dtype=np.float64)

            chunk_name_data = f"chunk_{chunk_idx:03d}_data.npy"
            chunk_name_times = f"chunk_{chunk_idx:03d}_times.npy"
            np.save(output_dir / chunk_name_data, chunk_arr)
            np.save(output_dir / chunk_name_times, times_arr)
            print(f"  -> Saved {chunk_name_data}: {chunk_arr.shape}")

            chunk_data.clear()
            chunk_times.clear()
            chunk_idx += 1

    print(f"\nDone. {chunk_idx} chunks saved to {output_dir}")


if __name__ == "__main__":
    main()