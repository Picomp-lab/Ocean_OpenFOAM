"""
Precompute Graph Laplacian Eigenbasis from OpenFOAM Mesh Connectivity
=====================================================================

This script builds a graph Laplacian from OpenFOAM's owner/neighbour files,
restricts it to the cropped subdomain (via cellSet indices), and computes
the top-k eigenvectors for use as spectral basis in HPM.

Usage:
    python compute_lbo_eigenbasis.py \
        --case /path/to/openfoam/case \
        --cellset subdomainCells \
        --coords /path/to/coords.npy \
        --k 64 \
        --output /path/to/output_dir

Output:
    output_dir/
        lbo_eigenvectors.npy   # (N_crop, k) float32 - eigenvectors
        lbo_eigenvalues.npy    # (k,) float32 - eigenvalues
        laplacian_info.txt     # metadata

Notes:
    - owner/neighbour define which two cells share each internal face
    - Graph Laplacian L = D - W, where W is the adjacency/weight matrix
    - We use face-area weighting when available, otherwise binary adjacency
    - eigsh solves the generalized eigenvalue problem for the smallest
      non-trivial eigenvalues (skip the constant mode lambda_0 = 0)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import argparse
import time
import os


# ============================================================
# OpenFOAM File Parsers
# ============================================================

def read_label_list(filepath):
    """Read OpenFOAM label list file (owner, neighbour, cellSet).
    
    Handles both formats:
      - Plain list: one integer per line between ( and )
      - Compact binary/ascii with header specifying count
    
    Returns: numpy int64 array
    """
    with open(filepath, 'r') as f:
        text = f.read()

    # Find the data block between ( and )
    start = text.index('(') + 1
    end = text.rindex(')')
    block = text[start:end]

    # Parse all integers
    values = np.fromstring(block.replace('\n', ' '), dtype=np.int64, sep=' ')
    return values


def read_cell_set(filepath):
    """Read OpenFOAM cellSet file -> sorted integer array of cell indices."""
    values = read_label_list(filepath)
    values.sort()
    return values


# ============================================================
# Graph Laplacian Construction
# ============================================================

def build_cropped_graph_laplacian(owner, neighbour, cell_indices, coords=None):
    """
    Build graph Laplacian restricted to cropped subdomain.

    Steps:
      1. Filter internal faces: keep only faces where BOTH owner and 
         neighbour are in cell_indices
      2. Re-index cell IDs from global (0..N_total-1) to local (0..N_crop-1)
      3. Build sparse adjacency matrix W
      4. Compute degree matrix D and Laplacian L = D - W

    Parameters:
        owner:        (n_internal_faces,) int array - owner cell of each face
        neighbour:    (n_internal_faces,) int array - neighbour cell of each face
        cell_indices: (N_crop,) sorted int array - global cell indices in subdomain
        coords:       (N_crop, 3) optional - cell coordinates for distance weighting

    Returns:
        L: (N_crop, N_crop) sparse CSC matrix - graph Laplacian
        D: (N_crop, N_crop) sparse diagonal - degree matrix
        W: (N_crop, N_crop) sparse CSC matrix - adjacency/weight matrix
    """
    N_crop = len(cell_indices)
    print(f"  Building graph for {N_crop} cells...")

    # --- Step 1: Build global-to-local index mapping ---
    t0 = time.time()
    # Use a lookup array for O(1) mapping (memory-intensive but fast)
    max_global_id = cell_indices[-1] + 1  # cell_indices is sorted
    global_to_local = np.full(max_global_id, -1, dtype=np.int64)
    global_to_local[cell_indices] = np.arange(N_crop, dtype=np.int64)
    print(f"    Global-to-local mapping built ({time.time()-t0:.1f}s)")

    # --- Step 2: Filter faces - keep only internal subdomain faces ---
    t0 = time.time()
    # A face is internal to the subdomain if both its owner and neighbour
    # are in cell_indices. Vectorized: clamp indices to valid range for
    # lookup, then check the lookup result.
    o_clamped = np.minimum(owner, max_global_id - 1)
    n_clamped = np.minimum(neighbour, max_global_id - 1)
    owner_in = (owner < max_global_id) & (global_to_local[o_clamped] >= 0)
    neighbour_in = (neighbour < max_global_id) & (global_to_local[n_clamped] >= 0)
    mask = owner_in & neighbour_in

    n_internal_faces = mask.sum()
    print(f"    Filtered {n_internal_faces} internal faces from {len(owner)} total ({time.time()-t0:.1f}s)")

    # Get local indices for the kept faces
    kept_owner = global_to_local[owner[mask]]
    kept_neighbour = global_to_local[neighbour[mask]]

    # --- Step 3: Compute edge weights ---
    t0 = time.time()
    if coords is not None:
        # Distance-based weighting: w_ij = 1 / ||x_i - x_j||
        # This approximates the FVM flux coupling strength
        dx = coords[kept_owner] - coords[kept_neighbour]
        distances = np.sqrt(np.sum(dx**2, axis=1))
        distances = np.maximum(distances, 1e-10)  # avoid division by zero
        weights = 1.0 / distances
        print(f"    Distance-based weights computed ({time.time()-t0:.1f}s)")
    else:
        # Binary adjacency (unweighted)
        weights = np.ones(n_internal_faces, dtype=np.float64)
        print(f"    Using binary weights ({time.time()-t0:.1f}s)")

    # --- Step 4: Build symmetric adjacency matrix W ---
    t0 = time.time()
    # Each internal face contributes two entries (symmetric)
    rows = np.concatenate([kept_owner, kept_neighbour])
    cols = np.concatenate([kept_neighbour, kept_owner])
    data = np.concatenate([weights, weights])

    W = sp.csc_matrix((data, (rows, cols)), shape=(N_crop, N_crop))
    print(f"    Adjacency matrix built: {W.nnz} non-zeros ({time.time()-t0:.1f}s)")

    # --- Step 5: Compute Laplacian L = D - W ---
    t0 = time.time()
    degree = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(degree, format='csc')
    L = D - W
    print(f"    Laplacian computed ({time.time()-t0:.1f}s)")

    # Sanity checks
    print(f"    Degree stats: min={degree.min():.2f}, max={degree.max():.2f}, "
          f"mean={degree.mean():.2f}, median={np.median(degree):.2f}")
    n_isolated = (degree == 0).sum()
    if n_isolated > 0:
        print(f"    WARNING: {n_isolated} isolated cells (degree=0)!")

    return L, D, W


# ============================================================
# Eigenvalue Computation
# ============================================================

def compute_eigenbasis(L, k, method='eigsh'):
    """
    Compute the first k non-trivial eigenvectors of the graph Laplacian.

    Uses LU-based shift-invert mode: factor (L - sigma*I) once via splu,
    then each ARPACK iteration is just a triangular solve — O(nnz) per step.

    Automatically detects and discards all zero modes (one per connected
    component), so the returned k eigenvectors are always non-trivial.

    Parameters:
        L: (N, N) sparse matrix - graph Laplacian
        k: int - number of non-trivial eigenvectors to compute
        method: 'eigsh' (default) - scipy ARPACK

    Returns:
        eigenvalues:  (k,) float32 array
        eigenvectors: (N, k) float32 array
    """
    N = L.shape[0]
    print(f"\n  Computing {k} eigenvectors for matrix of size {N}x{N}...")
    print(f"  Method: LU-based shift-invert (ARPACK)")
    print(f"  This may take a while for large meshes...")

    # --- Step 1: Detect connected components to know how many zero modes ---
    t0 = time.time()
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(L, directed=False)
    print(f"  Connected components: {n_components}")
    if n_components > 1:
        comp_sizes = np.bincount(labels)
        print(f"    Component sizes: {sorted(comp_sizes, reverse=True)[:10]}"
              f"{'...' if n_components > 10 else ''}")
        print(f"    WARNING: {n_components} components means {n_components} zero modes.")
        print(f"    Will request {k + n_components} eigenpairs to get {k} non-trivial ones.")

    n_request = k + n_components  # request enough to cover all zero modes + k useful ones
    if n_request >= N:
        n_request = N - 1
        print(f"    Clamped request to {n_request} (matrix size {N})")

    # --- Step 2: LU-based shift-invert ---
    sigma = 1e-8
    print(f"  Computing LU decomposition of (L - {sigma}*I)...")
    t_lu = time.time()
    try:
        L_shifted = L.tocsc() - sigma * sp.eye(N, format='csc')
        lu = sla.splu(L_shifted)
        print(f"  LU done ({time.time()-t_lu:.1f}s)")

        op_inv = sla.LinearOperator(
            shape=L.shape, matvec=lu.solve, dtype=L.dtype
        )
        eigenvalues, eigenvectors = sla.eigsh(
            L, k=n_request, sigma=sigma, which='LM',
            OPinv=op_inv,
            tol=1e-6, maxiter=1000
        )
    except Exception as e:
        print(f"  Shift-invert failed: {e}")
        print(f"  Falling back to smallest-magnitude mode (slower, less stable)...")
        eigenvalues, eigenvectors = sla.eigsh(
            L, k=n_request, which='SM',
            tol=1e-6, maxiter=2000
        )

    elapsed = time.time() - t0
    print(f"  Eigendecomposition done ({elapsed:.1f}s)")

    # --- Step 3: Sort and discard all zero modes dynamically ---
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Identify zero modes: eigenvalues close to 0 (threshold relative to
    # first clearly non-zero eigenvalue)
    zero_threshold = 1e-6 * max(abs(eigenvalues[-1]), 1e-10)
    n_zero = np.sum(np.abs(eigenvalues) < zero_threshold)
    n_zero = max(n_zero, n_components)  # at least n_components zero modes

    print(f"  Eigenvalue summary:")
    print(f"    Zero modes detected: {n_zero} (threshold: {zero_threshold:.2e})")
    for i in range(min(n_zero + 2, len(eigenvalues))):
        tag = " (zero mode)" if i < n_zero else ""
        print(f"    λ_{i} = {eigenvalues[i]:.6e}{tag}")
    print(f"    λ_{len(eigenvalues)-1} = {eigenvalues[-1]:.6e} (largest computed)")

    # Discard zero modes
    eigenvalues = eigenvalues[n_zero:]
    eigenvectors = eigenvectors[:, n_zero:]

    # Take exactly k
    if len(eigenvalues) < k:
        print(f"  WARNING: Only {len(eigenvalues)} non-trivial eigenvectors available, "
              f"requested {k}. Returning all available.")
        k = len(eigenvalues)

    eigenvalues = eigenvalues[:k].astype(np.float32)
    eigenvectors = eigenvectors[:, :k].astype(np.float32)

    return eigenvalues, eigenvectors


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute Graph Laplacian Eigenbasis from OpenFOAM mesh'
    )
    parser.add_argument('--case', type=str, required=True,
                        help='Path to OpenFOAM case directory')
    parser.add_argument('--cellset', type=str, default='subdomainCells',
                        help='Name of the cellSet (default: subdomainCells)')
    parser.add_argument('--coords', type=str, default=None,
                        help='Path to coords.npy (for distance-based weighting)')
    parser.add_argument('--k', type=int, default=64,
                        help='Number of eigenvectors to compute (default: 64)')
    parser.add_argument('--no-distance-weight', action='store_true',
                        help='Use binary adjacency instead of distance weighting')
    parser.add_argument('--output', type=str, default='./lbo_output',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- Step 1: Read OpenFOAM mesh connectivity ----
    print("=" * 60)
    print("Step 1: Reading OpenFOAM mesh files")
    print("=" * 60)

    mesh_dir = os.path.join(args.case, 'constant', 'polyMesh')

    t0 = time.time()
    print(f"  Reading owner...")
    owner = read_label_list(os.path.join(mesh_dir, 'owner'))
    print(f"    {len(owner)} internal faces ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print(f"  Reading neighbour...")
    neighbour = read_label_list(os.path.join(mesh_dir, 'neighbour'))
    print(f"    {len(neighbour)} entries ({time.time()-t0:.1f}s)")

    # Sanity: neighbour should be same length as owner for internal faces
    # Actually in OpenFOAM, len(owner) >= len(neighbour) because boundary
    # faces have an owner but no neighbour. We only use the first
    # len(neighbour) faces (internal faces).
    n_internal = len(neighbour)
    owner = owner[:n_internal]
    print(f"  Using {n_internal} internal faces")

    # ---- Step 2: Read cellSet ----
    print(f"\n{'=' * 60}")
    print("Step 2: Reading cellSet indices")
    print("=" * 60)

    cellset_path = os.path.join(mesh_dir, 'sets', args.cellset)
    t0 = time.time()
    cell_indices = read_cell_set(cellset_path)
    print(f"  {len(cell_indices)} cells in set '{args.cellset}' ({time.time()-t0:.1f}s)")
    print(f"  Index range: [{cell_indices[0]}, {cell_indices[-1]}]")

    # ---- Step 3: Load coordinates (optional, for distance weighting) ----
    coords = None
    if args.coords and not args.no_distance_weight:
        print(f"\n{'=' * 60}")
        print("Step 3: Loading coordinates for distance weighting")
        print("=" * 60)
        t0 = time.time()
        coords = np.load(args.coords).astype(np.float64)
        print(f"  Loaded coords: shape {coords.shape} ({time.time()-t0:.1f}s)")
        assert len(coords) == len(cell_indices), \
            f"coords ({len(coords)}) != cellSet ({len(cell_indices)})"
    else:
        print(f"\n  Skipping coordinate loading (binary adjacency mode)")

    # ---- Step 4: Build Graph Laplacian ----
    print(f"\n{'=' * 60}")
    print("Step 4: Building Graph Laplacian")
    print("=" * 60)

    L, D, W = build_cropped_graph_laplacian(owner, neighbour, cell_indices, coords)

    # ---- Step 5: Compute Eigenbasis ----
    print(f"\n{'=' * 60}")
    print(f"Step 5: Computing top-{args.k} eigenvectors")
    print("=" * 60)

    eigenvalues, eigenvectors = compute_eigenbasis(L, args.k)

    # ---- Step 6: Save results ----
    print(f"\n{'=' * 60}")
    print("Step 6: Saving results")
    print("=" * 60)

    ev_path = os.path.join(args.output, 'lbo_eigenvectors.npy')
    el_path = os.path.join(args.output, 'lbo_eigenvalues.npy')
    np.save(ev_path, eigenvectors)
    np.save(el_path, eigenvalues)
    print(f"  Saved eigenvectors: {ev_path} — shape {eigenvectors.shape}")
    print(f"  Saved eigenvalues:  {el_path} — shape {eigenvalues.shape}")

    # Save metadata
    info_path = os.path.join(args.output, 'laplacian_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"OpenFOAM case: {os.path.abspath(args.case)}\n")
        f.write(f"cellSet: {args.cellset}\n")
        f.write(f"N_crop: {len(cell_indices)}\n")
        f.write(f"N_internal_faces (total): {n_internal}\n")
        f.write(f"N_internal_faces (cropped): {W.nnz // 2}\n")
        f.write(f"k (eigenvectors): {args.k}\n")
        f.write(f"Distance weighting: {coords is not None}\n")
        f.write(f"Eigenvalue range: [{eigenvalues[0]:.6e}, {eigenvalues[-1]:.6e}]\n")
        f.write(f"Eigenvector shape: {eigenvectors.shape}\n")
    print(f"  Saved metadata:     {info_path}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
