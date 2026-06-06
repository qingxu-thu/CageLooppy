from __future__ import annotations

import heapq
from collections.abc import Iterator

import numpy as np

from cagingloop.nearest import nn_prepare, nn_search
from cagingloop.types import DistanceMapResult, VoxelizationResult


def _validate_surface_id(voxelization: VoxelizationResult, point_id: int, name: str) -> None:
    if point_id < 0 or point_id >= len(voxelization.index.on_index):
        raise ValueError(f"{name} is outside the surface point range")


def _spacing(voxelization: VoxelizationResult) -> np.ndarray:
    def axis_spacing(axis: np.ndarray) -> float:
        return float(axis[1] - axis[0]) if len(axis) > 1 else 1.0

    return np.array(
        [
            axis_spacing(voxelization.grid_x),
            axis_spacing(voxelization.grid_y),
            axis_spacing(voxelization.grid_z),
        ],
        dtype=float,
    )


def _six_neighbors(index: tuple[int, int, int], shape: tuple[int, int, int]) -> Iterator[tuple[int, int, int]]:
    x, y, z = index
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            yield nx, ny, nz


def _dijkstra_distance(
    traversable: np.ndarray,
    source: tuple[int, int, int],
    spacing: np.ndarray,
) -> np.ndarray:
    distance = np.full(traversable.shape, np.inf, dtype=float)
    if not traversable[source]:
        return distance

    distance[source] = 0.0
    queue: list[tuple[float, tuple[int, int, int]]] = [(0.0, source)]
    step_cost = {
        (1, 0, 0): abs(float(spacing[0])),
        (-1, 0, 0): abs(float(spacing[0])),
        (0, 1, 0): abs(float(spacing[1])),
        (0, -1, 0): abs(float(spacing[1])),
        (0, 0, 1): abs(float(spacing[2])),
        (0, 0, -1): abs(float(spacing[2])),
    }

    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance > distance[current]:
            continue
        cx, cy, cz = current
        for neighbor in _six_neighbors(current, traversable.shape):
            if not traversable[neighbor]:
                continue
            nx, ny, nz = neighbor
            cost = step_cost[(nx - cx, ny - cy, nz - cz)]
            candidate = current_distance + cost
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distance


def _fmm_distance(
    traversable: np.ndarray,
    source: tuple[int, int, int],
    spacing: np.ndarray,
) -> np.ndarray:
    import skfmm

    phi = np.ones(traversable.shape, dtype=float)
    phi[source] = -1.0
    speed = traversable.astype(float)
    speed[~traversable] = 0.0
    distance = skfmm.travel_time(phi, speed, dx=tuple(np.abs(spacing)))
    return np.asarray(distance.filled(np.inf) if hasattr(distance, "filled") else distance, dtype=float)


# --- Multistencil Fast Marching (MSFM) -------------------------------------
# Faithful NumPy port of D. Kroon's msfm3d.c (Code/tool/FastMarching_version3b),
# the multistencil fast marching of Hassouna & Farag (2007) used by MATLAB's
# DistanceMapByFastMarching.m. Six stencils — the axis stencil plus three
# face-diagonal and two body-diagonal stencils — each solving the upwind Eikonal
# quadratic (2nd order where the 2-step neighbour is smaller); the smallest
# travel-time across stencils is kept. This removes the diagonal anisotropy that
# single-stencil FMM (skfmm) and especially Dijkstra leave behind.
#
# The C code's custom binary-tree narrow band is just a min-priority queue; we use
# heapq with lazy deletion (push on improvement, skip stale pops). MSFM is solved
# in voxel units (spacing = 1, as MATLAB does) and scaled by the mean voxel size
# at the end so its magnitude matches the other solvers (world units).

_MSFM_EPS = 2.2204460492503131e-16

# Direction vector of each of the 18 derivative slots (msfm3d.c Tm[0..17]).
# Sign is irrelevant (both ends are read), so aliased slots reuse a representative.
_MSFM_DIRS = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),            # stencil 0: axes
    (1, 0, 0), (0, 1, 1), (0, 1, -1),           # stencil 1: x + 2 face-diagonals
    (0, 1, 0), (1, 0, -1), (1, 0, 1),           # stencil 2: y + 2 face-diagonals
    (0, 0, 1), (1, 1, 0), (1, -1, 0),           # stencil 3: z + 2 face-diagonals
    (1, 0, 1), (1, 1, -1), (1, -1, -1),         # stencil 4: face-diag + 2 body-diagonals
    (1, 0, -1), (1, 1, 1), (1, -1, 1),          # stencil 5: face-diag + 2 body-diagonals
)
# First- and second-order stencil weights (1/h^2 and (3/2h)^2 = 9/4h^2); msfm3d.c G1,G2.
_MSFM_G1 = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5,
            0.5, 1.0 / 3.0, 1.0 / 3.0, 0.5, 1.0 / 3.0, 1.0 / 3.0)
_MSFM_G2 = tuple(2.25 * g for g in _MSFM_G1)


def _msfm_second_derivative(m1: float, m2: float, p1: float, p2: float) -> float:
    """msfm3d.c second_derivative: 2nd-order upwind value where the 2-step
    neighbour is smaller than the 1-step (else INF, meaning 'fall back to 1st order')."""
    ch1 = (m2 < m1) and np.isfinite(m1)
    ch2 = (p2 < p1) and np.isfinite(p1)
    if ch1 and ch2:
        return min((4.0 * m1 - m2) / 3.0, (4.0 * p1 - p2) / 3.0)
    if ch1:
        return (4.0 * m1 - m2) / 3.0
    if ch2:
        return (4.0 * p1 - p2) / 3.0
    return np.inf


def _msfm_roots_max(a: float, b: float, c: float) -> float:
    """Largest root of a*T^2 + b*T + c = 0 (common.c roots; a > 0 in our calls)."""
    d = b * b - 4.0 * a * c
    if d < 0.0:
        d = 0.0
    sq = d ** 0.5
    if a * a > _MSFM_EPS:
        return max((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a))
    # a ~ 0 (no neighbours) — degenerate; never reached once an axis neighbour exists
    return 0.0


def _msfm_point(
    T: np.ndarray,
    frozen: np.ndarray,
    shape: tuple[int, int, int],
    x: int,
    y: int,
    z: int,
    use_second: bool,
    use_cross: bool,
) -> float:
    """Travel-time update at (x,y,z) from frozen neighbours — msfm3d.c CalculateDistance
    with speed F = 1 (voxel units)."""
    nx, ny, nz = shape

    def rd(dx: int, dy: int, dz: int) -> float:
        i, j, k = x + dx, y + dy, z + dz
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and frozen[i, j, k]:
            return float(T[i, j, k])
        return np.inf

    n_slots = 18 if use_cross else 3
    Tm = [np.inf] * n_slots
    Tm2 = [0.0] * n_slots
    order = [0] * n_slots
    for t in range(n_slots):
        dx, dy, dz = _MSFM_DIRS[t]
        m1 = rd(-dx, -dy, -dz)
        p1 = rd(dx, dy, dz)
        v = m1 if m1 < p1 else p1
        Tm[t] = v
        order[t] = 1 if np.isfinite(v) else 0
        if use_second and order[t]:
            s = _msfm_second_derivative(m1, rd(-2 * dx, -2 * dy, -2 * dz), p1, rd(2 * dx, 2 * dy, 2 * dz))
            if np.isfinite(s):
                Tm2[t] = s
                order[t] = 2

    finv2 = 1.0  # 1/F^2 with F = 1

    def accumulate(t: int, c0: float, c1: float, c2: float) -> tuple[float, float, float]:
        if order[t] == 1:
            g = _MSFM_G1[t]
            return c0 + g, c1 - 2.0 * Tm[t] * g, c2 + Tm[t] * Tm[t] * g
        if order[t] == 2:
            g = _MSFM_G2[t]
            return c0 + g, c1 - 2.0 * Tm2[t] * g, c2 + Tm2[t] * Tm2[t] * g
        return c0, c1, c2

    # Stencil 0 (axes). Coeff accumulates across stencils exactly as msfm3d.c does.
    c0, c1, c2 = 0.0, 0.0, -finv2
    for t in range(3):
        c0, c1, c2 = accumulate(t, c0, c1, c2)
    tt = _msfm_roots_max(c0, c1, c2)

    if use_cross:
        for q in range(1, 6):
            c2 += -finv2
            for t in range(q * 3, (q + 1) * 3):
                c0, c1, c2 = accumulate(t, c0, c1, c2)
            if c0 > 0.0:
                tt = min(tt, _msfm_roots_max(c0, c1, c2))

    # Upwind/causality fix: the result must not be below a frozen neighbour used.
    finite = [v for v in Tm if np.isfinite(v)]
    if finite and tt < max(finite):
        tt = min(finite) + 1.0  # + 1/F
    return tt


def _msfm_distance(
    traversable: np.ndarray,
    source: tuple[int, int, int],
    spacing: np.ndarray,
    *,
    use_second: bool = True,
    use_cross: bool = True,
) -> np.ndarray:
    shape = traversable.shape
    nx, ny, nz = shape
    T = np.full(shape, np.inf, dtype=float)
    frozen = np.zeros(shape, dtype=bool)
    T[source] = 0.0
    frozen[source] = True

    axes = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    band: list[tuple[float, int, int, int]] = []

    def relax(px: int, py: int, pz: int) -> None:
        for dx, dy, dz in axes:
            i, j, k = px + dx, py + dy, pz + dz
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and traversable[i, j, k] and not frozen[i, j, k]:
                t = _msfm_point(T, frozen, shape, i, j, k, use_second, use_cross)
                if t < T[i, j, k]:
                    T[i, j, k] = t
                    heapq.heappush(band, (t, i, j, k))

    relax(*source)
    while band:
        t, x, y, z = heapq.heappop(band)
        if frozen[x, y, z] or t > T[x, y, z]:
            continue
        frozen[x, y, z] = True
        relax(x, y, z)

    # Solved in voxel units; scale to world units so magnitudes match the other solvers.
    h = float(np.mean(np.abs(spacing)))
    out = np.where(np.isfinite(T), T * h, np.inf)
    return out


def _surface_distances(distance_grid: np.ndarray, voxelization: VoxelizationResult) -> np.ndarray:
    on_index = voxelization.index.on_index
    dismap = distance_grid[on_index[:, 0], on_index[:, 1], on_index[:, 2]].astype(float)
    invalid = ~np.isfinite(dismap)
    if not np.any(invalid):
        return dismap

    finite = np.isfinite(dismap)
    if not np.any(finite):
        raise ValueError("no reachable surface points in distance grid")

    tree = nn_prepare(voxelization.grid_on)
    k = min(9, len(voxelization.grid_on))
    for point_id in np.where(invalid)[0]:
        indices, _ = nn_search(voxelization.grid_on, tree, voxelization.grid_on[point_id : point_id + 1], k)
        neighbor_values = dismap[indices[0]]
        neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
        if len(neighbor_values):
            dismap[point_id] = float(np.min(neighbor_values))
    return dismap


def distance_map_by_fast_marching(
    voxelization: VoxelizationResult,
    start_point_id: int,
    *,
    prefer_fmm: bool = True,
    solver: str | None = None,
    traversable_mask: np.ndarray | None = None,
    max_distance: float | None = None,
) -> DistanceMapResult:
    """Geodesic distance field rooted at a surface point, through the grasping space.

    `solver` selects the metric: ``"fmm"`` (single-stencil Eikonal fast marching via
    scikit-fmm, Euclidean), ``"msfm"`` (multistencil fast marching, MATLAB-parity,
    lowest anisotropy), or ``"dijkstra"`` (6-neighbour grid graph, Manhattan; no
    skfmm needed). If `solver` is None it follows the legacy `prefer_fmm` flag
    (True -> "fmm", False -> "dijkstra").

    `traversable_mask` overrides the default speed region (surface + exterior); pass a
    convex-hull-bounded mask for paper Theorem 3.2. `max_distance` caps the sweep
    (paper's `2h` radius): cells farther than this become unreached (inf)."""
    _validate_surface_id(voxelization, start_point_id, "start_point_id")
    source = tuple(int(v) for v in voxelization.index.on_index[start_point_id])
    if traversable_mask is not None:
        traversable = np.asarray(traversable_mask, dtype=bool)
        traversable[source] = True  # the source must always be traversable
    else:
        traversable = voxelization.output_grid != 0
    spacing = _spacing(voxelization)

    if solver is None:
        solver = "fmm" if prefer_fmm else "dijkstra"
    if solver not in ("fmm", "msfm", "dijkstra"):
        raise ValueError(f"unknown solver {solver!r}; expected 'fmm', 'msfm', or 'dijkstra'")

    if solver == "msfm":
        distance_grid = _msfm_distance(traversable, source, spacing)
    elif solver == "fmm":
        try:
            distance_grid = _fmm_distance(traversable, source, spacing)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"fast marching unavailable ({exc!r}); falling back to the Dijkstra "
                "grid-graph solver (Manhattan metric, slightly inflated distances). "
                "Install scikit-fmm for the Euclidean geodesic, or use solver='msfm'.",
                RuntimeWarning,
                stacklevel=2,
            )
            distance_grid = _dijkstra_distance(traversable, source, spacing)
    else:
        distance_grid = _dijkstra_distance(traversable, source, spacing)

    if max_distance is not None:
        distance_grid = np.where(distance_grid <= max_distance, distance_grid, np.inf)

    return DistanceMapResult(
        dismap=_surface_distances(distance_grid, voxelization),
        distance_grid=distance_grid,
    )


def _index_to_xyz(index: tuple[int, int, int], voxelization: VoxelizationResult) -> np.ndarray:
    x, y, z = index
    return np.array([voxelization.grid_x[x], voxelization.grid_y[y], voxelization.grid_z[z]], dtype=float)


def _voxel_to_world(positions: np.ndarray, voxelization: VoxelizationResult) -> np.ndarray:
    """Map continuous voxel-index positions (N x 3) to world XYZ coordinates."""

    def axis_origin_spacing(axis: np.ndarray) -> tuple[float, float]:
        origin = float(axis[0])
        spacing = float(axis[1] - axis[0]) if len(axis) > 1 else 1.0
        return origin, spacing

    ox, sx = axis_origin_spacing(voxelization.grid_x)
    oy, sy = axis_origin_spacing(voxelization.grid_y)
    oz, sz = axis_origin_spacing(voxelization.grid_z)
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    origin = np.array([ox, oy, oz])
    spacing = np.array([sx, sy, sz])
    return origin + positions * spacing


def _greedy_descent_indices(
    distance_grid: np.ndarray,
    start: tuple[int, int, int],
    source: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Integer-voxel steepest descent over 6-neighbors (fallback for thin grids)."""
    current = start
    path_indices = [current]
    visited = {current}
    max_steps = int(np.prod(distance_grid.shape))
    for _ in range(max_steps):
        if current == source:
            break
        candidates = [
            neighbor
            for neighbor in _six_neighbors(current, distance_grid.shape)
            if np.isfinite(distance_grid[neighbor])
        ]
        if not candidates:
            raise ValueError("no descending path to source")
        current_distance = distance_grid[current]
        descending = [candidate for candidate in candidates if distance_grid[candidate] < current_distance]
        pool = descending or [candidate for candidate in candidates if candidate not in visited]
        if not pool:
            raise ValueError("shortest path descent reached a local minimum")
        current = min(pool, key=lambda idx: distance_grid[idx])
        path_indices.append(current)
        visited.add(current)
    else:
        raise ValueError("shortest path exceeded grid size")
    return path_indices


def _gradient_descent_path(
    distance_grid: np.ndarray,
    start: tuple[int, int, int],
    source: tuple[int, int, int],
    *,
    step: float = 0.5,
) -> np.ndarray | None:
    """Continuous sub-voxel descent down the distance field.

    Mirrors the Fast-Marching-toolbox ``shortestpath``: trace a streamline along
    ``-grad(D)`` from ``start`` to the ``source`` minimum, producing a smooth
    sub-voxel path instead of a blocky voxel-stepped one. Returns ``None`` if the
    descent stalls (e.g. on a 1-voxel-thin structure), so the caller can fall
    back to the integer descent.
    """
    from scipy.interpolate import RegularGridInterpolator

    shape = distance_grid.shape
    if min(shape) < 2:
        return None

    # Fill blocked/unreachable cells with a high plateau so the streamline is
    # pushed away from the object interior rather than into it.
    finite = np.isfinite(distance_grid)
    if not finite.any():
        return None
    fill_value = float(distance_grid[finite].max()) + 1.0
    filled = np.where(finite, distance_grid, fill_value)

    axes = tuple(np.arange(n, dtype=float) for n in shape)
    gx, gy, gz = np.gradient(filled)
    interp_kwargs = {"bounds_error": False, "fill_value": None}
    grad_interp = [
        RegularGridInterpolator(axes, gx, **interp_kwargs),
        RegularGridInterpolator(axes, gy, **interp_kwargs),
        RegularGridInterpolator(axes, gz, **interp_kwargs),
    ]
    upper = np.array([n - 1 for n in shape], dtype=float)

    pos = np.array(start, dtype=float)
    source_pos = np.array(source, dtype=float)
    positions = [pos.copy()]
    max_iter = int(8 * sum(shape)) + 100
    reached = False
    for _ in range(max_iter):
        if np.linalg.norm(pos - source_pos) <= 1.0:
            reached = True
            break
        query = np.clip(pos, 0.0, upper)
        grad = np.array([float(g(query)[0]) for g in grad_interp])
        norm = float(np.linalg.norm(grad))
        if not np.isfinite(norm) or norm < 1e-9:
            return None
        pos = np.clip(pos - step * grad / norm, 0.0, upper)
        if not np.all(np.isfinite(pos)):
            return None
        positions.append(pos.copy())
    if not reached:
        return None

    positions.append(source_pos.copy())
    return np.vstack(positions)


# 26-neighbour offsets in the exact order of pointmin.m's `Ne` (so tie-breaking
# matches MATLAB), with (0,0,0) skipped.
_POINTMIN_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)


def _pointmin_descent_field(distance_grid: np.ndarray) -> np.ndarray | None:
    """Per-voxel steepest-descent direction — a NumPy port of pointmin.m.

    For each voxel, scan its 26 neighbours and record the unit vector toward the
    one with the smallest distance (the downhill direction). This is the discrete
    direction field MATLAB's `shortestpath` traces with RK4 — *not* a central
    difference. Unreachable cells are lifted to a high plateau so the descent never
    points into the obstacle. Returns an (nx, ny, nz, 3) field, or None if empty."""
    finite = np.isfinite(distance_grid)
    if not finite.any():
        return None
    high = float(distance_grid[finite].max()) + 1.0
    field = np.where(finite, distance_grid, high)
    nx, ny, nz = field.shape

    # Pad by one voxel with an even higher value so out-of-bounds neighbours are
    # never selected as the minimum (mirrors pointmin.m's J = max(I) border).
    padded = np.full((nx + 2, ny + 2, nz + 2), high + 1.0, dtype=float)
    padded[1:-1, 1:-1, 1:-1] = field

    best = field.copy()
    direction = np.zeros((nx, ny, nz, 3), dtype=float)
    for dx, dy, dz in _POINTMIN_OFFSETS:
        neighbour = padded[1 + dx : 1 + dx + nx, 1 + dy : 1 + dy + ny, 1 + dz : 1 + dz + nz]
        improved = neighbour < best
        if not improved.any():
            continue
        best = np.where(improved, neighbour, best)
        norm = (dx * dx + dy * dy + dz * dz) ** 0.5
        unit = np.array([dx, dy, dz], dtype=float) / norm
        direction[improved] = unit
    return direction


def _rk4_descent_path(
    distance_grid: np.ndarray,
    start: tuple[int, int, int],
    source: tuple[int, int, int],
    *,
    step: float = 0.5,
) -> np.ndarray | None:
    """Trace start -> source with Runge-Kutta 4 over the pointmin descent field —
    a port of the FastMarching toolbox `shortestpath` (Method='rk4'). Each stage's
    increment is normalised to `step` length, exactly like rk4.c. Returns voxel-index
    positions, or None if the trace leaves the domain or stalls before the source."""
    from scipy.interpolate import RegularGridInterpolator

    field = _pointmin_descent_field(distance_grid)
    if field is None:
        return None
    shape = distance_grid.shape
    upper = np.array([n - 1 for n in shape], dtype=float)
    axes = tuple(np.arange(n, dtype=float) for n in shape)
    interp = RegularGridInterpolator(axes, field, method="linear", bounds_error=False, fill_value=None)

    def descent(p: np.ndarray) -> np.ndarray:
        return np.asarray(interp(np.clip(p, 0.0, upper)[None])[0], dtype=float)

    def in_bounds(p: np.ndarray) -> bool:
        return bool(np.all(p >= 0.0) and np.all(p <= upper))

    def increment(p: np.ndarray) -> np.ndarray | None:
        g = descent(p)
        n = float(np.linalg.norm(g))
        if not np.isfinite(n) or n < 1e-12:
            return None
        return g * step / n

    def rk4_step(p: np.ndarray) -> np.ndarray | None:
        k1 = increment(p)
        if k1 is None or not in_bounds(p + 0.5 * k1):
            return None
        k2 = increment(p + 0.5 * k1)
        if k2 is None or not in_bounds(p + 0.5 * k2):
            return None
        k3 = increment(p + 0.5 * k2)
        if k3 is None or not in_bounds(p + k3):
            return None
        k4 = increment(p + k3)
        if k4 is None:
            return None
        nxt = p + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return nxt if in_bounds(nxt) else None

    source_pos = np.array(source, dtype=float)
    positions = [np.array(start, dtype=float)]
    max_iter = int(8 * sum(shape)) + 100
    reached = False
    for i in range(max_iter):
        nxt = rk4_step(positions[-1])
        if nxt is None:
            break
        # No-progress guard (shortestpath.m: movement over the last 10 steps).
        if i > 10 and np.linalg.norm(nxt - positions[i - 10]) < step:
            break
        positions.append(nxt)
        if np.linalg.norm(nxt - source_pos) < step:
            positions.append(source_pos)
            reached = True
            break
    if not reached:
        return None
    return np.vstack(positions)


def compute_shortest_path(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    start_point_id: int,
    source_point_id: int,
    *,
    connectivity: int = 6,
    integrator: str = "euler",
) -> np.ndarray:
    """Trace the caging-loop arc from `start` down to `source` through the distance
    field. `integrator='euler'` (default) uses a normalised-gradient streamline on a
    central-difference gradient; `integrator='rk4'` reproduces MATLAB's `shortestpath`
    (RK4 over the pointmin descent field). Both fall back to an integer-voxel greedy
    descent on thin structures."""
    if connectivity != 6:
        raise ValueError("only 6-connectivity is supported")
    if integrator not in ("euler", "rk4"):
        raise ValueError(f"unknown integrator {integrator!r}; expected 'euler' or 'rk4'")
    _validate_surface_id(voxelization, start_point_id, "start_point_id")
    _validate_surface_id(voxelization, source_point_id, "source_point_id")

    source = tuple(int(v) for v in voxelization.index.on_index[source_point_id])
    start = tuple(int(v) for v in voxelization.index.on_index[start_point_id])
    distance_grid = np.asarray(distance_grid, dtype=float)
    if distance_grid.shape != voxelization.output_grid.shape:
        raise ValueError("distance_grid shape must match output_grid")
    if not np.isfinite(distance_grid[start]):
        raise ValueError("start point is not reachable")

    if integrator == "rk4":
        traced = _rk4_descent_path(distance_grid, start, source)
    else:
        traced = _gradient_descent_path(distance_grid, start, source)
    if traced is not None:
        path = _voxel_to_world(traced, voxelization)
        # Snap the endpoints exactly onto the start and source surface points.
        path[0] = _index_to_xyz(start, voxelization)
        path[-1] = _index_to_xyz(source, voxelization)
        return path

    path_indices = _greedy_descent_indices(distance_grid, start, source)
    return np.vstack([_index_to_xyz(index, voxelization) for index in path_indices])


DistanceMapByFastMarching = distance_map_by_fast_marching
compute_shortestpath = compute_shortest_path
