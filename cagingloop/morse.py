"""Volumetric Morse-saddle caging loops (paper Fig. 3 + Theorem 3.1).

The released MATLAB code detects saddles only on the surface. The paper instead
finds Morse saddles of the *volumetric* distance field `D_p` (every grasping-space
voxel, by its 6 axis-neighbours), and traces a loop from each by descending in the
two opposite "downhill" directions back to the base point. Free-space saddles —
e.g. inside a handle's hole — generate loops that thread the hole, which the
surface method cannot find.
"""

from __future__ import annotations

import numpy as np

from cagingloop.grasp import CagingCandidate, loop_enclosed_area, smooth_closed_path
from cagingloop.types import CagingPath, VoxelizationResult

_AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]
# 26-connectivity adjacency between the shell offsets (for lower-link components)
_ADJ = [
    [
        j
        for j in range(26)
        if j != i
        and max(abs(_OFFSETS[i][k] - _OFFSETS[j][k]) for k in range(3)) <= 1
    ]
    for i in range(26)
]


def _shift_full(field: np.ndarray, off: tuple[int, int, int]) -> np.ndarray:
    """Neighbour value in offset direction `off` at each voxel; inf past the edge."""
    out = np.full_like(field, np.inf)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for k, o in enumerate(off):
        if o == 1:
            dst[k] = slice(0, -1)
            src[k] = slice(1, None)
        elif o == -1:
            dst[k] = slice(1, None)
            src[k] = slice(0, -1)
    out[tuple(dst)] = field[tuple(src)]
    return out


def _lower_components(lower_local: list[int]) -> list[list[int]]:
    """Connected components of a subset of the 26 shell offsets under 26-adjacency."""
    members = set(lower_local)
    seen: set = set()
    comps: list[list[int]] = []
    for start in lower_local:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in _ADJ[i]:
                if j in members and j not in seen:
                    seen.add(j)
                    stack.append(j)
        comps.append(comp)
    return comps


def detect_morse_saddles_3d(distance_grid: np.ndarray):
    """Discrete Morse saddles of `distance_grid` via lower-link connectivity (ref [23]).

    A voxel is a saddle if its **lower** neighbours (over the 26-neighbourhood) form
    **≥2 connected components** — i.e. the downhill region splits, so the wavefront
    arrives from two separated directions. Returns (voxel, [descending offsets], ncomp),
    one representative downhill offset per lower component. Ties broken by flat index.
    """
    D = np.asarray(distance_grid, dtype=float)
    shape = D.shape
    finite = np.isfinite(D)
    # vectorised pre-filter: count strictly-lower / finite neighbours
    lower_count = np.zeros(shape, dtype=int)
    finite_count = np.zeros(shape, dtype=int)
    for off in _OFFSETS:
        nb = _shift_full(D, off)
        f = np.isfinite(nb)
        finite_count += f
        lower_count += (f & (nb < D)).astype(int)
    cand = finite & (lower_count >= 2) & (lower_count < finite_count)
    cand[0, :, :] = cand[-1, :, :] = False
    cand[:, 0, :] = cand[:, -1, :] = False
    cand[:, :, 0] = cand[:, :, -1] = False
    flat = np.arange(D.size).reshape(shape)

    out = []
    for vox in np.argwhere(cand):
        x, y, z = int(vox[0]), int(vox[1]), int(vox[2])
        dv = D[x, y, z]
        fv = flat[x, y, z]
        lower_local = []
        for i, (dx, dy, dz) in enumerate(_OFFSETS):
            nv = D[x + dx, y + dy, z + dz]
            if not np.isfinite(nv):
                continue
            nf = flat[x + dx, y + dy, z + dz]
            if nv < dv or (nv == dv and nf < fv):  # tie-break = simulated simplicity
                lower_local.append(i)
        comps = _lower_components(lower_local)
        if len(comps) < 2:
            continue
        # representative downhill offset per component = the lowest-D one
        reps = []
        for comp in comps:
            best = min(comp, key=lambda i: D[x + _OFFSETS[i][0], y + _OFFSETS[i][1], z + _OFFSETS[i][2]])
            reps.append(_OFFSETS[best])
        out.append(((x, y, z), reps, len(comps)))
    return out


def _index_to_xyz(voxel: tuple[int, int, int], voxelization: VoxelizationResult) -> np.ndarray:
    x, y, z = voxel
    return np.array(
        [voxelization.grid_x[x], voxelization.grid_y[y], voxelization.grid_z[z]], dtype=float
    )


def _steepest_descent(D: np.ndarray, start: tuple[int, int, int], source: tuple[int, int, int]) -> list:
    """Integer steepest descent from `start` to the `source` (global min of D)."""
    cur = tuple(start)
    path = [cur]
    visited = {cur}
    shape = D.shape
    for _ in range(int(D.size)):
        if cur == tuple(source):
            break
        best, best_d = None, D[cur]
        for dx, dy, dz in _AXES:
            nb = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
            if 0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and 0 <= nb[2] < shape[2]:
                if np.isfinite(D[nb]) and D[nb] < best_d:
                    best, best_d = nb, D[nb]
        if best is None or best in visited:
            break
        cur = best
        path.append(cur)
        visited.add(cur)
    return path


def trace_loop_from_saddle(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    saddle: tuple[int, int, int],
    descending_offsets,
    source: tuple[int, int, int],
) -> CagingPath | None:
    """Theorem 3.1: descend from the saddle's two separated downhill directions back to
    the source, then join the two arcs into a closed loop through saddle and source."""
    d1, d2 = descending_offsets[0], descending_offsets[1]
    n1 = tuple(saddle[i] + d1[i] for i in range(3))
    n2 = tuple(saddle[i] + d2[i] for i in range(3))
    p1 = _steepest_descent(distance_grid, n1, source)
    p2 = _steepest_descent(distance_grid, n2, source)
    if p1[-1] != tuple(source) or p2[-1] != tuple(source):
        return None  # an arc stalled at a non-source local min
    # loop: source -> ... -> n1 -> saddle -> n2 -> ... -> source
    arc1 = list(reversed(p1)) + [tuple(saddle)]      # source -> n1 -> saddle
    arc2 = [tuple(saddle)] + p2                        # saddle -> n2 -> source
    voxels = arc1 + p2                                 # full loop (saddle once)
    if len(voxels) < 4:
        return None
    world = np.vstack([_index_to_xyz(v, voxelization) for v in voxels])
    world = np.vstack([world, world[0]])               # close
    final = smooth_closed_path(world)
    n1len = len(arc1)
    return CagingPath(final_path=final, path1=final[:n1len], path2=final[n1len - 1:])


def caging_loop_space(
    voxelization: VoxelizationResult,
    surface_normals: np.ndarray,
    *,
    base_points: int = 20,
    method: str = "morse",
    use_convex_hull: bool = False,
    sweep_radius: float | None = None,
    curvature_filter: bool = False,
    base_sampling: str = "farthest",
    base_height: float | None = None,
    base_height_band: float = 0.1,
    up_axis: int = 1,
    k: int = 9,
    max_loops_per_source: int = 6,
    prefer_fmm: bool = True,
    solver: str | None = None,
):
    """Paper **Algorithm 2**: build the caging-loop space L̃ with its filtering rules.

    - base points: farthest-point sampled, optionally restricted to positive-curvature
      points (Theorem 3.3);
    - grasping space: optionally bounded by the convex hull (Theorem 3.2);
    - distance field: optionally swept only to `sweep_radius` = 2h (gripper reach);
    - saddles: volumetric Morse (`method='morse'`, paper Fig. 3) or surface
      (`method='surface'`); a loop is traced per saddle and pooled.
    """
    from cagingloop.curvature import positive_curvature_points
    from cagingloop.distance import distance_map_by_fast_marching
    from cagingloop.grasp import (
        detect_saddle_point,
        farthest_point_sample,
        generate_caging_grasps,
        uniform_sample,
    )
    from cagingloop.voxelization import convex_hull_grasping_mask

    grid_on = voxelization.grid_on
    if curvature_filter:  # Theorem 3.3
        candidates = positive_curvature_points(grid_on, surface_normals)
        if len(candidates) == 0:
            candidates = np.arange(len(grid_on))
    else:
        candidates = np.arange(len(grid_on))
    if base_height is not None:  # restrict base points to a height band along up_axis
        coord = grid_on[:, up_axis]
        lo, hi = float(coord.min()), float(coord.max())
        target = lo + base_height * (hi - lo)
        half = 0.5 * base_height_band * (hi - lo)
        in_band = np.abs(grid_on[candidates, up_axis] - target) <= half
        if in_band.any():
            candidates = candidates[in_band]
    n = min(base_points, len(candidates))
    if base_sampling == "uniform":  # paper: "uniformly distributed sample points"
        local = uniform_sample(grid_on[candidates], n)
    else:
        local = farthest_point_sample(grid_on[candidates], n)
    sources = [int(candidates[i]) for i in local]

    mask = convex_hull_grasping_mask(voxelization) if use_convex_hull else None  # Theorem 3.2

    pool: list[CagingCandidate] = []
    for src in sources:
        distance = distance_map_by_fast_marching(
            voxelization, src, prefer_fmm=prefer_fmm, solver=solver,
            traversable_mask=mask, max_distance=sweep_radius,
        )
        if method == "morse":
            pool.extend(volumetric_caging_loops(voxelization, src, distance=distance, max_loops=max_loops_per_source))
        else:
            saddles = detect_saddle_point(distance.dismap, grid_on, src, surface_normals, keep=len(grid_on))
            if len(saddles):
                pool.extend(
                    generate_caging_grasps(
                        distance.distance_grid, voxelization, src, distance.dismap, saddles,
                        k=k, max_loops=max_loops_per_source,
                    )
                )
    return pool


def volumetric_caging_loops(
    voxelization: VoxelizationResult,
    source_point_id: int,
    *,
    distance=None,
    min_separation_ratio: float = 0.02,
    max_saddles: int = 4000,
    max_loops: int | None = None,
    prefer_fmm: bool = True,
    solver: str | None = None,
) -> list[CagingCandidate]:
    """Caging loops from one base point via volumetric saddles (paper Algorithm 1
    inner loop). Returns non-degenerate `CagingCandidate`s, deduplicated; if
    `max_loops` is set, keeps only the `max_loops` largest-area ones."""
    from cagingloop.distance import distance_map_by_fast_marching
    from cagingloop.grasp import _path_separation

    if distance is None:
        distance = distance_map_by_fast_marching(voxelization, source_point_id, prefer_fmm=prefer_fmm, solver=solver)
    D = distance.distance_grid
    source = tuple(int(v) for v in voxelization.index.on_index[source_point_id])
    grid_on = voxelization.grid_on
    scale = float(np.linalg.norm(grid_on.max(axis=0) - grid_on.min(axis=0))) or 1.0

    saddles = detect_morse_saddles_3d(D)
    if len(saddles) > max_saddles:  # process the deepest (largest-D) saddles first
        saddles.sort(key=lambda s: D[s[0]], reverse=True)
        saddles = saddles[:max_saddles]

    seen: set = set()
    out: list[CagingCandidate] = []
    for voxel, offsets, _ncomp in saddles:
        loop = trace_loop_from_saddle(D, voxelization, voxel, offsets, source)
        if loop is None:
            continue
        sep = _path_separation(loop)
        if sep < min_separation_ratio * scale:
            continue
        area = loop_enclosed_area(loop.final_path)
        key = (round(float(loop.final_path.mean(0)[0]), 2),
               round(float(loop.final_path.mean(0)[1]), 2),
               round(float(loop.final_path.mean(0)[2]), 2),
               round(area, 3))
        if key in seen:
            continue
        seen.add(key)
        flat = int(np.ravel_multi_index(voxel, D.shape))
        out.append(CagingCandidate(flat, loop, area, sep, int(source_point_id)))
    if max_loops is not None and len(out) > max_loops:
        out = sorted(out, key=lambda c: c.area, reverse=True)[:max_loops]
    return out
