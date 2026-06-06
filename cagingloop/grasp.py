from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from cagingloop.distance import compute_shortest_path, distance_map_by_fast_marching
from cagingloop.nearest import nn_prepare, nn_search
from cagingloop.saddle import detect_saddle_point
from cagingloop.types import CagingPath, VoxelizationResult


def _as_path(path: np.ndarray) -> np.ndarray:
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("path must be an N x 3 array")
    return arr


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return vector
    return vector / length


def _local_frame(grid_on: np.ndarray, point_id: int, frame_k: int) -> tuple[np.ndarray, np.ndarray]:
    tree = nn_prepare(grid_on)
    count = min(frame_k, len(grid_on))
    indices, _ = nn_search(grid_on, tree, grid_on[point_id : point_id + 1], count)
    neighbor_ids = [idx for idx in indices[0].tolist() if idx != point_id]
    if len(neighbor_ids) < 2:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    offsets = grid_on[neighbor_ids] - grid_on[point_id]
    # Match MATLAB pca(X', 3): center the neighbour offsets before extracting axes.
    centered = offsets - offsets.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u_direct = _normalize(vh[0])
    v_direct = _normalize(vh[1]) if len(vh) > 1 else np.array([0.0, 1.0, 0.0])
    return u_direct, v_direct


def _boundary_order(points_2d: np.ndarray) -> np.ndarray:
    if len(points_2d) < 3:
        return np.arange(len(points_2d), dtype=int)
    try:
        return ConvexHull(points_2d).vertices.astype(int)
    except QhullError:
        angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])
        return np.argsort(angles).astype(int)


def smooth_closed_path(path: np.ndarray) -> np.ndarray:
    path = _as_path(path)
    if len(path) == 0:
        return path.copy()
    if len(path) < 3:
        result = path.copy()
        result[-1] = result[0]
        return result

    result = path.copy()
    result[1:-1] = 0.2 * path[1:-1] + 0.4 * path[:-2] + 0.4 * path[2:]
    start_point = 0.4 * path[1] + 0.4 * path[-1] + 0.2 * path[0]
    result[0] = start_point
    result[-1] = start_point
    return result


def get_cage_points(dismap: np.ndarray, grid_on: np.ndarray, point_id: int, k: int = 9) -> np.ndarray:
    dismap = np.asarray(dismap, dtype=float)
    grid_on = np.asarray(grid_on, dtype=float)
    if grid_on.ndim != 2 or grid_on.shape[1] != 3:
        raise ValueError("grid_on must be an N x 3 array")
    if len(dismap) != len(grid_on):
        raise ValueError("dismap length must match grid_on")
    if point_id < 0 or point_id >= len(grid_on):
        raise ValueError("point_id is outside grid_on")

    tree = nn_prepare(grid_on)
    count = min(max(k, 3), len(grid_on))
    indices, _ = nn_search(grid_on, tree, grid_on[point_id : point_id + 1], count)
    neighbor_ids = np.array([idx for idx in indices[0].tolist() if idx != point_id], dtype=int)
    if len(neighbor_ids) < 2:
        raise ValueError("not enough neighbors to choose cage points")

    u_direct, v_direct = _local_frame(grid_on, point_id, min(15, len(grid_on)))
    offsets = grid_on[neighbor_ids] - grid_on[point_id]
    projected = np.column_stack((offsets @ u_direct, offsets @ v_direct))
    boundary = _boundary_order(projected)
    if len(boundary) == 0:
        boundary = np.arange(len(neighbor_ids), dtype=int)

    judge = dismap[neighbor_ids[boundary]] - dismap[point_id]
    first_pos = int(np.argmin(judge))
    chosen_positions = [int(boundary[first_pos])]

    first_vector = projected[chosen_positions[0]]
    negative = np.where(judge < 0.0)[0]
    if len(negative) > 1 and np.linalg.norm(first_vector) > 0.0:
        negative_positions = boundary[negative]
        negative_vectors = projected[negative_positions]
        lengths = np.linalg.norm(negative_vectors, axis=1)
        valid = lengths > 0.0
        if np.any(valid):
            negative_positions = negative_positions[valid]
            negative_vectors = negative_vectors[valid] / lengths[valid, None]
            products = negative_vectors @ _normalize(first_vector)
            second = int(negative_positions[int(np.argmin(products))])
            if second not in chosen_positions:
                chosen_positions.append(second)

    if len(chosen_positions) < 2:
        fallback = np.argsort(dismap[neighbor_ids])
        for pos in fallback:
            pos = int(pos)
            if pos not in chosen_positions:
                chosen_positions.append(pos)
            if len(chosen_positions) == 2:
                break
    if len(chosen_positions) < 2:
        raise ValueError("failed to choose two cage points")
    return neighbor_ids[chosen_positions[:2]].astype(int)


def generate_caging_grasp(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    source_point_id: int,
    dismap: np.ndarray,
    saddle_point_id: int,
    *,
    k: int = 9,
) -> CagingPath:
    if source_point_id < 0 or source_point_id >= len(voxelization.grid_on):
        raise ValueError("source_point_id is outside grid_on")
    if saddle_point_id < 0 or saddle_point_id >= len(voxelization.grid_on):
        raise ValueError("saddle_point_id is outside grid_on")

    cage_point_ids = get_cage_points(dismap, voxelization.grid_on, saddle_point_id, k=k)
    path1 = compute_shortest_path(distance_grid, voxelization, int(cage_point_ids[0]), source_point_id)
    path2 = compute_shortest_path(distance_grid, voxelization, int(cage_point_ids[1]), source_point_id)

    saddle = voxelization.grid_on[saddle_point_id : saddle_point_id + 1]
    path1 = np.vstack((saddle, path1))
    path2 = np.vstack((saddle, path2))
    path2 = np.flipud(path2)[1:]
    final_path = np.vstack((path1, path2))
    final_path = smooth_closed_path(final_path)
    return CagingPath(
        final_path=final_path,
        path1=final_path[: len(path1)],
        path2=final_path[len(path1) - 1 :],
    )


def loop_enclosed_area(path: np.ndarray) -> float:
    """Approximate area enclosed by a closed 3D loop (degenerate slivers ~ 0)."""
    path = _as_path(path)
    if len(path) < 3:
        return 0.0
    centroid = path.mean(axis=0)
    spokes = path - centroid
    cross = np.cross(spokes[:-1], spokes[1:])
    return float(np.linalg.norm(cross.sum(axis=0)) * 0.5)


def _path_separation(caging: CagingPath) -> float:
    """Mean distance between the loop's two arcs (≈0 for a degenerate out-and-back sliver)."""
    p1 = np.asarray(caging.path1, dtype=float)
    p2 = np.asarray(caging.path2, dtype=float)
    m = min(len(p1), len(p2))
    if m < 2:
        return 0.0
    a = p1[np.linspace(0, len(p1) - 1, m).astype(int)]
    b = p2[np.linspace(0, len(p2) - 1, m).astype(int)]
    return float(np.linalg.norm(a - b, axis=1).mean())


@dataclass(frozen=True)
class CagingCandidate:
    saddle_id: int
    path: CagingPath
    area: float          # enclosed area (large = body-wrapping, small = tight/handle loop)
    separation: float    # mean gap between the two arcs (clearness; ~0 = degenerate)
    source_id: int = -1  # the base point this loop was traced from


def generate_caging_grasps(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    source_point_id: int,
    dismap: np.ndarray,
    saddle_point_ids: np.ndarray,
    *,
    k: int = 9,
    max_loops: int = 10,
    min_separation_ratio: float = 0.02,
) -> list[CagingCandidate]:
    """Generate a caging loop per candidate saddle and return the top `max_loops`.

    `detect_saddle_point` ranks saddles by MATLAB's `diversityEval`, which does not
    correlate with caging quality. This builds the actual loop for every candidate,
    discards only the **degenerate** ones (the two arcs collapse onto the same
    geodesic — `separation ≈ 0`), and returns the rest ranked by enclosed area.
    Keeping several candidates preserves the small-but-clear loops (e.g. tight loops
    around a single handle) that a single largest-area pick would discard.
    """
    grid_on = voxelization.grid_on
    scale = float(np.linalg.norm(grid_on.max(axis=0) - grid_on.min(axis=0))) if len(grid_on) else 1.0
    candidates: list[CagingCandidate] = []
    for saddle_id in np.asarray(saddle_point_ids, dtype=int).tolist():
        try:
            caging = generate_caging_grasp(distance_grid, voxelization, source_point_id, dismap, saddle_id, k=k)
        except ValueError:
            continue
        separation = _path_separation(caging)
        if scale > 0.0 and separation < min_separation_ratio * scale:
            continue  # degenerate sliver: both arcs follow the same path, not a real loop
        candidates.append(
            CagingCandidate(
                saddle_id, caging, loop_enclosed_area(caging.final_path), separation, int(source_point_id)
            )
        )
    candidates.sort(key=lambda c: c.area, reverse=True)
    return candidates[:max_loops]


def generate_best_caging_grasp(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    source_point_id: int,
    dismap: np.ndarray,
    saddle_point_ids: np.ndarray,
    *,
    k: int = 9,
) -> tuple[CagingPath, int]:
    """Return the single best (largest-area, non-degenerate) caging loop and its saddle id."""
    candidates = generate_caging_grasps(
        distance_grid, voxelization, source_point_id, dismap, saddle_point_ids, k=k, max_loops=1
    )
    if not candidates:
        raise ValueError("no caging loop could be generated from the given saddle candidates")
    best = candidates[0]
    return best.path, best.saddle_id


def horizontal_slice_loop(
    voxelization: VoxelizationResult,
    height: float,
    *,
    up_axis: int = 1,
    margin: float = 0.0,
) -> CagingPath:
    """Caging loop as a horizontal ring encircling the solid at a given height.

    Unlike the saddle-based loops (which, at a flaring neck, come out vertical), this
    directly takes the object's cross-section at `height` (world units along `up_axis`)
    and traces the convex ring around it — a clean, horizontal loop at exactly the
    requested height, matching a hand-drawn waist grasp. `margin` (world units) expands
    the ring outward so the loop sits in the free space around the object."""
    grid = voxelization.output_grid
    axes = [voxelization.grid_x, voxelization.grid_y, voxelization.grid_z]
    up = axes[up_axis]
    plane_idx = int(np.argmin(np.abs(np.asarray(up, dtype=float) - height)))
    solid = grid >= 0  # surface ∪ inner
    sel = [slice(None)] * 3
    sel[up_axis] = plane_idx
    plane = solid[tuple(sel)]
    cells = np.argwhere(plane)
    if len(cells) < 3:
        raise ValueError("no solid cross-section at that height")
    other = [a for a in range(3) if a != up_axis]
    coords = np.column_stack((axes[other[0]][cells[:, 0]], axes[other[1]][cells[:, 1]]))
    try:
        order = ConvexHull(coords).vertices
    except QhullError as exc:
        raise ValueError("degenerate cross-section at that height") from exc
    ring2d = coords[order]
    center = ring2d.mean(axis=0)
    if margin > 0.0:
        directions = ring2d - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        ring2d = ring2d + directions / norms * margin
    n = len(ring2d)
    loop = np.zeros((n + 1, 3), dtype=float)
    loop[:n, other[0]] = ring2d[:, 0]
    loop[:n, other[1]] = ring2d[:, 1]
    loop[:n, up_axis] = float(up[plane_idx])
    loop[n] = loop[0]
    return CagingPath(final_path=loop, path1=loop[: n // 2 + 1], path2=loop[n // 2 :])


def farthest_point_sample(points: np.ndarray, k: int, seed: int | None = None) -> list[int]:
    """Greedy farthest-point sampling of `k` indices (well-spread base points)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if k >= n:
        return list(range(n))
    first = seed if seed is not None else int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    chosen = [first]
    dist = np.linalg.norm(points - points[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def sweep_caging_loops(
    voxelization: VoxelizationResult,
    grid_on_normals: np.ndarray,
    *,
    base_points: int = 20,
    k: int = 9,
    max_loops_per_source: int = 6,
    prefer_fmm: bool = True,
) -> list[CagingCandidate]:
    """Paper Algorithm 2 (simplified): sample `base_points` source points across the
    surface (farthest-point sampling), compute each one's distance field + saddles,
    and pool the resulting caging loops. A single base point only exposes saddles near
    itself; sweeping many is what surfaces loops elsewhere on the object (e.g. a waist
    loop when the seed is on top). Returns the pooled, de-duplicated candidates."""
    grid_on = voxelization.grid_on
    sources = farthest_point_sample(grid_on, base_points)
    pool: list[CagingCandidate] = []
    for src in sources:
        distance = distance_map_by_fast_marching(voxelization, src, prefer_fmm=prefer_fmm)
        saddles = detect_saddle_point(distance.dismap, grid_on, src, grid_on_normals, keep=len(grid_on))
        if len(saddles) == 0:
            continue
        pool.extend(
            generate_caging_grasps(
                distance.distance_grid, voxelization, src, distance.dismap, saddles,
                k=k, max_loops=max_loops_per_source,
            )
        )
    return pool


def waist_score(
    candidate: CagingCandidate,
    voxelization: VoxelizationResult,
    *,
    up_axis: int = 1,
    target_height: float | None = None,
    horizontal_weight: float = 0.5,
) -> float:
    """Paper §II.A mechanical score (lower = better): normalised distance of the loop
    centre from the centre of gravity (or `target_height` along `up_axis`) plus a tilt
    penalty (0 = perfectly horizontal)."""
    grid_on = voxelization.grid_on
    cog = grid_on.mean(axis=0)
    scale = float(np.linalg.norm(grid_on.max(axis=0) - grid_on.min(axis=0))) or 1.0
    fp = candidate.path.final_path
    center = fp.mean(axis=0)
    extent = fp.max(axis=0) - fp.min(axis=0)
    tilt = float(extent[up_axis] / (extent.max() + 1e-12))
    if target_height is not None:
        # Height is explicitly requested -> make matching it dominate, with tilt as a
        # tie-breaker among loops at that height (otherwise a flatter loop elsewhere wins).
        dist = abs(float(center[up_axis]) - target_height)
        return 4.0 * dist / scale + horizontal_weight * tilt
    dist = float(np.linalg.norm(center - cog))
    return dist / scale + horizontal_weight * tilt


def rank_caging_loops(
    pool: list[CagingCandidate],
    voxelization: VoxelizationResult,
    *,
    mode: str = "area",
    up_axis: int = 1,
    target_height: float | None = None,
    horizontal_weight: float = 0.5,
    min_area: float = 0.0,
) -> list[CagingCandidate]:
    """Rank pooled loops. `mode='area'` = largest first (body-wrapping); `mode='small'`
    = smallest non-degenerate first (tight loops, e.g. around a handle); `mode='waist'`
    = paper's mechanical criteria (near centre-of-gravity + horizontal)."""
    kept = [c for c in pool if c.area >= min_area]
    if mode == "waist":
        return sorted(
            kept,
            key=lambda c: waist_score(
                c, voxelization, up_axis=up_axis, target_height=target_height,
                horizontal_weight=horizontal_weight,
            ),
        )
    if mode == "small":
        # smallest first, but only loops whose two arcs are well separated (a real
        # tight loop, not a degenerate sliver) — these tend to wrap a thin handle.
        return sorted(kept, key=lambda c: c.area)
    return sorted(kept, key=lambda c: c.area, reverse=True)


def select_caging_loop(
    pool: list[CagingCandidate],
    voxelization: VoxelizationResult,
    *,
    up_axis: int = 1,
    target_height: float | None = None,
    horizontal_weight: float = 0.5,
    min_area: float = 0.0,
) -> CagingCandidate | None:
    """Pick the single best graspable loop (paper §II.A: near centre-of-gravity + horizontal)."""
    ranked = rank_caging_loops(
        pool, voxelization, mode="waist", up_axis=up_axis, target_height=target_height,
        horizontal_weight=horizontal_weight, min_area=min_area,
    )
    return ranked[0] if ranked else None


generateCagingGrasp = generate_caging_grasp
