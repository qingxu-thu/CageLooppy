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
    integrator: str = "euler",
) -> CagingPath:
    if source_point_id < 0 or source_point_id >= len(voxelization.grid_on):
        raise ValueError("source_point_id is outside grid_on")
    if saddle_point_id < 0 or saddle_point_id >= len(voxelization.grid_on):
        raise ValueError("saddle_point_id is outside grid_on")

    cage_point_ids = get_cage_points(dismap, voxelization.grid_on, saddle_point_id, k=k)
    path1 = compute_shortest_path(distance_grid, voxelization, int(cage_point_ids[0]), source_point_id, integrator=integrator)
    path2 = compute_shortest_path(distance_grid, voxelization, int(cage_point_ids[1]), source_point_id, integrator=integrator)

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


def _segment_penetrates_solid(a: np.ndarray, b: np.ndarray, voxelization: VoxelizationResult, *, samples: int = 8) -> bool:
    """True if the straight chord a->b passes through the object interior (an inner
    voxel). Used to tell a free-space shortcut from one blocked by the surface."""
    gx, gy, gz = voxelization.grid_x, voxelization.grid_y, voxelization.grid_z
    og = voxelization.output_grid

    def to_idx(coord: float, axis: np.ndarray) -> int:
        origin = float(axis[0])
        spacing = float(axis[1] - axis[0]) if len(axis) > 1 else 1.0
        return int(np.clip(round((coord - origin) / spacing), 0, len(axis) - 1))

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    for t in np.linspace(0.0, 1.0, samples):
        q = a + t * (b - a)
        if og[to_idx(q[0], gx), to_idx(q[1], gy), to_idx(q[2], gz)] == 0:  # interior solid
            return True
    return False


def loop_locally_shortest_at_base(
    candidate: CagingCandidate,
    voxelization: VoxelizationResult,
    *,
    window: int = 3,
    min_straightness: float = 0.95,
) -> bool:
    """Paper L̃ -> L test (Remark after Property 2.7; Fig. 4f): keep a p-based loop only
    if it is **locally shortest at its base point p**. A p-based loop is locally shortest
    everywhere except possibly at p (Def. 2.6), so we only inspect the neighbourhood of p.

    Direct shortenability test (the literal definition, robust to voxel jaggedness because
    it fits a whole window rather than two segments): take the `window` loop points on each
    side of p, and the straight chord between the window's two ends. Let
    `straightness = |chord| / (sub-path length)` (1 = already straight, < 1 = there is a
    corner to cut). The loop is NOT locally shortest at p iff that corner is genuinely
    removable: `straightness < min_straightness` AND the chord stays in free space (a chord
    that would penetrate the solid is a real taut touch held by the surface -> keep).

    The paper states this filter but does not specify the test, so this is our interpretation."""
    path = np.asarray(candidate.path.final_path, dtype=float)
    n = len(path)
    if n < 5:
        return True  # too short to judge a corner; don't over-filter
    sid = getattr(candidate, "source_id", -1)
    if sid is None or sid < 0 or sid >= len(voxelization.grid_on):
        return True
    p = voxelization.grid_on[sid]
    idx = int(np.argmin(np.linalg.norm(path - p, axis=1)))  # where the loop touches p
    w = max(1, min(window, n // 2 - 1))
    ring = [(idx + k) % n for k in range(-w, w + 1)]
    sub = path[ring]
    subpath_len = float(np.linalg.norm(np.diff(sub, axis=0), axis=1).sum())
    a, b = sub[0], sub[-1]
    chord_len = float(np.linalg.norm(b - a))
    if subpath_len < 1e-12:
        return True
    straightness = chord_len / subpath_len
    if straightness >= min_straightness:
        return True  # already ~straight through p: not shortenable -> locally shortest
    return _segment_penetrates_solid(a, b, voxelization)  # corner: keep only if held by surface


def filter_locally_shortest(
    pool: list[CagingCandidate],
    voxelization: VoxelizationResult,
    *,
    window: int = 3,
    min_straightness: float = 0.95,
) -> list[CagingCandidate]:
    """Apply `loop_locally_shortest_at_base` to a pool — the paper's L̃ -> L reduction."""
    return [
        c for c in pool
        if loop_locally_shortest_at_base(c, voxelization, window=window, min_straightness=min_straightness)
    ]


def isoperimetric_ratio(path: np.ndarray) -> float:
    """Shape-degeneracy measure rho = area / perimeter^2: ~0 for a loop collapsed toward a
    line, 1/(4*pi) ~= 0.0796 for a circle. Scale-free, so it rejects line-like loops at any
    size (unlike an absolute area floor)."""
    path = _as_path(path)
    if len(path) < 3:
        return 0.0
    perimeter = float(np.linalg.norm(np.diff(np.vstack((path, path[:1])), axis=0), axis=1).sum())
    if perimeter < 1e-12:
        return 0.0
    return loop_enclosed_area(path) / (perimeter * perimeter)


def filter_by_isoperimetric(pool: list[CagingCandidate], min_rho: float) -> list[CagingCandidate]:
    """Drop loops whose isoperimetric ratio is below `min_rho` (line-like / degenerate)."""
    if min_rho <= 0.0:
        return list(pool)
    return [c for c in pool if isoperimetric_ratio(c.path.final_path) >= min_rho]


def _points_inside_polygon(px: np.ndarray, py: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many sample points (px,py) against polygon (vx,vy)."""
    inside = np.zeros(len(px), dtype=bool)
    n = len(vx)
    j = n - 1
    for i in range(n):
        cond = ((vy[i] > py) != (vy[j] > py)) & (
            px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-12) + vx[i]
        )
        inside ^= cond
        j = i
    return inside


def loop_encircles_solid(
    candidate: CagingCandidate,
    voxelization: VoxelizationResult,
    *,
    samples: int = 16,
    shrink: float = 0.6,
    min_frac: float = 0.06,
) -> bool:
    """Does the loop go *around object material* (link the solid)? A real caging loop — even a
    tiny one around a thin handle — has a spanning disk whose **interior** passes through the
    object; a degenerate surface stub only grazes the solid at its rim. Size-independent, so it
    keeps small genuine handle loops and drops 'encircles nothing' stubs (unlike a length floor).

    We fit the loop plane (normal n) and sample the disk **shrunk toward the centroid** (factor
    `shrink`, ignoring the rim). For each inner sample we test the solid a step off-plane on *both*
    sides (+n and -n): the loop links the solid iff material **passes through** the plane somewhere
    inside it (solid on both sides). A bar or the body pierces the plane; a surface patch only has
    solid on one side, so it is dropped. `min_frac` = min fraction of inner samples that straddle."""
    path = _as_path(candidate.path.final_path)
    if len(path) < 3:
        return False
    centroid = path.mean(axis=0)
    centered = path - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return False
    u, v, n = vh[0], vh[1], vh[2]  # in-plane basis (u, v) and loop-plane normal n
    poly = np.column_stack((centered @ u, centered @ v))
    test_poly = poly * shrink  # inner region only: ignore the rim where a tangent disk clips solid
    lo, hi = test_poly.min(axis=0), test_poly.max(axis=0)
    su = np.linspace(lo[0], hi[0], samples)
    sv = np.linspace(lo[1], hi[1], samples)
    gu, gv = np.meshgrid(su, sv)
    gu, gv = gu.ravel(), gv.ravel()
    inside = _points_inside_polygon(gu, gv, test_poly[:, 0], test_poly[:, 1])
    n_inside = int(inside.sum())
    if n_inside == 0:
        return False
    gx, gy, gz = voxelization.grid_x, voxelization.grid_y, voxelization.grid_z
    og = voxelization.output_grid
    spacing = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    step = 1.5 * spacing * n  # one-voxel offset along the loop-plane normal

    def is_solid(points: np.ndarray) -> np.ndarray:
        def to_idx(coord: np.ndarray, axis: np.ndarray) -> np.ndarray:
            origin = float(axis[0])
            sp = float(axis[1] - axis[0]) if len(axis) > 1 else 1.0
            return np.clip(np.round((coord - origin) / sp).astype(int), 0, len(axis) - 1)
        return og[to_idx(points[:, 0], gx), to_idx(points[:, 1], gy), to_idx(points[:, 2], gz)] == 0

    world = centroid + np.outer(gu[inside], u) + np.outer(gv[inside], v)
    straddle = is_solid(world + step) & is_solid(world - step)  # material pierces the loop plane
    return float(np.mean(straddle)) >= min_frac


def filter_by_linking(
    pool: list[CagingCandidate], voxelization: VoxelizationResult, *, min_frac: float = 0.06
) -> list[CagingCandidate]:
    """Keep only loops that encircle object material (drop 'encircles nothing' surface stubs).
    `min_frac` = minimum fraction of the loop's inner disk that must be solid (higher = stricter)."""
    return [c for c in pool if loop_encircles_solid(c, voxelization, min_frac=min_frac)]


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
    integrator: str = "euler",
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
            caging = generate_caging_grasp(distance_grid, voxelization, source_point_id, dismap, saddle_id, k=k, integrator=integrator)
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


def uniform_sample(points: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """`k` indices drawn uniformly at random (≈ uniform over the surface, as surface
    voxels have roughly equal area) — matches the paper's '500 uniformly distributed
    sample points'. Deterministic given `seed`."""
    n = len(points)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


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
    = smallest non-degenerate first (tight loops, e.g. around a handle); `mode='mechanical'`
    (alias `'waist'`) = the paper's §II.A mechanical considerations: the loop centre as
    close as possible to the object's centre of gravity (minimise moment of inertia) and
    the loop roughly horizontal (so the object lifts up steadily)."""
    kept = [c for c in pool if c.area >= min_area]
    if mode in ("mechanical", "waist"):
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
