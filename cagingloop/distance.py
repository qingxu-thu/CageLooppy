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
) -> DistanceMapResult:
    _validate_surface_id(voxelization, start_point_id, "start_point_id")
    source = tuple(int(v) for v in voxelization.index.on_index[start_point_id])
    traversable = voxelization.output_grid != 0
    spacing = _spacing(voxelization)

    if prefer_fmm:
        try:
            distance_grid = _fmm_distance(traversable, source, spacing)
        except Exception:
            distance_grid = _dijkstra_distance(traversable, source, spacing)
    else:
        distance_grid = _dijkstra_distance(traversable, source, spacing)

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


def compute_shortest_path(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    start_point_id: int,
    source_point_id: int,
    *,
    connectivity: int = 6,
) -> np.ndarray:
    if connectivity != 6:
        raise ValueError("only 6-connectivity is supported")
    _validate_surface_id(voxelization, start_point_id, "start_point_id")
    _validate_surface_id(voxelization, source_point_id, "source_point_id")

    source = tuple(int(v) for v in voxelization.index.on_index[source_point_id])
    start = tuple(int(v) for v in voxelization.index.on_index[start_point_id])
    distance_grid = np.asarray(distance_grid, dtype=float)
    if distance_grid.shape != voxelization.output_grid.shape:
        raise ValueError("distance_grid shape must match output_grid")
    if not np.isfinite(distance_grid[start]):
        raise ValueError("start point is not reachable")

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
