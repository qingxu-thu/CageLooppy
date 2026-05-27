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
    traversable = voxelization.output_grid >= 0
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
    current = tuple(int(v) for v in voxelization.index.on_index[start_point_id])
    distance_grid = np.asarray(distance_grid, dtype=float)
    if distance_grid.shape != voxelization.output_grid.shape:
        raise ValueError("distance_grid shape must match output_grid")
    if not np.isfinite(distance_grid[current]):
        raise ValueError("start point is not reachable")

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

    return np.vstack([_index_to_xyz(index, voxelization) for index in path_indices])


DistanceMapByFastMarching = distance_map_by_fast_marching
compute_shortestpath = compute_shortest_path
