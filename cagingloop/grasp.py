from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from cagingloop.distance import compute_shortest_path
from cagingloop.nearest import nn_prepare, nn_search
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


def generate_best_caging_grasp(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    source_point_id: int,
    dismap: np.ndarray,
    saddle_point_ids: np.ndarray,
    *,
    k: int = 9,
) -> tuple[CagingPath, int]:
    """Generate a caging loop for each candidate saddle and return the best one.

    `detect_saddle_point` ranks saddles by MATLAB's `diversityEval`, which does not
    correlate with how well the resulting loop wraps the object — the top-ranked
    saddle often yields a degenerate sliver where both paths follow the same
    geodesic. This evaluates each candidate's actual loop and keeps the one that
    encloses the most area (a genuine wrapping loop), returning it with its
    saddle id.
    """
    best: CagingPath | None = None
    best_saddle = -1
    best_area = -1.0
    for saddle_id in np.asarray(saddle_point_ids, dtype=int).tolist():
        try:
            caging = generate_caging_grasp(distance_grid, voxelization, source_point_id, dismap, saddle_id, k=k)
        except ValueError:
            continue
        area = loop_enclosed_area(caging.final_path)
        if area > best_area:
            best_area = area
            best = caging
            best_saddle = saddle_id
    if best is None:
        raise ValueError("no caging loop could be generated from the given saddle candidates")
    return best, best_saddle


generateCagingGrasp = generate_caging_grasp
