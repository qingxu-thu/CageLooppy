from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from cagingloop.nearest import nn_prepare, nn_search


def _as_xyz_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    return arr


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError(f"{name} must not be zero length")
    return np.asarray(vector, dtype=float) / length


def diversity_eval(p1: np.ndarray, p2: np.ndarray, normal1: np.ndarray, normal2: np.ndarray) -> float:
    d = _normalize(np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float), "point difference")
    n1 = _normalize(np.asarray(normal1, dtype=float), "normal1")
    n2 = _normalize(np.asarray(normal2, dtype=float), "normal2")
    angle1 = np.arccos(np.clip(np.dot(-n1, d), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.dot(n2, d), -1.0, 1.0))
    return float(np.exp(-0.5 * max(angle1, angle2) ** 4))


def _local_frame(grid_on: np.ndarray, point_id: int, frame_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = nn_prepare(grid_on)
    count = min(frame_k, len(grid_on))
    indices, _ = nn_search(grid_on, tree, grid_on[point_id : point_id + 1], count)
    neighbor_ids = [idx for idx in indices[0].tolist() if idx != point_id]
    if len(neighbor_ids) < 2:
        raise ValueError("not enough neighbors to estimate a local frame")
    offsets = grid_on[neighbor_ids] - grid_on[point_id]
    _, _, vh = np.linalg.svd(offsets, full_matrices=False)
    u_direct = vh[0]
    v_direct = vh[1] if len(vh) > 1 else np.array([0.0, 1.0, 0.0])
    n_direct = vh[2] if len(vh) > 2 else np.cross(u_direct, v_direct)
    return _normalize(u_direct, "u_direct"), _normalize(v_direct, "v_direct"), _normalize(n_direct, "n_direct")


def _project_neighbors(
    grid_on: np.ndarray,
    point_id: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = nn_prepare(grid_on)
    count = min(max(k, 2), len(grid_on))
    indices, _ = nn_search(grid_on, tree, grid_on[point_id : point_id + 1], count)
    neighbor_ids = np.array([idx for idx in indices[0].tolist() if idx != point_id], dtype=int)
    if len(neighbor_ids) < 3:
        return neighbor_ids, np.zeros((0, 2), dtype=float)
    u_direct, v_direct, _ = _local_frame(grid_on, point_id, min(15, len(grid_on)))
    offsets = grid_on[neighbor_ids] - grid_on[point_id]
    projected = np.column_stack((offsets @ u_direct, offsets @ v_direct))
    return neighbor_ids, projected


def _boundary_order(points_2d: np.ndarray) -> np.ndarray:
    if len(points_2d) < 3:
        return np.arange(len(points_2d), dtype=int)
    try:
        return ConvexHull(points_2d).vertices.astype(int)
    except QhullError:
        angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])
        return np.argsort(angles).astype(int)


def _point_in_polygon_strict(point: np.ndarray, polygon: np.ndarray, tol: float = 1e-12) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        edge = np.array([x2 - x1, y2 - y1])
        to_point = np.array([x - x1, y - y1])
        cross = abs(float(np.cross(edge, to_point)))
        dot = float(np.dot(to_point, edge))
        if cross <= tol and 0.0 <= dot <= float(np.dot(edge, edge)):
            return False
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def _filter_boundary(boundary_points: np.ndarray, boundary_order: np.ndarray, judge: np.ndarray) -> np.ndarray:
    if len(boundary_order) < 3:
        return boundary_order
    norms = np.linalg.norm(boundary_points, axis=1)
    norms[norms == 0.0] = 1.0
    dis_toler = judge / norms
    start = int(np.argmax(np.abs(dis_toler)))
    order = np.concatenate((boundary_order[start:], boundary_order[:start]))
    points = np.vstack((boundary_points[start:], boundary_points[:start]))
    values = np.concatenate((judge[start:], judge[:start]))

    norms = np.linalg.norm(points, axis=1)
    norms[norms == 0.0] = 1.0
    dis_toler = values / norms
    delete_small = np.where(np.abs(dis_toler) <= 50.0)[0]

    a = points[:-1]
    b = points[1:]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0.0] = 1.0
    angles = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1) / denom, -1.0, 1.0)))
    delete_angle = np.where(angles <= 15.0)[0] + 1
    delete = np.unique(np.concatenate((delete_small, delete_angle))).astype(int)
    if len(order) - len(delete) < 3:
        return boundary_order
    return np.delete(order, delete)


def _count_transitions(judge: np.ndarray) -> int:
    if len(judge) == 0:
        return 0
    signs = (judge > 0.0).astype(int)
    return int(np.count_nonzero(signs != np.roll(signs, -1)))


def calculate_iter_num(dismap: np.ndarray, grid_on: np.ndarray, point_id: int, k: int = 9) -> int:
    dismap = np.asarray(dismap, dtype=float)
    grid_on = _as_xyz_array(grid_on, "grid_on")
    if point_id < 0 or point_id >= len(grid_on):
        raise ValueError("point_id is outside grid_on")
    if len(dismap) != len(grid_on):
        raise ValueError("dismap length must match grid_on")

    neighbor_ids, projected = _project_neighbors(grid_on, point_id, k)
    if len(projected) < 3:
        return -1
    boundary = _boundary_order(projected)
    boundary_points = projected[boundary]
    if not _point_in_polygon_strict(np.array([0.0, 0.0]), boundary_points):
        return -1

    judge = dismap[neighbor_ids[boundary]] - dismap[point_id]
    filtered_boundary = _filter_boundary(boundary_points, boundary, judge)
    judge = dismap[neighbor_ids[filtered_boundary]] - dismap[point_id]
    return _count_transitions(judge)


def detect_saddle_point(
    dismap: np.ndarray,
    grid_on: np.ndarray,
    source_point_id: int,
    grid_on_normals: np.ndarray,
    *,
    k: int = 9,
) -> np.ndarray:
    dismap = np.asarray(dismap, dtype=float)
    grid_on = _as_xyz_array(grid_on, "grid_on")
    normals = _as_xyz_array(grid_on_normals, "grid_on_normals")
    if len(dismap) != len(grid_on):
        raise ValueError("dismap length must match grid_on")
    if normals.shape != grid_on.shape:
        raise ValueError("grid_on_normals must have the same shape as grid_on")
    if source_point_id < 0 or source_point_id >= len(grid_on):
        raise ValueError("source_point_id is outside grid_on")

    candidates: list[int] = []
    for point_id in range(len(grid_on)):
        if calculate_iter_num(dismap, grid_on, point_id, k=k) >= 4:
            candidates.append(point_id)
    if not candidates:
        return np.zeros((0,), dtype=int)

    source_point = grid_on[source_point_id]
    source_normal = normals[source_point_id]
    scores = np.array(
        [diversity_eval(source_point, grid_on[point_id], source_normal, normals[point_id]) for point_id in candidates],
        dtype=float,
    )
    order = np.argsort(scores)[::-1]
    keep = max(1, int(round(len(scores) / 30.0)))
    return np.array(candidates, dtype=int)[order[:keep]]


detectSaddlePoint = detect_saddle_point
