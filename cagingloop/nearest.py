from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    return arr


@dataclass(frozen=True)
class NearestTree:
    points: np.ndarray

    def __post_init__(self) -> None:
        points = _as_points(self.points, "points")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "tree", cKDTree(points))


def nn_prepare(pointset: np.ndarray) -> NearestTree:
    return NearestTree(pointset)


def nn_search(
    pointset: np.ndarray,
    tree: NearestTree,
    query: np.ndarray,
    k: int,
    exclude: int | None = None,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    points = _as_points(pointset, "pointset")
    if points.shape != tree.points.shape or not np.allclose(points, tree.points):
        raise ValueError("pointset must match the prepared nearest-neighbor tree")
    if k < 1:
        raise ValueError("k must be at least 1")

    query_arr = np.asarray(query)
    query_is_indices = query_arr.ndim == 1 and np.issubdtype(query_arr.dtype, np.integer)
    if query_is_indices:
        query_indices = query_arr.astype(int)
        if np.any(query_indices < 0) or np.any(query_indices >= len(points)):
            raise ValueError("query indices are outside pointset")
        query_points = points[query_indices]
    else:
        query_points = _as_points(query_arr, "query")
        query_indices = None

    ask_k = k
    if query_indices is not None and exclude is not None:
        ask_k = min(len(points), k + 2 * exclude + 1)

    distances, indices = tree.tree.query(query_points, k=ask_k, eps=epsilon)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    if query_indices is None or exclude is None:
        return indices[:, :k].astype(int), distances[:, :k].astype(float)

    out_indices: list[np.ndarray] = []
    out_distances: list[np.ndarray] = []
    for row, center in enumerate(query_indices):
        mask = np.abs(indices[row] - center) > exclude
        kept_indices = indices[row][mask][:k]
        kept_distances = distances[row][mask][:k]
        if len(kept_indices) < k:
            raise ValueError("not enough neighbors after applying exclude")
        out_indices.append(kept_indices.astype(int))
        out_distances.append(kept_distances.astype(float))
    return np.vstack(out_indices), np.vstack(out_distances)
