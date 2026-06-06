"""Principal-curvature estimate and the Theorem 3.3 base-point filter.

The paper restricts caging-loop base points to surface points with **at least one
positive principal curvature** — a point where the surface is concave in both
directions cannot determine a caging loop (Theorem 3.3). Curvature is estimated by
fitting a local quadric in the tangent frame defined by the surface normal.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - (seed @ n) * n
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v


def principal_curvatures(
    grid_on: np.ndarray,
    normals: np.ndarray,
    point_id: int,
    k: int = 16,
    tree: cKDTree | None = None,
) -> tuple[float, float]:
    """Estimate the two principal curvatures at `grid_on[point_id]` (convex = positive,
    measured against the outward `normals[point_id]`). Fits a local quadric to the
    neighbourhood; returns (0, 0) if there are too few neighbours."""
    grid_on = np.asarray(grid_on, dtype=float)
    tree = tree if tree is not None else cKDTree(grid_on)
    count = min(max(k, 6), len(grid_on))
    _, idx = tree.query(grid_on[point_id], k=count)
    idx = np.atleast_1d(idx)
    neigh = [int(j) for j in idx if int(j) != point_id]
    if len(neigh) < 6:
        return 0.0, 0.0
    p = grid_on[point_id]
    n = np.asarray(normals[point_id], dtype=float)
    u, v = _tangent_frame(n)
    offs = grid_on[neigh] - p
    a = offs @ u
    b = offs @ v
    h = -(offs @ n)  # height "outward" so a convex bulge gives positive curvature
    # full quadratic fit: h = c0 + c1 a + c2 b + 0.5 A a^2 + B ab + 0.5 C b^2
    M = np.column_stack((np.ones_like(a), a, b, 0.5 * a * a, a * b, 0.5 * b * b))
    coef, *_ = np.linalg.lstsq(M, h, rcond=None)
    A, B, C = coef[3], coef[4], coef[5]
    k1, k2 = np.linalg.eigvalsh(np.array([[A, B], [B, C]]))
    return float(k1), float(k2)


def positive_curvature_points(
    grid_on: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 16,
    tol: float = 0.0,
) -> np.ndarray:
    """Theorem 3.3 base-point filter: indices of surface points with at least one
    principal curvature above `tol` (i.e. not concave in both directions)."""
    grid_on = np.asarray(grid_on, dtype=float)
    tree = cKDTree(grid_on)
    keep = []
    for i in range(len(grid_on)):
        k1, k2 = principal_curvatures(grid_on, normals, i, k=k, tree=tree)
        if max(k1, k2) > tol:
            keep.append(i)
    return np.array(keep, dtype=int)
