from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from cagingloop.types import GridIndex, VoxelizationResult


def _as_xyz_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    if len(arr) == 0:
        raise ValueError(f"{name} must contain at least one point")
    return arr


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("pt_normals contains zero-length normals")
    return normals / lengths[:, None]


def _validate_voxel_counts(counts: tuple[int, int, int]) -> None:
    if any(int(count) < 3 for count in counts):
        raise ValueError("voxel dimensions must be at least 3")


def _grid_coordinates(
    pt_cloud: np.ndarray,
    voxel_xnum: int,
    voxel_ynum: int,
    voxel_znum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = pt_cloud.min(axis=0)
    maxs = pt_cloud.max(axis=0)
    same = mins == maxs
    if np.any(same):
        mins = mins.copy()
        maxs = maxs.copy()
        mins[same] -= 0.5
        maxs[same] += 0.5
    return (
        np.linspace(mins[0], maxs[0], voxel_xnum),
        np.linspace(mins[1], maxs[1], voxel_ynum),
        np.linspace(mins[2], maxs[2], voxel_znum),
    )


def _surface_mask_from_field(field: np.ndarray, threshold: float) -> np.ndarray:
    surface = np.abs(field) <= threshold
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        sign_change = field[tuple(lower)] * field[tuple(upper)] <= 0.0
        surface[tuple(lower)] |= sign_change
        surface[tuple(upper)] |= sign_change
    return surface


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    # MATLAB moves only the x-min/x-max, y-min/y-max and z-max boundary inner
    # voxels to the outer set (the z-min face `tf_z2` is commented out), so the
    # bottom z=0 face stays classified as inner.
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, -1] = True
    return mask


def _coords_from_indices(
    indices: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.column_stack((grid_x[indices[:, 0]], grid_y[indices[:, 1]], grid_z[indices[:, 2]]))


def _extract_mesh(field: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> dict[str, np.ndarray]:
    try:
        from skimage import measure
    except ImportError as exc:
        raise ImportError("extract_mesh=True requires scikit-image") from exc

    spacing = (
        float(grid_x[1] - grid_x[0]),
        float(grid_y[1] - grid_y[0]),
        float(grid_z[1] - grid_z[0]),
    )
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=spacing)
    verts += np.array([grid_x[0], grid_y[0], grid_z[0]])
    return {"verts": verts, "faces": faces.astype(int)}


def point_cloud_voxelization_by_rbf(
    pt_cloud: np.ndarray,
    pt_normals: np.ndarray,
    voxel_xnum: int,
    voxel_ynum: int,
    voxel_znum: int,
    *,
    normal_offset: float | None = None,
    rbf_neighbors: int | None = 64,
    smoothing: float = 0.0,
    extract_mesh: bool = False,
) -> VoxelizationResult:
    pt_cloud = _as_xyz_array(pt_cloud, "pt_cloud")
    pt_normals = _as_xyz_array(pt_normals, "pt_normals")
    if pt_cloud.shape != pt_normals.shape:
        raise ValueError("pt_cloud and pt_normals must have the same shape")
    _validate_voxel_counts((voxel_xnum, voxel_ynum, voxel_znum))

    normals = _normalize_normals(pt_normals)
    grid_x, grid_y, grid_z = _grid_coordinates(pt_cloud, voxel_xnum, voxel_ynum, voxel_znum)
    spacings = np.array([grid_x[1] - grid_x[0], grid_y[1] - grid_y[0], grid_z[1] - grid_z[0]], dtype=float)
    offset = float(normal_offset) if normal_offset is not None else 0.1 * float(np.min(np.abs(spacings)))
    if offset <= 0.0:
        raise ValueError("normal_offset must be positive")

    samples = np.vstack((pt_cloud, pt_cloud + normals * offset, pt_cloud - normals * offset))
    values = np.concatenate(
        (
            np.zeros(len(pt_cloud), dtype=float),
            np.ones(len(pt_cloud), dtype=float),
            -np.ones(len(pt_cloud), dtype=float),
        )
    )

    neighbors = None if rbf_neighbors is None else min(int(rbf_neighbors), len(samples))
    try:
        rbf = RBFInterpolator(samples, values, neighbors=neighbors, smoothing=smoothing, kernel="linear")
    except np.linalg.LinAlgError:
        rbf = RBFInterpolator(samples, values, neighbors=neighbors, smoothing=max(smoothing, 1e-10), kernel="linear")

    x, y, z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    field = np.asarray(rbf(grid_points), dtype=float).reshape((voxel_xnum, voxel_ynum, voxel_znum))

    threshold = 0.05 * max(1.0, float(np.nanmax(np.abs(field))))
    surface = _surface_mask_from_field(field, threshold)
    inner = (field < 0.0) & ~surface
    outer = ~surface & ~inner

    boundary_inner = inner & _boundary_mask(inner.shape)
    inner[boundary_inner] = False
    outer[boundary_inner] = True

    output_grid = np.full(field.shape, -1, dtype=int)
    output_grid[inner] = 0
    output_grid[surface] = 1

    on_index = np.argwhere(surface).astype(int)
    inner_index = np.argwhere(inner).astype(int)
    outer_index = np.argwhere(outer).astype(int)

    mesh = _extract_mesh(field, grid_x, grid_y, grid_z) if extract_mesh else None
    return VoxelizationResult(
        output_grid=output_grid,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        index=GridIndex(on_index=on_index, inner_index=inner_index, outer_index=outer_index),
        grid_on=_coords_from_indices(on_index, grid_x, grid_y, grid_z),
        grids_inner=_coords_from_indices(inner_index, grid_x, grid_y, grid_z),
        grids_outer=_coords_from_indices(outer_index, grid_x, grid_y, grid_z),
        mesh=mesh,
    )


pointCloudVoxelizationByRBF = point_cloud_voxelization_by_rbf


def generalized_winding_number(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_points: np.ndarray,
    *,
    chunk: int = 512,
) -> np.ndarray:
    """Generalized winding number of a triangle mesh at each query point.

    ~1 inside a watertight mesh, ~0 outside, robust to noise. Uses the
    Van Oosterom-Strackee solid-angle formula summed over faces.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
    tri = vertices[faces]
    a0, b0, c0 = tri[:, 0], tri[:, 1], tri[:, 2]
    out = np.empty(len(query_points), dtype=float)
    for start in range(0, len(query_points), chunk):
        p = query_points[start : start + chunk][:, None, :]
        a = a0[None] - p
        b = b0[None] - p
        c = c0[None] - p
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        num = np.einsum("nmi,nmi->nm", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("nmi,nmi->nm", a, b) * lc
            + np.einsum("nmi,nmi->nm", b, c) * la
            + np.einsum("nmi,nmi->nm", c, a) * lb
        )
        out[start : start + chunk] = np.arctan2(num, den).sum(axis=1) / (2.0 * np.pi)
    return out


def _surface_mask_from_inside(inside: np.ndarray) -> np.ndarray:
    """Voxels straddling the inside/outside interface (a thin shell on both sides)."""
    surface = np.zeros(inside.shape, dtype=bool)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        boundary = inside[tuple(lower)] != inside[tuple(upper)]
        surface[tuple(lower)] |= boundary
        surface[tuple(upper)] |= boundary
    return surface


def voxelize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    voxel_xnum: int,
    voxel_ynum: int,
    voxel_znum: int,
    *,
    padding: float = 0.2,
    winding_chunk: int = 512,
) -> VoxelizationResult:
    """Voxelize a watertight triangle mesh directly (no surface reconstruction).

    Classifies every grid cell inside/outside via the generalized winding number,
    which preserves the mesh topology (handles/holes) that the RBF reconstruction
    cannot. The grid is padded around the mesh so exterior space exists for the
    geodesic to wrap through and so the surface is not clipped at the boundary.
    """
    vertices = _as_xyz_array(vertices, "vertices")
    faces = np.asarray(faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an M x 3 array")
    _validate_voxel_counts((voxel_xnum, voxel_ynum, voxel_znum))
    if padding < 0:
        raise ValueError("padding must be non-negative")

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extent = maxs - mins
    extent[extent == 0] = 1.0
    lo = mins - extent * padding
    hi = maxs + extent * padding
    grid_x = np.linspace(lo[0], hi[0], voxel_xnum)
    grid_y = np.linspace(lo[1], hi[1], voxel_ynum)
    grid_z = np.linspace(lo[2], hi[2], voxel_znum)

    x, y, z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    winding = generalized_winding_number(vertices, faces, grid_points, chunk=winding_chunk)
    inside = (winding > 0.5).reshape((voxel_xnum, voxel_ynum, voxel_znum))

    surface = _surface_mask_from_inside(inside)
    inner = inside & ~surface
    outer = ~inside & ~surface

    output_grid = np.full(inside.shape, -1, dtype=int)
    output_grid[inner] = 0
    output_grid[surface] = 1

    on_index = np.argwhere(surface).astype(int)
    inner_index = np.argwhere(inner).astype(int)
    outer_index = np.argwhere(outer).astype(int)

    return VoxelizationResult(
        output_grid=output_grid,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        index=GridIndex(on_index=on_index, inner_index=inner_index, outer_index=outer_index),
        grid_on=_coords_from_indices(on_index, grid_x, grid_y, grid_z),
        grids_inner=_coords_from_indices(inner_index, grid_x, grid_y, grid_z),
        grids_outer=_coords_from_indices(outer_index, grid_x, grid_y, grid_z),
        mesh={"verts": vertices, "faces": faces},
    )
