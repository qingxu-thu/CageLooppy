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
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
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
