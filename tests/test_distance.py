import numpy as np

from cagingloop.distance import compute_shortest_path, distance_map_by_fast_marching
from cagingloop.types import GridIndex, VoxelizationResult


def _line_voxelization():
    grid = np.zeros((5, 3, 3), dtype=int)
    on_index = np.array([[x, 1, 1] for x in range(5)], dtype=int)
    for idx in on_index:
        grid[tuple(idx)] = 1
    return VoxelizationResult(
        output_grid=grid,
        grid_x=np.arange(5, dtype=float),
        grid_y=np.arange(3, dtype=float),
        grid_z=np.arange(3, dtype=float),
        index=GridIndex(
            on_index=on_index,
            inner_index=np.zeros((0, 3), dtype=int),
            outer_index=np.zeros((0, 3), dtype=int),
        ),
        grid_on=np.array([[x, 1.0, 1.0] for x in range(5)]),
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
    )


def test_distance_map_fallback_handles_simple_line():
    voxels = _line_voxelization()

    result = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    assert np.isclose(result.dismap[0], 0.0)
    assert result.dismap[-1] > result.dismap[1]


def test_distance_map_traverses_outer_and_surface_but_not_inner():
    grid = np.array([[[1], [-1], [0]]], dtype=int)
    voxels = VoxelizationResult(
        output_grid=grid,
        grid_x=np.array([0.0]),
        grid_y=np.arange(3, dtype=float),
        grid_z=np.array([0.0]),
        index=GridIndex(
            on_index=np.array([[0, 0, 0]], dtype=int),
            inner_index=np.array([[0, 2, 0]], dtype=int),
            outer_index=np.array([[0, 1, 0]], dtype=int),
        ),
        grid_on=np.array([[0.0, 0.0, 0.0]]),
        grids_inner=np.array([[0.0, 2.0, 0.0]]),
        grids_outer=np.array([[0.0, 1.0, 0.0]]),
    )

    result = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    assert np.isfinite(result.distance_grid[0, 1, 0])
    assert not np.isfinite(result.distance_grid[0, 2, 0])


def test_compute_shortest_path_returns_xyz_path_to_source():
    voxels = _line_voxelization()
    distance = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    path = compute_shortest_path(distance.distance_grid, voxels, start_point_id=4, source_point_id=0)

    assert np.allclose(path[0], [4.0, 1.0, 1.0])
    assert np.allclose(path[-1], [0.0, 1.0, 1.0])
    assert path.shape[1] == 3
