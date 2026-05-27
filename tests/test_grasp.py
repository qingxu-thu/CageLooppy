import numpy as np

from cagingloop.distance import distance_map_by_fast_marching
from cagingloop.grasp import generate_caging_grasp, smooth_closed_path
from cagingloop.types import GridIndex, VoxelizationResult


def test_smooth_closed_path_preserves_shape_and_closes_loop():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    smoothed = smooth_closed_path(path)

    assert smoothed.shape == path.shape
    assert np.allclose(smoothed[0], smoothed[-1])


def test_generate_caging_grasp_returns_paths():
    grid = np.zeros((5, 5, 3), dtype=int)
    on_index = np.array([[x, 2, 1] for x in range(5)] + [[2, y, 1] for y in range(5)], dtype=int)
    on_index = np.unique(on_index, axis=0)
    for idx in on_index:
        grid[tuple(idx)] = 1
    grid_on = np.array([[idx[0], idx[1], idx[2]] for idx in on_index], dtype=float)
    voxels = VoxelizationResult(
        output_grid=grid,
        grid_x=np.arange(5, dtype=float),
        grid_y=np.arange(5, dtype=float),
        grid_z=np.arange(3, dtype=float),
        index=GridIndex(
            on_index=on_index,
            inner_index=np.zeros((0, 3), dtype=int),
            outer_index=np.zeros((0, 3), dtype=int),
        ),
        grid_on=grid_on,
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
    )
    source = int(np.where((on_index == [0, 2, 1]).all(axis=1))[0][0])
    saddle = int(np.where((on_index == [2, 2, 1]).all(axis=1))[0][0])
    distance = distance_map_by_fast_marching(voxels, source, prefer_fmm=False)

    caging = generate_caging_grasp(distance.distance_grid, voxels, source, distance.dismap, saddle, k=5)

    assert caging.final_path.shape[1] == 3
    assert len(caging.path1) > 0
    assert len(caging.path2) > 0
