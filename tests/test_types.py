import numpy as np

from cagingloop.types import CagingPath, DistanceMapResult, GridIndex, VoxelizationResult


def test_grid_index_arrays_are_integer_arrays():
    index = GridIndex(
        on_index=np.array([[0, 1, 2]]),
        inner_index=np.array([[1, 1, 1]]),
        outer_index=np.array([[2, 2, 2]]),
    )

    assert index.on_index.dtype.kind in "iu"
    assert index.inner_index.shape == (1, 3)
    assert index.outer_index.tolist() == [[2, 2, 2]]


def test_result_dataclasses_hold_expected_arrays():
    index = GridIndex(
        np.zeros((1, 3), dtype=int),
        np.zeros((0, 3), dtype=int),
        np.zeros((0, 3), dtype=int),
    )
    voxels = VoxelizationResult(
        output_grid=np.ones((2, 2, 2), dtype=int),
        grid_x=np.array([0.0, 1.0]),
        grid_y=np.array([0.0, 1.0]),
        grid_z=np.array([0.0, 1.0]),
        index=index,
        grid_on=np.zeros((1, 3)),
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
        mesh=None,
    )
    distance = DistanceMapResult(dismap=np.array([0.0]), distance_grid=np.zeros((2, 2, 2)))
    path = CagingPath(final_path=np.zeros((2, 3)), path1=np.zeros((1, 3)), path2=np.zeros((1, 3)))

    assert voxels.output_grid.shape == (2, 2, 2)
    assert distance.dismap.tolist() == [0.0]
    assert path.final_path.shape == (2, 3)
