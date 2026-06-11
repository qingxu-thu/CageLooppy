import numpy as np

from cagingloop.distance import distance_map_by_fast_marching
from cagingloop.grasp import (
    CagingCandidate,
    generate_caging_grasp,
    isoperimetric_ratio,
    loop_encircles_solid,
    loop_locally_shortest_at_base,
    smooth_closed_path,
)
from cagingloop.types import CagingPath, GridIndex, VoxelizationResult


def _free_voxelization():
    # 5^3 grid, no interior solid (output_grid all -1) so any shortcut is collision-free.
    n = 5
    grid = np.full((n, n, n), -1, dtype=int)
    on = np.array([[2, 2, 0]], dtype=int)  # base point p lives here (index 0)
    return VoxelizationResult(
        output_grid=grid,
        grid_x=np.arange(n, dtype=float),
        grid_y=np.arange(n, dtype=float),
        grid_z=np.arange(n, dtype=float),
        index=GridIndex(on_index=on, inner_index=np.zeros((0, 3), int), outer_index=np.zeros((0, 3), int)),
        grid_on=on.astype(float),
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
    )


def _candidate(final_path):
    fp = np.asarray(final_path, dtype=float)
    return CagingCandidate(0, CagingPath(final_path=fp, path1=fp, path2=fp), 1.0, 1.0, 0)


def test_ls_filter_keeps_loop_straight_through_base():
    vox = _free_voxelization()  # p = [2,2,0]
    # loop passes ~straight through p along +y: chord == sub-path -> straightness 1 -> keep
    loop = _candidate([[2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 4, 2], [2, 0, 2]])
    assert loop_locally_shortest_at_base(loop, vox, window=1) is True


def test_ls_filter_drops_freespace_kink_at_base():
    vox = _free_voxelization()  # p = [2,2,0], nothing solid -> shortcut is free
    # sharp corner at p ([2,2,0]); chord [1,1,0]->[1,3,0] is shorter than the sub-path and free
    loop = _candidate([[0, 0, 0], [1, 1, 0], [2, 2, 0], [1, 3, 0], [0, 4, 0], [0, 2, 2]])
    assert loop_locally_shortest_at_base(loop, vox, window=1) is False


def test_isoperimetric_ratio_line_vs_square():
    # a compact loop has rho ~ 0.05-0.08; a near-line sliver collapses to rho ~ 0
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    sliver = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0.001, 0]], dtype=float)
    rho_square = isoperimetric_ratio(square)
    rho_sliver = isoperimetric_ratio(sliver)
    assert rho_square > 0.03            # compact
    assert rho_sliver < 0.005           # line-like
    assert rho_square > 5.0 * rho_sliver


def test_link_filter_keeps_encircling_loop_drops_freespace_loop():
    # a solid bar runs along z at (x=3, y=3); the loop's encirclement decides keep/drop,
    # independent of the loop's size.
    n = 7
    grid = np.full((n, n, n), -1, dtype=int)
    grid[3, 3, :] = 0  # interior column = the "handle bar"
    vox = VoxelizationResult(
        output_grid=grid, grid_x=np.arange(n, dtype=float), grid_y=np.arange(n, dtype=float),
        grid_z=np.arange(n, dtype=float),
        index=GridIndex(on_index=np.array([[3, 3, 0]]), inner_index=np.zeros((0, 3), int),
                        outer_index=np.zeros((0, 3), int)),
        grid_on=np.array([[3.0, 3.0, 0.0]]), grids_inner=np.zeros((0, 3)), grids_outer=np.zeros((0, 3)),
    )
    around_bar = _candidate([[1, 1, 3], [5, 1, 3], [5, 5, 3], [1, 5, 3]])   # disk contains (3,3,3) interior
    in_free_space = _candidate([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])  # disk is all free
    assert loop_encircles_solid(around_bar, vox) is True
    assert loop_encircles_solid(in_free_space, vox) is False


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
