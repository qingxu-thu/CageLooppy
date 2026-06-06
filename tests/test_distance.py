import numpy as np

from cagingloop.distance import (
    _dijkstra_distance,
    _msfm_distance,
    compute_shortest_path,
    distance_map_by_fast_marching,
)
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


def test_msfm_recovers_euclidean_metric_better_than_dijkstra():
    # In an obstacle-free box, the geodesic distance to a centre source IS the
    # Euclidean radius. MSFM (multistencil) should be near-isotropic, while
    # Dijkstra's grid-graph metric inflates diagonals by exactly sqrt(2)/sqrt(3).
    n, c = 31, 15
    trav = np.ones((n, n, n), dtype=bool)
    src = (c, c, c)
    sp = np.array([1.0, 1.0, 1.0])
    msfm = _msfm_distance(trav, src, sp)
    dij = _dijkstra_distance(trav, src, sp)

    assert msfm[src] == 0.0
    r = 10
    # axis direction is exact for both
    assert abs(msfm[c + r, c, c] - r) < 0.1 * r
    # body diagonal: MSFM within a few %, Dijkstra ~73% over
    true_body = 6 * np.sqrt(3)
    msfm_body = msfm[c + 6, c + 6, c + 6]
    dij_body = dij[c + 6, c + 6, c + 6]
    assert abs(msfm_body - true_body) / true_body < 0.05
    assert dij_body / true_body > 1.5
    # MSFM must be strictly more isotropic than Dijkstra on the diagonal
    assert abs(msfm_body - true_body) < abs(dij_body - true_body)


def test_msfm_solver_routes_through_distance_map():
    voxels = _line_voxelization()
    result = distance_map_by_fast_marching(voxels, start_point_id=0, solver="msfm")
    assert np.isclose(result.dismap[0], 0.0)
    assert result.dismap[-1] > result.dismap[1]


def test_compute_shortest_path_returns_xyz_path_to_source():
    voxels = _line_voxelization()
    distance = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    path = compute_shortest_path(distance.distance_grid, voxels, start_point_id=4, source_point_id=0)

    assert np.allclose(path[0], [4.0, 1.0, 1.0])
    assert np.allclose(path[-1], [0.0, 1.0, 1.0])
    assert path.shape[1] == 3


def test_compute_shortest_path_rk4_reaches_source():
    # The RK4 integrator (MATLAB shortestpath parity) must trace from start to source
    # with both endpoints snapped exactly onto their surface points.
    voxels = _line_voxelization()
    distance = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    path = compute_shortest_path(
        distance.distance_grid, voxels, start_point_id=4, source_point_id=0, integrator="rk4"
    )

    assert np.allclose(path[0], [4.0, 1.0, 1.0])
    assert np.allclose(path[-1], [0.0, 1.0, 1.0])
    assert path.shape[1] == 3


def test_pointmin_descent_field_points_downhill():
    from cagingloop.distance import _pointmin_descent_field

    # A linear ramp in x: distance increases with x, so the descent must have a
    # negative x-component (toward smaller x). pointmin picks the first lowest
    # neighbour in scan order, so on a flat-in-yz ramp the chosen neighbour is the
    # (-1,-1,-1) corner (diagonal) — like MATLAB; we only require it heads downhill.
    d = np.tile(np.arange(6.0)[:, None, None], (1, 3, 3))
    field = _pointmin_descent_field(d)
    v = field[3, 1, 1]
    assert v[0] < 0  # heads toward decreasing distance
    assert np.isclose(np.linalg.norm(v), 1.0)  # unit direction
