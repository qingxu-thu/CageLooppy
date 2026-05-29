import numpy as np

from cagingloop.polyscope_visualization import register_pipeline_polyscope
from cagingloop.types import CagingPath, DistanceMapResult, GridIndex, VoxelizationResult


class FakeStructure:
    def __init__(self):
        self.scalar_quantities = []

    def add_scalar_quantity(self, name, values, enabled=False):
        self.scalar_quantities.append((name, np.asarray(values), enabled))


class FakePolyscope:
    def __init__(self):
        self.point_clouds = []
        self.curve_networks = []

    def register_point_cloud(self, name, points, **kwargs):
        structure = FakeStructure()
        self.point_clouds.append((name, np.asarray(points), kwargs, structure))
        return structure

    def register_curve_network(self, name, nodes, edges, **kwargs):
        self.curve_networks.append((name, np.asarray(nodes), np.asarray(edges), kwargs))
        return FakeStructure()


def _voxelization():
    on_index = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=int)
    grid_on = on_index.astype(float)
    return VoxelizationResult(
        output_grid=np.ones((3, 1, 1), dtype=int),
        grid_x=np.arange(3, dtype=float),
        grid_y=np.array([0.0]),
        grid_z=np.array([0.0]),
        index=GridIndex(
            on_index=on_index,
            inner_index=np.zeros((0, 3), dtype=int),
            outer_index=np.zeros((0, 3), dtype=int),
        ),
        grid_on=grid_on,
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.array([[3.0, 0.0, 0.0]]),
    )


def test_register_pipeline_polyscope_registers_points_scalars_and_path():
    fake = FakePolyscope()
    voxels = _voxelization()
    distance = DistanceMapResult(dismap=np.array([0.0, 1.0, 2.0]), distance_grid=np.zeros((3, 1, 1)))
    caging = CagingPath(
        final_path=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        path1=np.zeros((1, 3)),
        path2=np.zeros((1, 3)),
    )

    registered = register_pipeline_polyscope(
        voxels,
        distance=distance,
        saddles=np.array([1]),
        caging=caging,
        ps_module=fake,
    )

    names = [item[0] for item in fake.point_clouds]
    assert "surface" in names
    assert "outer voxels" in names
    assert "saddle points" in names
    assert fake.point_clouds[0][3].scalar_quantities[0][0] == "distance"
    assert fake.curve_networks[0][0] == "caging path"
    assert fake.curve_networks[0][2].tolist() == [[0, 1], [1, 2], [2, 0]]
    assert set(registered) >= {"surface", "outer", "saddles", "caging_path"}
