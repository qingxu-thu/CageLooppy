import numpy as np
import pytest

from cagingloop.voxelization import point_cloud_voxelization_by_rbf


def test_voxelization_classifies_simple_sphere_points():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    normals = points.copy()

    result = point_cloud_voxelization_by_rbf(points, normals, 7, 7, 7, rbf_neighbors=6)

    assert result.output_grid.shape == (7, 7, 7)
    assert result.grid_on.shape[1] == 3
    assert result.index.on_index.shape[1] == 3
    assert len(result.index.on_index) > 0
    assert np.all(result.index.on_index >= 0)
    assert set(np.unique(result.output_grid)).issubset({-1, 0, 1})


def test_voxelization_rejects_mismatched_normals():
    points = np.zeros((3, 3))
    normals = np.zeros((2, 3))

    with pytest.raises(ValueError, match="same shape"):
        point_cloud_voxelization_by_rbf(points, normals, 5, 5, 5)
