import numpy as np
import pytest

from cagingloop.nearest import NearestTree, nn_prepare, nn_search


def test_nn_search_returns_sorted_neighbors_for_query_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    tree = nn_prepare(points)

    indices, distances = nn_search(points, tree, np.array([[0.2, 0.0, 0.0]]), 2)

    assert indices.tolist() == [[0, 1]]
    assert np.allclose(distances, [[0.2, 0.8]])


def test_nn_search_accepts_query_indices_and_excludes_self():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    tree = NearestTree(points)

    indices, distances = nn_search(points, tree, np.array([1]), 1, exclude=0)

    assert indices.tolist() == [[0]]
    assert np.allclose(distances, [[1.0]])


def test_nn_prepare_rejects_non_xyz_arrays():
    with pytest.raises(ValueError, match="N x 3"):
        nn_prepare(np.array([1.0, 2.0, 3.0]))
