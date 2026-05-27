import numpy as np

from cagingloop.saddle import calculate_iter_num, detect_saddle_point, diversity_eval


def test_diversity_eval_prefers_aligned_source_and_saddle_normals():
    source = np.array([0.0, 0.0, 0.0])
    saddle = np.array([1.0, 0.0, 0.0])

    good = diversity_eval(source, saddle, np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    bad = diversity_eval(source, saddle, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    assert good > bad


def test_calculate_iter_num_detects_four_sign_transitions_on_ring():
    center = np.array([[0.0, 0.0, 0.0]])
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    grid_on = np.vstack([center, ring])
    dismap = np.array([1.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])

    assert calculate_iter_num(dismap, grid_on, point_id=0, k=9) >= 4


def test_detect_saddle_point_returns_valid_indices():
    center = np.array([[0.0, 0.0, 0.0]])
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    grid_on = np.vstack([center, ring])
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(grid_on), 1))
    dismap = np.array([1.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])

    saddles = detect_saddle_point(dismap, grid_on, source_point_id=1, grid_on_normals=normals, k=9)

    assert saddles.ndim == 1
    assert np.all(saddles >= 0)
    assert np.all(saddles < len(grid_on))
