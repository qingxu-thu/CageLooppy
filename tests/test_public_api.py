import cagingloop
from cagingloop.visualization import plot_caging_path, plot_points


def test_public_api_exports_main_workflow_functions():
    for name in [
        "point_cloud_voxelization_by_rbf",
        "distance_map_by_fast_marching",
        "detect_saddle_point",
        "generate_caging_grasp",
    ]:
        assert hasattr(cagingloop, name)


def test_visualization_helpers_are_importable():
    assert callable(plot_points)
    assert callable(plot_caging_path)
