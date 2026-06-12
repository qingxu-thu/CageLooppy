import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cagingloop.rgbd import depth_to_pointclouds, make_camera


def test_make_camera_intrinsics():
    V = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cam = make_camera(V, 40.0, 15.0, 5.0, fov_deg=60.0, width=640, height=480)
    assert cam.K[1, 1] == pytest.approx(240.0 / np.tan(np.radians(30.0)))
    assert cam.K[0, 0] == cam.K[1, 1]
    assert cam.K[0, 2] == 320.0 and cam.K[1, 2] == 240.0
    assert cam.width == 640 and cam.height == 480
    assert cam.distance == 5.0


def test_make_camera_extrinsic_looks_at_target():
    V = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])  # AABB center = origin
    cam = make_camera(V, 40.0, 15.0, 5.0)
    R, t = cam.T[:3, :3], cam.T[:3, 3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    center = -R.T @ t  # camera position in world
    assert np.linalg.norm(center) == pytest.approx(5.0)
    target_cam = t  # origin mapped to camera frame
    assert np.allclose(target_cam[:2], 0.0, atol=1e-9)  # target on optical axis
    assert target_cam[2] == pytest.approx(5.0)  # +Z forward
    assert float(R[1] @ np.array([0.0, 1.0, 0.0])) < 0.0  # +Y is downward


def test_make_camera_auto_distance_fits_object():
    V = np.random.default_rng(0).uniform(-1.0, 1.0, (100, 3))
    cam = make_camera(V, 10.0, 20.0, None, fov_deg=60.0, width=640, height=480)
    R, t = cam.T[:3, :3], cam.T[:3, 3]
    pts_cam = V @ R.T + t
    assert np.all(pts_cam[:, 2] > 0)  # everything in front of the camera
    u = cam.K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + cam.K[0, 2]
    v = cam.K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + cam.K[1, 2]
    assert u.min() >= 0.0 and u.max() < cam.width
    assert v.min() >= 0.0 and v.max() < cam.height


def test_make_camera_handles_top_down_view():
    V = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cam = make_camera(V, 0.0, 89.0, 5.0)  # nearly parallel to world up
    R = cam.T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_depth_to_pointclouds_roundtrip():
    V = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cam = make_camera(V, 33.0, -25.0, 4.0, width=64, height=48)
    depth = np.full((48, 64), np.inf, dtype=np.float32)
    depth[10:30, 20:50] = 2.5
    rgb = np.full((48, 64, 3), 128, dtype=np.uint8)
    p_cam, p_world, colors = depth_to_pointclouds(depth, rgb, cam)
    assert p_cam.shape == p_world.shape == (20 * 30, 3)
    assert colors.shape == (20 * 30, 3)
    assert np.allclose(p_cam[:, 2], 2.5)
    R, t = cam.T[:3, :3], cam.T[:3, 3]
    assert np.allclose(p_world @ R.T + t, p_cam, atol=1e-9)  # frame round-trip
    assert np.allclose(colors, 128.0 / 255.0)


def test_depth_to_pointclouds_center_pixel_on_axis():
    V = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cam = make_camera(V, 0.0, 0.0, 4.0, width=64, height=48)
    depth = np.full((48, 64), np.inf, dtype=np.float32)
    depth[24, 32] = 2.0  # pixel just below/right of the exact center
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    p_cam, _, _ = depth_to_pointclouds(depth, rgb, cam)
    # pixel center (32.5, 24.5) vs principal point (32, 24): offset 0.5px * z/f
    assert np.allclose(p_cam[0, :2], 0.5 * 2.0 / cam.K[0, 0], atol=1e-9)
    assert p_cam[0, 2] == pytest.approx(2.0)


@pytest.fixture(scope="module")
def o3d():
    return pytest.importorskip("open3d")


@pytest.fixture(scope="module")
def box_capture(o3d):
    from cagingloop.rgbd import load_mesh, render_rgbd  # noqa: F401 (load_mesh: import check)

    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    mesh.translate((-0.5, -0.5, -0.5))  # center the unit box at the origin
    mesh.compute_vertex_normals()
    # az=90, el=0 -> camera at (0, 0, 3) looking down -Z at the z=+0.5 face
    cam = make_camera(np.asarray(mesh.vertices), 90.0, 0.0, 3.0, width=160, height=120)
    rgb, depth = render_rgbd(mesh, cam)
    return mesh, cam, rgb, depth


def test_render_center_depth(box_capture):
    _, _, _, depth = box_capture
    assert depth[60, 80] == pytest.approx(2.5, rel=1e-3)  # 3.0 - half box depth


def test_render_rgb_object_visible(box_capture):
    _, _, rgb, depth = box_capture
    assert rgb.shape == (120, 160, 3) and rgb.dtype == np.uint8
    obj = np.isfinite(depth)
    assert obj.sum() > 100
    assert rgb[obj].mean() < rgb[~obj].mean()  # object darker than white background


def test_backprojection_lies_on_box_surface(box_capture):
    _, cam, rgb, depth = box_capture
    _, p_world, _ = depth_to_pointclouds(depth, rgb, cam)
    tol = 0.02
    assert np.all(np.abs(p_world) <= 0.5 + tol)  # inside the (inflated) box
    on_face = np.any(np.abs(np.abs(p_world) - 0.5) <= tol, axis=1)
    assert on_face.mean() > 0.99


def test_visible_shell_only(box_capture):
    _, cam, rgb, depth = box_capture
    _, p_world, _ = depth_to_pointclouds(depth, rgb, cam)
    assert p_world[:, 2].min() > -0.4  # nothing from the occluded back face (z=-0.5)


def test_render_smooth_shading(o3d):
    from cagingloop.rgbd import render_rgbd

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    sphere.compute_vertex_normals()
    cam = make_camera(np.asarray(sphere.vertices), 0.0, 0.0, 3.0, width=320, height=240)

    # flat reference: explode into per-face vertices so normals are face normals
    V = np.asarray(sphere.vertices)
    F = np.asarray(sphere.triangles)
    flat = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V[F.reshape(-1)]),
        o3d.utility.Vector3iVector(np.arange(3 * len(F)).reshape(-1, 3)),
    )
    flat.compute_vertex_normals()

    def p99_frontfacing_gradient(rgb, depth):
        # front-facing pixels only (small depth gradient): smooth shading is
        # near-constant there, while flat shading still shows facet jumps
        gray = rgb.astype(float).mean(axis=2)
        fin = np.isfinite(depth)
        d = np.where(fin, depth, 0.0)
        sel = fin[:, :-1] & fin[:, 1:] & (np.abs(np.diff(d, axis=1)) < 0.01)
        return np.percentile(np.abs(np.diff(gray, axis=1))[sel], 99)

    smooth_g = p99_frontfacing_gradient(*render_rgbd(sphere, cam))
    flat_g = p99_frontfacing_gradient(*render_rgbd(flat, cam))
    assert smooth_g < 0.5 * flat_g  # interpolated normals: no facet-boundary jumps


def test_load_mesh_missing_file(o3d, tmp_path):
    from cagingloop.rgbd import load_mesh

    with pytest.raises(ValueError, match="no triangles"):
        load_mesh(tmp_path / "nothing.obj")


def test_cli_smoke(o3d, tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "view"
    r = subprocess.run(
        [sys.executable, "examples/render_rgbd.py", "Models/torus.obj",
         "--azimuth", "30", "--elevation", "20", "--out", str(out)],
        cwd=root, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    for name in ["rgb.png", "depth.npy", "cloud_cam.ply", "cloud_world.ply", "camera.json"]:
        assert (out / name).exists(), name
    meta = json.loads((out / "camera.json").read_text())
    assert {"K", "T_world_to_cam", "width", "height", "fov_deg",
            "azimuth_deg", "elevation_deg", "distance", "model"} <= set(meta)
    assert np.array(meta["K"]).shape == (3, 3)
    assert np.array(meta["T_world_to_cam"]).shape == (4, 4)
    depth = np.load(out / "depth.npy")
    assert depth.shape == (480, 640)
    assert np.isnan(depth).any() and np.isfinite(depth).any()
