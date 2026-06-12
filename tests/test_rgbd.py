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
