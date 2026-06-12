"""Single-view RGB-D rendering: perspective pinhole camera + Open3D offscreen backend.

Produces what an RGB-D camera would capture from a known mesh: an RGB render, a
metric view-space depth map, and the back-projected visible-shell point cloud in
both camera and world frames (a single view can only ever observe the front
surface, so the cloud is inherently a shell). OpenCV camera convention: +Z
forward, +X right, +Y down; the extrinsic T maps world to camera. Open3D is
imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # no wheels for Python 3.14 — see pyproject extra `rgbd`
        raise ImportError(
            "open3d is required for RGB-D rendering (pip install open3d; "
            "needs a Python version with open3d wheels, e.g. 3.10-3.12)"
        ) from exc
    return o3d


@dataclass
class Camera:
    """Pinhole camera: K (3x3 intrinsics), T (4x4 world-to-camera extrinsic)."""

    K: np.ndarray
    T: np.ndarray
    width: int
    height: int
    fov_deg: float
    azimuth_deg: float
    elevation_deg: float
    distance: float


def make_camera(
    vertices: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float | None = None,
    *,
    fov_deg: float = 60.0,
    width: int = 640,
    height: int = 480,
    target=None,
) -> Camera:
    """Orbit camera looking at `target` (default: AABB center of `vertices`).

    `distance=None` auto-fits the bounding sphere of `vertices` into ~80% of the
    frame. Azimuth/elevation follow `view_dir_from_angles` (y is world up).
    """
    V = np.asarray(vertices, dtype=float)
    if target is None:
        target = 0.5 * (V.min(axis=0) + V.max(axis=0))
    target = np.asarray(target, dtype=float)

    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    cam_dir = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])

    half_v = np.radians(fov_deg) / 2.0
    fy = (height / 2.0) / np.tan(half_v)
    half_h = np.arctan(np.tan(half_v) * width / height)
    if distance is None:
        radius = float(np.linalg.norm(V - target, axis=1).max())
        distance = radius / np.sin(0.8 * min(half_v, half_h))
    center = target + distance * cam_dir

    forward = -cam_dir  # +Z looks at the target
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(up, forward))) > 0.95:  # view nearly parallel to up
        up = np.array([1.0, 0.0, 0.0])
    x_cam = np.cross(forward, up)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(forward, x_cam)  # right-handed, +Y down in image

    R = np.stack([x_cam, y_cam, forward])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    K = np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return Camera(
        K=K, T=T, width=int(width), height=int(height), fov_deg=float(fov_deg),
        azimuth_deg=float(azimuth_deg), elevation_deg=float(elevation_deg),
        distance=float(distance),
    )


def depth_to_pointclouds(depth, rgb, camera):  # implemented in Task 3
    raise NotImplementedError
