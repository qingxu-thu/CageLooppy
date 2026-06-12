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


def depth_to_pointclouds(depth: np.ndarray, rgb: np.ndarray, camera: Camera):
    """Back-project all finite-depth pixels through the pinhole model.

    Returns `(points_cam (N,3), points_world (N,3), colors (N,3) in [0,1])`.
    Both clouds are the same visible shell; only the frame differs. World frame
    equals the source mesh's coordinates.
    """
    depth = np.asarray(depth, dtype=float)
    mask = np.isfinite(depth)
    v, u = np.nonzero(mask)
    z = depth[mask]
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    x = (u + 0.5 - cx) * z / fx  # +0.5: depth samples live at pixel centers
    y = (v + 0.5 - cy) * z / fy
    points_cam = np.column_stack([x, y, z])
    R, t = camera.T[:3, :3], camera.T[:3, 3]
    points_world = (points_cam - t) @ R  # row-wise R^T @ (p - t)
    colors = np.asarray(rgb, dtype=float)[mask] / 255.0
    return points_cam, points_world, colors


def load_mesh(path):
    """Read an .obj/.stl/.ply triangle mesh via Open3D; ensure vertex normals."""
    o3d = _require_open3d()
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"no triangles read from {path} (missing, empty, or unsupported file)")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        # files shipping vn normals skip compute_vertex_normals, but the legacy
        # renderer needs triangle normals for lighting (else unlit flat output)
        mesh.compute_triangle_normals()
    mesh.normalize_normals()  # some Models/*Normal.obj store non-unit normals
    return mesh


_FILAMENT_UNAVAILABLE = False  # set on first failed probe (e.g. Windows wheels)


def render_rgbd(mesh, camera: Camera, *, base_color=(0.7, 0.7, 0.7)):
    """One offscreen pass -> (rgb (H,W,3) uint8, depth (H,W) float32 view-space z).

    Background pixels have depth inf. The untextured mesh is shaded with a
    uniform Lambertian material. Uses the Filament OffscreenRenderer where
    available and falls back to a hidden legacy Visualizer window elsewhere
    (Windows wheels lack EGL headless support).
    """
    global _FILAMENT_UNAVAILABLE
    o3d = _require_open3d()
    if not _FILAMENT_UNAVAILABLE:
        try:
            return _render_rgbd_filament(mesh, camera, base_color)
        except RuntimeError:
            _FILAMENT_UNAVAILABLE = True  # probe once; don't re-log every call
    return _render_rgbd_hidden_window(o3d, mesh, camera, base_color)


def _render_rgbd_filament(mesh, camera: Camera, base_color):
    from open3d.visualization import rendering

    renderer = rendering.OffscreenRenderer(camera.width, camera.height)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (*base_color, 1.0)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.add_geometry("object", mesh, mat)
    sun_dir = camera.T[2, :3]  # camera forward in world: headlight-style sun
    renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, sun_dir)
    renderer.setup_camera(
        camera.K.astype(np.float64), camera.T.astype(np.float64), camera.width, camera.height
    )
    rgb = np.asarray(renderer.render_to_image())[:, :, :3].copy()
    depth_img = renderer.render_to_depth_image(z_in_view_space=True)
    depth = np.asarray(depth_img, dtype=np.float32).copy()
    return rgb, depth


def _render_rgbd_hidden_window(o3d, mesh, camera: Camera, base_color):
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=camera.width, height=camera.height, visible=False):
        raise RuntimeError("open3d could not create a hidden render window")
    try:
        shaded = o3d.geometry.TriangleMesh(mesh)  # copy: don't mutate the caller's mesh
        shaded.paint_uniform_color(base_color)
        vis.add_geometry(shaded)
        # binding name quirk: MeshShadeOption.Color == smooth shading (interpolate
        # vertex normals); the default is flat shading, which looks faceted
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height,
            camera.K[0, 0], camera.K[1, 1], camera.K[0, 2], camera.K[1, 2],
        )
        params.extrinsic = camera.T
        if not vis.get_view_control().convert_from_pinhole_camera_parameters(
            params, allow_arbitrary=True
        ):
            raise RuntimeError("open3d rejected the pinhole camera parameters")
        rgb_f = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        rgb = np.clip(np.round(rgb_f * 255.0), 0, 255).astype(np.uint8)
        depth = np.asarray(vis.capture_depth_float_buffer(do_render=True), dtype=np.float32).copy()
        depth[depth <= 0.0] = np.inf  # legacy buffer marks background as 0
    finally:
        vis.destroy_window()
    return rgb, depth
