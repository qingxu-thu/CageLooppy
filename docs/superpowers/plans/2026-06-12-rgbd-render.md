# Single-View RGB-D Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a mesh from one user-specified monocular viewpoint into RGB + metric depth + the visible-shell point cloud in camera and world frames (spec: `docs/superpowers/specs/2026-06-12-rgbd-render-design.md`).

**Architecture:** A numpy-only camera/back-projection core in `cagingloop/rgbd.py` (testable without open3d) plus Open3D `OffscreenRenderer` calls behind a lazy import; `examples/render_rgbd.py` wraps it as a one-view-per-run CLI writing 5 files to an output dir.

**Tech Stack:** numpy, Open3D (optional extra `rgbd`; lazy import), PIL (PNG), polyscope (optional preview), pytest.

**Environment note:** The repo's main interpreter is Python 3.14, for which open3d has **no wheels**. The user installs open3d themselves (e.g. with the installed Python 3.10). All open3d-dependent tests use `pytest.importorskip` so the 3.14 suite still passes. To fully verify Tasks 4–6, use an env with open3d, e.g.:
`py -3.10 -m venv .venv-o3d && .venv-o3d\Scripts\pip install -e .[rgbd] pillow pytest` then run pytest/CLI with `.venv-o3d\Scripts\python`.

**Conventions (locked by spec):** OpenCV pinhole — camera +Z forward, +X right, +Y down; extrinsic `T` (4×4) maps world→camera (`X_cam = R @ X_world + t`); depth = view-space z, `inf`/NaN background; azimuth/elevation y-up like `view_dir_from_angles` in `cagingloop/depth.py`.

---

### Task 1: Optional dependency + spec amendment

**Files:**
- Modify: `pyproject.toml` (optional-dependencies block)
- Modify: `docs/superpowers/specs/2026-06-12-rgbd-render-design.md`

- [ ] **Step 1: Add the `rgbd` extra**

In `pyproject.toml`, extend `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
plot = ["matplotlib>=3.7"]
mesh = ["scikit-image>=0.21"]
polyscope = ["polyscope>=2.3"]
rgbd = ["open3d>=0.18"]
dev = ["pytest>=7.4"]
```

- [ ] **Step 2: Amend the spec's backend row**

In the spec's Decisions table, change the backend cell to:
`Open3D OffscreenRenderer (optional extra; NO Python 3.14 wheels — user installs open3d in a compatible env, tests skip when absent)`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml docs/superpowers/specs/2026-06-12-rgbd-render-design.md
git commit -m "feat: add open3d optional extra for rgbd rendering"
```

---

### Task 2: `Camera` dataclass + `make_camera` (pure numpy, TDD)

**Files:**
- Create: `cagingloop/rgbd.py`
- Create: `tests/test_rgbd.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rgbd.py`:

```python
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
    assert target_cam[2] == pytest.approx(5.0)           # +Z forward
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rgbd.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'cagingloop.rgbd'`

- [ ] **Step 3: Implement `Camera` + `make_camera`**

Create `cagingloop/rgbd.py`:

```python
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
```

(`depth_to_pointclouds` is imported by the test file but added in Task 3 — the
Task 2 test run will fail on import until Step 3 of Task 3; to keep Task 2
green on its own, add a temporary stub at the bottom of `cagingloop/rgbd.py`:)

```python
def depth_to_pointclouds(depth, rgb, camera):  # implemented in Task 3
    raise NotImplementedError
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rgbd.py -v`
Expected: 4 PASS (`test_make_camera_*`)

- [ ] **Step 5: Commit**

```bash
git add cagingloop/rgbd.py tests/test_rgbd.py
git commit -m "feat: pinhole Camera and make_camera for rgbd rendering"
```

---

### Task 3: `depth_to_pointclouds` (pure numpy, TDD)

**Files:**
- Modify: `cagingloop/rgbd.py` (replace the stub)
- Modify: `tests/test_rgbd.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rgbd.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rgbd.py -v -k depth_to_pointclouds`
Expected: 2 FAIL with `NotImplementedError`

- [ ] **Step 3: Replace the stub with the implementation**

In `cagingloop/rgbd.py`, replace the `depth_to_pointclouds` stub with:

```python
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
```

- [ ] **Step 4: Run the full test file**

Run: `python -m pytest tests/test_rgbd.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add cagingloop/rgbd.py tests/test_rgbd.py
git commit -m "feat: pinhole back-projection to camera/world shell clouds"
```

---

### Task 4: `load_mesh` + `render_rgbd` (Open3D, skippable tests)

**Files:**
- Modify: `cagingloop/rgbd.py` (append)
- Modify: `tests/test_rgbd.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rgbd.py`:

```python
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


def test_load_mesh_missing_file(o3d, tmp_path):
    from cagingloop.rgbd import load_mesh

    with pytest.raises(ValueError, match="no triangles"):
        load_mesh(tmp_path / "nothing.obj")
```

- [ ] **Step 2: Run the tests to verify they fail (or skip without open3d)**

Run: `python -m pytest tests/test_rgbd.py -v`
Expected on Python 3.14 (no open3d): the 5 new tests SKIP, the 6 old ones PASS.
Expected in an open3d env: new tests ERROR with `ImportError: cannot import name 'load_mesh'`.

- [ ] **Step 3: Implement `load_mesh` + `render_rgbd`**

Append to `cagingloop/rgbd.py`:

```python
def load_mesh(path):
    """Read an .obj/.stl/.ply triangle mesh via Open3D; ensure vertex normals."""
    o3d = _require_open3d()
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"no triangles read from {path} (missing, empty, or unsupported file)")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def render_rgbd(mesh, camera: Camera, *, base_color=(0.7, 0.7, 0.7)):
    """One offscreen pass -> (rgb (H,W,3) uint8, depth (H,W) float32 view-space z).

    Background pixels have depth inf. The untextured mesh is shaded with a
    uniform Lambertian material lit along the view direction.
    """
    _require_open3d()
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
    depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32).copy()
    return rgb, depth
```

- [ ] **Step 4: Run the tests in an open3d-capable environment**

Run (open3d env): `<o3d-python> -m pytest tests/test_rgbd.py -v`
Expected: 11 PASS. (On plain 3.14: 6 PASS, 5 SKIP.)
If `render_to_depth_image(z_in_view_space=True)` raises a TypeError on the
installed open3d version, that version is too old — require `open3d>=0.16`.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/rgbd.py tests/test_rgbd.py
git commit -m "feat: open3d offscreen RGB-D rendering of meshes"
```

---

### Task 5: CLI `examples/render_rgbd.py` + smoke test

**Files:**
- Create: `examples/render_rgbd.py`
- Modify: `tests/test_rgbd.py` (append)

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_rgbd.py` (also add `import json`, `import subprocess`,
`import sys`, `from pathlib import Path` at the top of the file):

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run (open3d env): `<o3d-python> -m pytest tests/test_rgbd.py::test_cli_smoke -v`
Expected: FAIL — `examples/render_rgbd.py` does not exist (returncode != 0).

- [ ] **Step 3: Implement the CLI**

Create `examples/render_rgbd.py`:

```python
"""Single-view RGB-D capture of a mesh: RGB render + metric depth + the
visible-shell point cloud in camera AND world frames (world = mesh coordinates,
directly comparable with the source model as completion ground truth).

    python examples/render_rgbd.py Models/knotty.obj --azimuth 40 --elevation 15

Writes into --out (default out/<model>_az<az>_el<el>/): rgb.png, depth.npy
(float32 view-space z, NaN background), cloud_cam.ply, cloud_world.ply,
camera.json (intrinsics K, world->camera extrinsic T, view parameters).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cagingloop.rgbd import depth_to_pointclouds, load_mesh, make_camera, render_rgbd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Single-view RGB-D capture of a mesh.")
    ap.add_argument("model", help="mesh file (.obj/.stl/.ply)")
    ap.add_argument("--azimuth", type=float, default=40.0, help="camera azimuth (deg)")
    ap.add_argument("--elevation", type=float, default=15.0, help="camera elevation (deg)")
    ap.add_argument("--distance", default="auto",
                    help="'auto' (fit object in frame) or a float in mesh units")
    ap.add_argument("--fov", type=float, default=60.0, help="vertical field of view (deg)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=None,
                    help="output directory (default out/<model>_az<az>_el<el>)")
    ap.add_argument("--preview", action="store_true",
                    help="show world-frame cloud + mesh in polyscope")
    args = ap.parse_args(argv)

    import open3d as o3d  # required anyway by load_mesh; explicit here for PLY IO
    from PIL import Image

    mesh = load_mesh(args.model)
    distance = None if str(args.distance).lower() == "auto" else float(args.distance)
    camera = make_camera(
        np.asarray(mesh.vertices), args.azimuth, args.elevation, distance,
        fov_deg=args.fov, width=args.width, height=args.height,
    )
    rgb, depth = render_rgbd(mesh, camera)
    if not np.isfinite(depth).any():
        raise SystemExit(
            "no object pixels in view — try other --azimuth/--elevation or a larger --distance"
        )
    points_cam, points_world, colors = depth_to_pointclouds(depth, rgb, camera)

    out = Path(args.out) if args.out else (
        Path("out") / f"{Path(args.model).stem}_az{args.azimuth:g}_el{args.elevation:g}"
    )
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth.npy", np.where(np.isfinite(depth), depth, np.nan).astype(np.float32))
    for name, pts in (("cloud_cam.ply", points_cam), ("cloud_world.ply", points_world)):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out / name), pc)
    meta = {
        "model": str(args.model),
        "width": camera.width, "height": camera.height, "fov_deg": camera.fov_deg,
        "azimuth_deg": camera.azimuth_deg, "elevation_deg": camera.elevation_deg,
        "distance": camera.distance,
        "K": camera.K.tolist(),
        "T_world_to_cam": camera.T.tolist(),
        "depth_definition": "float32 view-space z along the optical axis; NaN = background; mesh units",
        "cloud_world_frame": "source mesh coordinates",
        "convention": "OpenCV pinhole: +Z forward, +X right, +Y down; X_cam = T[:3,:3] @ X_world + T[:3,3]",
    }
    (out / "camera.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  ({len(points_world)} visible-shell points)")

    if args.preview:
        import polyscope as ps

        ps.init()
        cloud = ps.register_point_cloud("cloud_world", points_world)
        cloud.add_color_quantity("rgb", colors, enabled=True)
        m = ps.register_surface_mesh(
            "mesh", np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        )
        m.set_transparency(0.4)
        ps.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run (open3d env): `<o3d-python> -m pytest tests/test_rgbd.py::test_cli_smoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/render_rgbd.py tests/test_rgbd.py
git commit -m "feat: render_rgbd CLI writing rgb/depth/shell clouds/camera meta"
```

---

### Task 6: Package exports, README, full verification

**Files:**
- Modify: `cagingloop/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Export the rgbd API**

In `cagingloop/__init__.py`, after the `from cagingloop.depth import (...)` block add:

```python
from cagingloop.rgbd import (
    Camera,
    depth_to_pointclouds,
    load_mesh,
    make_camera,
    render_rgbd,
)
```

and extend `__all__` (after `"view_dir_from_angles",`):

```python
    "Camera",
    "depth_to_pointclouds",
    "load_mesh",
    "make_camera",
    "render_rgbd",
```

(`cagingloop/rgbd.py` imports open3d lazily, so this keeps the package importable without open3d. `tests/test_public_api.py` may assert over `__all__` — check whether it needs the five new names added.)

- [ ] **Step 2: Document the tool in README.md**

Append a section (match the README's existing style):

```markdown
## Single-view RGB-D capture (examples/render_rgbd.py)

Render a model from one monocular viewpoint into RGB + depth + the visible-shell
point cloud (camera frame and world/mesh frame). Requires open3d
(`pip install open3d`; needs a Python version with open3d wheels):

    python examples/render_rgbd.py Models/knotty.obj --azimuth 40 --elevation 15

Outputs in out/knotty_az40_el15/: rgb.png, depth.npy (view-space z, NaN
background), cloud_cam.ply, cloud_world.ply, camera.json (K, world->camera T).
Add --preview to inspect the cloud against the mesh in polyscope.
```

- [ ] **Step 3: Run the full suite in both environments**

Run: `python -m pytest -v` (3.14: rgbd open3d tests SKIP, everything else PASS)
Run (open3d env): `<o3d-python> -m pytest tests/test_rgbd.py -v` (all PASS)

- [ ] **Step 4: Render a real model and eyeball the artifacts**

Run (open3d env):
`<o3d-python> examples/render_rgbd.py Models/knotty.obj --azimuth 40 --elevation 15`
Expected: `out/knotty_az40_el15/` with the five files; `rgb.png` shows the shaded
knot; `cloud_world.ply` point count equals finite pixels in `depth.npy`.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/__init__.py README.md
git commit -m "feat: export rgbd API and document the render CLI"
```
