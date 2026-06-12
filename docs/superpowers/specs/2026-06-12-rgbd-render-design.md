# Single-View RGB-D Rendering Design

Date: 2026-06-12
Status: approved pending user review

## Purpose

Generate, from any mesh in `Models/` and an arbitrary user-specified monocular
viewpoint, the data a real RGB-D camera would capture:

- an RGB image of the object (shaded render),
- the corresponding metric depth map,
- the back-projected partial point cloud — the **visible shell only** (a
  single-view depth map can never observe occluded geometry, so the cloud is
  inherently the front surface under that view, not a closed/complete shape),
  in **both camera frame and world frame**.

Primary use: building an offline dataset of partial observations to feed
point-cloud completion / 3D generation methods (GenPC, ComPC, PCDreamer, ...
see `docs/2025plus-depth-partial-pointcloud-3d-generation-papers.md` in the
main repo). One CLI invocation renders one model from one viewpoint; the user
scripts dataset batching externally (or imports the module functions).

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Purpose | Offline dataset generation |
| Camera model | Perspective pinhole (intrinsics saved per view) |
| Rendering backend | Open3D `OffscreenRenderer` (optional extra; NO Python 3.14 wheels — user installs open3d in a compatible env, tests skip when absent) |
| Viewpoint selection | Single user-specified view per run (az/el/distance CLI args) |
| Point cloud frames | Both: camera frame and world frame (same shell, two frames) |

## Architecture

Two new files in the python-port worktree:

```
cagingloop/rgbd.py          # rendering + back-projection library
examples/render_rgbd.py     # CLI wrapper, one view -> one output directory
tests/test_rgbd.py          # unit tests on a synthetic mesh
```

Open3D is imported lazily inside `cagingloop/rgbd.py` so the rest of the
package keeps working without it. `pyproject.toml` gains an optional extra:
`rgbd = ["open3d>=0.18"]`.

## Module: `cagingloop/rgbd.py`

### Camera conventions

OpenCV pinhole convention throughout: camera looks down **+Z**, **+X** right,
**+Y** down. The extrinsic `T` (4x4) maps world to camera:
`X_cam = T[:3,:3] @ X_world + T[:3,3]`. Depth is **view-space z** (distance
along the optical axis, not ray length). Units are the source mesh's units.

### Functions

- `load_mesh(path) -> o3d.geometry.TriangleMesh`
  Reads `.obj` / `.stl` / `.ply` via Open3D IO; computes vertex normals when
  missing; raises a clear error for empty/unreadable meshes.

- `make_camera(azimuth_deg, elevation_deg, distance=None, *, fov_deg=60.0,
  width=640, height=480, target=None) -> Camera`
  Builds intrinsics `K` (from vertical FOV and image size, principal point at
  center) and extrinsic `T`. The camera orbits `target` (default: mesh AABB
  center) at `distance`; `distance=None` auto-fits so the mesh bounding sphere
  fills ~80% of the frame. Azimuth/elevation use the same convention as the
  existing `view_dir_from_angles` in `cagingloop/depth.py` (y is world up).
  `Camera` is a small dataclass: `K, T, width, height, fov_deg` plus the
  originating `azimuth/elevation/distance` for provenance.

- `render_rgbd(mesh, camera, *, base_color=(0.7, 0.7, 0.7)) -> (rgb, depth)`
  One `OffscreenRenderer` pass. The untextured mesh gets a uniform `defaultLit`
  material under the renderer's default lighting. Returns `rgb` (H, W, 3)
  uint8 and `depth` (H, W) float32 view-space z with `inf` on background
  pixels (converted to NaN at save time).

- `depth_to_pointclouds(depth, rgb, camera) -> (points_cam, points_world, colors)`
  Pinhole back-projection of every finite-depth pixel:
  `x = (u - cx) * z / fx`, `y = (v - cy) * z / fy` -> `points_cam` (N, 3).
  `points_world = R^T @ (points_cam - t)` using the extrinsic -> aligned with
  the source mesh (directly comparable to the GT model). `colors` (N, 3) are
  the corresponding RGB pixel values. Both clouds are the same visible shell;
  only the frame differs.

## CLI: `examples/render_rgbd.py`

```
python examples/render_rgbd.py Models/knotty.obj \
    --azimuth 40 --elevation 15 --distance auto \
    --fov 60 --width 640 --height 480 \
    --out out/knotty_az40_el15 [--preview]
```

Defaults: `--azimuth 40 --elevation 15 --distance auto --fov 60 --width 640
--height 480`; `--out` defaults to `out/<model-stem>_az<az>_el<el>/`.

Writes into `--out`:

| File | Content |
|---|---|
| `rgb.png` | 8-bit color render |
| `depth.npy` | float32 view-space depth, NaN = background, mesh units |
| `cloud_cam.ply` | colored visible-shell cloud, **camera frame** |
| `cloud_world.ply` | colored visible-shell cloud, **world frame** (mesh coords) |
| `camera.json` | `K` (3x3), `T` (4x4 world->camera), width/height, fov_deg, azimuth/elevation/distance, model path, depth definition note |

`--preview` opens the world-frame cloud (plus the source mesh, half-
transparent) in polyscope for a visual sanity check; never required for
dataset runs.

## Error handling

- Missing open3d: `ImportError` with message pointing at
  `pip install open3d` / the `rgbd` extra.
- Unreadable or empty mesh: explicit `ValueError` naming the file.
- Zero finite depth pixels (camera inside the object, or object out of
  frame): exit with a message suggesting different angles/distance, write
  nothing.
- Output directory is created with `mkdir(parents=True, exist_ok=True)`;
  existing files are overwritten (re-running a view regenerates it).

## Testing: `tests/test_rgbd.py`

Tests use a synthetic axis-aligned box (`o3d.geometry.TriangleMesh.create_box`)
so ground truth is analytic; they `pytest.importorskip("open3d")` so the suite
still passes where open3d is absent.

1. **Depth correctness**: camera on the +Z axis at known distance looking at a
   box face -> depth at image center equals (distance - half box depth) within
   1e-3 relative tolerance.
2. **Back-projection consistency**: every world-frame point lies on the box
   surface (distance to the box mesh < 2 depth-pixel footprints).
3. **Frame round-trip**: applying the saved extrinsic to `cloud_world`
   reproduces `cloud_cam` to numerical precision.
4. **Shell property**: with the camera on +Z, no world-frame point has z below
   the box's front face minus tolerance (nothing from the occluded back side).
5. **CLI smoke test**: run the CLI on `Models/torus.obj` into a temp dir,
   assert all five output files exist and `camera.json` parses with the
   expected keys.

## Out of scope (YAGNI)

- Batch / multi-view sweeps inside the tool (user scripts these externally).
- Sensor noise models, textures/materials beyond uniform Lambertian.
- Orthographic projection (the existing `cagingloop/depth.py` already covers
  the pipeline's orthographic capture; this tool does not replace it).
- 16-bit PNG depth export (`depth.npy` is lossless; add a converter later if
  some downstream tool demands PNG).
