"""Single-view depth capture + occlusion-shadow voxelization.

Simulates the paper's on-the-fly RGB-D scenario from a known mesh/point cloud:
render a single orthographic depth view (z-buffer hidden-point removal) to get the
visible front-facing shell, then voxelize it. The key switch is `occlusion_solid`:
treating the camera's line-of-sight shadow (everything behind the visible shell)
as solid (speed 0) — the field-level fix that stops the wavefront leaking behind an
open shell and forming phantom saddles (Defense 2 / occlusion-as-solid). With it off
you get the leaky open shell, for A/B comparison.
"""

from __future__ import annotations

import numpy as np

from cagingloop.model_io import transfer_point_normals
from cagingloop.types import GridIndex, VoxelizationResult


def _camera_frame(view_dir: np.ndarray, world_up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal camera frame (right, up, dir); `dir` points from camera into the scene."""
    d = np.asarray(view_dir, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    wu = np.asarray(world_up, dtype=float)
    if abs(float(np.dot(wu, d))) > 0.95:  # view nearly parallel to up -> pick another reference
        wu = np.array([1.0, 0.0, 0.0])
    right = np.cross(wu, d)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(d, right)
    return right, up, d


def view_dir_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Camera-to-scene direction from azimuth/elevation (degrees); y is world up."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    cam = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    return -cam  # look from the camera toward the origin


def render_depth_cloud(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    view_dir: np.ndarray,
    resolution: int = 120,
):
    """Orthographic single-view capture: keep only the front-facing point nearest the
    camera in each pixel (z-buffer hidden-point removal). Returns the visible points and
    normals plus the camera frame + depth buffer (reused by `depth_voxelization`)."""
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    right, up, d = _camera_frame(view_dir)

    a = points @ right
    b = points @ up
    depth = points @ d  # increases away from the camera
    front = (normals @ d) < 0.0  # normal points back toward the camera

    amin, amax = float(a.min()), float(a.max())
    bmin, bmax = float(b.min()), float(b.max())

    def to_px(coord, lo, hi):
        return np.clip(np.round((coord - lo) / (hi - lo + 1e-12) * (resolution - 1)).astype(int), 0, resolution - 1)

    cand = np.where(front)[0]
    pa, pb = to_px(a[cand], amin, amax), to_px(b[cand], bmin, bmax)
    order = cand[np.argsort(depth[cand])]  # nearest-to-camera first
    pa_o, pb_o = to_px(a[order], amin, amax), to_px(b[order], bmin, bmax)

    depth_buffer = np.full((resolution, resolution), np.inf, dtype=float)
    seen = np.zeros((resolution, resolution), dtype=bool)
    visible = []
    for idx, px, py in zip(order, pa_o, pb_o):
        if not seen[px, py]:
            seen[px, py] = True
            depth_buffer[px, py] = depth[idx]
            visible.append(idx)
    visible = np.array(visible, dtype=int)

    frame = {
        "right": right, "up": up, "dir": d,
        "amin": amin, "amax": amax, "bmin": bmin, "bmax": bmax,
        "resolution": resolution, "depth_buffer": depth_buffer,
    }
    return points[visible], normals[visible], frame


def occlusion_shadow_voxelization(
    visible_points: np.ndarray,
    visible_normals: np.ndarray,
    frame: dict,
    *,
    voxel_count: int = 48,
    padding: float = 0.15,
    surface_band: float = 1.2,
    occlusion_solid: bool = True,
):
    """Voxelize the single-view shell. Each voxel is projected into the camera; comparing
    its depth to the visible shell depth at that pixel classifies it:

    - within `surface_band` voxels of the shell depth -> surface (1);
    - behind the shell -> solid (0) if `occlusion_solid` else free (-1)  [the switch];
    - in front of the shell, or background pixels -> free (-1).

    With `occlusion_solid=True` the line-of-sight shadow becomes an impassable pillar so
    the fast-marching front cannot leak behind the shell (Defense 2)."""
    right, up, d = frame["right"], frame["up"], frame["dir"]
    res = frame["resolution"]
    amin, amax, bmin, bmax = frame["amin"], frame["amax"], frame["bmin"], frame["bmax"]
    depth_buffer = frame["depth_buffer"]

    lo = visible_points.min(axis=0)
    hi = visible_points.max(axis=0)
    pad = padding * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx = np.linspace(lo[0], hi[0], voxel_count)
    gy = np.linspace(lo[1], hi[1], voxel_count)
    gz = np.linspace(lo[2], hi[2], voxel_count)
    spacing = float(min(gx[1] - gx[0], gy[1] - gy[0], gz[1] - gz[0]))

    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    centers = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    a = centers @ right
    b = centers @ up
    depth_c = centers @ d

    def to_px(coord, c_lo, c_hi):
        return np.clip(np.round((coord - c_lo) / (c_hi - c_lo + 1e-12) * (res - 1)).astype(int), 0, res - 1)

    px, py = to_px(a, amin, amax), to_px(b, bmin, bmax)
    shell_depth = depth_buffer[px, py]
    has_shell = np.isfinite(shell_depth)
    band = surface_band * spacing
    delta = depth_c - shell_depth

    output = np.full(centers.shape[0], -1, dtype=int)  # default free / exterior
    surface = has_shell & (np.abs(delta) <= band)
    behind = has_shell & (delta > band)
    output[surface] = 1
    if occlusion_solid:
        output[behind] = 0  # line-of-sight shadow -> solid
    output = output.reshape((voxel_count, voxel_count, voxel_count))

    on_index = np.argwhere(output == 1).astype(int)
    inner_index = np.argwhere(output == 0).astype(int)
    outer_index = np.argwhere(output == -1).astype(int)

    def coords(idx):
        if len(idx) == 0:
            return np.zeros((0, 3), dtype=float)
        return np.column_stack((gx[idx[:, 0]], gy[idx[:, 1]], gz[idx[:, 2]]))

    grid_on = coords(on_index)
    voxels = VoxelizationResult(
        output_grid=output,
        grid_x=gx, grid_y=gy, grid_z=gz,
        index=GridIndex(on_index=on_index, inner_index=inner_index, outer_index=outer_index),
        grid_on=grid_on,
        grids_inner=coords(inner_index),
        grids_outer=coords(outer_index),
    )
    surface_normals = (
        transfer_point_normals(visible_points, visible_normals, grid_on)
        if len(grid_on) else np.zeros((0, 3))
    )
    return voxels, surface_normals
