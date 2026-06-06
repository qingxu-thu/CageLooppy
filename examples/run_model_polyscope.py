from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cagingloop import (
    detect_saddle_point,
    distance_map_by_fast_marching,
    generate_caging_grasps,
    horizontal_slice_loop,
    load_obj_point_cloud,
    loop_enclosed_area,
    offset_voxelization,
    point_cloud_voxelization_by_rbf,
    rank_caging_loops,
    sweep_caging_loops,
    transfer_point_normals,
    voxelize_mesh,
)
from cagingloop.grasp import CagingCandidate
from cagingloop.model_io import compute_vertex_normals, load_obj_mesh
from cagingloop.polyscope_visualization import show_caging_loops_polyscope, show_pipeline_polyscope


def run_model(
    path: Path,
    *,
    voxel_count: int,
    max_points: int | None,
    source_point_id: int,
    normal_offset: float,
    max_loops: int = 10,
    backend: str = "rbf",
    sweep: int = 0,
    select: str = "area",
    gripper_radius: float = 0.0,
    target_height: float | None = None,
    slice_height: float | None = None,
    slice_margin: float = 0.0,
):
    if backend == "mesh":
        # Voxelize the actual triangle mesh via winding number: crisper, topology-correct
        # surface (no RBF smoothing). Slower — cost is O(grid x faces).
        vertices, faces = load_obj_mesh(path)
        voxels = voxelize_mesh(vertices, faces, voxel_count, voxel_count, voxel_count)
        points = vertices
        vertex_normals = compute_vertex_normals(vertices, faces)
        normals_source_pts, normals_source_n = vertices, vertex_normals
    else:
        points, normals, _ = load_obj_point_cloud(path, max_points=max_points)
        voxels = point_cloud_voxelization_by_rbf(
            points,
            normals,
            voxel_count,
            voxel_count,
            voxel_count,
            rbf_neighbors=min(64, len(points)),
            smoothing=1e-8,
            normal_offset=normal_offset,
        )
        normals_source_pts, normals_source_n = points, normals
    if len(voxels.grid_on) == 0:
        raise ValueError("voxelization produced no surface points")
    if gripper_radius > 0.0:
        # r-offset surface (paper §IV): grow the solid by the gripper radius so thin
        # split features (e.g. trophy columns) fuse into one graspable bundle.
        spacing = float(voxels.grid_x[1] - voxels.grid_x[0])
        radius_voxels = max(1, round(gripper_radius / spacing))
        voxels = offset_voxelization(voxels, radius_voxels)
        print(f"r-offset: gripper_radius={gripper_radius} -> {radius_voxels} voxels, surface now {len(voxels.grid_on)}")
        if len(voxels.grid_on) == 0:
            raise ValueError("r-offset removed all surface points (radius too large)")
    if slice_height is not None:
        # Direct horizontal ring at a chosen height fraction (0..1) of the model — the
        # most reliable way to get a clean horizontal waist loop at an exact height.
        ylo, yhi = voxels.grid_on[:, 1].min(), voxels.grid_on[:, 1].max()
        world_h = ylo + slice_height * (yhi - ylo)
        loop = horizontal_slice_loop(voxels, world_h, up_axis=1, margin=slice_margin)
        cand = CagingCandidate(-1, loop, loop_enclosed_area(loop.final_path), 0.0, -1)
        return points, voxels, None, np.zeros((0,), dtype=int), [cand]

    # Transfer the model's true surface normals to each surface voxel, matching
    # MATLAB's `grid_on_normals` input rather than approximating radially.
    surface_normals = transfer_point_normals(normals_source_pts, normals_source_n, voxels.grid_on)

    if sweep > 0:
        # Paper Algorithm 2: pool loops from many base points so loops anywhere on the
        # object (e.g. a waist loop) are found, not just near one seed.
        distance = None
        saddles = np.zeros((0,), dtype=int)
        pool = sweep_caging_loops(voxels, surface_normals, base_points=sweep, max_loops_per_source=6)
        candidates = rank_caging_loops(pool, voxels, mode=select, target_height=target_height)[:max_loops]
        return points, voxels, distance, saddles, candidates

    if source_point_id < 0:
        # Auto-seed: the surface point farthest from the centroid behaves like a
        # fingertip contact and yields a much better wrapping loop than an
        # arbitrary grid corner.
        grid_on = voxels.grid_on
        source_point_id = int(np.argmax(np.linalg.norm(grid_on - grid_on.mean(axis=0), axis=1)))
    else:
        source_point_id = min(source_point_id, len(voxels.grid_on) - 1)
    distance = distance_map_by_fast_marching(voxels, source_point_id, prefer_fmm=True)
    # Evaluate every saddle candidate (keep wide) and keep the loop that actually
    # wraps the object, instead of the single diversity-top saddle that MATLAB's
    # heuristic returns (which is often a degenerate sliver).
    saddles = detect_saddle_point(
        distance.dismap, voxels.grid_on, source_point_id, surface_normals, keep=len(voxels.grid_on)
    )

    # Keep the top `max_loops` loops ranked by `select` (area = body-wrapping, waist =
    # paper's near-CoG + horizontal graspable loop).
    candidates = []
    if len(saddles) > 0:
        pool = generate_caging_grasps(
            distance.distance_grid, voxels, source_point_id, distance.dismap, saddles, max_loops=10 * max_loops
        )
        candidates = rank_caging_loops(pool, voxels, mode=select, target_height=target_height)[:max_loops]
    return points, voxels, distance, saddles, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CagingLoop model pipeline and show it in Polyscope.")
    parser.add_argument("model", nargs="?", default="Models/knotty.obj")
    parser.add_argument("--voxel-count", type=int, default=17)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--source-point-id", type=int, default=-1, help="-1 = auto (farthest from centroid)")
    parser.add_argument("--normal-offset", type=float, default=1e-4)
    parser.add_argument("--backend", choices=["rbf", "mesh"], default="rbf",
                        help="rbf = RBF reconstruction (like MATLAB); mesh = winding-number voxelization (crisper, slower).")
    parser.add_argument("--gripper-radius", type=float, default=0.0,
                        help="r-offset (paper S_r): grow the solid by this radius (world units) to fuse thin gaps.")
    parser.add_argument("--sweep", type=int, default=0,
                        help="Sample N base points (paper Algorithm 2) and pool their loops. 0 = single source.")
    parser.add_argument("--select", choices=["area", "small", "waist"], default="area",
                        help="Loop ranking: area = largest (body); small = smallest non-degenerate (handle); waist = near CoG + horizontal.")
    parser.add_argument("--target-height", type=float, default=None,
                        help="With --select waist: aim the loop at this height (world units, along the up axis).")
    parser.add_argument("--slice-height", type=float, default=None,
                        help="Direct horizontal ring at this height FRACTION (0..1) of the model (e.g. 0.55 = neck).")
    parser.add_argument("--slice-margin", type=float, default=0.02,
                        help="Expand the slice ring outward by this much (world units) into the free space.")
    parser.add_argument("--max-loops", type=int, default=10, help="How many top caging-loop candidates to keep.")
    parser.add_argument("--loop-rank", type=int, default=0, help="Which candidate to display (rank 0 = best).")
    parser.add_argument("--show-all", action="store_true", help="Show all candidate loops at once (distinct colors).")
    parser.add_argument("--no-show", action="store_true", help="Build and register data without opening the UI.")
    args = parser.parse_args()

    model_path = Path(args.model)
    points, voxels, distance, saddles, candidates = run_model(
        model_path,
        voxel_count=args.voxel_count,
        max_points=args.max_points,
        source_point_id=args.source_point_id,
        normal_offset=args.normal_offset,
        max_loops=args.max_loops,
        backend=args.backend,
        sweep=args.sweep,
        select=args.select,
        gripper_radius=args.gripper_radius,
        target_height=args.target_height,
        slice_height=args.slice_height,
        slice_margin=args.slice_margin,
    )
    print(f"model: {model_path}")
    print(f"backend: {args.backend}  sweep: {args.sweep}  select: {args.select}")
    print(f"input points: {len(points)}")
    print(f"normal offset: {args.normal_offset}")
    print(f"surface points: {len(voxels.grid_on)}")
    print(f"inner voxels: {len(voxels.grids_inner)}")
    print(f"outer voxels: {len(voxels.grids_outer)}")
    print(f"saddle candidates: {len(saddles)}")
    print(f"caging loop candidates: {len(candidates)}")
    if candidates:
        ylo, yhi = voxels.grid_on[:, 1].min(), voxels.grid_on[:, 1].max()
        print(f"{'rank':>4} {'src':>6} {'saddle':>7} {'area':>8} {'sep':>6} {'height':>7} {'tilt':>6}")
        for rank, c in enumerate(candidates):
            fp = c.path.final_path
            hfrac = (fp[:, 1].mean() - ylo) / (yhi - ylo + 1e-9)
            ext = fp.max(0) - fp.min(0)
            tilt = ext[1] / (ext.max() + 1e-9)
            mark = "  <-- shown" if rank == args.loop_rank else ""
            print(f"{rank:>4} {c.source_id:>6} {c.saddle_id:>7} {c.area:>8.4f} {c.separation:>6.2f} "
                  f"{hfrac:>7.2f} {tilt:>6.2f}{mark}")

    if args.show_all and candidates:
        labels = [
            f"loop{r:02d} s{c.saddle_id} a{c.area:.3f}" for r, c in enumerate(candidates)
        ]
        show_caging_loops_polyscope(
            voxels,
            [c.path for c in candidates],
            labels=labels,
            distance=distance,
            saddles=saddles,
            show=not args.no_show,
        )
    else:
        chosen = candidates[args.loop_rank] if 0 <= args.loop_rank < len(candidates) else None
        caging = chosen.path if chosen is not None else None
        show_pipeline_polyscope(voxels, distance=distance, saddles=saddles, caging=caging, show=not args.no_show)


if __name__ == "__main__":
    main()
