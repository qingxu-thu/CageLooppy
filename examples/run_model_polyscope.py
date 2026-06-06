from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cagingloop import (
    caging_loop_space,
    detect_saddle_point,
    distance_map_by_fast_marching,
    generate_caging_grasps,
    load_obj_point_cloud,
    offset_voxelization,
    point_cloud_voxelization_by_rbf,
    rank_caging_loops,
    transfer_point_normals,
    volumetric_caging_loops,
    voxelize_mesh,
)
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
    morse: bool = False,
    use_convex_hull: bool = False,
    sweep_radius: float | None = None,
    curvature_filter: bool = False,
    base_sampling: str = "farthest",
    rings_per_point: int = 6,
    base_height: float | None = None,
    base_height_band: float = 0.1,
    solver: str = "fmm",
    integrator: str = "euler",
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

    # Transfer the model's true surface normals to each surface voxel, matching
    # MATLAB's `grid_on_normals` input rather than approximating radially.
    surface_normals = transfer_point_normals(normals_source_pts, normals_source_n, voxels.grid_on)

    if sweep > 0:
        # Paper Algorithm 2: pool loops from many base points, with its filtering rules
        # (Thm 3.2 convex-hull grasping space, Thm 3.3 curvature base-point filter,
        # 2h sweep radius). method='morse' uses volumetric saddles (Fig.3), else surface.
        pool = caging_loop_space(
            voxels, surface_normals, base_points=sweep,
            method=("morse" if morse else "surface"),
            use_convex_hull=use_convex_hull, sweep_radius=sweep_radius, curvature_filter=curvature_filter,
            base_sampling=base_sampling, max_loops_per_source=rings_per_point,
            base_height=base_height, base_height_band=base_height_band,
            solver=solver,
        )
        candidates = rank_caging_loops(pool, voxels, mode=select, target_height=target_height)[:max_loops]
        return points, voxels, None, np.zeros((0,), dtype=int), candidates

    if morse:
        # Single base point, volumetric Morse-saddle loops (Fig. 3 + Thm 3.1).
        if source_point_id < 0:
            g = voxels.grid_on
            src = int(np.argmax(np.linalg.norm(g - g.mean(axis=0), axis=1)))
        else:
            src = min(source_point_id, len(voxels.grid_on) - 1)
        pool = volumetric_caging_loops(voxels, src, solver=solver)
        candidates = rank_caging_loops(pool, voxels, mode=select, target_height=target_height)[:max_loops]
        return points, voxels, None, np.zeros((0,), dtype=int), candidates

    if source_point_id < 0:
        # Auto-seed: the surface point farthest from the centroid behaves like a
        # fingertip contact and yields a much better wrapping loop than an
        # arbitrary grid corner.
        grid_on = voxels.grid_on
        source_point_id = int(np.argmax(np.linalg.norm(grid_on - grid_on.mean(axis=0), axis=1)))
    else:
        source_point_id = min(source_point_id, len(voxels.grid_on) - 1)
    distance = distance_map_by_fast_marching(voxels, source_point_id, solver=solver)
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
            distance.distance_grid, voxels, source_point_id, distance.dismap, saddles,
            max_loops=10 * max_loops, integrator=integrator,
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
    parser.add_argument("--morse", action="store_true",
                        help="Use volumetric Morse-saddle loops (paper Fig.3+Thm3.1); can thread holes. Pair with --backend mesh + high --voxel-count.")
    parser.add_argument("--grasp-hull", action="store_true",
                        help="Algorithm 2 / Thm 3.2: bound the grasping space by the surface's convex hull (with --sweep).")
    parser.add_argument("--curvature-filter", action="store_true",
                        help="Algorithm 2 / Thm 3.3: only seed base points with a positive principal curvature (with --sweep).")
    parser.add_argument("--sweep-radius", type=float, default=None,
                        help="Algorithm 2: cap the distance sweep at this radius = 2h (gripper reach), world units (with --sweep).")
    parser.add_argument("--base-sampling", choices=["farthest", "uniform"], default="farthest",
                        help="How to choose the --sweep base points: farthest-point (well-spread) or uniform random (paper).")
    parser.add_argument("--rings-per-point", type=int, default=6,
                        help="Max caging loops kept per base point (largest-area). 16 pts x 6 = up to 96 rings.")
    parser.add_argument("--base-height", type=float, default=None,
                        help="Restrict --sweep base points to this height FRACTION (0..1) of the model (e.g. 0.55).")
    parser.add_argument("--base-height-band", type=float, default=0.1,
                        help="Width of the base-point height band as a fraction of model height (default 0.1).")
    parser.add_argument("--target-height", type=float, default=None,
                        help="With --select waist: aim the loop at this height (world units, along the up axis).")
    parser.add_argument("--max-loops", type=int, default=10, help="How many top caging-loop candidates to keep.")
    parser.add_argument("--loop-rank", type=int, default=0, help="Which candidate to display (rank 0 = best).")
    parser.add_argument("--show-all", action="store_true", help="Show all candidate loops at once (distinct colors).")
    parser.add_argument("--slider", action="store_true", help="With --show-all: add a UI slider to browse loops one at a time.")
    parser.add_argument("--no-show", action="store_true", help="Build and register data without opening the UI.")
    parser.add_argument("--slice-height", type=float, default=None,
                        help="Show the distance field on a horizontal cross-section at this height FRACTION (0..1). "
                             "Needs a volumetric distance field (single-source surface mode).")
    parser.add_argument("--show-mesh", action="store_true", help="Register the actual OBJ triangle mesh (semi-transparent).")
    parser.add_argument("--mesh-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"),
                        help="Translate the OBJ mesh by this world-space offset (to separate it from the voxel layers).")
    parser.add_argument("--solver", choices=["fmm", "msfm", "dijkstra"], default="fmm",
                        help="Distance-field solver: fmm = single-stencil Eikonal fast marching (Euclidean, needs skfmm); "
                             "msfm = multistencil fast marching (MATLAB-parity, lowest anisotropy, pure Python); "
                             "dijkstra = 6-neighbor grid-graph geodesic (Manhattan, no skfmm needed).")
    parser.add_argument("--path-integrator", choices=["euler", "rk4"], default="euler",
                        help="Shortest-path tracer (surface method): euler = normalised-gradient streamline (default); "
                             "rk4 = MATLAB shortestpath parity (RK4 over the pointmin descent field).")
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
        morse=args.morse,
        use_convex_hull=args.grasp_hull,
        sweep_radius=args.sweep_radius,
        curvature_filter=args.curvature_filter,
        base_sampling=args.base_sampling,
        rings_per_point=args.rings_per_point,
        base_height=args.base_height,
        base_height_band=args.base_height_band,
        solver=args.solver,
        integrator=args.path_integrator,
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

    obj_mesh = load_obj_mesh(model_path) if args.show_mesh else None
    mesh_shift = tuple(args.mesh_shift)
    if (args.show_all or args.slider) and candidates:
        labels = [
            f"loop{r:02d} s{c.saddle_id} a{c.area:.3f}" for r, c in enumerate(candidates)
        ]
        show_caging_loops_polyscope(
            voxels,
            [c.path for c in candidates],
            labels=labels,
            distance=distance,
            saddles=saddles,
            slider=args.slider,
            slice_height=args.slice_height,
            obj_mesh=obj_mesh,
            mesh_shift=mesh_shift,
            show=not args.no_show,
        )
    else:
        chosen = candidates[args.loop_rank] if 0 <= args.loop_rank < len(candidates) else None
        caging = chosen.path if chosen is not None else None
        show_pipeline_polyscope(
            voxels, distance=distance, saddles=saddles, caging=caging,
            slice_height=args.slice_height, obj_mesh=obj_mesh, mesh_shift=mesh_shift,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
