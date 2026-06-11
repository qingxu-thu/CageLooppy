"""Single-view depth-capture caging-loop demo.

Renders one orthographic depth view of a model (simulating an RGB-D capture), voxelizes
the visible shell, and computes caging loops. The headline switch is `--occlusion-solid`:
treat the camera's line-of-sight shadow as solid so the fast-marching front cannot leak
behind the open shell (Defense 2). Run it both ways to see the difference:

    # leaky open shell (phantom saddles, contractible rim loops)
    python examples/run_depth_polyscope.py Models/knotty.obj --azimuth 40 --no-occlusion-solid
    # occlusion-as-solid (front forced through visible features; collision-safe)
    python examples/run_depth_polyscope.py Models/knotty.obj --azimuth 40

Base points are sampled ONLY on the genuine visible shell (never the fabricated shadow).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cagingloop import (
    detect_saddle_point,
    distance_map_by_fast_marching,
    filter_by_isoperimetric,
    filter_by_linking,
    filter_locally_shortest,
    generate_caging_grasps,
    occlusion_shadow_voxelization,
    rank_caging_loops,
    render_depth_cloud,
    view_dir_from_angles,
)
from cagingloop.model_io import compute_vertex_normals, load_obj_mesh
from cagingloop.morse import detect_morse_saddles_3d, volumetric_caging_loops
from cagingloop.polyscope_visualization import show_internals_polyscope


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-view depth caging-loop demo (occlusion-shadow switch).")
    ap.add_argument("model", nargs="?", default="Models/knotty.obj")
    ap.add_argument("--azimuth", type=float, default=40.0, help="Camera azimuth (deg).")
    ap.add_argument("--elevation", type=float, default=15.0, help="Camera elevation (deg).")
    ap.add_argument("--depth-res", type=int, default=140, help="Depth-map resolution (z-buffer pixels).")
    ap.add_argument("--voxel-count", type=int, default=48)
    ap.add_argument("--padding", type=float, default=0.15)
    ap.add_argument("--occlusion-solid", dest="occlusion_solid", action="store_true", default=True,
                    help="Treat the camera line-of-sight shadow as solid (Defense 2; default on).")
    ap.add_argument("--no-occlusion-solid", dest="occlusion_solid", action="store_false",
                    help="Leaky open shell (no shadow solid) for A/B comparison.")
    ap.add_argument("--method", choices=["morse", "surface"], default="morse")
    ap.add_argument("--path-integrator", choices=["euler", "rk4"], default="euler")
    ap.add_argument("--solver", choices=["fmm", "msfm", "dijkstra"], default="fmm")
    ap.add_argument("--sweep", type=int, default=1, help="Base points sampled on the VISIBLE shell only.")
    ap.add_argument("--max-loops", type=int, default=8)
    ap.add_argument("--link-filter", action="store_true")
    ap.add_argument("--ls-filter", action="store_true")
    ap.add_argument("--min-rho", type=float, default=0.0)
    ap.add_argument("--select", choices=["area", "small", "mechanical"], default="area")
    ap.add_argument("--slider", action="store_true")
    ap.add_argument("--show-mesh", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    from cagingloop import farthest_point_sample, uniform_sample

    t = time.time()
    V, F = load_obj_mesh(args.model)
    normals = compute_vertex_normals(V, F)
    view = view_dir_from_angles(args.azimuth, args.elevation)
    vis_pts, vis_nrm, frame = render_depth_cloud(V, normals, view_dir=view, resolution=args.depth_res)
    voxels, surf_normals = occlusion_shadow_voxelization(
        vis_pts, vis_nrm, frame,
        voxel_count=args.voxel_count, padding=args.padding, occlusion_solid=args.occlusion_solid,
    )
    go = voxels.grid_on
    print(f"depth view az={args.azimuth} el={args.elevation}: visible points={len(vis_pts)}, "
          f"surface voxels={len(go)}, shadow-solid={len(voxels.grids_inner)} "
          f"(occlusion_solid={args.occlusion_solid}), {time.time()-t:.0f}s")
    if len(go) == 0:
        raise SystemExit("no surface voxels — try a different --azimuth or higher --depth-res")

    # Base points: VISIBLE shell only (it IS grid_on here, so this is automatic), farthest-spread.
    n = min(max(args.sweep, 1), len(go))
    sel = [int(np.argmax(np.linalg.norm(go - go.mean(0), axis=1)))] if n == 1 else farthest_point_sample(go, n)
    base_ids = [int(i) for i in sel]

    pool, saddle_voxels, saddle_points, saddle_types = [], [], [], []
    first_distance = None
    for bid in base_ids:
        d = distance_map_by_fast_marching(voxels, bid, solver=args.solver)
        if first_distance is None:
            first_distance = d
        if args.method == "morse":
            for (v, _off, ncomp) in detect_morse_saddles_3d(d.distance_grid):
                saddle_voxels.append(v); saddle_types.append(ncomp)
            pool.extend(volumetric_caging_loops(voxels, bid, distance=d, max_loops=max(args.max_loops, 50)))
        else:
            sad = detect_saddle_point(d.dismap, go, bid, surf_normals, keep=len(go))
            for sid in np.atleast_1d(sad).astype(int):
                saddle_points.append(go[sid])
            if len(sad):
                pool.extend(generate_caging_grasps(
                    d.distance_grid, voxels, bid, d.dismap, sad,
                    max_loops=max(args.max_loops, 50), integrator=args.path_integrator,
                ))

    n_raw = len(pool)
    if args.min_rho > 0.0:
        pool = filter_by_isoperimetric(pool, args.min_rho)
    if args.link_filter:
        pool = filter_by_linking(pool, voxels)
    n_link = len(pool)
    if args.ls_filter:
        pool = filter_locally_shortest(pool, voxels)
    ranked = rank_caging_loops(pool, voxels, mode=args.select)[: args.max_loops]
    loops = [c.path for c in ranked]
    print(f"method={args.method}  loops: raw={n_raw} -> link({n_link}) -> {args.select} top {len(loops)}")

    show_internals_polyscope(
        voxels,
        distance=first_distance,
        grasp_mask=(voxels.output_grid != 0),
        base_point_id=base_ids,
        saddle_voxels=saddle_voxels if args.method == "morse" else None,
        saddle_points=saddle_points if args.method == "surface" else None,
        saddle_types=saddle_types if args.method == "morse" else None,
        saddle_scalar_label=("saddle_order" if args.method == "morse" else "transitions"),
        loops=loops,
        slider=args.slider,
        obj_mesh=((V, F) if args.show_mesh else None),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
