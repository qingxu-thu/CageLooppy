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
    generate_caging_grasps,
    load_obj_point_cloud,
    offset_voxelization,
    point_cloud_voxelization_by_rbf,
    transfer_point_normals,
    voxelize_mesh,
)
from scipy.spatial import ConvexHull

from cagingloop.curvature import positive_curvature_points
from cagingloop.morse import detect_morse_saddles_3d, volumetric_caging_loops
from cagingloop.model_io import compute_vertex_normals, load_obj_mesh
from cagingloop.polyscope_visualization import show_internals_polyscope
from cagingloop.saddle import _boundary_order, _project_neighbors, calculate_iter_num
from cagingloop.voxelization import convex_hull_grasping_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="Show CagingLoop pipeline internals in Polyscope.")
    ap.add_argument("model", nargs="?", default="Models/knotty.obj")
    ap.add_argument("--backend", choices=["rbf", "mesh"], default="mesh")
    ap.add_argument("--voxel-count", type=int, default=37)
    ap.add_argument("--padding", type=float, default=0.1,
                    help="Mesh-backend bbox padding fraction; smaller = more voxels on the object (finer surface).")
    ap.add_argument("--max-points", type=int, default=1500)
    ap.add_argument("--method", choices=["morse", "surface"], default="morse",
                    help="Saddle method: morse = volumetric Morse saddles in the void (Fig.3+Thm3.1, default); "
                         "surface = MATLAB detectSaddlePoint on the surface (tangent-ring sign flips).")
    ap.add_argument("--path-integrator", choices=["euler", "rk4"], default="euler",
                    help="Surface-method loop tracer: euler (default) or rk4 (MATLAB shortestpath parity).")
    ap.add_argument("--grasp-hull", action="store_true", help="Bound the grasp space by the convex hull (Thm 3.2).")
    ap.add_argument("--curvature-filter", action="store_true", help="Seed base points from positive-curvature points (Thm 3.3).")
    ap.add_argument("--sweep", type=int, default=1, help="Number of base points to seed (each gets its own saddles).")
    ap.add_argument("--base-sampling", choices=["farthest", "uniform"], default="farthest")
    ap.add_argument("--base-height", type=float, default=None, help="Restrict base points to this height FRACTION (0..1).")
    ap.add_argument("--base-height-band", type=float, default=0.1, help="Height-band width (fraction of model height).")
    ap.add_argument("--sweep-radius", type=float, default=None, help="Cap the distance front at 2h (world units).")
    ap.add_argument("--solver", choices=["fmm", "msfm", "dijkstra"], default="fmm",
                    help="Distance-field solver: fmm = single-stencil Eikonal fast marching (Euclidean, needs skfmm); "
                         "msfm = multistencil fast marching (MATLAB-parity, lowest anisotropy, pure Python); "
                         "dijkstra = 6-neighbor grid-graph geodesic (Manhattan, no skfmm needed).")
    ap.add_argument("--max-loops", type=int, default=5, help="How many traced loops to draw per base point.")
    ap.add_argument("--slider", action="store_true", help="Add a UI slider to browse the loops one at a time.")
    ap.add_argument("--slice-plane", action="store_true",
                    help="Add a movable scene slice plane to cut into the D_p-coloured grasp-space volume.")
    ap.add_argument("--slice-height", type=float, default=None,
                    help="Show the distance field on a horizontal cross-section at this height FRACTION (0..1).")
    ap.add_argument("--show-mesh", action="store_true", help="Register the actual OBJ triangle mesh (semi-transparent).")
    ap.add_argument("--mesh-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"),
                    help="Translate the OBJ mesh by this world-space offset (to separate it from the voxel layers).")
    ap.add_argument("--gripper-radius", type=float, default=0.0,
                    help="r-offset surface S_r (paper SS IV): grow the solid by this radius (world units) to fuse "
                         "thin gaps before computing the grasp space / saddles.")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    t = time.time()
    if args.backend == "mesh":
        V, F = load_obj_mesh(args.model)
        voxels = voxelize_mesh(V, F, args.voxel_count, args.voxel_count, args.voxel_count, padding=args.padding)
        normals_src_pts, normals_src_n = V, compute_vertex_normals(V, F)
    else:
        pts, nrm, _ = load_obj_point_cloud(args.model, max_points=args.max_points)
        voxels = point_cloud_voxelization_by_rbf(
            pts, nrm, args.voxel_count, args.voxel_count, args.voxel_count,
            rbf_neighbors=min(64, len(pts)), smoothing=1e-8, normal_offset=1e-4,
        )
        normals_src_pts, normals_src_n = pts, nrm
    if args.gripper_radius > 0.0:
        # r-offset surface S_r (paper SS IV): dilate the solid by the gripper radius so
        # thin gaps fuse, BEFORE the grasp space and saddles are computed.
        spacing = float(voxels.grid_x[1] - voxels.grid_x[0])
        radius_voxels = max(1, round(args.gripper_radius / spacing))
        voxels = offset_voxelization(voxels, radius_voxels)
        print(f"r-offset: gripper_radius={args.gripper_radius} -> {radius_voxels} voxels")
    normals = transfer_point_normals(normals_src_pts, normals_src_n, voxels.grid_on)
    print(f"voxelized ({args.backend}, {args.voxel_count}^3): {time.time()-t:.0f}s, surface={len(voxels.grid_on)}")

    from cagingloop import farthest_point_sample, uniform_sample

    go = voxels.grid_on
    # candidate base points: optional curvature filter (Thm 3.3) + optional height band
    candidates = np.arange(len(go))
    if args.curvature_filter:
        pc = positive_curvature_points(go, normals)
        if len(pc):
            candidates = pc
    if args.base_height is not None:
        coord = go[:, 1]
        lo, hi = float(coord.min()), float(coord.max())
        tgt = lo + args.base_height * (hi - lo)
        half = 0.5 * args.base_height_band * (hi - lo)
        inb = np.abs(go[candidates, 1] - tgt) <= half
        if inb.any():
            candidates = candidates[inb]
    n = min(max(args.sweep, 1), len(candidates))
    if n == 1:
        sel = [int(np.argmax(np.linalg.norm(go[candidates] - go.mean(axis=0), axis=1)))]
    elif args.base_sampling == "uniform":
        sel = uniform_sample(go[candidates], n)
    else:
        sel = farthest_point_sample(go[candidates], n)
    base_ids = [int(candidates[i]) for i in sel]

    mask = convex_hull_grasping_mask(voxels) if args.grasp_hull else (voxels.output_grid != 0)
    axes_off = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    single = len(base_ids) == 1  # only draw the +/- field for one base point (else cluttered)
    saddle_voxels, saddle_points, saddle_types, saddle_source, saddle_descents, all_loops = [], [], [], [], [], []
    npts, signs = [], []
    first_distance = None
    for si, bid in enumerate(base_ids):
        d = distance_map_by_fast_marching(
            voxels, bid, solver=args.solver, traversable_mask=mask, max_distance=args.sweep_radius
        )
        if first_distance is None:
            first_distance = d

        if args.method == "morse":
            D = d.distance_grid
            sh = D.shape
            for (v, _off, ncomp) in detect_morse_saddles_3d(D):
                saddle_voxels.append(v)
                saddle_types.append(ncomp)       # lower-link component count
                saddle_source.append(si)
                saddle_descents.append(_off)
                if single:                       # 6-neighbour +/- field (Fig. 3)
                    a, b, c = v
                    for dx, dy, dz in axes_off:
                        na, nb, nc = a + dx, b + dy, c + dz
                        if 0 <= na < sh[0] and 0 <= nb < sh[1] and 0 <= nc < sh[2] and np.isfinite(D[na, nb, nc]):
                            npts.append([voxels.grid_x[na], voxels.grid_y[nb], voxels.grid_z[nc]])
                            signs.append(1.0 if D[na, nb, nc] > D[a, b, c] else -1.0)
            for cand in sorted(volumetric_caging_loops(voxels, bid, distance=d), key=lambda c: c.area, reverse=True)[: args.max_loops]:
                all_loops.append(cand.path)
        else:  # surface method: MATLAB detectSaddlePoint on the surface point cloud
            sad_ids = detect_saddle_point(d.dismap, go, bid, normals, keep=len(go))
            for sid in np.atleast_1d(sad_ids).astype(int):
                saddle_points.append(go[sid])
                saddle_types.append(float(calculate_iter_num(d.dismap, go, int(sid))))  # tangent-ring sign-flip count
                saddle_source.append(si)
                if single:  # tangent-plane ring around the saddle, coloured by sign(D-neighbour - D-saddle)
                    neighbour_ids, projected = _project_neighbors(go, int(sid), 9)
                    if len(projected) >= 3:
                        ring = neighbour_ids[_boundary_order(projected)]
                        judge = d.dismap[ring] - d.dismap[sid]
                        for w, jv in zip(go[ring], judge):
                            npts.append(list(w))
                            signs.append(1.0 if jv > 0 else -1.0)
            if len(sad_ids) > 0:
                pool = generate_caging_grasps(
                    d.distance_grid, voxels, bid, d.dismap, sad_ids,
                    max_loops=args.max_loops, integrator=args.path_integrator,
                )
                for cand in sorted(pool, key=lambda c: c.area, reverse=True)[: args.max_loops]:
                    all_loops.append(cand.path)

    saddle_neighbors = (np.array(npts), np.array(signs)) if npts else None
    hull = (go, ConvexHull(go).simplices)
    n_saddles = len(saddle_voxels) if args.method == "morse" else len(saddle_points)
    print(f"method={args.method}  base points={len(base_ids)} (ids {base_ids[:8]}{'...' if len(base_ids) > 8 else ''}), "
          f"grasp-space voxels={int(mask.sum())}, saddles={n_saddles}, loops drawn={len(all_loops)}")

    show_internals_polyscope(
        voxels,
        distance=first_distance,           # D_p of the first base point (colours the grasp space)
        grasp_mask=mask,
        base_point_id=base_ids,            # all base points
        saddle_voxels=saddle_voxels,       # morse: void voxels
        saddle_points=saddle_points,       # surface: surface points
        saddle_types=saddle_types,
        saddle_source=saddle_source,       # which base point each saddle came from
        saddle_descents=saddle_descents,   # opposite descending dirs -> loop arcs (Thm 3.1; morse only)
        saddle_scalar_label=("saddle_order" if args.method == "morse" else "transitions"),
        hull=hull,
        saddle_neighbors=saddle_neighbors,
        loops=all_loops,
        slider=args.slider,
        slice_plane=args.slice_plane,
        slice_height=args.slice_height,
        up_axis=1,
        obj_mesh=(load_obj_mesh(args.model) if args.show_mesh else None),
        mesh_shift=tuple(args.mesh_shift),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
