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
    generate_best_caging_grasp,
    load_obj_point_cloud,
    point_cloud_voxelization_by_rbf,
    transfer_point_normals,
)
from cagingloop.polyscope_visualization import show_pipeline_polyscope


def run_model(
    path: Path,
    *,
    voxel_count: int,
    max_points: int | None,
    source_point_id: int,
    normal_offset: float,
):
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
    if len(voxels.grid_on) == 0:
        raise ValueError("voxelization produced no surface points")
    if source_point_id < 0:
        # Auto-seed: the surface point farthest from the centroid behaves like a
        # fingertip contact and yields a much better wrapping loop than an
        # arbitrary grid corner.
        grid_on = voxels.grid_on
        source_point_id = int(np.argmax(np.linalg.norm(grid_on - grid_on.mean(axis=0), axis=1)))
    else:
        source_point_id = min(source_point_id, len(voxels.grid_on) - 1)
    distance = distance_map_by_fast_marching(voxels, source_point_id, prefer_fmm=True)
    # Transfer the model's true surface normals to each surface voxel, matching
    # MATLAB's `grid_on_normals` input rather than approximating radially.
    surface_normals = transfer_point_normals(points, normals, voxels.grid_on)
    # Evaluate every saddle candidate (keep wide) and keep the loop that actually
    # wraps the object, instead of the single diversity-top saddle that MATLAB's
    # heuristic returns (which is often a degenerate sliver).
    saddles = detect_saddle_point(
        distance.dismap, voxels.grid_on, source_point_id, surface_normals, keep=len(voxels.grid_on)
    )

    caging = None
    best_saddle = -1
    if len(saddles) > 0:
        try:
            caging, best_saddle = generate_best_caging_grasp(
                distance.distance_grid, voxels, source_point_id, distance.dismap, saddles
            )
        except ValueError as exc:
            print(f"no caging loop: {exc}")
    return points, voxels, distance, saddles, caging, best_saddle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CagingLoop model pipeline and show it in Polyscope.")
    parser.add_argument("model", nargs="?", default="Models/knotty.obj")
    parser.add_argument("--voxel-count", type=int, default=17)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--source-point-id", type=int, default=-1, help="-1 = auto (farthest from centroid)")
    parser.add_argument("--normal-offset", type=float, default=1e-4)
    parser.add_argument("--no-show", action="store_true", help="Build and register data without opening the UI.")
    args = parser.parse_args()

    model_path = Path(args.model)
    points, voxels, distance, saddles, caging, best_saddle = run_model(
        model_path,
        voxel_count=args.voxel_count,
        max_points=args.max_points,
        source_point_id=args.source_point_id,
        normal_offset=args.normal_offset,
    )
    print(f"model: {model_path}")
    print(f"input points: {len(points)}")
    print(f"normal offset: {args.normal_offset}")
    print(f"surface points: {len(voxels.grid_on)}")
    print(f"inner voxels: {len(voxels.grids_inner)}")
    print(f"outer voxels: {len(voxels.grids_outer)}")
    print(f"saddle candidates: {len(saddles)}")
    if caging is None:
        print("caging path points: 0")
    else:
        from cagingloop import loop_enclosed_area

        print(f"chosen saddle: {best_saddle}")
        print(f"caging path points: {len(caging.final_path)}")
        print(f"caging loop enclosed area: {loop_enclosed_area(caging.final_path):.4f}")

    show_pipeline_polyscope(voxels, distance=distance, saddles=saddles, caging=caging, show=not args.no_show)


if __name__ == "__main__":
    main()
