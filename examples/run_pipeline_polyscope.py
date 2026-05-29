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
    generate_caging_grasp,
    point_cloud_voxelization_by_rbf,
)
from cagingloop.polyscope_visualization import show_pipeline_polyscope
from examples.run_pipeline import sphere_points, surface_normals


def build_pipeline(voxel_count: int):
    points, normals = sphere_points()
    voxels = point_cloud_voxelization_by_rbf(
        points,
        normals,
        voxel_count,
        voxel_count,
        voxel_count,
        rbf_neighbors=8,
    )
    source_point_id = 0
    distance = distance_map_by_fast_marching(voxels, source_point_id, prefer_fmm=False)
    saddles = detect_saddle_point(distance.dismap, voxels.grid_on, source_point_id, surface_normals(voxels.grid_on))
    caging = None
    if len(saddles) > 0:
        try:
            caging = generate_caging_grasp(distance.distance_grid, voxels, source_point_id, distance.dismap, int(saddles[0]))
        except ValueError as exc:
            print(f"could not generate caging path: {exc}")
    return voxels, distance, saddles, caging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CagingLoop synthetic example with Polyscope visualization.")
    parser.add_argument("--voxel-count", type=int, default=9)
    parser.add_argument("--no-show", action="store_true", help="Build and register data without opening the UI.")
    args = parser.parse_args()

    voxels, distance, saddles, caging = build_pipeline(args.voxel_count)
    print(f"surface points: {len(voxels.grid_on)}")
    print(f"inner voxels: {len(voxels.grids_inner)}")
    print(f"outer voxels: {len(voxels.grids_outer)}")
    print(f"saddle candidates: {len(saddles)}")
    if caging is not None:
        print(f"caging path points: {len(caging.final_path)}")

    show_pipeline_polyscope(voxels, distance=distance, saddles=saddles, caging=caging, show=not args.no_show)


if __name__ == "__main__":
    main()
