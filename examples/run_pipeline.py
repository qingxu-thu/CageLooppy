from __future__ import annotations

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


def sphere_points() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.7, 0.7, 0.0],
            [-0.7, 0.7, 0.0],
            [0.7, -0.7, 0.0],
            [-0.7, -0.7, 0.0],
        ],
        dtype=float,
    )
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    return points, normals


def surface_normals(points: np.ndarray) -> np.ndarray:
    normals = points.copy()
    lengths = np.linalg.norm(normals, axis=1)
    zero = lengths == 0.0
    lengths[zero] = 1.0
    normals = normals / lengths[:, None]
    normals[zero] = np.array([0.0, 0.0, 1.0])
    return normals


def main() -> None:
    points, normals = sphere_points()
    voxels = point_cloud_voxelization_by_rbf(points, normals, 9, 9, 9, rbf_neighbors=8)
    print(f"surface points: {len(voxels.grid_on)}")
    print(f"inner voxels: {len(voxels.grids_inner)}")
    print(f"outer voxels: {len(voxels.grids_outer)}")

    if len(voxels.grid_on) == 0:
        print("no surface points generated")
        return

    source_point_id = 0
    distance = distance_map_by_fast_marching(voxels, source_point_id, prefer_fmm=False)
    saddles = detect_saddle_point(distance.dismap, voxels.grid_on, source_point_id, surface_normals(voxels.grid_on))
    print(f"saddle candidates: {len(saddles)}")

    if len(saddles) == 0:
        print("no saddle candidate found for this synthetic example")
        return

    try:
        caging = generate_caging_grasp(distance.distance_grid, voxels, source_point_id, distance.dismap, int(saddles[0]))
    except ValueError as exc:
        print(f"could not generate caging path: {exc}")
        return
    print(f"caging path points: {len(caging.final_path)}")


if __name__ == "__main__":
    main()
