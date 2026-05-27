# CagingLoop Python Port Design

## Goal

Port the full MATLAB CagingLoop implementation to a maintainable Python package. The Python version should preserve the four public workflow stages from the original code:

1. Point cloud voxelization and surface classification.
2. Distance field computation from a source surface point.
3. Saddle point detection on the surface samples.
4. Caging loop generation through shortest paths from saddle-adjacent points.

The port should remove the runtime dependency on MATLAB, MEX files, and `FastRBF.exe`. It should be algorithmically equivalent at the workflow level, while accepting small numerical differences from replacing old external solvers with Python libraries.

## Non-Goals

- Bit-for-bit or point-for-point equality with MATLAB output.
- Full reproduction of every legacy visualization side effect.
- Reimplementation of third-party numerical solvers from scratch.
- A complete command-line product beyond a working example pipeline.

## Package Layout

The Python implementation will live in a new `cagingloop/` package:

- `cagingloop/types.py`: shared dataclasses for voxelization outputs, index sets, distance maps, and caging paths.
- `cagingloop/nearest.py`: nearest-neighbor helpers replacing `nn_prepare.m` and `nn_search.m`.
- `cagingloop/voxelization.py`: point-cloud voxelization and surface/inside/outside grid classification.
- `cagingloop/distance.py`: distance field computation and shortest path extraction.
- `cagingloop/saddle.py`: saddle candidate detection and geometry-based filtering.
- `cagingloop/grasp.py`: caging path construction and smoothing.
- `cagingloop/visualization.py`: optional plotting helpers using Matplotlib.
- `examples/run_pipeline.py`: end-to-end example matching the README workflow.
- `tests/`: focused tests for indexing, nearest neighbors, distance paths, and local saddle logic.

The original MATLAB files will stay in `Code/` for reference.

## Dependency Strategy

Use common Python scientific packages:

- `numpy` for array operations.
- `scipy` for KD-trees, interpolation, convex hulls, and fallback graph/path utilities.
- `scikit-fmm` for fast marching when available.
- `scikit-learn` only if needed for PCA; otherwise use `numpy.linalg.svd`.
- `matplotlib` for optional 3D visualization.
- `trimesh` or `scikit-image` for optional mesh/surface extraction if needed.
- `pytest` for tests.

Nearest-neighbor preprocessing from TSTOOL MEX files will be replaced by `scipy.spatial.cKDTree`. The tree object will provide repeated k-nearest-neighbor queries over the same point set.

## Voxelization

The MATLAB function `pointCloudVoxelizationByRBF.m` uses `FastRBF.exe` to fit an implicit function, evaluate it on grid samples, classify surface/inside/outside voxels, and reconstruct a mesh.

The Python port will provide a native approximation:

1. Build positive, negative, and zero constraints from the point cloud and normals.
2. Fit an implicit scalar field with `scipy.interpolate.RBFInterpolator`.
3. Sample that scalar field on a regular grid.
4. Mark surface voxels where local neighbor values cross zero.
5. Classify remaining voxels as inside or outside by the sign of the implicit field.
6. Move boundary voxels to the outside set, matching the MATLAB boundary handling.

The function will return a dataclass containing:

- `output_grid`: integer grid with `1` on surface, `0` inside, `-1` outside.
- `grid_x`, `grid_y`, `grid_z`: coordinate vectors.
- `index.on_index`, `index.inner_index`, `index.outer_index`: zero-based voxel indices.
- `grid_on`, `grids_inner`, `grids_outer`: coordinate arrays.
- Optional mesh data when surface extraction is enabled.

MATLAB is one-based, while Python is zero-based. All Python APIs will use zero-based indices consistently.

## Distance Field And Paths

`DistanceMapByFastMarching.m` builds a speed/weight grid and calls `msfm`. The Python port will:

1. Build a traversability/speed array from the voxel grid.
2. Use `scikit-fmm` when installed to compute travel time from the source voxel.
3. Provide a deterministic fallback based on Dijkstra over the regular grid if `scikit-fmm` is unavailable.
4. Extract surface distances into `dismap`.
5. Replace unreachable or invalid surface distances with the minimum valid value among local neighbors, matching the MATLAB repair step.

`compute_shortestpath.m` calls MATLAB's `shortestpath` on the distance field. The Python implementation will trace a monotone path by descending the distance field through neighboring voxels from the endpoint to the source. The fallback Dijkstra backend can also return predecessor links directly.

## Saddle Detection

`detectSaddlePoint.m` will be ported with the same local structure:

1. For each surface point, query local neighbors.
2. Estimate a local tangent frame with SVD/PCA.
3. Project neighboring points into that local 2D frame.
4. Compute the local boundary ordering with `scipy.spatial.ConvexHull`.
5. Require the origin to lie inside the local polygon.
6. Compare neighbor distance-map values against the center value.
7. Count sign transitions after filtering boundary points.
8. Keep points whose transition count is at least four.
9. Filter candidate saddles by the same source/saddle normal diversity score.

The MATLAB code keeps approximately the top `1/30` of scored candidates. The Python port will keep that rule, with a minimum of one candidate when candidates exist.

## Caging Grasp Generation

`generateCagingGrasp.m` will be ported as:

1. At the chosen saddle point, find two cage extension points using the local tangent-plane distance-map descent logic.
2. Compute two shortest paths from those points back to the source.
3. Concatenate the two paths through the saddle point.
4. Apply the same one-pass Laplacian-style smoothing.
5. Return a `CagingPath` dataclass with `final_path`, `path1`, and `path2`.

The path arrays will use XYZ coordinates, not voxel indices, matching the MATLAB output.

## Visualization

Plotting will be optional and separated from computation. No core function will open a figure by default. `visualization.py` will provide helpers for:

- Voxelization and reconstructed surface preview.
- Distance-map coloring on surface points.
- Saddle candidate display.
- Final caging path display.

This keeps tests and batch execution headless-friendly.

## Error Handling

Functions will validate input shapes and raise `ValueError` with clear messages for:

- Point clouds or normals that are not `N x 3`.
- Mismatched point/normal counts.
- Invalid voxel dimensions.
- Source or saddle indices outside valid surface-point ranges.
- Empty surface sets or missing reachable paths.

Optional dependencies will fail with actionable messages. For example, if `scikit-fmm` is unavailable, the fallback path will run automatically; if optional mesh extraction is requested but unavailable, the error will name the missing package.

## Testing Strategy

Tests will focus on stable behavior rather than exact MATLAB parity:

- Nearest-neighbor query returns sorted neighbors and distances.
- Voxel grid index conversion is zero-based and consistent with coordinate vectors.
- Distance fallback produces finite distances on simple synthetic grids.
- Shortest path descends toward the source and returns XYZ coordinates.
- Saddle local transition counting behaves as expected on controlled synthetic neighborhoods.
- Caging path concatenation and smoothing preserve expected shape and endpoints.

The first implementation pass should include small synthetic fixtures so tests do not require large model files or external executables.

## Migration Notes

The original MATLAB API names can be preserved as aliases if useful, but the primary Python names should follow snake_case. The README should be updated after implementation with installation instructions, dependency notes, and a Python usage example.

Because the RBF and fast marching backends are being replaced, validation should compare qualitative outputs and invariants:

- A non-empty surface set is produced.
- Source distances are near zero.
- Saddle candidates are valid surface indices.
- Generated caging paths are continuous coordinate sequences.

