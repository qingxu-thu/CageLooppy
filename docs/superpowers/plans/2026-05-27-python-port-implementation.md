# CagingLoop Python Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python port of the MATLAB CagingLoop workflow without MATLAB, MEX files, or `FastRBF.exe`.

**Architecture:** Create a small `cagingloop` package with dataclasses for shared results, focused modules for each MATLAB interface, optional plotting separated from computation, and synthetic tests that do not require external model assets. The first pass favors clear, deterministic behavior and Python-native substitutes over exact MATLAB numerical parity.

**Tech Stack:** Python 3.10+, `numpy`, `scipy`, optional `scikit-fmm`, optional `matplotlib`, optional `scikit-image`, `pytest`.

---

## File Map

- Create `pyproject.toml`: package metadata, dependencies, pytest configuration.
- Create `cagingloop/__init__.py`: public API exports.
- Create `cagingloop/types.py`: `GridIndex`, `VoxelizationResult`, `DistanceMapResult`, `CagingPath`.
- Create `cagingloop/nearest.py`: KD-tree wrapper and nearest-neighbor query.
- Create `cagingloop/voxelization.py`: RBF-based point-cloud voxelization and grid classification.
- Create `cagingloop/distance.py`: distance-map computation and shortest-path extraction.
- Create `cagingloop/saddle.py`: local saddle detection and scoring.
- Create `cagingloop/grasp.py`: caging path generation and smoothing.
- Create `cagingloop/visualization.py`: optional Matplotlib helpers.
- Create `examples/run_pipeline.py`: synthetic sphere pipeline example.
- Modify `README.md`: Python installation and usage.
- Create tests under `tests/`.

## Task 1: Scaffold Package Types

**Files:**
- Create: `pyproject.toml`
- Create: `cagingloop/__init__.py`
- Create: `cagingloop/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_types.py`:

```python
import numpy as np

from cagingloop.types import CagingPath, DistanceMapResult, GridIndex, VoxelizationResult


def test_grid_index_arrays_are_integer_arrays():
    index = GridIndex(
        on_index=np.array([[0, 1, 2]]),
        inner_index=np.array([[1, 1, 1]]),
        outer_index=np.array([[2, 2, 2]]),
    )

    assert index.on_index.dtype.kind in "iu"
    assert index.inner_index.shape == (1, 3)
    assert index.outer_index.tolist() == [[2, 2, 2]]


def test_result_dataclasses_hold_expected_arrays():
    index = GridIndex(np.zeros((1, 3), dtype=int), np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=int))
    voxels = VoxelizationResult(
        output_grid=np.ones((2, 2, 2), dtype=int),
        grid_x=np.array([0.0, 1.0]),
        grid_y=np.array([0.0, 1.0]),
        grid_z=np.array([0.0, 1.0]),
        index=index,
        grid_on=np.zeros((1, 3)),
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
        mesh=None,
    )
    distance = DistanceMapResult(dismap=np.array([0.0]), distance_grid=np.zeros((2, 2, 2)))
    path = CagingPath(final_path=np.zeros((2, 3)), path1=np.zeros((1, 3)), path2=np.zeros((1, 3)))

    assert voxels.output_grid.shape == (2, 2, 2)
    assert distance.dismap.tolist() == [0.0]
    assert path.final_path.shape == (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cagingloop'`.

- [ ] **Step 3: Write minimal package scaffold**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cagingloop"
version = "0.1.0"
description = "Python port of CagingLoop MATLAB interfaces"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.23",
    "scipy>=1.10",
]

[project.optional-dependencies]
fmm = ["scikit-fmm>=2023.4.2"]
plot = ["matplotlib>=3.7"]
mesh = ["scikit-image>=0.21"]
dev = ["pytest>=7.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `cagingloop/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class GridIndex:
    on_index: IntArray
    inner_index: IntArray
    outer_index: IntArray


@dataclass(frozen=True)
class VoxelizationResult:
    output_grid: IntArray
    grid_x: FloatArray
    grid_y: FloatArray
    grid_z: FloatArray
    index: GridIndex
    grid_on: FloatArray
    grids_inner: FloatArray
    grids_outer: FloatArray
    mesh: Any | None = None


@dataclass(frozen=True)
class DistanceMapResult:
    dismap: FloatArray
    distance_grid: FloatArray


@dataclass(frozen=True)
class CagingPath:
    final_path: FloatArray
    path1: FloatArray
    path2: FloatArray
```

Create `cagingloop/__init__.py` with imports for the dataclasses only; later tasks will expand it:

```python
from cagingloop.types import CagingPath, DistanceMapResult, GridIndex, VoxelizationResult

__all__ = [
    "CagingPath",
    "DistanceMapResult",
    "GridIndex",
    "VoxelizationResult",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml cagingloop tests/test_types.py
git commit -m "Add Python package scaffold"
```

## Task 2: Nearest-Neighbor Helpers

**Files:**
- Create: `cagingloop/nearest.py`
- Modify: `cagingloop/__init__.py`
- Test: `tests/test_nearest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nearest.py`:

```python
import numpy as np
import pytest

from cagingloop.nearest import NearestTree, nn_prepare, nn_search


def test_nn_search_returns_sorted_neighbors_for_query_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    tree = nn_prepare(points)

    indices, distances = nn_search(points, tree, np.array([[0.2, 0.0, 0.0]]), 2)

    assert indices.tolist() == [[0, 1]]
    assert np.allclose(distances, [[0.2, 0.8]])


def test_nn_search_accepts_query_indices_and_excludes_self():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    tree = NearestTree(points)

    indices, distances = nn_search(points, tree, np.array([1]), 1, exclude=0)

    assert indices.tolist() == [[0]]
    assert np.allclose(distances, [[1.0]])


def test_nn_prepare_rejects_non_xyz_arrays():
    with pytest.raises(ValueError, match="N x 3"):
        nn_prepare(np.array([1.0, 2.0, 3.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nearest.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cagingloop.nearest'`.

- [ ] **Step 3: Implement nearest-neighbor helpers**

Create `cagingloop/nearest.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 array")
    return arr


@dataclass(frozen=True)
class NearestTree:
    points: np.ndarray

    def __post_init__(self) -> None:
        points = _as_points(self.points, "points")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "tree", cKDTree(points))


def nn_prepare(pointset: np.ndarray) -> NearestTree:
    return NearestTree(pointset)


def nn_search(
    pointset: np.ndarray,
    tree: NearestTree,
    query: np.ndarray,
    k: int,
    exclude: int | None = None,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    points = _as_points(pointset, "pointset")
    if points.shape != tree.points.shape or not np.allclose(points, tree.points):
        raise ValueError("pointset must match the prepared nearest-neighbor tree")
    if k < 1:
        raise ValueError("k must be at least 1")

    query_arr = np.asarray(query)
    query_is_indices = query_arr.ndim == 1 and np.issubdtype(query_arr.dtype, np.integer)
    if query_is_indices:
        query_indices = query_arr.astype(int)
        if np.any(query_indices < 0) or np.any(query_indices >= len(points)):
            raise ValueError("query indices are outside pointset")
        query_points = points[query_indices]
    else:
        query_points = _as_points(query_arr, "query")
        query_indices = None

    ask_k = k
    if query_indices is not None and exclude is not None:
        ask_k = min(len(points), k + 2 * exclude + 1)

    distances, indices = tree.tree.query(query_points, k=ask_k, eps=epsilon)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    if query_indices is None or exclude is None:
        return indices[:, :k].astype(int), distances[:, :k].astype(float)

    out_indices: list[np.ndarray] = []
    out_distances: list[np.ndarray] = []
    for row, center in enumerate(query_indices):
        mask = np.abs(indices[row] - center) > exclude
        kept_indices = indices[row][mask][:k]
        kept_distances = distances[row][mask][:k]
        if len(kept_indices) < k:
            raise ValueError("not enough neighbors after applying exclude")
        out_indices.append(kept_indices.astype(int))
        out_distances.append(kept_distances.astype(float))
    return np.vstack(out_indices), np.vstack(out_distances)
```

Update `cagingloop/__init__.py` to export `NearestTree`, `nn_prepare`, and `nn_search`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nearest.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/__init__.py cagingloop/nearest.py tests/test_nearest.py
git commit -m "Add nearest neighbor helpers"
```

## Task 3: Voxelization

**Files:**
- Create: `cagingloop/voxelization.py`
- Modify: `cagingloop/__init__.py`
- Test: `tests/test_voxelization.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voxelization.py`:

```python
import numpy as np
import pytest

from cagingloop.voxelization import point_cloud_voxelization_by_rbf


def test_voxelization_classifies_simple_sphere_points():
    points = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    normals = points.copy()

    result = point_cloud_voxelization_by_rbf(points, normals, 7, 7, 7, rbf_neighbors=6)

    assert result.output_grid.shape == (7, 7, 7)
    assert result.grid_on.shape[1] == 3
    assert result.index.on_index.shape[1] == 3
    assert len(result.index.on_index) > 0
    assert np.all(result.index.on_index >= 0)
    assert set(np.unique(result.output_grid)).issubset({-1, 0, 1})


def test_voxelization_rejects_mismatched_normals():
    points = np.zeros((3, 3))
    normals = np.zeros((2, 3))

    with pytest.raises(ValueError, match="same shape"):
        point_cloud_voxelization_by_rbf(points, normals, 5, 5, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_voxelization.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cagingloop.voxelization'`.

- [ ] **Step 3: Implement voxelization**

Implement `point_cloud_voxelization_by_rbf` with this public signature and the exact behaviors below:

```python
point_cloud_voxelization_by_rbf(
    pt_cloud: np.ndarray,
    pt_normals: np.ndarray,
    voxel_xnum: int,
    voxel_ynum: int,
    voxel_znum: int,
    *,
    normal_offset: float | None = None,
    rbf_neighbors: int | None = 64,
    smoothing: float = 0.0,
    extract_mesh: bool = False,
) -> VoxelizationResult
```

Implementation details:

- Validate both arrays are `N x 3` and have the same shape.
- Validate every voxel dimension is at least 3.
- Normalize normals defensively, raising `ValueError("pt_normals contains zero-length normals")` for zero normals.
- Use `normal_offset` if provided; otherwise use `0.1 * min(grid spacing)`.
- Build constraint samples from original, positive offset, and negative offset points with values `0`, `1`, and `-1`.
- Fit `scipy.interpolate.RBFInterpolator(samples, values, neighbors=rbf_neighbors, smoothing=smoothing, kernel="linear")`.
- Build grid vectors with `np.linspace`.
- Evaluate all grid coordinates.
- Mark surface voxels where the field value is close to zero or any 6-neighbor edge crosses zero.
- Mark non-surface voxels as inner when value `< 0`, outer otherwise.
- Move any boundary inner voxel to the outer set.
- Return zero-based indices and coordinate arrays.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_voxelization.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/__init__.py cagingloop/voxelization.py tests/test_voxelization.py
git commit -m "Add Python voxelization"
```

## Task 4: Distance Map And Shortest Paths

**Files:**
- Create: `cagingloop/distance.py`
- Modify: `cagingloop/__init__.py`
- Test: `tests/test_distance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distance.py`:

```python
import numpy as np

from cagingloop.distance import compute_shortest_path, distance_map_by_fast_marching
from cagingloop.types import GridIndex, VoxelizationResult


def _line_voxelization():
    grid = np.zeros((5, 3, 3), dtype=int)
    on_index = np.array([[x, 1, 1] for x in range(5)], dtype=int)
    for idx in on_index:
        grid[tuple(idx)] = 1
    return VoxelizationResult(
        output_grid=grid,
        grid_x=np.arange(5, dtype=float),
        grid_y=np.arange(3, dtype=float),
        grid_z=np.arange(3, dtype=float),
        index=GridIndex(on_index=on_index, inner_index=np.zeros((0, 3), dtype=int), outer_index=np.zeros((0, 3), dtype=int)),
        grid_on=np.array([[x, 1.0, 1.0] for x in range(5)]),
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
    )


def test_distance_map_fallback_handles_simple_line():
    voxels = _line_voxelization()

    result = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    assert np.isclose(result.dismap[0], 0.0)
    assert result.dismap[-1] > result.dismap[1]


def test_compute_shortest_path_returns_xyz_path_to_source():
    voxels = _line_voxelization()
    distance = distance_map_by_fast_marching(voxels, start_point_id=0, prefer_fmm=False)

    path = compute_shortest_path(distance.distance_grid, voxels, start_point_id=4, source_point_id=0)

    assert np.allclose(path[0], [4.0, 1.0, 1.0])
    assert np.allclose(path[-1], [0.0, 1.0, 1.0])
    assert path.shape[1] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_distance.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cagingloop.distance'`.

- [ ] **Step 3: Implement distance and path functions**

Implement these public signatures with the exact behaviors below:

```python
distance_map_by_fast_marching(
    voxelization: VoxelizationResult,
    start_point_id: int,
    *,
    prefer_fmm: bool = True,
) -> DistanceMapResult


compute_shortest_path(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    start_point_id: int,
    source_point_id: int,
    *,
    connectivity: int = 6,
) -> np.ndarray
```

Implementation details:

- Validate source and endpoint ids against `len(voxelization.index.on_index)`.
- Build a traversable mask where `output_grid >= 0`.
- If `prefer_fmm` is true, try importing `skfmm`; if unavailable or fails, use Dijkstra fallback.
- Dijkstra fallback uses a priority queue over 6-connected grid neighbors and unit/Euclidean step costs.
- Surface `dismap` is `distance_grid` sampled at `on_index`.
- Repair non-finite surface values by querying nearest surface neighbors with `nn_search`.
- `compute_shortest_path` starts at `on_index[start_point_id]`, repeatedly moves to the finite 6-neighbor with smallest distance, and stops when it reaches the source voxel.
- Convert voxel indices to XYZ coordinates with `grid_x[x]`, `grid_y[y]`, and `grid_z[z]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_distance.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/__init__.py cagingloop/distance.py tests/test_distance.py
git commit -m "Add distance map and shortest paths"
```

## Task 5: Saddle Detection

**Files:**
- Create: `cagingloop/saddle.py`
- Modify: `cagingloop/__init__.py`
- Test: `tests/test_saddle.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_saddle.py`:

```python
import numpy as np

from cagingloop.saddle import calculate_iter_num, detect_saddle_point, diversity_eval


def test_diversity_eval_prefers_aligned_source_and_saddle_normals():
    source = np.array([0.0, 0.0, 0.0])
    saddle = np.array([1.0, 0.0, 0.0])

    good = diversity_eval(source, saddle, np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    bad = diversity_eval(source, saddle, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    assert good > bad


def test_calculate_iter_num_detects_four_sign_transitions_on_ring():
    center = np.array([[0.0, 0.0, 0.0]])
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    grid_on = np.vstack([center, ring])
    dismap = np.array([1.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])

    assert calculate_iter_num(dismap, grid_on, point_id=0, k=9) >= 4


def test_detect_saddle_point_returns_valid_indices():
    center = np.array([[0.0, 0.0, 0.0]])
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    grid_on = np.vstack([center, ring])
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(grid_on), 1))
    dismap = np.array([1.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])

    saddles = detect_saddle_point(dismap, grid_on, source_point_id=1, grid_on_normals=normals, k=9)

    assert saddles.ndim == 1
    assert np.all(saddles >= 0)
    assert np.all(saddles < len(grid_on))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_saddle.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cagingloop.saddle'`.

- [ ] **Step 3: Implement saddle detection**

Implement these public signatures with the exact behaviors below:

```python
diversity_eval(p1: np.ndarray, p2: np.ndarray, normal1: np.ndarray, normal2: np.ndarray) -> float


calculate_iter_num(dismap: np.ndarray, grid_on: np.ndarray, point_id: int, k: int = 9) -> int


detect_saddle_point(
    dismap: np.ndarray,
    grid_on: np.ndarray,
    source_point_id: int,
    grid_on_normals: np.ndarray,
    *,
    k: int = 9,
) -> np.ndarray
```

Implementation details:

- Use `NearestTree` and `nn_search` for local neighborhoods.
- Use SVD on centered neighbor offsets to compute tangent axes.
- Use `scipy.spatial.ConvexHull` to order projected local boundary points.
- Use `matplotlib.path.Path` if available for origin-in-polygon; otherwise use a small ray-crossing helper.
- Filter boundary points with the same distance tolerance and angle checks as MATLAB.
- Count sign transitions circularly so the last-to-first edge is included.
- Keep candidates with transition count at least four.
- Score candidates with `diversity_eval`, sort descending, and keep `max(1, round(len(score) / 30))`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_saddle.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/__init__.py cagingloop/saddle.py tests/test_saddle.py
git commit -m "Add saddle point detection"
```

## Task 6: Caging Grasp Generation

**Files:**
- Create: `cagingloop/grasp.py`
- Modify: `cagingloop/__init__.py`
- Test: `tests/test_grasp.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grasp.py`:

```python
import numpy as np

from cagingloop.distance import distance_map_by_fast_marching
from cagingloop.grasp import generate_caging_grasp, smooth_closed_path
from cagingloop.types import GridIndex, VoxelizationResult


def test_smooth_closed_path_preserves_shape_and_closes_loop():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    smoothed = smooth_closed_path(path)

    assert smoothed.shape == path.shape
    assert np.allclose(smoothed[0], smoothed[-1])


def test_generate_caging_grasp_returns_paths():
    grid = np.zeros((5, 5, 3), dtype=int)
    on_index = np.array([[x, 2, 1] for x in range(5)] + [[2, y, 1] for y in range(5)], dtype=int)
    on_index = np.unique(on_index, axis=0)
    for idx in on_index:
        grid[tuple(idx)] = 1
    grid_on = np.array([[idx[0], idx[1], idx[2]] for idx in on_index], dtype=float)
    voxels = VoxelizationResult(
        output_grid=grid,
        grid_x=np.arange(5, dtype=float),
        grid_y=np.arange(5, dtype=float),
        grid_z=np.arange(3, dtype=float),
        index=GridIndex(on_index=on_index, inner_index=np.zeros((0, 3), dtype=int), outer_index=np.zeros((0, 3), dtype=int)),
        grid_on=grid_on,
        grids_inner=np.zeros((0, 3)),
        grids_outer=np.zeros((0, 3)),
    )
    source = int(np.where((on_index == [0, 2, 1]).all(axis=1))[0][0])
    saddle = int(np.where((on_index == [2, 2, 1]).all(axis=1))[0][0])
    distance = distance_map_by_fast_marching(voxels, source, prefer_fmm=False)

    caging = generate_caging_grasp(distance.distance_grid, voxels, source, distance.dismap, saddle, k=5)

    assert caging.final_path.shape[1] == 3
    assert len(caging.path1) > 0
    assert len(caging.path2) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grasp.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cagingloop.grasp'`.

- [ ] **Step 3: Implement grasp generation**

Implement these public signatures with the exact behaviors below:

```python
smooth_closed_path(path: np.ndarray) -> np.ndarray


get_cage_points(dismap: np.ndarray, grid_on: np.ndarray, point_id: int, k: int = 9) -> np.ndarray


generate_caging_grasp(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    source_point_id: int,
    dismap: np.ndarray,
    saddle_point_id: int,
    *,
    k: int = 9,
) -> CagingPath
```

Implementation details:

- `get_cage_points` mirrors the MATLAB local tangent-plane selection.
- If the strict opposite-direction rule has too few negative neighbors, choose the two lowest-distance local neighbors as a deterministic fallback.
- `generate_caging_grasp` computes both paths with `compute_shortest_path`, prepends the saddle coordinate, flips the second path, concatenates, smooths, and returns `CagingPath`.
- `smooth_closed_path` applies `0.2 * current + 0.4 * previous + 0.4 * next` to interior points and closes start/end to the same smoothed point.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grasp.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cagingloop/__init__.py cagingloop/grasp.py tests/test_grasp.py
git commit -m "Add caging grasp generation"
```

## Task 7: Visualization, Example, And README

**Files:**
- Create: `cagingloop/visualization.py`
- Create: `examples/run_pipeline.py`
- Modify: `README.md`
- Test: `tests/test_public_api.py`

- [ ] **Step 1: Write the failing public API test**

Create `tests/test_public_api.py`:

```python
import cagingloop


def test_public_api_exports_main_workflow_functions():
    for name in [
        "point_cloud_voxelization_by_rbf",
        "distance_map_by_fast_marching",
        "detect_saddle_point",
        "generate_caging_grasp",
    ]:
        assert hasattr(cagingloop, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_public_api.py -v`

Expected: FAIL until all workflow functions are exported.

- [ ] **Step 3: Add optional visualization helpers and example**

Create `cagingloop/visualization.py` with optional imports inside these functions:

```python
plot_points(points, *, values=None, ax=None, title=None)


plot_caging_path(path, *, ax=None, title="Caging path")
```

Create `examples/run_pipeline.py` that:

- Builds a small sphere point cloud and normals.
- Calls voxelization.
- Chooses the first surface point as source.
- Computes distance map.
- Detects saddles.
- Generates a caging path if any saddle exists.
- Prints counts and path length.

- [ ] **Step 4: Update README**

Add Python usage notes:

```markdown
## Python Port

The Python implementation lives in `cagingloop/` and mirrors the MATLAB workflow without requiring MATLAB, MEX files, or `FastRBF.exe`.

Install for development:

```powershell
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

Run the synthetic example:

```powershell
python examples/run_pipeline.py
```
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_public_api.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cagingloop/__init__.py cagingloop/visualization.py examples/run_pipeline.py README.md tests/test_public_api.py
git commit -m "Add Python example and documentation"
```

## Task 8: Final Verification

**Files:**
- Modify only if verification reveals a defect.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -v`

Expected: all tests pass.

- [ ] **Step 2: Run the example**

Run: `python examples/run_pipeline.py`

Expected: it prints surface count, saddle count, and either generated path length or a no-saddle message.

- [ ] **Step 3: Inspect Git status**

Run: `git status --short`

Expected: no unstaged or uncommitted files.

- [ ] **Step 4: Commit verification fixes if needed**

If Step 1 or Step 2 reveals a defect, fix the narrow issue, rerun the failed command, and commit:

```bash
git add <changed-files>
git commit -m "Fix Python port verification issues"
```
