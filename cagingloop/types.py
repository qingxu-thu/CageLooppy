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
