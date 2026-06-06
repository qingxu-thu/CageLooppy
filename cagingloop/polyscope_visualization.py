from __future__ import annotations

from typing import Any

import numpy as np

from cagingloop.types import CagingPath, DistanceMapResult, VoxelizationResult


def _load_polyscope(ps_module=None):
    if ps_module is not None:
        return ps_module
    try:
        import polyscope as ps
    except ImportError as exc:
        raise ImportError("Polyscope visualization requires `python -m pip install polyscope`") from exc
    return ps


def _register_points(ps, name: str, points: np.ndarray, **kwargs):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    return ps.register_point_cloud(name, points, **kwargs)


def _loop_edges(point_count: int) -> np.ndarray:
    if point_count < 2:
        return np.zeros((0, 2), dtype=int)
    starts = np.arange(point_count, dtype=int)
    ends = np.roll(starts, -1)
    return np.column_stack((starts, ends))


def register_pipeline_polyscope(
    voxelization: VoxelizationResult,
    *,
    distance: DistanceMapResult | None = None,
    saddles: np.ndarray | None = None,
    caging: CagingPath | None = None,
    ps_module=None,
) -> dict[str, Any]:
    ps = _load_polyscope(ps_module)
    registered: dict[str, Any] = {}

    surface = _register_points(
        ps,
        "surface",
        voxelization.grid_on,
        radius=0.008,
        color=(0.9, 0.1, 0.1),
    )
    if surface is not None:
        registered["surface"] = surface
        if distance is not None:
            if len(distance.dismap) != len(voxelization.grid_on):
                raise ValueError("distance.dismap length must match voxelization.grid_on")
            surface.add_scalar_quantity("distance", np.asarray(distance.dismap, dtype=float), enabled=True)

    inner = _register_points(
        ps,
        "inner voxels",
        voxelization.grids_inner,
        radius=0.005,
        color=(0.1, 0.65, 0.25),
        enabled=False,
    )
    if inner is not None:
        registered["inner"] = inner

    outer = _register_points(
        ps,
        "outer voxels",
        voxelization.grids_outer,
        radius=0.003,
        color=(0.35, 0.45, 0.95),
        enabled=False,
    )
    if outer is not None:
        registered["outer"] = outer

    if saddles is not None and len(saddles) > 0:
        saddle_ids = np.asarray(saddles, dtype=int)
        if np.any(saddle_ids < 0) or np.any(saddle_ids >= len(voxelization.grid_on)):
            raise ValueError("saddle indices are outside voxelization.grid_on")
        saddle_cloud = _register_points(
            ps,
            "saddle points",
            voxelization.grid_on[saddle_ids],
            radius=0.02,
            color=(1.0, 0.85, 0.0),
        )
        if saddle_cloud is not None:
            registered["saddles"] = saddle_cloud

    if caging is not None and len(caging.final_path) >= 2:
        path = np.asarray(caging.final_path, dtype=float)
        curve = ps.register_curve_network(
            "caging path",
            path,
            _loop_edges(len(path)),
            radius=0.01,
            color=(1.0, 0.2, 0.05),
        )
        registered["caging_path"] = curve

    return registered


def _distinct_color(i: int, n: int) -> tuple[float, float, float]:
    """Evenly-spaced hue -> RGB, so many loops are easy to tell apart."""
    import colorsys

    hue = (i / max(n, 1)) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 0.95)


def register_caging_loops_polyscope(
    ps,
    loops,
    *,
    labels=None,
    radius: float = 0.008,
) -> dict[str, Any]:
    """Register a list of caging loops (each a `CagingPath` or an N x 3 array) as
    separately-toggleable, distinctly-coloured curve networks."""
    registered: dict[str, Any] = {}
    loops = list(loops)
    for i, loop in enumerate(loops):
        path = np.asarray(getattr(loop, "final_path", loop), dtype=float)
        if len(path) < 2:
            continue
        label = labels[i] if labels is not None and i < len(labels) else f"loop {i:02d}"
        curve = ps.register_curve_network(
            label,
            path,
            _loop_edges(len(path)),
            radius=radius,
            color=_distinct_color(i, len(loops)),
        )
        registered[label] = curve
    return registered


def show_caging_loops_polyscope(
    voxelization: VoxelizationResult,
    loops,
    *,
    labels=None,
    distance: DistanceMapResult | None = None,
    saddles: np.ndarray | None = None,
    ps_module=None,
    show: bool = True,
) -> dict[str, Any]:
    """Show the surface (+ optional distance/saddles) with *all* given caging loops at once."""
    ps = _load_polyscope(ps_module)
    ps.init()
    registered = register_pipeline_polyscope(
        voxelization, distance=distance, saddles=saddles, caging=None, ps_module=ps
    )
    registered.update(register_caging_loops_polyscope(ps, loops, labels=labels))
    if show:
        ps.show()
    return registered


def show_pipeline_polyscope(
    voxelization: VoxelizationResult,
    *,
    distance: DistanceMapResult | None = None,
    saddles: np.ndarray | None = None,
    caging: CagingPath | None = None,
    ps_module=None,
    show: bool = True,
) -> dict[str, Any]:
    ps = _load_polyscope(ps_module)
    ps.init()
    registered = register_pipeline_polyscope(
        voxelization,
        distance=distance,
        saddles=saddles,
        caging=caging,
        ps_module=ps,
    )
    if show:
        ps.show()
    return registered
