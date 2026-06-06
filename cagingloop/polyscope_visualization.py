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


def _install_loop_slider(ps, handles: list, labels: list) -> None:
    """Polyscope UI slider that shows one loop at a time (good for browsing 100+ loops)."""
    import polyscope.imgui as psim

    for j, h in enumerate(handles):
        h.set_enabled(j == 0)
    state = {"i": 0}

    def callback():
        n = len(handles)
        if n == 0:
            return
        changed, state["i"] = psim.SliderInt("loop", state["i"], 0, n - 1)
        psim.TextUnformatted(f"showing {state['i'] + 1}/{n}: {labels[state['i']]}")
        if psim.Button("show all"):
            for h in handles:
                h.set_enabled(True)
        psim.SameLine()
        if psim.Button("show one"):
            for j, h in enumerate(handles):
                h.set_enabled(j == state["i"])
        if changed:
            for j, h in enumerate(handles):
                h.set_enabled(j == state["i"])

    ps.set_user_callback(callback)


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
    slider: bool = False,
    slice_height: float | None = None,
    up_axis: int = 1,
    obj_mesh=None,
    mesh_shift=(0.0, 0.0, 0.0),
    ps_module=None,
    show: bool = True,
) -> dict[str, Any]:
    """Show the surface (+ optional distance/saddles) with the given caging loops.
    With `slider=True`, a UI slider shows one loop at a time (good for browsing many)."""
    ps = _load_polyscope(ps_module)
    ps.init()
    registered = register_pipeline_polyscope(
        voxelization, distance=distance, saddles=saddles, caging=None, ps_module=ps
    )
    if obj_mesh is not None:
        m = register_obj_mesh(ps, obj_mesh, shift=mesh_shift)
        if m is not None:
            registered["object_mesh"] = m
    if slice_height is not None:
        sl = register_distance_slice(ps, voxelization, distance, height_frac=slice_height, up_axis=up_axis)
        if sl is not None:
            registered["distance_slice"] = sl
    loops = list(loops)
    loop_reg = register_caging_loops_polyscope(ps, loops, labels=labels)
    registered.update(loop_reg)
    if slider and loop_reg:
        lbls = labels if labels is not None else [f"loop {i:02d}" for i in range(len(loops))]
        _install_loop_slider(ps, list(loop_reg.values()), list(lbls))
    if show:
        ps.show()
    return registered


def register_obj_mesh(
    ps,
    obj_mesh,
    *,
    name: str = "object mesh",
    transparency: float = 0.4,
    shift=(0.0, 0.0, 0.0),
):
    """Register the actual OBJ triangle mesh as a Polyscope surface mesh, optionally
    translated by `shift` (world units) so it can be offset from the voxel layers.
    `obj_mesh` is a (vertices, faces) tuple; returns the handle or None."""
    if obj_mesh is None:
        return None
    verts, faces = obj_mesh
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = verts + np.asarray(shift, dtype=float)
    mesh = ps.register_surface_mesh(name, verts, faces)
    try:
        mesh.set_transparency(transparency)
        mesh.set_color((0.7, 0.72, 0.78))
    except Exception:
        pass
    return mesh


def register_distance_slice(
    ps,
    voxelization: VoxelizationResult,
    distance: DistanceMapResult,
    *,
    height_frac: float = 0.5,
    up_axis: int = 1,
    name: str | None = None,
):
    """Register a horizontal cross-section of the 3D distance field D_p as a point
    cloud coloured by D_p (jet) — the Polyscope analogue of a 2D distance-field slice.
    The slice is taken at `height_frac` (0..1) along `up_axis`. Returns the handle or
    None if `distance` has no volumetric grid or the slice is empty."""
    if distance is None or getattr(distance, "distance_grid", None) is None:
        return None
    grid = np.asarray(distance.distance_grid, dtype=float)
    coords = [voxelization.grid_x, voxelization.grid_y, voxelization.grid_z]
    axis_coords = coords[up_axis]
    n = len(axis_coords)
    idx = min(max(int(round(height_frac * (n - 1))), 0), n - 1)
    sl = [slice(None)] * 3
    sl[up_axis] = idx
    plane = grid[tuple(sl)]  # 2D slice
    finite = np.isfinite(plane)
    if not finite.any():
        return None
    others = [a for a in range(3) if a != up_axis]
    ii, jj = np.where(finite)
    pts = np.zeros((len(ii), 3), dtype=float)
    pts[:, others[0]] = coords[others[0]][ii]
    pts[:, others[1]] = coords[others[1]][jj]
    pts[:, up_axis] = axis_coords[idx]
    label = name or f"D_p slice @ {'xyz'[up_axis]}={axis_coords[idx]:.3f}"
    pc = _register_points(ps, label, pts, radius=0.006)
    if pc is not None:
        pc.add_scalar_quantity("D_p", plane[ii, jj], enabled=True, cmap="jet")
    return pc


def show_internals_polyscope(
    voxelization: VoxelizationResult,
    *,
    distance: DistanceMapResult | None = None,
    grasp_mask: np.ndarray | None = None,
    base_point_id=None,
    saddle_voxels=None,
    saddle_points=None,
    saddle_types=None,
    saddle_source=None,
    saddle_descents=None,
    saddle_scalar_label="saddle_order",
    hull=None,
    saddle_neighbors=None,
    loops=None,
    slider: bool = False,
    slice_plane: bool = False,
    slice_height: float | None = None,
    up_axis: int = 1,
    obj_mesh=None,
    mesh_shift=(0.0, 0.0, 0.0),
    ps_module=None,
    show: bool = True,
) -> dict[str, Any]:
    """Show pipeline internals: surface, grasping space (coloured by D_p), base point
    p, volumetric Morse saddles, and traced loops — each a toggleable layer.
    With `slider=True`, a UI slider browses the loops one at a time.
    With `slice_plane=True`, a movable scene slice plane lets you cut into the
    D_p-coloured grasp-space volume (like the 2D slices in docs/saddle-distance-field.png)."""
    ps = _load_polyscope(ps_module)
    ps.init()
    gx, gy, gz = voxelization.grid_x, voxelization.grid_y, voxelization.grid_z
    reg: dict[str, Any] = {}

    s = _register_points(ps, "surface", voxelization.grid_on, radius=0.004, color=(0.8, 0.8, 0.85))
    if s is not None:
        reg["surface"] = s

    if obj_mesh is not None:
        m = register_obj_mesh(ps, obj_mesh, shift=mesh_shift)
        if m is not None:
            reg["object_mesh"] = m

    if grasp_mask is not None:
        # show only the exterior part of the grasp space, so it doesn't overlap the
        # separately-drawn `surface` layer (the mask itself includes the surface band)
        gmask = np.asarray(grasp_mask, dtype=bool) & (voxelization.output_grid != 1)
        gi = np.argwhere(gmask)
        gw = np.column_stack((gx[gi[:, 0]], gy[gi[:, 1]], gz[gi[:, 2]]))
        gc = _register_points(ps, "grasp space", gw, radius=0.003, color=(0.4, 0.6, 1.0))
        if gc is not None:
            reg["grasp_space"] = gc
            if distance is not None:
                gc.add_scalar_quantity("D_p", distance.distance_grid[gi[:, 0], gi[:, 1], gi[:, 2]], enabled=True)

    if slice_height is not None:
        sl = register_distance_slice(ps, voxelization, distance, height_frac=slice_height, up_axis=up_axis)
        if sl is not None:
            reg["distance_slice"] = sl

    if hull is not None:
        hv, hf = hull
        mesh = ps.register_surface_mesh("convex hull", np.asarray(hv, dtype=float), np.asarray(hf, dtype=int))
        try:
            mesh.set_transparency(0.25)
            mesh.set_color((0.95, 0.75, 0.2))
        except Exception:
            pass
        reg["hull"] = mesh

    if base_point_id is not None:
        ids = np.atleast_1d(np.asarray(base_point_id, dtype=int))
        bp = _register_points(
            ps, "base points", voxelization.grid_on[ids], radius=0.02, color=(0.1, 0.9, 0.2)
        )
        if bp is not None:
            reg["base_points"] = bp

    if saddle_neighbors is not None:
        pts, signs = saddle_neighbors
        if len(pts) > 0:
            nc = _register_points(ps, "saddle +/- neighbors", np.asarray(pts, dtype=float), radius=0.006)
            if nc is not None:
                # +1 = neighbour has higher D (uphill), -1 = lower D (downhill) — the Fig. 3 labels
                nc.add_scalar_quantity("sign", np.asarray(signs, dtype=float), enabled=True, cmap="coolwarm")
                reg["saddle_neighbors"] = nc

    # Saddle markers: either void voxels (morse) or surface points (surface method).
    saddle_world = None
    saddle_name = "morse saddles"
    if saddle_points is not None and len(saddle_points) > 0:
        saddle_world = np.asarray(saddle_points, dtype=float)
        saddle_name = "surface saddles"
    elif saddle_voxels is not None and len(saddle_voxels) > 0:
        sv = np.asarray(saddle_voxels, dtype=int)
        saddle_world = np.column_stack((gx[sv[:, 0]], gy[sv[:, 1]], gz[sv[:, 2]]))
    if saddle_world is not None and len(saddle_world) > 0:
        n_sad = len(saddle_world)
        sd = _register_points(ps, saddle_name, saddle_world, radius=0.012, color=(1.0, 0.0, 1.0))
        if sd is not None:
            reg["saddles"] = sd
            if saddle_types is not None and len(saddle_types) == n_sad:
                # morse: lower-link component count (2 = 1-saddle, 3+ = monkey).
                # surface: tangent-ring sign-flip count (>=4 = saddle).
                sd.add_scalar_quantity(saddle_scalar_label, np.asarray(saddle_types, dtype=float), enabled=True)
            if saddle_source is not None and len(saddle_source) == n_sad:
                # which base point each saddle came from (when several base points are shown)
                sd.add_scalar_quantity("base_point", np.asarray(saddle_source, dtype=float), enabled=False)
            if saddle_descents is not None and len(saddle_descents) == n_sad:
                # the (>=2) opposite descending directions of each saddle, in world units.
                # These are the directions traced back to p that join into a caging loop
                # (Thm 3.1). Drawn as the first two descents per saddle.
                spacing = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
                v1 = np.zeros((n_sad, 3))
                v2 = np.zeros((n_sad, 3))
                for i, offs in enumerate(saddle_descents):
                    offs = list(offs)
                    if len(offs) >= 1:
                        v1[i] = np.asarray(offs[0], dtype=float) * spacing
                    if len(offs) >= 2:
                        v2[i] = np.asarray(offs[1], dtype=float) * spacing
                sd.add_vector_quantity("descent A", v1, enabled=True, color=(0.1, 0.9, 0.1), radius=0.004)
                sd.add_vector_quantity("descent B", v2, enabled=True, color=(0.1, 0.6, 0.1), radius=0.004)

    if slice_plane:
        try:
            sp = ps.add_scene_slice_plane()
            # start it cutting through the middle along the up (y) axis
            sp.set_pose((0.0, float(gy.mean()), 0.0), (0.0, 1.0, 0.0))
            reg["slice_plane"] = sp
        except Exception:
            pass

    loop_handles, loop_labels = [], []
    if loops:
        for i, lp in enumerate(loops):
            path = np.asarray(getattr(lp, "final_path", lp), dtype=float)
            if len(path) >= 2:
                name = f"loop {i:02d}"
                h = ps.register_curve_network(name, path, _loop_edges(len(path)), radius=0.008, color=_distinct_color(i, len(loops)))
                reg[name] = h
                loop_handles.append(h)
                loop_labels.append(name)
        if slider and loop_handles:
            _install_loop_slider(ps, loop_handles, loop_labels)
    if show:
        ps.show()
    return reg


def show_pipeline_polyscope(
    voxelization: VoxelizationResult,
    *,
    distance: DistanceMapResult | None = None,
    saddles: np.ndarray | None = None,
    caging: CagingPath | None = None,
    slice_height: float | None = None,
    up_axis: int = 1,
    obj_mesh=None,
    mesh_shift=(0.0, 0.0, 0.0),
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
    if obj_mesh is not None:
        m = register_obj_mesh(ps, obj_mesh, shift=mesh_shift)
        if m is not None:
            registered["object_mesh"] = m
    if slice_height is not None:
        sl = register_distance_slice(ps, voxelization, distance, height_frac=slice_height, up_axis=up_axis)
        if sl is not None:
            registered["distance_slice"] = sl
    if show:
        ps.show()
    return registered
