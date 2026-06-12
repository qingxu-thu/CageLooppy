"""GPU (fp32) batched RK4 streamline tracer for caging-loop arcs.

The CPU tracer steps one arc at a time (~1000 RK4 steps x 4 scipy interpolation calls each),
dominated by per-call overhead. This runs **all arcs of one base point in lockstep**: each
step is a single batched trilinear gather over the pointmin descent field, in float32 on the
GPU. The per-step logic (normalised increments, the k1..k4 bounds checks, the no-progress and
reached termination) mirrors `_rk4_descent_path` exactly, so only fp32 precision differs.

Arcs that fail RK4 on the GPU (out of bounds / stalled / never reach the source) fall back to
the CPU `compute_shortest_path` for that arc — identical to the CPU path's own RK4->greedy
fallback. Falls back wholesale to the CPU generator if torch/CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np

from cagingloop.distance import (
    _index_to_xyz,
    _pointmin_descent_field,
    _voxel_to_world,
    compute_shortest_path,
)
from cagingloop.grasp import (
    CagingCandidate,
    _path_separation,
    get_cage_points,
    loop_enclosed_area,
    smooth_closed_path,
)
from cagingloop.types import CagingPath, VoxelizationResult


def _arc_world(voxel_path, start_idx, src_idx, voxelization):
    """Convert a traced voxel path to world coords with endpoints snapped to the start/source
    surface points — the tail of `compute_shortest_path`."""
    path = _voxel_to_world(voxel_path, voxelization)
    path[0] = _index_to_xyz(tuple(int(v) for v in start_idx), voxelization)
    path[-1] = _index_to_xyz(tuple(int(v) for v in src_idx), voxelization)
    return path


def _batch_rk4(fields: np.ndarray, base_idx: np.ndarray, starts: np.ndarray, sources: np.ndarray, *, step: float, device):
    """Trace many arcs in lockstep on GPU, each arc `b` over its own field `fields[base_idx[b]]`
    from `starts[b]` to `sources[b]`. `fields` is [Nb, nx, ny, nz, 3]. Single-field callers pass
    Nb=1 with base_idx all-zero. Returns a list of voxel-index paths ([M,3]) or None per arc."""
    import torch

    dev = torch.device(device)
    F = torch.as_tensor(np.ascontiguousarray(fields, dtype=np.float32), device=dev)  # [Nb,nx,ny,nz,3]
    _, nx, ny, nz, _ = F.shape
    upper = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=torch.float32, device=dev)
    nmax = torch.tensor([nx - 1, ny - 1, nz - 1], device=dev)
    P0 = torch.as_tensor(np.asarray(starts, dtype=np.float32), device=dev)   # [B,3]
    S = torch.as_tensor(np.asarray(sources, dtype=np.float32), device=dev)   # [B,3]
    bse = torch.as_tensor(np.asarray(base_idx, dtype=np.int64), device=dev)  # [B]
    B = P0.shape[0]

    def interp(Q):  # clamped trilinear gather over each arc's own field, [B,3] -> [B,3]
        Qc = torch.clamp(Q, torch.zeros(3, device=dev), upper)
        p0 = torch.clamp(torch.floor(Qc).long(), torch.zeros(3, dtype=torch.long, device=dev), nmax)
        f = Qc - p0.float()
        x0, y0, z0 = p0[:, 0], p0[:, 1], p0[:, 2]
        x1 = torch.clamp(x0 + 1, max=nx - 1); y1 = torch.clamp(y0 + 1, max=ny - 1); z1 = torch.clamp(z0 + 1, max=nz - 1)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        out = torch.zeros((B, 3), device=dev)
        for xi, wx in ((x0, 1 - fx), (x1, fx)):
            for yi, wy in ((y0, 1 - fy), (y1, fy)):
                for zi, wz in ((z0, 1 - fz), (z1, fz)):
                    out = out + (wx * wy * wz)[:, None] * F[bse, xi, yi, zi]
        return out

    def increment(Q):
        g = interp(Q)
        n = g.norm(dim=1)
        valid = torch.isfinite(n) & (n >= 1e-12)
        inc = torch.where(valid[:, None], g * step / n.clamp(min=1e-12)[:, None], torch.zeros_like(g))
        return inc, valid

    def in_bounds(Q):
        return (Q >= 0).all(dim=1) & (Q <= upper).all(dim=1)

    def rk4_step(Q):
        k1, v1 = increment(Q); ok = v1 & in_bounds(Q + 0.5 * k1)
        k2, v2 = increment(Q + 0.5 * k1); ok = ok & v2 & in_bounds(Q + 0.5 * k2)
        k3, v3 = increment(Q + 0.5 * k2); ok = ok & v3 & in_bounds(Q + k3)
        k4, v4 = increment(Q + k3); ok = ok & v4
        nxt = Q + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return nxt, ok & in_bounds(nxt)

    max_iter = int(8 * (nx + ny + nz)) + 100
    buf = torch.empty((max_iter + 2, B, 3), dtype=torch.float32, device=dev)
    buf[0] = P0
    status = torch.zeros(B, dtype=torch.int8, device=dev)  # 0 active, 1 reached, 2 failed
    reach_idx = torch.full((B,), -1, dtype=torch.long, device=dev)
    for i in range(max_iter):
        active = status == 0
        if not bool(active.any()):
            break
        nxt, ok = rk4_step(buf[i])
        ok_active = active & ok
        if i > 10:
            noprog = ok_active & ((nxt - buf[i - 10]).norm(dim=1) < step)  # checked before appending (CPU order)
        else:
            noprog = torch.zeros(B, dtype=torch.bool, device=dev)
        advance = ok_active & ~noprog
        buf[i + 1] = torch.where(advance[:, None], nxt, buf[i])
        reached_now = advance & ((nxt - S).norm(dim=1) < step)
        status = torch.where(reached_now, torch.ones_like(status), status)
        reach_idx = torch.where(reached_now, torch.full_like(reach_idx, i + 1), reach_idx)
        status = torch.where(active & (~ok | noprog), torch.full_like(status, 2), status)

    buf_c = buf.cpu().numpy()
    st = status.cpu().numpy()
    ri = reach_idx.cpu().numpy()
    src = np.asarray(sources, dtype=float)  # [B,3], per-arc
    out = []
    for b in range(B):
        if st[b] == 1:  # reached: path = buf[0..reach_idx] + this arc's source
            out.append(np.vstack((buf_c[: ri[b] + 1, b, :], src[b][None, :])))
        else:
            out.append(None)
    return out


def generate_caging_grasps_gpu(
    distance_grid: np.ndarray,
    voxelization: VoxelizationResult,
    source_point_id: int,
    dismap: np.ndarray,
    saddle_point_ids: np.ndarray,
    *,
    k: int = 9,
    max_loops: int = 10,
    min_separation_ratio: float = 0.02,
    step: float = 0.5,
    device: str = "cuda",
) -> list[CagingCandidate]:
    """GPU fp32 batched version of `generate_caging_grasps` for the rk4 tracer. Falls back to
    the CPU generator if torch/CUDA is unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
    except Exception:
        from cagingloop.grasp import generate_caging_grasps
        return generate_caging_grasps(distance_grid, voxelization, source_point_id, dismap, saddle_point_ids,
                                      k=k, max_loops=max_loops, min_separation_ratio=min_separation_ratio,
                                      integrator="rk4")

    distance_grid = np.asarray(distance_grid, dtype=float)
    field = _pointmin_descent_field(distance_grid)
    grid_on = voxelization.grid_on
    scale = float(np.linalg.norm(grid_on.max(axis=0) - grid_on.min(axis=0))) if len(grid_on) else 1.0
    source_vox = tuple(int(v) for v in voxelization.index.on_index[source_point_id])

    # gather the two cage points (arc starts) per saddle
    sad_ids, starts_idx, cage_pairs = [], [], []
    for sid in np.asarray(saddle_point_ids, dtype=int).tolist():
        try:
            cps = get_cage_points(dismap, grid_on, sid, k=k)
        except ValueError:
            continue
        c0, c1 = int(cps[0]), int(cps[1])
        sad_ids.append(sid)
        cage_pairs.append((c0, c1))
        starts_idx.append(voxelization.index.on_index[c0])
        starts_idx.append(voxelization.index.on_index[c1])

    if not sad_ids or field is None:
        return []
    starts = np.asarray(starts_idx, dtype=float)
    n_arcs = len(starts)
    traced = _batch_rk4(
        field[None], np.zeros(n_arcs, dtype=np.int64), starts,
        np.tile(np.asarray(source_vox, dtype=float), (n_arcs, 1)), step=step, device=device,
    )

    def arc_world(voxel_path, start_idx, src_idx):
        return _arc_world(voxel_path, start_idx, src_idx, voxelization)

    candidates: list[CagingCandidate] = []
    for j, sid in enumerate(sad_ids):
        c0, c1 = cage_pairs[j]
        results = []
        for arc, cage in ((traced[2 * j], c0), (traced[2 * j + 1], c1)):
            if arc is not None and len(arc) >= 2:
                results.append(arc_world(arc, voxelization.index.on_index[cage], source_vox))
            else:  # GPU arc failed -> CPU fallback (its own rk4->greedy), identical to CPU path
                try:
                    results.append(compute_shortest_path(distance_grid, voxelization, cage, source_point_id, integrator="rk4"))
                except ValueError:
                    results.append(None)
        if results[0] is None or results[1] is None:
            continue
        saddle = grid_on[sid : sid + 1]
        p1 = np.vstack((saddle, results[0]))
        p2 = np.vstack((saddle, results[1]))
        p2 = np.flipud(p2)[1:]
        final_path = smooth_closed_path(np.vstack((p1, p2)))
        caging = CagingPath(final_path=final_path, path1=final_path[: len(p1)], path2=final_path[len(p1) - 1 :])
        separation = _path_separation(caging)
        if scale > 0.0 and separation < min_separation_ratio * scale:
            continue
        candidates.append(CagingCandidate(sid, caging, loop_enclosed_area(final_path), separation, int(source_point_id)))
    candidates.sort(key=lambda c: c.area, reverse=True)
    return candidates[:max_loops]


def sweep_surface_loops_gpu(
    voxelization: VoxelizationResult,
    base_ids,
    surface_normals: np.ndarray,
    padded_topo: dict,
    *,
    solver: str = "fmm",
    traversable_mask=None,
    sweep_radius=None,
    k: int = 9,
    min_separation_ratio: float = 0.02,
    step: float = 0.5,
    device: str = "cuda",
):
    """Cross-base-batched surface caging-loop sweep. Phase 1 (per base, CPU+GPU-saddle):
    distance field, GPU saddle detection, cage points. Phase 2: trace **all** arcs of **all**
    base points in ONE batched GPU run, each arc over its own base's field — amortising the
    per-step launch overhead over the whole sweep. Phase 3: assemble + filter loops.

    Returns (pool, saddle_points, saddle_types, saddle_source, first_distance)."""
    from cagingloop import distance_map_by_fast_marching
    from cagingloop.saddle import _iter_num_cached
    from cagingloop.saddle_gpu import detect_saddle_point_gpu

    go = voxelization.grid_on
    topo = padded_topo["topo"]
    fields: list[np.ndarray] = []
    distances: list = []
    saddle_points, saddle_types, saddle_source = [], [], []
    meta = []   # per saddle: (base_local, sid, cage0, cage1)
    starts, sources, base_of = [], [], []
    first_distance = None
    for bi, bid in enumerate(base_ids):
        d = distance_map_by_fast_marching(
            voxelization, bid, solver=solver, traversable_mask=traversable_mask, max_distance=sweep_radius
        )
        if first_distance is None:
            first_distance = d
        fields.append(_pointmin_descent_field(d.distance_grid))
        distances.append(d)
        sad = detect_saddle_point_gpu(d.dismap, go, bid, surface_normals, padded=padded_topo, keep=len(go))
        src_idx = voxelization.index.on_index[bid]
        for sid in np.atleast_1d(sad).astype(int):
            saddle_points.append(go[sid]); saddle_source.append(bi)
            saddle_types.append(float(_iter_num_cached(d.dismap, topo, int(sid))))
            try:
                cps = get_cage_points(d.dismap, go, sid, k=k)
            except ValueError:
                continue
            c0, c1 = int(cps[0]), int(cps[1])
            meta.append((bi, sid, c0, c1))
            for cage in (c0, c1):
                starts.append(voxelization.index.on_index[cage]); sources.append(src_idx); base_of.append(bi)

    if not meta or any(f is None for f in fields):
        return [], saddle_points, saddle_types, saddle_source, first_distance

    fields_stacked = np.stack(fields, axis=0)  # [Nb, nx, ny, nz, 3]
    traced = _batch_rk4(fields_stacked, np.asarray(base_of), np.asarray(starts, float),
                        np.asarray(sources, float), step=step, device=device)

    grid_on = go
    scale = float(np.linalg.norm(grid_on.max(axis=0) - grid_on.min(axis=0))) if len(grid_on) else 1.0
    pool: list[CagingCandidate] = []
    for mi, (bi, sid, c0, c1) in enumerate(meta):
        d = distances[bi]; bid = base_ids[bi]
        src_vox = tuple(int(v) for v in voxelization.index.on_index[bid])
        results = []
        for arc, cage in ((traced[2 * mi], c0), (traced[2 * mi + 1], c1)):
            if arc is not None and len(arc) >= 2:
                results.append(_arc_world(arc, voxelization.index.on_index[cage], src_vox, voxelization))
            else:  # GPU arc failed -> CPU fallback (rk4 -> greedy), identical to CPU path
                try:
                    results.append(compute_shortest_path(d.distance_grid, voxelization, cage, bid, integrator="rk4"))
                except ValueError:
                    results.append(None)
        if results[0] is None or results[1] is None:
            continue
        saddle = grid_on[sid : sid + 1]
        p1 = np.vstack((saddle, results[0]))
        p2 = np.flipud(np.vstack((saddle, results[1])))[1:]
        final_path = smooth_closed_path(np.vstack((p1, p2)))
        caging = CagingPath(final_path=final_path, path1=final_path[: len(p1)], path2=final_path[len(p1) - 1 :])
        separation = _path_separation(caging)
        if scale > 0.0 and separation < min_separation_ratio * scale:
            continue
        pool.append(CagingCandidate(sid, caging, loop_enclosed_area(final_path), separation, int(bid)))
    return pool, saddle_points, saddle_types, saddle_source, first_distance
