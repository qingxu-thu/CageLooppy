"""GPU (fp32) batched surface saddle test.

Same algorithm as the CPU `detect_saddle_point`, but the per-base-point work — judge,
the flat/collinear boundary filter, and the cyclic sign-transition count — is run as
dense [N, maxk] torch tensors on the GPU in float32. The base-point-independent boundary
geometry is precomputed once (`build_saddle_topology_padded`, CPU).

float32 only affects results where a `D[neighbour] - D[point]` sign is a near-tie, so a
few saddles can appear/disappear vs the fp64 CPU path — measure the divergence with
`saddle_set_divergence` and keep the CPU path as the exact reference.

Exact-equivalence notes (so the logic — not just precision — matches the CPU):
- `_filter_boundary` rotates the boundary to start at the max |grad| neighbour; the only
  effect on the result is that the *incoming* edge of that neighbour is exempt from the
  <=15deg collinear cull. So delete_angle = {q != argmax : inc_angle[q] <= 15}.
- delete_small and the final cyclic transition count are rotation-invariant.
"""

from __future__ import annotations

import numpy as np

from cagingloop.saddle import build_saddle_topology, diversity_eval


def build_saddle_topology_padded(grid_on: np.ndarray, k: int = 9) -> dict:
    """Dense, padded version of `build_saddle_topology` for batched GPU evaluation.
    Returns [N, maxk] arrays: boundary neighbour ids, norms, incoming edge angles, a
    validity mask, plus per-point boundary lengths and the enclosed flag."""
    topo = build_saddle_topology(grid_on, k=k)
    n = len(grid_on)
    lengths = np.array(
        [len(topo.bnd_ids[i]) if topo.enclosed[i] else 0 for i in range(n)], dtype=np.int64
    )
    maxk = int(lengths.max()) if lengths.max() > 0 else 1
    bnd_ids = np.zeros((n, maxk), dtype=np.int64)
    norms = np.ones((n, maxk), dtype=np.float32)
    inc_ang = np.full((n, maxk), 999.0, dtype=np.float32)  # 999 = "not collinear" for pads
    valid = np.zeros((n, maxk), dtype=bool)
    for i in range(n):
        if not topo.enclosed[i]:
            continue
        ids = topo.bnd_ids[i]
        pts = topo.bnd_pts[i]
        m = len(ids)
        bnd_ids[i, :m] = ids
        valid[i, :m] = True
        nrm = np.linalg.norm(pts, axis=1)
        nrm[nrm == 0.0] = 1.0
        norms[i, :m] = nrm
        prev = np.roll(pts, 1, axis=0)  # prev[q] = pts[q-1] (cyclic)
        denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(pts, axis=1)
        denom[denom == 0.0] = 1.0
        inc_ang[i, :m] = np.degrees(np.arccos(np.clip(np.sum(prev * pts, axis=1) / denom, -1.0, 1.0)))
    return {
        "bnd_ids": bnd_ids, "norms": norms, "inc_ang": inc_ang, "valid": valid,
        "lengths": lengths, "enclosed": topo.enclosed, "maxk": maxk, "k": k,
        "topo": topo,  # the CPU SaddleTopology (so callers reuse one build)
    }


def _iter_num_batch_gpu(dismap, padded, device):
    """Per-point iter_num (transition count) for all points at once, on GPU fp32."""
    import torch

    dev = torch.device(device)
    D = torch.as_tensor(np.asarray(dismap, dtype=np.float32), device=dev)
    bnd_ids = torch.as_tensor(padded["bnd_ids"], device=dev)
    norms = torch.as_tensor(padded["norms"], device=dev)
    inc_ang = torch.as_tensor(padded["inc_ang"], device=dev)
    valid = torch.as_tensor(padded["valid"], device=dev)
    lengths = torch.as_tensor(padded["lengths"], device=dev)
    enclosed = torch.as_tensor(padded["enclosed"], device=dev)
    n, maxk = bnd_ids.shape
    col = torch.arange(maxk, device=dev)[None, :]

    D = torch.nan_to_num(D, nan=0.0, posinf=1e30, neginf=-1e30)
    pt = torch.arange(n, device=dev)
    judge = D[bnd_ids] - D[pt][:, None]                       # [N, maxk]
    judge = torch.where(valid, judge, torch.zeros_like(judge))
    dis_toler = torch.where(valid, torch.abs(judge) / norms, torch.full_like(judge, -1.0))
    peak = dis_toler.max(dim=1).values                        # [N]
    argmax = dis_toler.argmax(dim=1)                          # [N] = rotation start
    delete_small = valid & (dis_toler < (0.1 * peak)[:, None])
    delete_angle = valid & (inc_ang <= 15.0) & (col != argmax[:, None])
    keep = valid & ~(delete_small | delete_angle)

    sign = (judge > 0.0).to(torch.int8)                       # [N, maxk]
    m = lengths.clamp(min=1)[:, None]
    nxt = torch.full((n, maxk), -1, dtype=torch.int8, device=dev)
    found = torch.zeros((n, maxk), dtype=torch.bool, device=dev)
    for d in range(1, maxk + 1):                              # next kept sign, cyclically
        cand_col = (col + d) % m
        cand_keep = torch.gather(keep, 1, cand_col)
        cand_sign = torch.gather(sign, 1, cand_col)
        take = keep & cand_keep & ~found
        nxt = torch.where(take, cand_sign, nxt)
        found |= take
    transitions = (keep & found & (sign != nxt)).sum(dim=1)   # [N]
    iter_num = torch.where(enclosed, transitions, torch.full_like(transitions, -1))
    return iter_num.cpu().numpy()


def detect_saddle_point_gpu(
    dismap, grid_on, source_point_id, grid_on_normals, *, padded=None, k=9, keep=None, device="cuda"
):
    """GPU fp32 drop-in for `detect_saddle_point`. Falls back to the CPU path if torch/CUDA
    is unavailable. Pass a prebuilt `padded` topology to amortise it over many base points."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    except Exception:
        from cagingloop.saddle import detect_saddle_point
        return detect_saddle_point(dismap, grid_on, source_point_id, grid_on_normals, k=k, keep=keep)

    grid_on = np.asarray(grid_on, dtype=float)
    normals = np.asarray(grid_on_normals, dtype=float)
    if padded is None:
        padded = build_saddle_topology_padded(grid_on, k=k)
    iter_num = _iter_num_batch_gpu(dismap, padded, device)
    candidates = np.nonzero(iter_num >= 4)[0]
    if len(candidates) == 0:
        return np.zeros((0,), dtype=int)

    sp, sn = grid_on[source_point_id], normals[source_point_id]
    scores = np.array([diversity_eval(sp, grid_on[c], sn, normals[c]) for c in candidates], dtype=float)
    order = np.argsort(scores)[::-1]
    n_keep = max(1, int(round(len(scores) / 30.0))) if keep is None else max(1, min(int(keep), len(scores)))
    return candidates[order[:n_keep]].astype(int)


def saddle_set_divergence(cpu_ids: np.ndarray, gpu_ids: np.ndarray) -> dict:
    """Report how the fp32-GPU saddle set differs from the fp64-CPU set (the 'a little
    inaccurate' measurement)."""
    c, g = set(int(x) for x in cpu_ids), set(int(x) for x in gpu_ids)
    inter = c & g
    return {
        "cpu": len(c), "gpu": len(g), "shared": len(inter),
        "only_cpu": len(c - g), "only_gpu": len(g - c),
        "jaccard": (len(inter) / len(c | g)) if (c | g) else 1.0,
    }
