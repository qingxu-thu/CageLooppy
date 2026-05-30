# MATLAB ↔ Python Differences

A function-by-function comparison of the original MATLAB code in [`Code/`](../Code/)
against the Python port in [`cagingloop/`](../cagingloop/).

**Bottom line: the port is _not_ bit-for-bit identical to MATLAB, and was never
intended to be.** The design spec
([`docs/superpowers/specs/2026-05-27-python-port-design.md`](superpowers/specs/2026-05-27-python-port-design.md))
explicitly lists "bit-for-bit or point-for-point equality with MATLAB output" as a
**non-goal**. It aims for *workflow-level algorithmic equivalence* while accepting
numerical drift from replacing the external solvers.

Differences are grouped as:

- **(A) Fundamental backend replacements** — intended; full parity impossible.
- **(B) Algorithmic / logic differences** — the same step done differently.
- **(C) Aligned in this session** — divergences that were fixed to match MATLAB.
- **(D) Non-algorithmic** — I/O and visualization, no effect on results.

---

## (A) Fundamental backend replacements — cannot match MATLAB numerically

| Stage | MATLAB | Python | Consequence |
|-------|--------|--------|-------------|
| Implicit field | `FastRBF.exe` (`fit -accuracy=1.0 -rho=0.01`), evaluated via shelled-out `.exe` calls and temp files | `scipy.interpolate.RBFInterpolator(kernel="linear", neighbors=..., smoothing=...)` | Different scalar field → **different surface voxel set**, different inner/outer split. **This is now the single largest remaining numerical gap.** |
| Fast marching | `msfm(W, source)` — Kroon multistencil Eikonal solver | `skfmm.travel_time(...)` — same Eikonal solve on the regular grid with a speed map (now a **core dependency**, used by default). Dijkstra only remains as a fallback for degenerate/thin grids | **Same method and domain.** Residual difference is solver order only (skfmm 1st/2nd order vs `msfm` multistencil) — see C4 |
| Nearest neighbors | TSTOOL MEX `nn_prepare` / `nn_search` | `scipy.spatial.cKDTree` | Equivalent k-NN results (not a meaningful difference) |
| Surface mesh | `FastRBF.exe mcubes` → `gridmesh.obj` | optional `skimage.measure.marching_cubes` (only if `extract_mesh=True`) | Cosmetic; not used by the caging pipeline |

---

## (B) Algorithmic / logic differences

### B1. Voxelization — surface detection criterion
- **MATLAB** ([`pointCloudVoxelizationByRBF.m:96-102`](../Code/pointCloudVoxelizationByRBF.m)):
  for every grid point, take its 6 nearest grid neighbors, compute
  `ff = max(rbf_neighbors) * min(rbf_neighbors)`, and mark it surface if `ff <= 1.0`.
- **Python** ([`voxelization.py:51-61`](../cagingloop/voxelization.py)):
  mark surface where `|field| <= 0.05 * max|field|` **or** any axis-aligned 6-neighbor
  edge crosses zero.
- **Effect:** different rule, and it operates on a different field (A), so the surface
  point set differs in count and placement.

### B2. Voxelization — surface index derivation
- **MATLAB** ([`pointCloudVoxelizationByRBF.m:106-118`](../Code/pointCloudVoxelizationByRBF.m)):
  recomputes a bounding box **from `grid_on` itself**, derives new spacings, and rounds
  coordinates back to integer indices (`round((p-min)/l)+1`).
- **Python** ([`voxelization.py:159`](../cagingloop/voxelization.py)):
  uses `np.argwhere(surface)` — the grid indices directly.
- **Effect:** Python is the cleaner/more direct mapping; MATLAB's re-binning can shift a
  point to a neighboring cell. Usually negligible, occasionally off-by-one.

### B3. Voxelization — fixed offset vs derived offset
- **MATLAB**: hard-coded `±0.0001` normal offset for the ± constraints; **does not
  normalize** `ptnormals`.
- **Python**: `normal_offset` parameter (default `0.1 * min(spacing)`); **normalizes**
  normals and rejects zero-length normals.
- **Note:** the model example passes `normal_offset=1e-4`, matching MATLAB's `0.0001`.
  The normalization difference matters only if input normals are non-unit.

### B4. Voxelization — degenerate axis handling
- **Python** ([`voxelization.py:38-43`](../cagingloop/voxelization.py)) expands any axis where
  `min == max` by ±0.5. **MATLAB has no such guard** (would divide by zero).

### B5. Distance map — unreachable-point repair
- **MATLAB** ([`DistanceMapByFastMarching.m:29-36`](../Code/DistanceMapByFastMarching.m)):
  a surface value is "bad" if `>= 10000`; replaced with the `min` over the raw `D` of its
  9 nearest surface neighbors (including any large values).
- **Python** ([`distance.py:91-110`](../cagingloop/distance.py)):
  "bad" means **non-finite** (`inf`); replaced with the `min` over the **finite-only**
  neighbor values.
- **Effect:** different bad-value threshold and a slightly different repair set.

### B6. Saddle detection — local frame: PCA vs SVD (residual)
- **MATLAB** ([`detectSaddlePoint.m:54-57`](../Code/detectSaddlePoint.m)): `pca(X', 3)`
  on the 14 nearest neighbors; axes = principal components.
- **Python** ([`saddle.py:32-46`](../cagingloop/saddle.py)): `np.linalg.svd` on the
  **mean-centered** offsets (see C6); axes = right-singular vectors.
- **Residual:** mathematically the same subspace, but **axis signs can flip** (PCA/SVD
  sign ambiguity). The convex-hull ordering and transition count are invariant to a mirror,
  so the sign ambiguity does not change the outcome. (The **centering** part of this was a
  real bug — see C6.)

### B7. Saddle detection — boundary polygon (verified equivalent)
- **MATLAB**: `boundary(adj_point, 0)` returns a **closed** loop (first vertex repeated);
  `filterBoundary` then rotates it, checks the wrap-edge angle, and deletes indices.
- **Python** ([`saddle.py:64-71`](../cagingloop/saddle.py)): `scipy.spatial.ConvexHull`
  (not closed); the wrap edge is restored by `np.roll` in
  [`_count_transitions`](../cagingloop/saddle.py#L123). Falls back to angular sort if Qhull
  fails (MATLAB has no fallback).
- **Verified:** a faithful closed-loop reimplementation of `filterBoundary` was compared
  against the current open-hull version over all 1812 surface points of `knotty` — the
  `flag>=4` saddle classification was **identical** (0 points reclassified). The open/closed
  difference is functionally equivalent, so it was left as-is.

### B8. Saddle detection — extra guards
- **Python** `_filter_boundary` ([`saddle.py:118-119`](../cagingloop/saddle.py)) keeps the
  original boundary if filtering would leave `< 3` points. **MATLAB has no such guard.**

### B9. Saddle selection — keep-count floor
- **MATLAB** ([`detectSaddlePoint.m:29`](../Code/detectSaddlePoint.m)):
  `saddle(opt_I(1:round(n/30)))` — can be **empty** when `round(n/30) == 0`.
- **Python** ([`saddle.py:184`](../cagingloop/saddle.py)): `max(1, round(n/30))` — always
  keeps **at least one** candidate.

### B10. Cage points — fallback
- **MATLAB** `getCagePoint` ([`generateCagingGrasp.m:31-74`](../Code/generateCagingGrasp.m))
  assumes ≥2 negative-direction neighbors exist.
- **Python** `get_cage_points` ([`grasp.py:109-118`](../cagingloop/grasp.py)) adds a
  deterministic fallback (two lowest-distance neighbors) when that assumption fails.

> Note: the **caging path concatenation and Laplacian smoothing**
> ([`grasp.py:122-150`](../cagingloop/grasp.py)) are an **exact** port of
> [`generateCagingGrasp.m:14-28`](../Code/generateCagingGrasp.m) — same `0.2/0.4/0.4`
> weights, same start/end closure.

---

## (C) Divergences aligned in this session

These were genuine mismatches that have now been corrected to match MATLAB:

### C1. `diversity_eval` — MATLAB `pow2` semantics ✅ fixed
- **MATLAB** ([`detectSaddlePoint.m:48`](../Code/detectSaddlePoint.m)):
  `exp(-0.5 * pow2(max(angle1,angle2), 4))`. MATLAB `pow2(F,E) = F * 2^E`, so
  `pow2(x,4) = 16x` → `exp(-8 * max_angle)`.
- **Was:** `exp(-0.5 * max_angle ** 4)` (misread as a 4th power).
- **Now** ([`saddle.py:23-31`](../cagingloop/saddle.py)): `exp(-0.5 * max_angle * 2**4)`.
  This is the saddle-ranking score, so the bug affected selection on every model.

### C2. z-min boundary face ✅ fixed
- **MATLAB** ([`pointCloudVoxelizationByRBF.m:181-196`](../Code/pointCloudVoxelizationByRBF.m)):
  moves only the **x±, y±, z-max** boundary inner voxels to "outer"; the `z-min` face
  (`tf_z2`) is deliberately commented out, keeping the bottom face inner.
- **Was:** Python moved **all 6 faces**.
- **Now** ([`voxelization.py:64-73`](../cagingloop/voxelization.py)): z-min face excluded.

### C3. Surface normals fed to saddle detection ✅ fixed (in the model example)
- **MATLAB**: `detectSaddlePoint` receives true surface normals (`grid_on_normals`).
- **Was:** the model example faked **radial** normals (`grid_on − centroid`), wrong for
  concave shapes.
- **Now** ([`model_io.py:transfer_point_normals`](../cagingloop/model_io.py) +
  [`run_model_polyscope.py`](../examples/run_model_polyscope.py)): each surface voxel takes
  the normal of its nearest input-cloud point. (The synthetic sphere example keeps radial
  normals, which are correct for a sphere.)

### C4. Distance field — Eikonal solver instead of Dijkstra ✅ aligned
- **MATLAB** ([`DistanceMapByFastMarching.m:24`](../Code/DistanceMapByFastMarching.m)):
  `msfm(W, source)` — fast marching solving the Eikonal equation on the voxel grid with a
  speed map (surface=1, exterior=1, interior=0).
- **Was:** with `scikit-fmm` absent, the code fell back to a **Dijkstra** graph distance
  (sum of axis steps), which is not an Eikonal geodesic.
- **Now**: `scikit-fmm` is a **core dependency** and the examples run `prefer_fmm=True`, so
  `skfmm.travel_time` performs the same Eikonal solve over the same domain
  ([`distance.py:76-88`](../cagingloop/distance.py)). Dijkstra remains only as a fallback
  for degenerate/thin grids (and in the synthetic unit tests).
- **Residual:** `skfmm` is 1st/2nd-order vs `msfm`'s multistencil — a small numerical
  difference, not a method/domain difference.

### C5. Shortest path — sub-voxel gradient descent ✅ aligned
- **MATLAB** ([`compute_shortestpath.m:15-29`](../Code/compute_shortestpath.m)): the
  Fast-Marching-toolbox `shortestpath(D, start, source)` traces a **continuous gradient
  descent** down the distance field, returning **sub-voxel** points.
- **Was:** Python did a greedy **integer-voxel** 6-neighbor descent → a blocky path.
- **Now** ([`distance.py:_gradient_descent_path`](../cagingloop/distance.py)): a streamline
  tracer follows `-grad(D)` (trilinear-interpolated, fixed-step) from start to the source
  minimum, producing a smooth sub-voxel path. Verified ~6× finer than the voxel size on
  `knotty`. Integer descent is kept only as a fallback when the tracer stalls (thin grids).

### C6. Local frame — PCA centering ✅ fixed (root cause of bad saddles)
- **MATLAB** ([`detectSaddlePoint.m:56`](../Code/detectSaddlePoint.m),
  [`generateCagingGrasp.m:36`](../Code/generateCagingGrasp.m)): `pca(X', 3)` **centers** the
  neighbour offsets by their mean before extracting the tangent/normal axes (it returns the
  mean as its 4th output).
- **Was:** `_local_frame` ran `np.linalg.svd` on the **raw** offsets-from-the-point, i.e.
  with no centering. The mean offset tilted the estimated tangent plane.
- **Symptom:** when neighbours were projected onto that tilted plane, the point's own
  origin frequently fell **outside** the convex hull of its projected neighbours, so
  `calculate_iter_num` rejected it (`flag = -1`). On `knotty` this wrongly rejected
  **432 / 1812** surface points (24%); the rest got distorted transition counts. Saddle
  detection — and therefore the whole caging loop — was computed on corrupted local frames.
- **Now** ([`saddle.py:32-46`](../cagingloop/saddle.py),
  [`grasp.py:25-37`](../cagingloop/grasp.py)): the offsets are mean-centered before SVD,
  matching MATLAB. Spurious `flag = -1` rejections dropped **432 → 53** (24% → 3%), and the
  origin now sits inside its neighbours' hull as it geometrically should.

---

## (E) Deliberate improvements beyond MATLAB (better caging loops)

These intentionally **deviate** from MATLAB in the example pipeline because MATLAB's
choices produced poor caging loops. The core library functions remain MATLAB-faithful
(and the deviations are opt-in via parameters / the example), so they're listed separately
from the alignment differences above.

### E1. Saddle selection — best loop instead of diversity-top
- **MATLAB** ([`detectSaddlePoint.m:28-29`](../Code/detectSaddlePoint.m)): keeps the
  `round(n/30)` saddles with the highest `diversityEval` score (for `knotty` that is a
  single saddle), and the caller feeds one saddle to `generateCagingGrasp`.
- **Problem:** `diversityEval` ranks by source/saddle normal alignment, which does **not**
  correlate with caging quality. On `knotty` the diversity-top saddle gives a near-degenerate
  loop (enclosed area 0.06) where both paths follow the same geodesic.
- **Improvement** ([`grasp.py:generate_best_caging_grasp`](../cagingloop/grasp.py)): the
  example widens `detect_saddle_point(..., keep=...)` to expose all candidates, generates the
  loop for each, and keeps the one enclosing the most area (a genuine wrapping loop). On
  `knotty` this lifted the loop area 0.06 → 0.26. `detect_saddle_point` still defaults to the
  MATLAB selection (`keep=None`).

### E2. Source seed — farthest-from-centroid instead of grid corner
- **Was:** the example hard-coded `source_point_id = 0`, i.e. the first `argwhere` voxel —
  an arbitrary min-index grid corner.
- **Improvement** ([`run_model_polyscope.py`](../examples/run_model_polyscope.py)): default
  `--source-point-id -1` now auto-seeds the surface point **farthest from the centroid**
  (a fingertip-like contact). Combined with E1 this lifted the `knotty` loop area to **0.51**
  (≈8.5× the original), and roughly doubled it on most models. Pass an explicit
  `--source-point-id` to override.

## (D) Non-algorithmic differences (no effect on results)

- **OBJ / normal loading** ([`model_io.py`](../cagingloop/model_io.py)): the MATLAB code
  used `readOBJ` from gptoolbox and assumed normals were supplied externally; the Python
  port has its own OBJ reader, vertex-normal computation, and normal transfer.
- **Visualization**: MATLAB opens figures inline (`pointCloud.plot`, `patch`, `scatter3`)
  inside each function. The Python port keeps computation headless and offers separate
  Matplotlib / Polyscope helpers
  ([`visualization.py`](../cagingloop/visualization.py),
  [`polyscope_visualization.py`](../cagingloop/polyscope_visualization.py)).
- **Indexing**: MATLAB is 1-based; Python is 0-based throughout.

---

## How to maximize alignment

1. **Fast marching is now built in:** `scikit-fmm` is a core dependency and used by
   default (C4); the smooth sub-voxel path tracer (C5) matches the toolbox `shortestpath`.
2. **Match the offset:** keep `normal_offset=1e-4` (MATLAB's hard-coded `0.0001`).
3. **Raise resolution:** higher `--voxel-count` reduces the blockiness from B1/B2 (the
   surface points are grid points, not shape points).
4. **Remaining hard limit:** the **RBF implicit field** (A) is the main thing that cannot
   be made identical without re-implementing `FastRBF.exe`, which the spec rules out. The
   `skfmm`-vs-`msfm` solver-order gap (C4) is the only other residual, and is minor.
