# MATLAB ↔ Python 差异说明

本文档逐函数比较原始 MATLAB 代码 [`Code/`](../Code/) 和 Python port
[`cagingloop/`](../cagingloop/)。

**结论先说：这个 port 不追求与 MATLAB bit-for-bit 完全相同，也从未以此为目标。**
设计文档
[`docs/superpowers/specs/2026-05-27-python-port-design.md`](superpowers/specs/2026-05-27-python-port-design.md)
明确把“bit-for-bit 或 point-for-point 等同 MATLAB 输出”列为 **non-goal**。
目标是工作流层面的算法等价，同时接受外部求解器替换带来的数值差异。

差异分为四类：

- **(A) 基础后端替换**：有意为之，因此完全数值一致不现实。
- **(B) 算法/逻辑差异**：同一阶段的实现方式不同。
- **(C) 本次已对齐的差异**：已经修正为与 MATLAB 一致的地方。
- **(D) 非算法差异**：I/O 和可视化，不影响结果。

---

## (A) 基础后端替换：无法与 MATLAB 数值完全一致

| 阶段 | MATLAB | Python | 后果 |
|---|---|---|---|
| 隐式场 | `FastRBF.exe`（`fit -accuracy=1.0 -rho=0.01`），通过外部 `.exe` 和临时文件求值 | `scipy.interpolate.RBFInterpolator(kernel="linear", neighbors=..., smoothing=...)` | 标量场不同，所以**表面体素集合**、inside/outside 划分都会不同。**这是目前最大的剩余数值差距。** |
| Fast marching | `msfm(W, source)`，Kroon multistencil Eikonal solver | `skfmm.travel_time(...)`，同样在规则网格 speed map 上解 Eikonal；现在是核心依赖，默认使用。Dijkstra 只作为退化/薄网格 fallback | **方法和传播域一致。** 剩余差异只是 solver 阶数，例如 `skfmm` 与 `msfm` multistencil 的数值差别，见 C4 |
| 最近邻 | TSTOOL MEX `nn_prepare` / `nn_search` | `scipy.spatial.cKDTree` | k-NN 结果等价，不是实质性差异 |
| 表面 mesh | `FastRBF.exe mcubes` 输出 `gridmesh.obj` | 可选 `skimage.measure.marching_cubes`（仅 `extract_mesh=True` 时） | 只影响显示，不进入 caging pipeline |

---

## (B) 算法/逻辑差异

### B1. 体素化：表面检测规则

- **MATLAB**（[`pointCloudVoxelizationByRBF.m:96-102`](../Code/pointCloudVoxelizationByRBF.m)）：
  对每个网格点，取 6 个最近网格邻居，计算 `ff = max(rbf_neighbors) * min(rbf_neighbors)`，
  若 `ff <= 1.0` 则标记为 surface。
- **Python**（[`voxelization.py:51-61`](../cagingloop/voxelization.py)）：
  如果 `|field| <= 0.05 * max|field|`，或任意轴向 6-neighbor 边跨过 0，则标记为 surface。
- **影响：** 规则不同，而且作用在不同的隐式场上，所以表面点数量和位置都会不同。

### B2. 体素化：surface index 的推导

- **MATLAB**（[`pointCloudVoxelizationByRBF.m:106-118`](../Code/pointCloudVoxelizationByRBF.m)）：
  从 `grid_on` 自身重新计算 bounding box，再推导新 spacing，把坐标 round 回整数索引：
  `round((p-min)/l)+1`。
- **Python**（[`voxelization.py:159`](../cagingloop/voxelization.py)）：
  直接用 `np.argwhere(surface)`，也就是网格索引本身。
- **影响：** Python 的映射更直接；MATLAB 的 re-binning 可能把点移动到相邻 cell。
  通常影响很小，偶尔会出现 off-by-one。

### B3. 体素化：固定 offset 与派生 offset

- **MATLAB：** 法线正负约束使用硬编码 `±0.0001`，并且**不归一化** `ptnormals`。
- **Python：** `normal_offset` 是参数，默认 `0.1 * min(spacing)`；会**归一化**法线并拒绝零长度法线。
- **说明：** 模型示例传入 `normal_offset=1e-4`，匹配 MATLAB 的 `0.0001`。
  归一化差异只在输入法线不是单位长度时有影响。

### B4. 体素化：退化轴处理

- **Python**（[`voxelization.py:38-43`](../cagingloop/voxelization.py)）会把 `min == max` 的轴扩展
  `±0.5`。**MATLAB 没有这个 guard**，会除以零。

### B5. 距离场：不可达点修补

- **MATLAB**（[`DistanceMapByFastMarching.m:29-36`](../Code/DistanceMapByFastMarching.m)）：
  如果某个 surface value `>= 10000`，就把它替换为 9 个最近表面邻居在原始 `D` 上的最小值
  （包括可能很大的值）。
- **Python**（[`distance.py:91-110`](../cagingloop/distance.py)）：
  “坏值”定义为 **non-finite**，即 `inf`；替换为有限邻居值中的最小值。
- **影响：** 坏值阈值和修补集合略有不同。

### B6. 鞍点检测：局部坐标系中的 PCA vs SVD

- **MATLAB**（[`detectSaddlePoint.m:54-57`](../Code/detectSaddlePoint.m)）：
  对 14 个最近邻执行 `pca(X', 3)`，主成分作为局部轴。
- **Python**（[`saddle.py:32-46`](../cagingloop/saddle.py)）：
  对**减去均值**的 offset 执行 `np.linalg.svd`，右奇异向量作为局部轴。
- **剩余差异：** 从数学上看，二者得到的是同一子空间，但轴的符号可能翻转，这是 PCA/SVD
  的固有符号不确定性。convex-hull 排序和 sign transition count 对镜像不敏感，所以不会改变结果。
  **是否中心化**曾经是真 bug，见 C6。

### B7. 鞍点检测：边界 polygon

- **MATLAB：** `boundary(adj_point, 0)` 返回闭合环，首点会重复；`filterBoundary` 会旋转环、
  检查 wrap-edge angle，并删除一些方向。
- **Python**（[`saddle.py:64-71`](../cagingloop/saddle.py)）：
  用 `scipy.spatial.ConvexHull`，返回的 hull 不是闭合的；wrap edge 通过
  [`_count_transitions`](../cagingloop/saddle.py#L123) 中的 `np.roll` 恢复。
  如果 Qhull 失败，会退回角度排序；MATLAB 没有这个 fallback。
- **验证结果：** 在 `knotty` 的全部 1812 个表面点上，忠实闭合版 `filterBoundary` 与当前 open-hull
  版本的 `flag>=4` saddle 分类完全一致，0 个点被重新分类。因此 open/closed 差异在功能上等价。

### B8. 鞍点检测：额外 guard

- **Python** 的 `_filter_boundary`（[`saddle.py:118-119`](../cagingloop/saddle.py)）在过滤后若少于
  3 个点，会保留原始边界。**MATLAB 没有这个 guard。**

### B9. 鞍点选择：保留数量下限

- **MATLAB**（[`detectSaddlePoint.m:29`](../Code/detectSaddlePoint.m)）：
  `saddle(opt_I(1:round(n/30)))`，当 `round(n/30) == 0` 时可能为空。
- **Python**（[`saddle.py:184`](../cagingloop/saddle.py)）：
  `max(1, round(n/30))`，只要有候选，就至少保留一个。

### B10. Cage points：fallback

- **MATLAB** 的 `getCagePoint`（[`generateCagingGrasp.m:31-74`](../Code/generateCagingGrasp.m)）
  假设至少存在两个负方向邻居。
- **Python** 的 `get_cage_points`（[`grasp.py:109-118`](../cagingloop/grasp.py)）
  在假设不成立时加入确定性 fallback：选择两个最低距离邻居。

> 注意：**caging path 的拼接和 Laplacian 平滑**
> （[`grasp.py:122-150`](../cagingloop/grasp.py)）是
> [`generateCagingGrasp.m:14-28`](../Code/generateCagingGrasp.m) 的精确 port：
> 权重同为 `0.2/0.4/0.4`，start/end 闭合逻辑也相同。

---

## (C) 本次已对齐的差异

下面这些曾经是真实 mismatch，现在已经修成与 MATLAB 一致：

### C1. `diversity_eval`：MATLAB `pow2` 语义 ✅ 已修复

- **MATLAB**（[`detectSaddlePoint.m:48`](../Code/detectSaddlePoint.m)）：
  `exp(-0.5 * pow2(max(angle1,angle2), 4))`。MATLAB 中 `pow2(F,E) = F * 2^E`，
  因此 `pow2(x,4) = 16x`，即 `exp(-8 * max_angle)`。
- **之前：** 错读成四次方，写成 `exp(-0.5 * max_angle ** 4)`。
- **现在**（[`saddle.py:23-31`](../cagingloop/saddle.py)）：
  `exp(-0.5 * max_angle * 2**4)`。这个分数用于 saddle 排名，所以之前的 bug 会影响每个模型的选择。

### C2. z-min boundary face ✅ 已修复

- **MATLAB**（[`pointCloudVoxelizationByRBF.m:181-196`](../Code/pointCloudVoxelizationByRBF.m)）：
  只把 **x±、y±、z-max** 边界上的 inner voxels 移到 outer；`z-min` 面的 `tf_z2` 被注释掉，
  因此底面仍保留为 inner。
- **之前：** Python 把 6 个面全部移动。
- **现在**（[`voxelization.py:64-73`](../cagingloop/voxelization.py)）：排除了 z-min 面。

### C3. 传给 saddle detection 的表面法线 ✅ 已在模型示例中修复

- **MATLAB：** `detectSaddlePoint` 接收真实 surface normals，即 `grid_on_normals`。
- **之前：** 模型示例使用了**径向**法线 `grid_on - centroid`，对凹形物体是错的。
- **现在**（[`model_io.py:transfer_point_normals`](../cagingloop/model_io.py) +
  [`run_model_polyscope.py`](../examples/run_model_polyscope.py)）：每个表面体素取最近输入点云点的法线。
  合成 sphere 示例仍使用径向法线，因为对球是正确的。

### C4. 距离场：从 Dijkstra 改为 Eikonal solver ✅ 已对齐

- **MATLAB**（[`DistanceMapByFastMarching.m:24`](../Code/DistanceMapByFastMarching.m)）：
  `msfm(W, source)`，即在 voxel grid 上用 speed map 解 Eikonal 方程；
  surface=1，exterior=1，interior=0。
- **之前：** 缺少 `scikit-fmm` 时，代码 fallback 到 **Dijkstra** 图距离，也就是轴向步长求和；
  这不是 Eikonal geodesic。
- **现在：** `scikit-fmm` 是**核心依赖**，示例默认 `prefer_fmm=True`，因此使用
  `skfmm.travel_time` 在相同传播域上求解 Eikonal
  （[`distance.py:76-88`](../cagingloop/distance.py)）。Dijkstra 只保留为退化/薄网格 fallback，
  以及合成单元测试使用。
- **剩余差异：** `skfmm` 是 1st/2nd-order，`msfm` 是 multistencil；这是较小的数值差异，
  不是方法或传播域差异。

### C5. 最短路径：亚体素梯度下降 ✅ 已对齐

- **MATLAB**（[`compute_shortestpath.m:15-29`](../Code/compute_shortestpath.m)）：
  Fast-Marching toolbox 的 `shortestpath(D, start, source)` 会沿距离场做**连续梯度下降**，
  返回**亚体素**点。
- **之前：** Python 做贪心的整数体素 6-neighbor descent，路径呈 blocky 阶梯。
- **现在**（[`distance.py:_gradient_descent_path`](../cagingloop/distance.py)）：
  streamline tracer 沿 `-grad(D)` 走，使用三线性插值和固定步长，从 start 到 source minimum，
  输出平滑的亚体素路径。已在 `knotty` 上验证，点间距约比体素尺寸细 6 倍。
  整数 descent 只作为 tracer 卡住时的 fallback。

### C6. 局部坐标系：PCA centering ✅ 已修复

- **MATLAB**（[`detectSaddlePoint.m:56`](../Code/detectSaddlePoint.m)，
  [`generateCagingGrasp.m:36`](../Code/generateCagingGrasp.m)）：
  `pca(X', 3)` 会在提取 tangent/normal 轴之前，把邻居 offset **减去均值**。
- **之前：** `_local_frame` 直接对“点到邻居的 raw offsets”做 SVD，没有 centering；
  均值 offset 会把估计的切平面拉歪。
- **症状：** 邻居投影到这个倾斜平面后，点自己的原点经常落在投影邻居的 convex hull 外，
  `calculate_iter_num` 就会拒绝它（`flag = -1`）。在 `knotty` 上，这错误拒绝了
  **432 / 1812** 个表面点（24%），其余点的 transition count 也被扭曲。
- **现在**（[`saddle.py:32-46`](../cagingloop/saddle.py)，
  [`grasp.py:25-37`](../cagingloop/grasp.py)）：SVD 前会先 mean-center offsets，匹配 MATLAB。
  错误的 `flag = -1` 从 **432 降到 53**（24% 到 3%），原点也如几何上应有的那样落在邻域 hull 内。

---

## (E) 有意超出 MATLAB 的改进

下面这些在示例 pipeline 中**有意偏离** MATLAB，因为 MATLAB 的选择在某些模型上会产生很差的 caging loop。
核心库函数仍尽量保持 MATLAB-faithful；这些偏离通过参数或示例入口启用，因此单独列出。

### E1. Saddle selection：选择最佳 loop，而不是 diversity-top

- **MATLAB**（[`detectSaddlePoint.m:28-29`](../Code/detectSaddlePoint.m)）：
  保留 `diversityEval` 分数最高的 `round(n/30)` 个 saddles。对 `knotty` 来说通常就是一个 saddle，
  调用方把这个 saddle 传给 `generateCagingGrasp`。
- **问题：** `diversityEval` 看的是 source/saddle 法线对齐程度，它和 caging quality 并不强相关。
  在 `knotty` 上，diversity-top saddle 会给出近似退化的 loop，两条路径几乎走同一条 geodesic，
  enclosed area 只有 0.06。
- **改进**（[`grasp.py:generate_best_caging_grasp`](../cagingloop/grasp.py)）：
  示例会放宽 `detect_saddle_point(..., keep=...)`，暴露所有候选；为每个候选生成 loop，
  并选 enclosed area 最大的那个。`knotty` 上 loop area 从 0.06 提升到 0.26。
  `detect_saddle_point` 默认仍保持 MATLAB 选择方式（`keep=None`）。

### E2. Source seed：离质心最远点，而不是网格角落

- **之前：** 示例硬编码 `source_point_id = 0`，也就是第一个 `argwhere` 体素，通常只是任意 min-index 角落。
- **改进**（[`run_model_polyscope.py`](../examples/run_model_polyscope.py)）：
  默认 `--source-point-id -1` 自动选择离质心最远的表面点，更像 fingertip contact。
  配合 E1，`knotty` 的 loop area 提升到 **0.51**，约为原始选择的 8.5 倍；在大多数模型上也大约翻倍。
  传显式 `--source-point-id` 可覆盖默认行为。

## (D) 非算法差异

- **OBJ / normal loading**（[`model_io.py`](../cagingloop/model_io.py)）：MATLAB 使用 gptoolbox 的
  `readOBJ`，并假设外部提供 normals；Python port 自带 OBJ reader、顶点法线计算和 normal transfer。
- **可视化**：MATLAB 在每个函数内部直接打开 figure（`pointCloud.plot`、`patch`、`scatter3`）。
  Python port 把计算保持为 headless，并提供独立的 Matplotlib / Polyscope helper：
  [`visualization.py`](../cagingloop/visualization.py)，
  [`polyscope_visualization.py`](../cagingloop/polyscope_visualization.py)。
- **索引**：MATLAB 全部是 1-based；Python 全部是 0-based。

---

## 如何最大化与 MATLAB 的一致性

1. **Fast marching 已内置：** `scikit-fmm` 是核心依赖并默认使用（C4）；
   平滑的亚体素 path tracer（C5）对应 toolbox 的 `shortestpath`。
2. **匹配 offset：** 保持 `normal_offset=1e-4`，也就是 MATLAB 硬编码的 `0.0001`。
3. **提高分辨率：** 更高的 `--voxel-count` 会减少 B1/B2 带来的体素粗糙感；surface points
   是网格点，不是原始形状点。
4. **剩余硬限制：** **RBF 隐式场**（A）是最大不可消除差异；除非重新实现 `FastRBF.exe`，
   否则无法做到完全一致。`skfmm` 与 `msfm` 的 solver-order 差异（C4）是另一个剩余项，但影响较小。
