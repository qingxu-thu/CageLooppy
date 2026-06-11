# CagingLoop Python Port 对话纪要

本文整理了围绕 MATLAB 到 Python 移植、论文算法对齐、Polyscope 可视化、过滤器设计和后续研究方向的主要讨论。它不是逐字聊天记录，而是按主题归纳的工作记录。

## 当前目标

- 将发布版 MATLAB CagingLoop 算法移植到 Python。
- 尽量和 MATLAB 行为对齐，同时补齐论文 Algorithm 2 中发布代码缺失的部分。
- 用 Polyscope 可视化模型、体素、距离场、Morse saddles、base points 和 caging loops。
- 解释并过滤大量候选 loops，最终得到更接近论文意义上的 caging candidates。

## 主要入口命令

模型 viewer：

```powershell
python examples/run_model_polyscope.py Models/knotty.obj --sweep 24 --morse --solver msfm
```

internals viewer：

```powershell
python examples/run_internals_polyscope.py Models/knotty.obj --backend mesh --voxel-count 37 --grasp-hull
```

带 slider 浏览候选 loops：

```powershell
python examples/run_model_polyscope.py Models/knotty.obj --sweep 16 --max-loops 100 --slider
```

完整内部可视化，包括 mesh、过滤和 mechanical selection：

```powershell
python examples/run_internals_polyscope.py .\Models\fertilityNormal.obj --voxel-count 60 --backend rbf --max-loops 400 --slider --solver msfm --method surface --path-integrator rk4 --base-height 0.54 --sweep 20 --show-mesh --gripper-radius 0.02 --base-sampling uniform --grasp-hull --curvature-filter --select mechanical --ls-filter --min-rho 0.01 --link-filter
```

## Voxelization

### RBF backend

RBF pipeline 复现 MATLAB 的 `pointCloudVoxelizationByRBF.m` recipe：

- surface constraints: `0`
- outward normal offset: `+1`
- inward normal offset: `-1`
- inner / outer 由 RBF field 符号决定
- boundary fix 复现 MATLAB 行为，`z-min` 保持 inner

但底层 solver 不同：

- MATLAB 使用 FastRBF.exe，全局 biharmonic / polyharmonic RBF。
- Python 使用 SciPy `RBFInterpolator(kernel="linear", neighbors=64)`，是局部近似。

注意：SciPy 的 `kernel="linear"` 在 3D 中就是 `phi(r)=r`，即 biharmonic polyharmonic spline。名字里的 linear 指 kernel 对半径 `r` 线性，不是普通线性插值。

### Mesh backend

mesh backend 使用 winding-number voxelization，拓扑更正确，适合 handles / holes / thin features。重要参数：

```powershell
--backend mesh
--voxel-count 49
--padding 0.08
```

`--padding` 会给 bbox 加空白边距：

- 太小：物体可能贴边或被裁切。
- 太大：有效物体分辨率变低。
- 对 knotty handle 这类细孔，`0.05-0.15` 通常更合适。

## Distance Field

`D_p` 是 grasp space 中从 base point `p` 出发的 geodesic shortest-path distance，不是欧氏直线距离，也不是表面距离。

MATLAB：

```matlab
W(inner)=0
W(outer/on)=1
[D,Y] = msfm(W, sourcePointID')
```

Python 对应：

```python
distance_map_by_fast_marching(...)
```

支持三个 solver：

- `--solver fmm`: scikit-fmm，single-stencil Eikonal fast marching。
- `--solver msfm`: Python port 的 multistencil FMM，更接近 MATLAB Kroon `msfm3d.c`。
- `--solver dijkstra`: 6-neighbor grid graph fallback，Manhattan-ish。

MSFM 验证结果：空盒子中心源点下，axis / face diagonal / body diagonal 方向误差约 2%，而 Dijkstra 对角线误差符合 `sqrt(2)` / `sqrt(3)` 预期。

## Shortest Path Tracing

MATLAB `compute_shortestpath.m` 使用：

- `pointmin.m`: 26-neighbor discrete steepest descent field
- `rk4.c`: RK4 integrator

Python 已加入：

```powershell
--path-integrator {euler,rk4}
```

`rk4` + `pointmin` 更贴近 MATLAB。该路径主要影响 surface method，也就是默认 `detectSaddlePoint` 路径。

## Saddle Detection

### Surface method

默认方法，不加 `--morse` 时使用。

对应 MATLAB `detectSaddlePoint`：

- 在 surface point cloud 上找 saddle。
- 对每个点取邻居，做 PCA tangent plane。
- 在 tangent plane 上绕点排序邻居。
- 统计 `D(neighbor)-D(point)` 的 sign transitions。
- `iterNum >= 4` 判为 saddle candidate。

### Volumetric Morse method

加 `--morse` 时使用，来自论文 Fig. 3 / Thm 3.1：

- 在 3D grasp space 中找 Morse saddles。
- 通过 lower-link / `+/-` 邻居配置判断 wavefront collision。
- 从 saddle 的两个下降方向 trace 回 base point，拼成 loop。

它比 surface method 更接近论文完整方法，能在 torus 这类简单孔洞上稳定产生穿孔 loop。

## Paper Algorithm 2 对应

Algorithm 2 的核心：

```text
初始化 L~
计算 S 和 convex hull H 之间的 grasping space G
对每个正主曲率 surface sample p:
    计算 Dp，传播半径限制为 2h
    对 Dp 的每个 Morse saddle q:
        trace p-based caging loop
        加入 L~
```

Python 对应 flags：

| 论文规则 | Python 实现 | flag |
| --- | --- | --- |
| S 到 H 之间的 grasping space | `convex_hull_grasping_mask` | `--grasp-hull` |
| positive principal curvature samples | `positive_curvature_points` | `--curvature-filter` |
| 2h sweep radius | `max_distance` | `--sweep-radius` |
| 500 surface samples | `--sweep N` | `--sweep 500` |
| volumetric Morse saddles | `detect_morse_saddles_3d` | `--morse` |

完整论文式配置示例：

```powershell
python examples/run_model_polyscope.py Models/knotty.obj --backend mesh --voxel-count 50 --gripper-radius 0.05 --grasp-hull --sweep 500 --curvature-filter --sweep-radius 0.5 --morse
```

注意：Python 实现功能上对应，但性能远慢于论文中的 C / MEX 实现。

## r-offset / Gripper Radius

`--gripper-radius R` 对应论文 §IV 的 r-offset surface `S_r`。

实现上是对 solid 做 voxel dilation：

```text
radius_voxels = round(R / spacing)
```

作用：

- 融合比 gripper 半径更窄的缝隙。
- 让 trophy columns 等细分裂结构变成一个可抓取 bundle。
- surface voxels 增加，grasp space 缩小。

可视化建议：

```powershell
python examples/run_internals_polyscope.py Models/torus.obj --sweep 1 --gripper-radius 0.2 --show-mesh
```

真实 OBJ mesh 仍显示原始几何，surface layer 显示膨胀后的 `S_r`。

## Loop Selection

`--select` 控制候选排序：

- `area`: 最大 enclosed area，常用于 body loop。
- `small`: 最小有效 loop，常用于 handle / neck。
- `mechanical`: 论文 §II.A mechanical considerations，即 loop center 接近 CoG 且大致水平。
- `waist`: 旧名，保留为 `mechanical` 的 alias。

论文没有使用 “waist” 这个词。`mechanical` 是更忠实的名字。

## Candidate Filters

候选池会很大，尤其是 `--sweep N` 时：

```text
N base points × rings_per_point
```

很多 loop 是重复、退化或只属于 `L~` 而不属于真实 `L` 的伪候选。

### `--ls-filter`

对应论文 Property 2.7 后的 `L~ -> L` reduction：去掉在 base point `p` 处不是 locally shortest 的 loops。

当前实现使用 windowed shortenability test：

```text
straightness = chord_length / subpath_length
```

如果 chord 明显更短且不穿 solid，则 loop 在 `p` 附近可缩短，丢弃。

相关参数：

```powershell
--ls-window 3
--ls-straightness 0.95
```

### `--min-rho`

isoperimetric ratio：

```text
rho = area / perimeter^2
```

- collapsed / line-like loop: `rho -> 0`
- perfect circle: `rho = 1/(4*pi) ~= 0.0796`

用于丢掉细长、线状退化 loop。常用：

```powershell
--min-rho 0.01
```

### `--link-filter`

用于丢掉没有 encircle material 的 surface stubs。

旧测试只问 loop disk 是否碰到 solid，容易误保留 bulky object 表面的 stubs。新测试问：

```text
material 是否穿过 loop plane？
```

做法：

- PCA 拟合 loop plane。
- 在 loop 投影 polygon 内采样。
- 对每个 sample 沿 plane normal 的 `+n` / `-n` 两侧探测 solid。
- 两侧都有 solid 才算 material pierces plane，保留。
- 只有一侧 solid 的 surface stub 丢弃。

默认：

```powershell
--link-filter
```

会自动使用：

```powershell
--link-min-frac 0.06
```

通常不用手动调。若 stubs 仍存活，提高到 `0.15`；若真实 thin handle wrap 被误删，降低到 `0.02`。

### 不推荐：`--min-loop-frac`

长度下限对小 handle loop 不公平，会误杀论文想保留的小尺度抓取。因此已从推荐 filter stack 中移除。

## Funnel 输出

internals viewer 会打印过滤漏斗：

```text
loops: raw=N -> min-rho(A) -> link(B) -> ls-filter(C) -> mechanical top M
```

含义：

- `raw=N`: 原始 `L~` pool。
- `min-rho(A)`: 去掉 line-like loops 后。
- `link(B)`: 去掉 encircles-nothing stubs 后。
- `ls-filter(C)`: 去掉 base-point spoke artifacts 后。
- `mechanical top M`: 排序并截断后显示。

## Polyscope 可视化

### Internals viewer layers

- `surface`: surface voxels
- `grasp space`: `S -> H` band，带 `D_p` scalar
- `base point p`: 绿色 seed point
- `morse saddles`: saddle markers，带 `saddle_order`
- `loop 00...NN`: traced loops
- `object mesh`: `--show-mesh` 时显示真实 OBJ mesh
- `distance slice`: `--slice-height F` 时显示距离场高度切片
- `slice plane`: `--slice-plane` 时显示可移动切平面

### 常用显示参数

```powershell
--show-mesh
--slider
--show-all
--slice-height 0.5
--slice-plane
```

`--mesh-shift DX DY DZ` 只是显示平移，不影响计算。

## Torus Benchmark

新增 `Models/torus.obj`：

- genus 1
- `R=1.0`, `r=0.4`
- y 轴为孔轴
- 用于验证 threaded loops

torus 是 sanity benchmark：正确算法应能找到穿过中心孔的 loop。MSFM、Morse saddles、link-filter 都用 torus 做过验证。

## Knotty / Mug Handle 结论

`knotty.obj` 实际是杯身 + knot / pretzel handle。

讨论结论：

- body / waist loops 能稳定生成。
- simple hole threading 在 torus 上验证可行。
- complex knotted handle threading 受分辨率、base point、selection 和几何复杂度限制，很难稳定由 saddle 方法直接产生。
- 高分辨率 mesh backend + 小 padding 可以打开 handle holes：

```powershell
python examples/run_internals_polyscope.py Models/knotty.obj --backend mesh --voxel-count 49 --padding 0.08 --grasp-hull
```

但高分辨率下候选数量巨大，必须依赖 filters 和 slider。

## Slice Methods

曾加入 direct slice loop，用于在指定高度生成干净水平 ring：

```powershell
--slice-height 0.55
--slice-region handle|body|all
```

它是直接几何构造，不是 saddle method。适合“我明确要这个高度的水平环”的场景，但不等价于论文自动发现 loop 的方式。

## Physics-aware / Task-weighted Caging 讨论

一个潜在新方法：

```text
Task-weighted caging loops:
replace uniform geodesic Dp with a physics-weighted metric.
```

当前 `D_p` 是：

```text
min ∫ c(s) ds
```

其中现在 `c=1`。可以改成：

```text
c(x) = c0 * (1 + lambda * f(x))
```

让 loops 被物理 potential 引导。

### 适合进 field 的局部物理

- hot / fragile / no-touch zones: 高 cost 或 impassable。
- friction preference: 高摩擦区域低 cost。
- surface affinity / reachability: 接触更容易的区域低 cost。

### 适合留在 selection 的全局物理

- true center of gravity
- inertia tensor
- torque / wrench feasibility

### mass potential 例子

lever-arm potential：

```text
f(x) = horizontal distance from x to vertical axis through true CoG
```

CoG-height well：

```text
f(x) = (height(x) - height(CoG))^2
```

这些会让 candidate loops 在生成阶段就偏向质量分布更合理的位置，而不仅仅是后处理排序。

重要 caveat：

- cost distance 和物理长度要分开追踪。
- cost-taut 不等于 geometrically taut。
- `2h` reach 应继续按真实几何长度判断。
- weighted metric 下 Thm 3.2 / reach bound 需要重新审视。

## 深度相机场景接入

论文真实部署场景是：

```text
two depth cameras
-> fused partial point cloud
-> RBF S_r
-> grasp space
-> distance field
-> saddles
-> loops
```

Python 后端已经基本对应这个流程：`point_cloud_voxelization_by_rbf(points, normals, ...)` 接收点云和法向，后续 field、saddles、loops、filters 都可以复用。新增部分主要是深度图前端和场景设置。

### Depth map -> point cloud + normals

对每个 depth pixel 用相机内参反投影：

```text
X = (u - cx) * z / fx
Y = (v - cy) * z / fy
Z = z
```

得到 camera-space point cloud 后，再用外参变换到 world frame。

depth map 的 normals 可以从相邻反投影点叉乘得到，也可以局部平面拟合。因为 normals 天然朝向相机，在可见表面上通常就是 outward orientation，正好适合 RBF 的 `+offset / -offset` constraints。

必须先分割物体：

- 用 RANSAC plane fit 或 height threshold 去掉桌面。
- 去掉背景。
- 对剩余点做 clustering，只保留目标物体。

否则 grasp field 会把桌子也当成场景一部分，saddles 会绕桌子生成。

### 关键陷阱：coverage

单张 depth map 是 2.5D，只看到前侧，背面缺失。caging loop 需要绕物体，wavefront 必须能在远侧相遇。如果远侧几何不存在，就不会形成正确 saddles / loops。

论文使用两台相机就是为了这个：

1. **Multi-view fusion**

   最忠实方案。用两台或多台 depth cameras，从前后视角获取点云，通过已知外参、ICP 或 KinectFusion-style depth fusion 融合成一份点云。这样才有足够 wrap-around coverage。

2. **RBF / hull closure**

   RBF implicit surface 是全局的，会自动闭合中等缺口，生成近似 watertight solid。这也是论文选择 RBF 的原因：它对不完整几何有容忍度。轻微遮挡时，RBF closure 可以让 field 在隐式补全的远侧形成 saddles。

3. **Symmetry / shape completion**

   如果只有单视角，可以镜像可见半边，或用 learned completion 补背面。但这是更强的先验。

核心选择是：投入多少 coverage。两侧视角是稳健 baseline；单视角只有在 RBF closure 能合理补全背面时才可用。

### 从点云得到 proper field

一旦有了 coverage 足够的 `(points, normals)`，后端流程不变：

```text
RBF r-offset implicit
-> voxelization
-> S_r 到 H_r 之间的 grasp space
-> base point sampling
-> D_p fast marching
-> Morse saddles
-> loop tracing
-> filters / mechanical selection
```

关键是 RBF closure：没有闭合 solid，就没有良定义的“绕物体的 geodesic field”。论文 Fig. 8 中的噪声鲁棒性主要来自 RBF smoothing 和 embedding-space distance field。后续还可以接入前面讨论过的 diffusion-distance variant 来增强 noisy RGB-D 场景的稳定性。

### 场景坐标系要素

- **Gravity / up axis**：可从桌面法向或相机外参获得。`mechanical` selection、CoG、horizontality、mass / torque 相关项都依赖它。
- **Object bbox + resolution**：对分割出的物体 bbox 建 voxel grid。论文使用 `50^3`。spacing 决定细节分辨率和 link-filter 采样可靠性。
- **Gripper size `2h`**：使用真实世界单位，设置 `--sweep-radius`，并决定小孔穿过还是大尺度包围。
- **Units / scale**：depth 是 metric，所以 `2h`、`r-offset`、CoG 都可以用米，不需要归一化猜测。

### 对代码的映射

真正新增的是前端模块：

```text
depth maps + intrinsics/extrinsics
-> segmented object points
-> oriented normals
```

之后直接调用：

```python
point_cloud_voxelization_by_rbf(points, normals, ...)
```

其余 pipeline 不需要改。实际 scene setup 是：

```text
fuse >= 2 views
-> segment object
-> fix gravity frame
-> set r and 2h in meters
-> run existing CagingLoop pipeline
```

## 当前推荐 filter stack

对需要大量候选可视化的 internals viewer：

```powershell
--select mechanical --ls-filter --min-rho 0.01 --link-filter
```

如需浏览：

```powershell
--slider --max-loops 400
```

如果 stubs 还存在：

```powershell
--link-min-frac 0.15
```

如果真实 thin handle loop 被误删：

```powershell
--link-min-frac 0.02
```

如果 locally-shortest filter 过严或过松：

```powershell
--ls-straightness
--ls-window
```

## 已知限制

- RBF backend 与 MATLAB FastRBF recipe 相同，但数值不完全一致。
- mesh backend 更适合拓扑，但高分辨率慢。
- `--link-filter` 当前基于 planar PCA disk；强非平面 loops 可能需要 triangulated fan spanning surface。
- complex knot handle threading 仍是困难 case，需要高分辨率、合适 base points 和 robust selection。
- physics-aware metric 是研究方向，尚不是当前稳定功能。
