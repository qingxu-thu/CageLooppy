# CagingLoop - 仓库概览与调试指南

这是一份实用地图，帮助你理解这个仓库的结构、算法流程，以及结果不对时该从哪里查。

## 这个项目是什么

这是论文 **"Caging Loops in Shape Embedding Space"**（Liu 等，ICRA 2018，
`docs/1807.11661`）的实现。给定一个 3D 物体，它会计算一条 **caging loop**：
一条位于物体周围自由空间中的闭合曲线，机械夹爪可以沿这条曲线闭合，从而 cage/grasp
物体。这个环的计算过程是：

1. 把物体周围空间离散成体素，并标记 inside / surface / outside；
2. 从一个源点出发，在**物体外部空间**中计算测地距离场，路径不能穿过实体内部；
3. 在距离场上寻找 **Morse 鞍点**，也就是波前绕过物体后相遇的位置；
4. 从鞍点追踪两条回到源点的最短路径，并把它们拼成闭环。

## 算法详解：理论到代码

### 精确目标

我们想要的是一条 **caging loop**：闭合曲线 `L` 位于物体周围的自由空间中，使物体不能从
它里面逃出。论文 §II.A 中接受两类环：

- **稳定环**：`L` 紧贴物体的狭窄部位，小扰动下不能滑脱或缩短，是真正的 cage。
- **不稳定环**：`L` 只是像“球的大圆”一样绕过物体。

`L` 必须满足三点：不能穿透实体；必须是局部最短的，也就是像拉紧的橡皮筋；尺寸要大致匹配夹爪。
论文的关键技巧是：不要在所有曲线空间里搜索，而是**把问题化成寻找距离场的鞍点**。

### 核心思想：距离场 + Morse 理论

选一个自由空间中的基点 `p`，实际代码里通常在表面体素上选。定义距离场：

```text
D_p(x) = 从 p 到 x 的最短路径长度，路径只能经过自由空间，不能穿过实体。
```

这是 shape embedding space 里的测地距离，不是沿表面的距离。正是这一点让环不依赖表面细节，
而是感知整个形状。

可以想象一个波前从 `p` 以单位速度膨胀；`D_p(x)` 就是到达时间。遇到障碍物、把手、细颈或整个身体时，
波前会分裂，从两侧绕过去，并在另一侧**自我碰撞**。碰撞轨迹形成 ridge，其中的低点就是
`D_p` 的 **Morse 鞍点**。

### 为什么鞍点能给出 caging loop

**论文定理 3.1：** `D_p` 的每个 Morse 鞍点或极大点 `q` 都定义了一条基于 `p` 的 caging loop。

直觉是：在鞍点 `q`，两个相遇的波前从**相反方向**抵达。因此存在两条从 `p` 到 `q` 的最短路径
`Π₁, Π₂`，它们离开 `q` 的方向相反。把它们粘起来：走 `Π₁`，再反向走 `Π₂`。结果就是一条经过
`p` 和 `q` 的闭合环，并且**包住了迫使波前分裂的障碍物**。由于两半都是最短路径，这条环除了
基点 `p` 附近以外都是局部最短的，也就是定义 2.6 里的 p-based caging loop。

**离散 Morse 测试（论文 Fig. 3）：** 用中心体素与邻居的 `D` 值差来标记邻居是高 `+` 还是低 `-`。
如果绕着邻域一圈，高低符号**交替**，具体说符号变化次数 `>= 4`，就是鞍点。例如：
`+ + - - + + - -`。普通点通常变化 2 次；极小/极大点变化 0 次。所以：

> **caging-loop seed 等价于：点周围 `D_p` 的符号变化次数 >= 4。**

这条事实就是整个方法的发动机。剩下的实现都是为了让它更稳健、更高效。

### Stage 1 - 体素化：构造抓取空间（[voxelization.py](../cagingloop/voxelization.py)）

我们需要一个离散自由空间来传播波前。规则网格中的每个 cell 会被分类为 **inside / surface / outside**。

- **RBF 路径**（`point_cloud_voxelization_by_rbf`，对应 MATLAB）：拟合隐式函数
  `f(x)=Σ wᵢ‖x-xᵢ‖ + P(x)`，约束为表面上 `f=0`，沿法线微小偏移处 `f=±1`。
  然后 `f<0` 表示 inside，`f>0` 表示 outside，`f≈0` 表示 surface。Python 里用
  `scipy.interpolate.RBFInterpolator`，不再依赖 `FastRBF.exe`。
- **Mesh 路径**（`voxelize_mesh`，**新增**）：对于 watertight mesh，用**广义 winding number**
  分类每个 cell；inside 约为 1，outside 约为 0。这个路径保持拓扑，并带 bounding-box padding，
  避免表面被裁掉。

输出是 `VoxelizationResult`：`output_grid` 中 `1`=surface、`0`=inside、`-1`=outside，
并带上匹配的索引数组和坐标数组。**表面体素 `grid_on` 是基点和鞍点所在的位置；inside 体素是
不可穿越的障碍物；outside 体素是环可以通过的自由空间。**

### Stage 2 - 距离场：测地波前（[distance.py](../cagingloop/distance.py)）

`distance_map_by_fast_marching` 在网格上求解 **Eikonal 方程** `‖∇D‖ = 1/speed`：

- surface + outside 上 **speed = 1**，波前可以自由传播；
- inside 上 **speed = 0**，波前不能进入实体内部，这迫使路径绕过物体。

求解器是 `scikit-fmm` 的 `skfmm.travel_time`，对应 MATLAB 的 `msfm`；如果 `skfmm` 不可用，
还有 Dijkstra 图搜索 fallback。`dismap` 是把 `D` 采样到表面体素上的结果。

`compute_shortest_path` 会沿距离场向下追踪路径：从某个点开始，反复沿 `-∇D` 方向走
（亚体素、三线性插值的梯度），直到到达源点极小值。这就是“沿最短路径回家”的离散版本，
也是实现 `Π₁, Π₂` 的方式。

### Stage 3 - 鞍点检测：找到 loop seed（[saddle.py](../cagingloop/saddle.py)）

对每个表面点应用 `>=4` 符号变化测试，并做一些稳健化处理：

1. **局部坐标系**（`_local_frame`）：对该点周围约 14 个最近表面邻居做 PCA。邻居 offset 会
   **减去均值**，匹配 MATLAB 的 `pca` 行为，得到切平面 `(u,v)` 和法线 `n`。
2. **投影和排序**：把约 8 个最近邻投影到 `(u,v)` 平面，用 2D convex hull 排成一圈方向。
3. **符号序列**：沿这一圈计算 `judge = D(neighbour) - D(centre)`；符号表示该方向上距离场是上升
   `+` 还是下降 `-`。
4. **要求中心被包住**：原点必须在 hull 内；否则这是边界伪影，拒绝，返回 `flag = -1`。
5. **过滤平坦方向**（`_filter_boundary`）：删除相对于该点最陡方向来说梯度太小的方向
   （`FILTER_TOLERANCE_FACTOR`，替代 MATLAB 中尺度不稳的硬编码 `50`），以及近似重复的方向。
6. **闭环计数符号变化**。如果 `>= 4`，该点就是 **saddle candidate**。

之后 `diversity_eval` 会根据 source→saddle 方向和两个表面法线的对齐程度打分：
`exp(-8·max_angle)`。注意 MATLAB 的 `pow2(x,4)` 表示 `16x`，不是 `x⁴`。排序后保留分数最高的一批。
这就是 `detect_saddle_point`。

### Stage 4 - 生成环：追踪并闭合（[grasp.py](../cagingloop/grasp.py)）

给定鞍点 `q` 后，`generate_caging_grasp` 做：

1. `get_cage_points` 在 `q` 附近找到**两个相反的下降方向**，也就是波前从两侧到达的方向：
   最陡下降邻居，以及切平面里与它最相反的另一个下降邻居。这两个点就是 cage extension points。
2. 从两个 cage point 分别调用 `compute_shortest_path` 回到 source，得到 `Π₁` 和 `Π₂`。
3. 拼接成 `saddle → Π₁ → source → reverse(Π₂) → saddle` 的闭合环。
4. `smooth_closed_path` 做一轮 Laplacian 平滑：`0.2·self + 0.4·prev + 0.4·next`，
   去掉体素阶梯感，同时保持闭环。

`generate_best_caging_grasp`（**新增**）会对所有候选鞍点生成环，并保留包围面积最大的环，
避免退化成几乎“去而复返”的细小环。

### 端到端流程

```text
object mesh / point cloud
   │  Stage 1: voxelize → inside / surface / outside  抓取空间
   ▼
choose a base point p on the surface
   │  Stage 2: 从 p 出发传播波前，实体阻挡 → 距离场 D_p
   ▼
   │  Stage 3: 扫描表面，找 D_p 符号变化 >=4 的 Morse saddles
   ▼
for the best saddle q
   │  Stage 4: 两条相反的最短路径 p→q，拼接并平滑
   ▼
caging loop L  — 拉紧、不穿透实体、绕住 q 对应的障碍结构
```

每个 Morse 鞍点都由定理 3.1 保证能产生有效的拉紧环；自由空间波前保证环不会穿过实体；
鞍点在障碍另一侧的位置让环确实“绕住”物体。对多个鞍点做选择，则决定具体 cage 哪一部分。

> **这个版本还没有做的内容**（论文中有，released MATLAB 里也没有）：用夹爪半径构造 `r`-offset
> surface，用 convex hull 限定抓取空间，采样多个 base points，按曲率过滤 base points。
> 这些机制会控制结果是穿过某个小把手，还是绕住整个身体。见
> [paper-alignment.md](paper-alignment.md)。

### 一个很小的手算例子

**(a) 鞍点测试。** 这个例子就是
`test_calculate_iter_num_detects_four_sign_transitions_on_ring`，位于
[tests/test_saddle.py](../tests/test_saddle.py)。

取一个中心点，周围有 8 个均匀分布在单位圆上的邻居（每 45° 一个）。假设距离场为：

```text
centre  D = 1.0
ring    D = [ 2, 2, 0, 0, 2, 2, 0, 0 ]      角度 0° 45° 90° 135° 180° 225° 270° 315°
```

Stage 3 会计算 `judge = D(neighbour) - D(centre)`，再看符号：

```text
neighbour D :  2   2   0   0   2   2   0   0
judge       : +1  +1  -1  -1  +1  +1  -1  -1
sign        :  +   +   -   -   +   +   -   -
                └flip┘   └flip┘   └flip┘   └flip(wrap)┘
```

绕一圈符号变化 **4 次**，所以 `calculate_iter_num` 返回 `4`，满足 `>=4`，中心点就是一个
**鞍点**，也就是 caging-loop seed。局部邻域可以这样理解：

```text
        +   +
      -       +          + = 距离场高于中心（上坡）
        (c)              - = 距离场低于中心（下坡，Π₁/Π₂ 离开的方向）
      +       -
        -   +
```

如果距离场只呈现一段上升、一段下降，比如 `+ + + + - - - -`，符号只变化 2 次，就是普通点；
如果全是 `-`，符号变化 0 次，是极大点，按定理 3.1 也可以产生 loop seed。

**(b) 为什么这会变成环：甜甜圈图像。** 把基点 `p` 放在甜甜圈外圈。波前从 `p` 出发，
沿 ring 的顺时针和逆时针两侧同时传播，并在远端相遇；相遇点 `q` 就是鞍点。两个抵达波前对应两条
最短路径 `Π₁` 和 `Π₂`。Stage 4 把 `p →Π₁→ q →Π₂→ p` 拼起来，得到一条绕过甜甜圈 ring 的闭环，
也就是穿过洞的 caging loop。球体上同样会得到类似 great circle 的不稳定环；在 `knotty` 上则得到
经过 MATLAB 对照验证的 body-wrapping loop。

运行下面命令可以看到数字：

```powershell
python -m pytest tests/test_saddle.py -k four_sign_transitions -q
python examples/run_model_polyscope.py Models/knotty.obj --no-show
```

## 仓库结构

```text
Code/                     MATLAB 参考实现（只有 main interfaces，需要 FastRBF.exe 和工具包）
  pointCloudVoxelizationByRBF.m   stage 1: 通过 RBF 体素化
  DistanceMapByFastMarching.m     stage 2: 测地距离场（msfm）
  detectSaddlePoint.m             stage 3: 鞍点检测
  generateCagingGrasp.m           stage 4: 环生成
  compute_shortestpath.m          路径追踪（Kroon shortestpath）
  tool/                           第三方依赖压缩包（FastMarching、toolbox_graph、TSTOOL nn 等）
Models/                   示例模型（.obj/.stl/.ply），如 knotty、kitten、deer、fertility 等
cagingloop/               Python port
examples/                 可运行 pipeline
tests/                    pytest 测试（使用合成 fixture，不依赖模型文件）
docs/
  1807.11661*             论文
  superpowers/specs|plans port 的设计说明和实现计划
  matlab-python-differences.md   MATLAB 与 Python 的差异和已修复项
  paper-alignment.md             MATLAB 代码 / port 与论文的对应关系
  overview.md                    本文件
```

## Python pipeline：文件到阶段

| 阶段 | 模块 | 关键函数 |
|---|---|---|
| 1. 体素化 | [cagingloop/voxelization.py](../cagingloop/voxelization.py) | `point_cloud_voxelization_by_rbf`（RBF，类似 MATLAB），`voxelize_mesh`（winding number，**新增**） |
| 2. 距离场 | [cagingloop/distance.py](../cagingloop/distance.py) | `distance_map_by_fast_marching`, `compute_shortest_path` |
| 3. 鞍点 | [cagingloop/saddle.py](../cagingloop/saddle.py) | `detect_saddle_point`, `calculate_iter_num`, `diversity_eval` |
| 4. 环 | [cagingloop/grasp.py](../cagingloop/grasp.py) | `generate_caging_grasp`, `generate_best_caging_grasp`（**新增**）, `smooth_closed_path` |
| 支撑 | [cagingloop/nearest.py](../cagingloop/nearest.py) | KD-tree 版 `nn_prepare`/`nn_search` |
| 支撑 | [cagingloop/model_io.py](../cagingloop/model_io.py) | OBJ 加载、顶点法线、`transfer_point_normals`（**新增**） |
| 可视化 | [cagingloop/visualization.py](../cagingloop/visualization.py), [polyscope_visualization.py](../cagingloop/polyscope_visualization.py) | 可选、支持 headless |
| 类型 | [cagingloop/types.py](../cagingloop/types.py) | `VoxelizationResult`, `DistanceMapResult`, `CagingPath` dataclasses |

数据在三个 dataclass 里以普通数组流动。最重要的不变量：
`grid_on[i]`、`index.on_index[i]`、`dismap[i]` 指向**同一个表面体素 `i`**。
Python 中是 0-based，MATLAB 中是 1-based。

## 如何运行

```powershell
cd .worktrees/python-port
python -m pip install -e ".[dev]"        # numpy, scipy, scikit-fmm, pytest
python examples/run_model_polyscope.py Models/knotty.obj            # 打开 3D viewer
python examples/run_model_polyscope.py Models/knotty.obj --no-show  # headless，只打印统计
python -m pytest -q                                                  # 21 tests
```

常用参数：`--voxel-count N`、`--max-points N`、`--source-point-id -1`（auto = 离质心最远点）、
`--normal-offset 1e-4`。

## 如何调试：入口与旋钮

- **单独检查每个阶段**：每个阶段都是数组上的纯函数。可以构造 `VoxelizationResult`，
  调用 `distance_map_by_fast_marching`，再调用 `detect_saddle_point` 等，并打印 shape/range。
  `tests/` 里的合成 fixture 展示了最小输入。
- **可视化**：安装 `matplotlib` 后，可以把 `voxels.grid_on` 按 `distance.dismap` 上色散点显示，
  再叠加 `caging.final_path`。这是检查 loop、saddle、field 的常用方式。
- **对照 MATLAB**：`E:/tmp/mwork/` 里的 harness 会把相同的 `grid_on`/`dismap`/`D` 输入 MATLAB 和
  Python 的 `detectSaddlePoint`/`generateCagingGrasp`，并比较输出。它已经验证过 98% 的逐点 saddle
  一致，以及近乎相同的 loop。可复用于任何模型或 saddle。
- **影响结果的关键旋钮**
  - `voxel_count`：分辨率。低分辨率如 17³ 会粗糙；论文使用 50³。越高越细，但更慢。
  - `normal_offset`：RBF 约束偏移；MATLAB 使用 `1e-4`。
  - `source_point_id`：距离场的源点。它**强烈影响**最终环。`-1` 表示离质心最远点，是一个还不错的默认值。
  - `FILTER_TOLERANCE_FACTOR`：在 `saddle.py` 里，是尺度相对的 saddle filter 阈值，
    替代 MATLAB 中尺度不稳的硬编码 `50`。
  - `prefer_fmm`：`True` 使用 `scikit-fmm`，对应 MATLAB 的 `msfm`；`False` 使用 Dijkstra fallback。
- **常见坑**
  - Python 是 0-based，MATLAB 是 1-based。
  - `dismap` 单位：`skfmm` 搭配 `dx=spacing` 是**世界坐标单位**；MATLAB `msfm` 是**体素单位**。
    现在 saddle filter 已改成尺度相对，因此这个差异对 saddle 检测不再关键。
  - 没有 padding 时，物体可能贴满 bbox，表面会被裁掉。`voxelize_mesh` 默认会 padding；
    RBF 路径则像 MATLAB 一样使用原始 bbox。

## 已知限制

详见 [paper-alignment.md](paper-alignment.md)。这个 port 匹配的是 released MATLAB 代码，而该代码只是论文的一个子集。
尚未实现：夹爪半径 `r` 的 offset surface、convex-hull-bounded grasping space、多 base point 采样、
曲率过滤（Thm 3.3）以及论文中的 loop-selection criteria。这些机制决定结果是穿过小把手，
还是绕住整个身体。
