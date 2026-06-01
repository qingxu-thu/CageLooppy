# 与论文的对齐情况

本文档检查当前实现与 **Liu 等，"Caging Loops in Shape Embedding Space:
Theory and Computation," ICRA 2018**（arXiv:1807.11661）之间的关系。

这里比较三件事：

- **论文**：正式发表的算法。
- **MATLAB**：作者发布的 `Code/`，按 README 说法只提供 main interfaces。
- **Python**：当前 port。它以 released MATLAB `Code/` 为主要目标，并通过 live MATLAB
  对照验证过：逐点 saddle 约 98% 一致，loop 几乎相同。

**核心结论：** released MATLAB 代码是论文的一个**简化子集**。它省略了论文里最有标志性的
**尺度感知**机制。Python port 复现的是 MATLAB 代码，因此也继承了这些省略项。重要的是，
我们得到的“绕住整个身体”的 loop 与论文核心思想并不矛盾；穿过单个把手是另一个 regime，
由夹爪半径 `r` 控制，而 released MATLAB 和当前 port 都没有实现这套 `r` 机制。

---

## 关键概念：为什么它没有穿过把手

论文的贡献是通过在 *embedding space* 中工作，把 caging loop 从单个 handle 的表面细节中
**解耦**出来。论文 §II.A, p.3 明确说这类 loop 可以：

- 像“包围球体的大圆”一样绕住目标物体；
- 大致保持“水平”，中心接近重心；
- 包住多个把手，或者填平小孔，而不是一定穿过某一个孔（见 Fig. 1 top、Fig. 2a 和摘要）。

loop 是**穿过某个把手**（Fig. 7 红色 loop），还是**绕住身体或多个把手**（Fig. 7 绿色 loop），
由 §IV 中的 **夹爪半径 `r`** 通过 `r`-offset surface `S_r` 控制：大的 `r` 会填平小孔，让 loop
绕住身体；小的 `r` 才允许 loop 穿过小孔。

所以 MATLAB 和当前 port 产生的近似平面、包住物体的 loop，是论文中 intended 的
“great-circle”行为，并不必然是 bug。要得到 handle-threading loop，需要实现 released code
没有提供的 `r`-offset / grasping-space 构造。

---

## 逐组件对照

### 1. 通过 RBF 构造 grasping space（§IV.A）

- **论文：** `f(x)=Σ wᵢφ(‖x-xᵢ‖)+P(x)`，`φ(t)=t`。约束包括表面上 `f(xⱼ)=0`，以及
  **`f(xⱼ+r·nⱼ)=r`**，即夹爪半径 offset surface。grasping space 为
  `G_r = { outside voxels : f(x) ≥ r }`，并且按定理 3.2 被限制在 `S_r` 和
  **convex hull** `H_r` 之间。
- **MATLAB：** 使用 FastRBF，三组约束：表面 `=0`，`±0.0001` 法线偏移处 `=±1`。
  按符号分类：inner 为 `f<0`，outer 为 `f>0`，surface 约为 `0`。**没有夹爪半径、
  没有 `f≥r` offset，也没有 convex-hull bound。**
- **Python：** 与 MATLAB 相同思路，用 `scipy.RBFInterpolator`，`±offset` 映射到 `±1`。
  另外提供可选的 `voxelize_mesh` winding-number 路径。
- **结论：** ⚠️ **与论文有偏离。** 基函数 `φ(t)=t` 是一致的，但 released code 使用的是
  符号指示 `±1`，不是论文中的到 `r`-offset 的 signed distance；而且整套尺度感知的
  `r` / convex hull 机制都缺失。代码里的 `offset` 只是 RBF 拟合用的微小参数，**不是**夹爪半径。

### 2. 距离场 `D_p`（§III.A）

- **论文：** 在 grasping space `G` 中计算测地最短路径长度；这是围绕物体的体空间距离。
  为效率，扫掠在半径 `2h`（夹爪最大张开尺度）处截断。
- **MATLAB / Python：** `msfm` / `scikit-fmm` Eikonal geodesic，传播域为表面+外部，
  内部被阻挡。单个 source point；没有 `2h` sweep limit。
- **结论：** ✅ **方法类型一致**，都是体空间外部测地距离。⚠️ 但只用单个基点，且没有夹爪尺寸截断。

### 3. 鞍点检测（§III.A, Fig. 3）

- **论文：** 在**体素**距离场上做离散 Morse-Smale：看 6 个轴向邻居相对中心是 `+` 还是 `-`。
  鞍点有 2 个相对的 `-`，其余为 `+`，对应 4 次符号变化；极大点全是 `-`。
- **MATLAB / Python：** 限制在**表面**体素上：用 PCA 构造切平面，把约 8 个邻居投到 2D，
  通过 convex hull 排成环，统计 `D` 的符号变化，保留 `>=4`；再加 `filterBoundary`
  去噪和 `diversityEval` 排名。
- **结论：** ✅ **概念一致**，核心都是 `>=4` 符号变化即 Morse saddle。⚠️ 具体实现不同：
  论文是体素 6 邻域，代码是表面切平面环；并且代码加入了论文没有的启发式。

### 4. Base point 限制（Thm 3.3）

- **论文：** 只使用至少一个 principal curvature 为正的表面点；完全凹的点不能作为 caging loop seed。
- **MATLAB / Python：** **没有曲率测试**；只使用单个任意或启发式 source point。
- **结论：** ❌ **未实现。**

### 5. Loop tracing（Thm 3.1）

- **论文：** 在鞍点 `q` 处，两条最短路径 `Π₁, Π₂` 从相反方向离开，并拼成 p-based loop。
- **MATLAB / Python：** `getCagePoint` 选择最陡下降方向和与它最相反的另一个下降方向；
  两条最短路径回到 source；拼接并做一轮 Laplacian 平滑。
- **结论：** ✅ **一致**，并且 Python port 与 MATLAB 的 loop 近乎相同。

### 6. Loop 池与选择（§II.B, §III.B, §IV.B）

- **论文：** 对**多个 base points × 所有 saddles** 建立 loop pool `L̃`，然后过滤：
  只保留在 base point 也局部最短的 loop、位于 `S_r` 和 `H_r` 之间、长度 `< 4h`、
  大致水平、中心接近**重心**；最后再按夹爪干涉和 opening angle 选择。
- **MATLAB：** 使用 `diversityEval`，按法线对齐程度排序，保留 top `round(n/30)`。
- **Python：** 与 MATLAB 相同，另外加入 `generate_best_caging_grasp`，按最大 enclosed area 选 loop。
- **结论：** ⚠️ **大幅偏离论文。** 论文中的几何/机械选择标准，例如水平、近重心、`<4h`、
  base point 局部最短、夹爪干涉等，在 released code 和 port 中都没有实现。

### 7. 离散化和采样（§V.D）

- **论文：** 使用 **50×50×50** 体素，**500** 个均匀采样 base points，约 2K 顶点 mesh，
  每个模型约 1.5 秒。
- **MATLAB / Python：** 分辨率是参数；默认只使用**单个** base point。
- **结论：** ⚠️ 默认分辨率更低，且只使用单个基点。

---

## 总结

| 组件 | 论文 ↔ released MATLAB | MATLAB ↔ 当前 Python port |
|---|---|---|
| RBF basis `φ(t)=t` | ✅ | ✅ |
| offset = gripper radius `r`，`f≥r`，convex-hull bound | ❌ MATLAB 省略 | ✅ port 匹配 MATLAB，也省略 |
| 体空间外部测地距离 | ✅ 类型一致 | ✅ |
| `2h` sweep limit | ❌ 省略 | ✅ 匹配，也省略 |
| saddle = `>=4` 符号变化 | ✅ 思想一致，实现不同 | ✅ 已验证高度一致 |
| 曲率 base-point filter（Thm 3.3） | ❌ 省略 | ✅ 匹配，也省略 |
| 两条相反路径拼 loop | ✅ | ✅ 近乎相同 |
| 多 base-point pool + 论文选择标准 | ❌ 省略 | ✅ 匹配，也省略 |

**一句话：**

1. 这个 port 的目标是**忠实复现 released MATLAB code**，并非复现论文中所有实验设置。
2. released MATLAB code 是论文的**简化子集**：实现了 distance-field + Morse-saddle +
   two-path-loop 的核心，但省略了 gripper-size `r`-offset surface、convex-hull-bounded
   grasping space、多 base-point 采样、曲率过滤和论文中的 loop-selection criteria。
3. “绕住身体”的 loop 是论文 intended 的 great-circle 行为。要得到 handle-threading loop，
   需要实现 `r`-offset 机制；这已经超出 released MATLAB code 的范围。
