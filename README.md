# CagingLoop in Matlab
This is an Matlab implementation of the paper "[Caging Loops in Shape Embedding Space: Theory and Computation](https://kevinkaixu.net/papers/liu_icra18_grasp.pdf)". The paper is about synthesizing feasible caging grasps for a target object through computing Caging Loops, a closed curve defined in the shape embedding space of the object. This code was originally written by Jian Liu from Shan Dong University and is being improved and maintained here in this repository.

Note that the current version approvides the main interfaces of CagingLoop.The main interfaces include grasping space construction, p-based distance field computation, saddle points detection and caging loop generation.

## Usage
CagingLoop should be run with Matlab R2015B. A function Library implemented with Matlab is provided in the folder of tool.
Before running the main interfaces of CagingLoop, you need to add Full Path of the tool folder in Matlab.

1. run pointCloudVoxelizationByRBF.m to discretize the grasping space based on RBF
2. run DistanceMapByFastMarching.m to compute the distance field rooted at source point
3. run detectSaddlePoint.m to detect the saddle points
4. run generateCagingGrasp.m to generate a caging loop connecting source point and saddle point

## Python Port

The Python implementation lives in `cagingloop/` and mirrors the MATLAB workflow without requiring MATLAB, MEX files, or `FastRBF.exe`. It uses Python-native numerical libraries, so outputs are intended to be workflow-equivalent rather than point-for-point identical to the MATLAB version.

Install for development:

```powershell
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

Run the synthetic example:

```powershell
python examples/run_pipeline.py
```

Run the same synthetic pipeline in Polyscope:

```powershell
python -m pip install -e ".[polyscope]"
python examples/run_pipeline_polyscope.py
```

For a non-interactive smoke test without opening the Polyscope window:

```powershell
python examples/run_pipeline_polyscope.py --no-show
```

Run `Models/knotty.obj` in Polyscope:

```powershell
python examples/run_model_polyscope.py Models\knotty.obj --voxel-count 17 --max-points 1200 --normal-offset 0.0001
```

Use `--no-show` for a dry run:

```powershell
python examples/run_model_polyscope.py Models\knotty.obj --voxel-count 17 --max-points 1200 --normal-offset 0.0001 --no-show
```

The primary Python workflow functions are:

1. `point_cloud_voxelization_by_rbf`
2. `distance_map_by_fast_marching`
3. `detect_saddle_point`
4. `generate_caging_grasp`

### Single-view RGB-D capture (examples/render_rgbd.py)

Render a model from one monocular viewpoint into RGB + depth + the visible-shell
point cloud (camera frame and world/mesh frame). Requires open3d
(`pip install open3d`; needs a Python version with open3d wheels, e.g. 3.10-3.12):

```powershell
python examples/render_rgbd.py Models/knotty.obj --azimuth 40 --elevation 15
```

Outputs in `out/knotty_az40_el15/`: `rgb.png`, `depth.npy` (view-space z, NaN
background), `cloud_cam.ply`, `cloud_world.ply`, `camera.json` (K, world->camera T).
Add `--preview` to inspect the cloud against the mesh in polyscope.

## Citation
If you use this code, please cite the following paper.
```
@article {liu2018RopeCA,
	title = {Caging Loops in Shape Embedding Space: Theory and Computation},
	author = {Jian Liu and Shiqing Xin and Zengfu Gao and Kai Xu and Changhe Tu and Baoquan Chen},
	journal = {2018 IEEE International Conference on Robotics and Automation (ICRA)},
  	year = {2018},
	pages = {?}
}
```
