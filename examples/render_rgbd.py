"""Single-view RGB-D capture of a mesh: RGB render + metric depth + the
visible-shell point cloud in camera AND world frames (world = mesh coordinates,
directly comparable with the source model as completion ground truth).

    python examples/render_rgbd.py Models/knotty.obj --azimuth 40 --elevation 15

Writes into --out (default out/<model>_az<az>_el<el>/): rgb.png, depth.npy
(float32 view-space z, NaN background), cloud_cam.ply, cloud_world.ply,
camera.json (intrinsics K, world->camera extrinsic T, view parameters).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cagingloop.rgbd import depth_to_pointclouds, load_mesh, make_camera, render_rgbd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Single-view RGB-D capture of a mesh.")
    ap.add_argument("model", help="mesh file (.obj/.stl/.ply)")
    ap.add_argument("--azimuth", type=float, default=40.0, help="camera azimuth (deg)")
    ap.add_argument("--elevation", type=float, default=15.0, help="camera elevation (deg)")
    ap.add_argument("--distance", default="auto",
                    help="'auto' (fit object in frame) or a float in mesh units")
    ap.add_argument("--fov", type=float, default=60.0, help="vertical field of view (deg)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=None,
                    help="output directory (default out/<model>_az<az>_el<el>)")
    ap.add_argument("--preview", action="store_true",
                    help="show world-frame cloud + mesh in polyscope")
    args = ap.parse_args(argv)

    import open3d as o3d  # required anyway by load_mesh; explicit here for PLY IO
    from PIL import Image

    mesh = load_mesh(args.model)
    distance = None if str(args.distance).lower() == "auto" else float(args.distance)
    camera = make_camera(
        np.asarray(mesh.vertices), args.azimuth, args.elevation, distance,
        fov_deg=args.fov, width=args.width, height=args.height,
    )
    rgb, depth = render_rgbd(mesh, camera)
    if not np.isfinite(depth).any():
        raise SystemExit(
            "no object pixels in view — try other --azimuth/--elevation or a larger --distance"
        )
    points_cam, points_world, colors = depth_to_pointclouds(depth, rgb, camera)

    out = Path(args.out) if args.out else (
        Path("out") / f"{Path(args.model).stem}_az{args.azimuth:g}_el{args.elevation:g}"
    )
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth.npy", np.where(np.isfinite(depth), depth, np.nan).astype(np.float32))
    for name, pts in (("cloud_cam.ply", points_cam), ("cloud_world.ply", points_world)):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out / name), pc)
    meta = {
        "model": str(args.model),
        "width": camera.width, "height": camera.height, "fov_deg": camera.fov_deg,
        "azimuth_deg": camera.azimuth_deg, "elevation_deg": camera.elevation_deg,
        "distance": camera.distance,
        "K": camera.K.tolist(),
        "T_world_to_cam": camera.T.tolist(),
        "depth_definition": "float32 view-space z along the optical axis; NaN = background; mesh units",
        "cloud_world_frame": "source mesh coordinates",
        "convention": "OpenCV pinhole: +Z forward, +X right, +Y down; X_cam = T[:3,:3] @ X_world + T[:3,3]",
    }
    (out / "camera.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  ({len(points_world)} visible-shell points)")

    if args.preview:
        import polyscope as ps

        ps.init()
        cloud = ps.register_point_cloud("cloud_world", points_world)
        cloud.add_color_quantity("rgb", colors, enabled=True)
        m = ps.register_surface_mesh(
            "mesh", np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        )
        m.set_transparency(0.4)
        ps.show()


if __name__ == "__main__":
    main()
