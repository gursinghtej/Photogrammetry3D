
import open3d as o3d
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLY_PATH = (
    ROOT / "outputs" / "reconstruction"
    / "reconstruction_pair13_14.ply"
)

pcd = o3d.io.read_point_cloud(str(PLY_PATH))

if pcd.is_empty():
    raise SystemExit("Point cloud is empty!")

points = np.asarray(pcd.points)

print("Number of points:", len(points))
print("\nCoordinate ranges:")
print("X:", points[:, 0].min(), points[:, 0].max())
print("Y:", points[:, 1].min(), points[:, 1].max())
print("Z:", points[:, 2].min(), points[:, 2].max())

# Remove invalid points
pcd = pcd.remove_non_finite_points()


# Create coordinate axes
axis_size = max(
    np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.15,
    0.1
)

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=axis_size,
    origin=[0, 0, 0]
)

# Configure visualizer
visualizer = o3d.visualization.Visualizer()

visualizer.create_window(
    window_name="Photogrammetry - 3D Point Cloud",
    width=1280,
    height=800
)

visualizer.add_geometry(pcd)
visualizer.add_geometry(coordinate_frame)

render = visualizer.get_render_option()

render.point_size = 5.0
render.background_color = np.array([0.08, 0.08, 0.08])

visualizer.run()

visualizer.destroy_window()
