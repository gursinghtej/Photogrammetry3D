
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "images"
OUTPUT_DIR = ROOT / "outputs" / "reconstruction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 1. Load selected image pair
# -------------------------------------------------

img1 = cv2.imread(str(IMAGE_DIR / "image13.jpg"))
img2 = cv2.imread(str(IMAGE_DIR / "image14.jpg"))

if img1 is None or img2 is None:
    raise SystemExit("Could not load image pair")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

height, width = gray1.shape

if gray2.shape != gray1.shape:
    raise SystemExit("Image dimensions do not match")

# -------------------------------------------------
# 2. Approximate camera intrinsics
# -------------------------------------------------

focal = width / (2 * np.tan(np.deg2rad(70.0 / 2)))

K = np.array([
    [focal, 0, width / 2],
    [0, focal, height / 2],
    [0, 0, 1]
], dtype=np.float64)

# -------------------------------------------------
# 3. Extract SIFT features
# -------------------------------------------------

sift = cv2.SIFT_create(nfeatures=2000)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise SystemExit("Feature extraction failed")

# -------------------------------------------------
# 4. Match features
# -------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_L2)

raw_matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for pair in raw_matches:
    if len(pair) == 2:
        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print("Good descriptor matches:", len(good_matches))

if len(good_matches) < 8:
    raise SystemExit("Insufficient matches")

pts1 = np.float64([
    kp1[m.queryIdx].pt for m in good_matches
])

pts2 = np.float64([
    kp2[m.trainIdx].pt for m in good_matches
])

# -------------------------------------------------
# 5. Estimate Essential Matrix
# -------------------------------------------------

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None or mask is None:
    raise SystemExit("Essential Matrix estimation failed")

if E.shape[0] > 3:
    E = E[:3, :3]

# -------------------------------------------------
# 6. Recover relative camera pose
# -------------------------------------------------

count, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K,
    mask=mask
)

print("Pose inliers:", count)
print("Rotation:\n", R)
print("Translation:\n", t)

# -------------------------------------------------
# 7. Keep only pose inlier correspondences
# -------------------------------------------------

valid = pose_mask.ravel() != 0

pts1_in = pts1[valid]
pts2_in = pts2[valid]

print("Correspondences for triangulation:", len(pts1_in))

# -------------------------------------------------
# 8. Convert pixel coordinates to normalized coordinates
# -------------------------------------------------

norm1 = cv2.undistortPoints(
    pts1_in.reshape(-1, 1, 2),
    K,
    None
).reshape(-1, 2)

norm2 = cv2.undistortPoints(
    pts2_in.reshape(-1, 1, 2),
    K,
    None
).reshape(-1, 2)

# -------------------------------------------------
# 9. Construct projection matrices
# -------------------------------------------------

P1 = np.hstack((
    np.eye(3),
    np.zeros((3, 1))
))

P2 = np.hstack((
    R,
    t
))

# -------------------------------------------------
# 10. Triangulate 3D points
# -------------------------------------------------

points_h = cv2.triangulatePoints(
    P1,
    P2,
    norm1.T,
    norm2.T
)

points_3d = (
    points_h[:3] / points_h[3]
).T

# -------------------------------------------------
# 11. Geometric filtering
# -------------------------------------------------

# Depth in camera 1
depth1 = points_3d[:, 2]

# Transform points into camera 2
points_cam2 = (
    R @ points_3d.T + t
).T

depth2 = points_cam2[:, 2]

# Reprojection in normalized image coordinates
projected1 = points_3d[:, :2] / depth1[:, None]
projected2 = points_cam2[:, :2] / depth2[:, None]

error1 = np.linalg.norm(projected1 - norm1, axis=1)
error2 = np.linalg.norm(projected2 - norm2, axis=1)

mean_error = (error1 + error2) / 2

valid_points = (
    np.isfinite(points_3d).all(axis=1)
    & np.isfinite(mean_error)
    & (depth1 > 0)
    & (depth2 > 0)
    & (mean_error < 0.003)
)

filtered_points = points_3d[valid_points]

print("Triangulated points:", len(points_3d))
print("Valid 3D points:", len(filtered_points))

if len(filtered_points) < 5:
    raise SystemExit(
        "Too few valid points. Inspect camera calibration and matches."
    )

print(
    "Mean normalized reprojection error:",
    np.mean(mean_error[valid_points])
)

# -------------------------------------------------
# 12. Extract point colors from image 1
# -------------------------------------------------

valid_pixels = pts1_in[valid_points]

colors = []

for x, y in valid_pixels:
    x = int(round(x))
    y = int(round(y))

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    color = img1[y, x][::-1] / 255.0
    colors.append(color)

colors = np.asarray(colors)

# -------------------------------------------------
# 13. Generate Open3D point cloud
# -------------------------------------------------

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    filtered_points
)

pcd.colors = o3d.utility.Vector3dVector(colors)

# Save PLY
ply_path = OUTPUT_DIR / "reconstruction_pair13_14.ply"

o3d.io.write_point_cloud(str(ply_path), pcd)

# Save coordinates
np.savetxt(
    OUTPUT_DIR / "points_3d.csv",
    filtered_points,
    delimiter=",",
    header="X,Y,Z",
    comments=""
)

# -------------------------------------------------
# 14. Visualize
# -------------------------------------------------

o3d.visualization.draw_geometries(
    [pcd],
    window_name="3D Reconstruction - Image 13 and 14"
)

print("\n3D reconstruction completed!")
print("PLY:", ply_path)
print("CSV:", OUTPUT_DIR / "points_3d.csv")
