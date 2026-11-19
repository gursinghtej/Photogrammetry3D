
import cv2
import numpy as np
import open3d as o3d

from pathlib import Path

# ============================================================
# 0. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "images"
OUTPUT_DIR = ROOT / "outputs" / "reconstruction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE1_PATH = IMAGE_DIR / "image13.jpg"
IMAGE2_PATH = IMAGE_DIR / "image14.jpg"

MAX_FEATURES = 5000
RATIO_THRESHOLD = 0.75

# Initial estimate only. Replace with calibrated intrinsics later.
HORIZONTAL_FOV_DEG = 70.0

# RANSAC settings
RANSAC_PROBABILITY = 0.999
ESSENTIAL_THRESHOLD_PX = 1.5

# ============================================================
# 1. LOAD IMAGES
# ============================================================

print("\n[1] Loading images...")

img1 = cv2.imread(str(IMAGE1_PATH))
img2 = cv2.imread(str(IMAGE2_PATH))

if img1 is None or img2 is None:
    raise SystemExit("ERROR: Could not load input images.")

height, width = img1.shape[:2]

if img2.shape[:2] != (height, width):
    raise SystemExit("ERROR: Image dimensions do not match.")

print("Image dimensions:", width, "x", height)

# ============================================================
# 2. INTERACTIVE OBJECT SELECTION
# ============================================================

def select_object_roi(image, window_name):

    screen_width = 1050
    screen_height = 720

    h, w = image.shape[:2]

    scale = min(
        screen_width / w,
        screen_height / h,
        1.0
    )

    preview = cv2.resize(
        image,
        (
            int(w * scale),
            int(h * scale)
        ),
        interpolation=cv2.INTER_AREA
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        preview.shape[1],
        preview.shape[0]
    )

    print("\nSelect the complete box in:", window_name)
    print("Drag the rectangle.")
    print("Press ENTER or SPACE to confirm.")
    print("Press C to cancel.")

    x, y, rw, rh = cv2.selectROI(
        window_name,
        preview,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    if rw == 0 or rh == 0:
        raise SystemExit("ERROR: ROI selection cancelled.")

    # Convert preview coordinates into original image coordinates.
    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(rw / scale)),
        int(round(rh / scale))
    )


print("\n[2] Select object regions.")

roi1 = select_object_roi(
    img1,
    "Select Box - Image 13"
)

roi2 = select_object_roi(
    img2,
    "Select Box - Image 14"
)

print("\nSelected regions:")
print("Image 13:", roi1)
print("Image 14:", roi2)

# ============================================================
# 3. CREATE ROI MASKS
# ============================================================

def create_roi_mask(image, roi):

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    x, y, w, h = roi

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)

    mask[y1:y2, x1:x2] = 255

    return mask


mask1 = create_roi_mask(img1, roi1)
mask2 = create_roi_mask(img2, roi2)

# ============================================================
# 4. SIFT FEATURE DETECTION
# ============================================================

print("\n[3] Detecting object features...")

gray1 = cv2.cvtColor(
    img1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    img2,
    cv2.COLOR_BGR2GRAY
)

sift = cv2.SIFT_create(
    nfeatures=MAX_FEATURES,
    contrastThreshold=0.02,
    edgeThreshold=12
)

kp1, des1 = sift.detectAndCompute(
    gray1,
    mask1
)

kp2, des2 = sift.detectAndCompute(
    gray2,
    mask2
)

if des1 is None or des2 is None:
    raise SystemExit("ERROR: No descriptors found.")

print("Image 13 features:", len(kp1))
print("Image 14 features:", len(kp2))

# ============================================================
# 5. MUTUAL FEATURE MATCHING
# ============================================================

print("\n[4] Matching object features...")

bf = cv2.BFMatcher(
    cv2.NORM_L2
)


def ratio_matches(des_a, des_b):

    raw = bf.knnMatch(
        des_a,
        des_b,
        k=2
    )

    accepted = []

    for pair in raw:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < RATIO_THRESHOLD * n.distance:
            accepted.append(m)

    return accepted


forward_matches = ratio_matches(
    des1,
    des2
)

reverse_matches = ratio_matches(
    des2,
    des1
)

# Retain only mutually consistent descriptor matches.
reverse_pairs = {
    (m.trainIdx, m.queryIdx)
    for m in reverse_matches
}

good_matches = [
    m for m in forward_matches
    if (m.queryIdx, m.trainIdx) in reverse_pairs
]

print("Forward ratio matches:", len(forward_matches))
print("Reverse ratio matches:", len(reverse_matches))
print("Mutual matches:", len(good_matches))

if len(good_matches) < 8:
    raise SystemExit(
        "ERROR: Insufficient mutual matches. "
        "Capture more overlapping object photographs."
    )

# ============================================================
# 6. EXTRACT MATCHED COORDINATES
# ============================================================

pts1 = np.float64([
    kp1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float64([
    kp2[m.trainIdx].pt
    for m in good_matches
])

# ============================================================
# 7. SAVE MATCH VISUALIZATION
# ============================================================

print("\n[5] Saving feature matching visualization...")

match_visualization = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches[:150],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "box_feature_matches.jpg"),
    match_visualization
)

# ============================================================
# 8. GEOMETRIC VERIFICATION
# ============================================================

print("\n[6] Running geometric verification...")

# Fundamental Matrix
F, fundamental_mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC,
    1.5,
    RANSAC_PROBABILITY,
    5000
)

if F is not None and fundamental_mask is not None:

    fundamental_inliers = int(
        np.count_nonzero(fundamental_mask)
    )

    print(
        "Fundamental Matrix inliers:",
        fundamental_inliers
    )

else:

    fundamental_inliers = 0
    print("Fundamental Matrix estimation failed.")

# Homography diagnostic
H, homography_mask = cv2.findHomography(
    pts1,
    pts2,
    cv2.RANSAC,
    3.0
)

if H is not None and homography_mask is not None:

    homography_inliers = int(
        np.count_nonzero(homography_mask)
    )

    print(
        "Homography inliers:",
        homography_inliers
    )

    if homography_inliers > 0.7 * len(good_matches):
        print(
            "WARNING: Matches are strongly planar. "
            "3D depth estimation may be poorly constrained."
        )

else:

    homography_inliers = 0
    print("Homography estimation failed.")

# ============================================================
# 9. CAMERA INTRINSIC MATRIX
# ============================================================

print("\n[7] Building approximate camera matrix...")

focal = width / (
    2.0 * np.tan(
        np.deg2rad(HORIZONTAL_FOV_DEG / 2.0)
    )
)

K = np.array([
    [focal, 0, width / 2.0],
    [0, focal, height / 2.0],
    [0, 0, 1]
], dtype=np.float64)

print("Camera matrix:")
print(K)

print(
    "NOTE: Focal length is estimated, not calibrated."
)

# ============================================================
# 10. ESSENTIAL MATRIX
# ============================================================

print("\n[8] Estimating Essential Matrix...")

E, essential_mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=RANSAC_PROBABILITY,
    threshold=ESSENTIAL_THRESHOLD_PX,
    maxIters=5000
)

if E is None or essential_mask is None:
    raise SystemExit(
        "ERROR: Essential Matrix estimation failed."
    )

if E.shape[0] > 3:
    E = E[:3, :3]

print("Essential Matrix:")
print(E)

# ============================================================
# 11. CAMERA POSE RECOVERY
# ============================================================

print("\n[9] Recovering relative camera pose...")

pose_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K,
    mask=essential_mask
)

print("Pose inliers:", pose_inliers)
print("Rotation:")
print(R)

print("Translation direction:")
print(t)

pose_valid = pose_mask.ravel() != 0

pts1_valid = pts1[pose_valid]
pts2_valid = pts2[pose_valid]

print(
    "Correspondences retained:",
    len(pts1_valid)
)

# Save pose-inlier visualization.
pose_matches = [
    m for m, valid in zip(good_matches, pose_valid)
    if valid
]

pose_visualization = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    pose_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "box_pose_inliers.jpg"),
    pose_visualization
)

if len(pts1_valid) < 8:
    raise SystemExit(
        "ERROR: Fewer than 8 pose inliers. "
        "Cannot reliably triangulate this pair."
    )

if len(pts1_valid) < 30:
    print(
        "WARNING: Sparse object reconstruction. "
        "More photographs are recommended."
    )

# ============================================================
# 12. NORMALIZE COORDINATES
# ============================================================

print("\n[10] Normalizing coordinates...")

norm1 = cv2.undistortPoints(
    pts1_valid.reshape(-1, 1, 2),
    K,
    None
).reshape(-1, 2)

norm2 = cv2.undistortPoints(
    pts2_valid.reshape(-1, 1, 2),
    K,
    None
).reshape(-1, 2)

# ============================================================
# 13. CAMERA PROJECTION MATRICES
# ============================================================

P1 = np.hstack([
    np.eye(3),
    np.zeros((3, 1))
])

P2 = np.hstack([
    R,
    t
])

# ============================================================
# 14. TRIANGULATION
# ============================================================

print("\n[11] Triangulating 3D points...")

points_h = cv2.triangulatePoints(
    P1,
    P2,
    norm1.T,
    norm2.T
)

# Avoid division by zero and non-finite homogeneous values.
w_values = points_h[3]

safe_w = np.abs(w_values) > 1e-10

points_h = points_h[:, safe_w]

norm1 = norm1[safe_w]
norm2 = norm2[safe_w]

pts1_valid = pts1_valid[safe_w]

points_3d = (
    points_h[:3] / points_h[3]
).T

print("Triangulated points:", len(points_3d))

# ============================================================
# 15. DEPTH AND REPROJECTION FILTERING
# ============================================================

print("\n[12] Filtering reconstructed points...")

depth1 = points_3d[:, 2]

points_cam2 = (
    R @ points_3d.T + t
).T

depth2 = points_cam2[:, 2]

safe_depth = (
    np.abs(depth1) > 1e-10
) & (
    np.abs(depth2) > 1e-10
)

projected1 = np.full(
    (len(points_3d), 2),
    np.nan
)

projected2 = np.full(
    (len(points_3d), 2),
    np.nan
)

projected1[safe_depth] = (
    points_3d[safe_depth, :2]
    / depth1[safe_depth, None]
)

projected2[safe_depth] = (
    points_cam2[safe_depth, :2]
    / depth2[safe_depth, None]
)

error1 = np.linalg.norm(
    projected1 - norm1,
    axis=1
)

error2 = np.linalg.norm(
    projected2 - norm2,
    axis=1
)

mean_error = (error1 + error2) / 2.0

valid_points_mask = (
    safe_depth
    & np.isfinite(points_3d).all(axis=1)
    & np.isfinite(mean_error)
    & (depth1 > 0)
    & (depth2 > 0)
    & (mean_error < 0.003)
)

filtered_points = points_3d[valid_points_mask]

valid_pixels = pts1_valid[valid_points_mask]

print("Valid 3D points:", len(filtered_points))

if len(filtered_points) == 0:
    raise SystemExit("No valid 3D points remain.")

mean_reprojection_error = np.mean(
    mean_error[valid_points_mask]
)

print(
    "Mean normalized reprojection error:",
    mean_reprojection_error
)

# ============================================================
# 16. ASSIGN COLORS
# ============================================================

colors = []

for x, y in valid_pixels:

    x = int(np.clip(round(x), 0, width - 1))
    y = int(np.clip(round(y), 0, height - 1))

    colors.append(
        img1[y, x][::-1] / 255.0
    )

colors = np.asarray(colors)

# ============================================================
# 17. GENERATE POINT CLOUD
# ============================================================

print("\n[13] Generating point cloud...")

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    filtered_points
)

pcd.colors = o3d.utility.Vector3dVector(
    colors
)

# ============================================================
# 18. SAVE OUTPUTS
# ============================================================

print("\n[14] Saving outputs...")

ply_path = OUTPUT_DIR / "box_reconstruction.ply"

csv_path = OUTPUT_DIR / "box_points.csv"

o3d.io.write_point_cloud(
    str(ply_path),
    pcd
)

np.savetxt(
    csv_path,
    filtered_points,
    delimiter=",",
    header="X,Y,Z",
    comments=""
)

# ============================================================
# 19. VISUALIZE POINT CLOUD
# ============================================================

coordinate_frame = (
    o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2
    )
)

o3d.visualization.draw_geometries(
    [pcd, coordinate_frame],
    window_name="Object-Specific 3D Reconstruction",
    width=1200,
    height=800
)

# ============================================================
# 20. FINAL REPORT
# ============================================================

print("\n===================================")
print("       RECONSTRUCTION REPORT")
print("===================================")

print("Initial mutual matches:", len(good_matches))
print("Fundamental Matrix inliers:", fundamental_inliers)
print("Homography inliers:", homography_inliers)
print("Pose inliers:", pose_inliers)
print("Triangulated points:", len(points_3d))
print("Valid 3D points:", len(filtered_points))

print(
    "Mean normalized reprojection error:",
    mean_reprojection_error
)

print("\nOUTPUT FILES:")
print("Point cloud:", ply_path)
print("Coordinates:", csv_path)
print("All matches:", OUTPUT_DIR / "box_feature_matches.jpg")
print("Pose inliers:", OUTPUT_DIR / "box_pose_inliers.jpg")

print("\nReconstruction pipeline completed!")
