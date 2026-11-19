
import cv2
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "images"
OUTPUT_DIR = ROOT / "outputs" / "camera_pose"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 1. Approximate camera intrinsics
# -------------------------------------------------

width = 2304
height = 4080

horizontal_fov_deg = 70.0

focal_length = width / (
    2.0 * np.tan(np.deg2rad(horizontal_fov_deg / 2.0))
)

K = np.array([
    [focal_length, 0, width / 2],
    [0, focal_length, height / 2],
    [0, 0, 1]
], dtype=np.float64)

print("Camera intrinsic matrix K:")
print(K)

# -------------------------------------------------
# 2. Initialize SIFT and matcher
# -------------------------------------------------

sift = cv2.SIFT_create(nfeatures=2000)

bf = cv2.BFMatcher(cv2.NORM_L2)

extensions = {".jpg", ".jpeg", ".png"}

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
])

if len(images) < 2:
    raise SystemExit("At least two images are required")

# -------------------------------------------------
# 3. Initialize camera trajectory
# -------------------------------------------------

R_world_to_camera = np.eye(3, dtype=np.float64)

t_world_to_camera = np.zeros((3, 1), dtype=np.float64)

camera_centers = [np.zeros(3)]

results = []

# -------------------------------------------------
# 4. Process consecutive image pairs
# -------------------------------------------------

for i in range(len(images) - 1):

    path1 = images[i]
    path2 = images[i + 1]

    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    if img1 is None or img2 is None:
        print("Image loading failed")
        continue

    if img1.shape[:2] != img2.shape[:2]:
        print("Image dimensions differ. Skipping pair.")
        continue

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Descriptors unavailable")
        continue

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 8:
        print(f"Insufficient matches: {path1.name}")
        continue

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
        cv2.RANSAC,
        0.999,
        1.0,
        5000
    )

    if E is None or mask is None:
        print("Essential Matrix estimation failed")
        continue

    # Handle stacked Essential Matrix solutions
    if E.shape[0] > 3:
        E = E[:3, :3]

    # -------------------------------------------------
    # 6. Recover relative camera pose
    # -------------------------------------------------

    recovered_count, R_relative, t_relative, pose_mask = (
        cv2.recoverPose(
            E,
            pts1,
            pts2,
            K,
            mask=mask
        )
    )

    print("\n--------------------------------")
    print(f"Pair: {path1.name} -> {path2.name}")
    print(f"Descriptor matches: {len(good_matches)}")
    print(f"Pose inliers: {recovered_count}")

    print("\nRelative rotation:")
    print(R_relative)

    print("\nRelative translation direction:")
    print(t_relative.ravel())

    # -------------------------------------------------
    # 7. Accumulate relative camera poses
    # -------------------------------------------------

    t_world_to_camera = (
        R_relative @ t_world_to_camera + t_relative
    )

    R_world_to_camera = (
        R_relative @ R_world_to_camera
    )

    # Camera center in the initial world coordinate frame
    camera_center = (
        -R_world_to_camera.T @ t_world_to_camera
    ).ravel()

    camera_centers.append(camera_center)

    results.append({
        "pair": f"{path1.name}-{path2.name}",
        "matches": len(good_matches),
        "pose_inliers": int(recovered_count),
        "rotation": R_relative.copy(),
        "translation": t_relative.copy()
    })

# -------------------------------------------------
# 8. Save camera trajectory
# -------------------------------------------------

trajectory_path = OUTPUT_DIR / "camera_trajectory.csv"

with open(trajectory_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Camera", "X", "Y", "Z"])

    for i, center in enumerate(camera_centers):

        writer.writerow([
            i + 1,
            center[0],
            center[1],
            center[2]
        ])

# -------------------------------------------------
# 9. Plot estimated camera positions
# -------------------------------------------------

camera_centers = np.array(camera_centers)

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    camera_centers[:, 0],
    camera_centers[:, 1],
    camera_centers[:, 2],
    marker="o"
)

for i, center in enumerate(camera_centers):
    ax.text(
        center[0],
        center[1],
        center[2],
        str(i + 1)
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Estimated Camera Trajectory")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "camera_trajectory.png",
    dpi=150
)

plt.close()

print("\nCamera pose estimation completed!")
print("Results:", OUTPUT_DIR)
print("Trajectory:", trajectory_path)
