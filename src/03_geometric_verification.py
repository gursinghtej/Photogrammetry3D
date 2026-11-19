
import cv2
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "images"
OUTPUT_DIR = ROOT / "outputs" / "geometry"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature detector and matcher
sift = cv2.SIFT_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_L2)

extensions = {".jpg", ".jpeg", ".png"}

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
])

print(f"Processing {len(images)} images")

for i in range(len(images) - 1):

    path1 = images[i]
    path2 = images[i + 1]

    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    if img1 is None or img2 is None:
        print("Image loading failed")
        continue

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("No descriptors available")
        continue

    # Descriptor matching
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 8:
        print(f"{path1.name} - {path2.name}: "
              "Not enough matches")
        continue

    # Extract corresponding coordinates
    pts1 = np.float32([
        kp1[m.queryIdx].pt for m in good_matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt for m in good_matches
    ])

    # Estimate Fundamental Matrix using RANSAC
    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        1.0,
        0.99,
        5000
    )

    if F is None or mask is None:
        print("Fundamental Matrix estimation failed")
        continue

    # Separate inliers and outliers
    mask = mask.ravel().astype(bool)

    inlier_matches = [
        m for m, valid in zip(good_matches, mask)
        if valid
    ]

    outlier_count = len(good_matches) - len(inlier_matches)

    inlier_ratio = (
        len(inlier_matches) / len(good_matches)
    ) * 100

    # Visualize geometrically verified matches
    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    output_path = OUTPUT_DIR / (
        f"{path1.stem}_{path2.stem}_inliers.jpg"
    )

    cv2.imwrite(str(output_path), result)

    print("\n--------------------------------")
    print(f"Pair: {path1.name} - {path2.name}")
    print(f"Initial matches: {len(good_matches)}")
    print(f"Inliers: {len(inlier_matches)}")
    print(f"Outliers: {outlier_count}")
    print(f"Inlier ratio: {inlier_ratio:.2f}%")
    print("Fundamental Matrix:")
    print(F)
    print("Saved:", output_path.name)

print("\nGeometric verification completed!")
