
import cv2
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "images"
OUTPUT_DIR = ROOT / "outputs" / "matches"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize SIFT
sift = cv2.SIFT_create(nfeatures=2000)

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

extensions = {".jpg", ".jpeg", ".png"}

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
])

if len(images) < 2:
    raise SystemExit("Need at least two images!")

print(f"Processing {len(images)} images")

# Process consecutive image pairs
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

    # Extract SIFT features
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Insufficient features:", path1.name, path2.name)
        continue

    # Find two nearest matches
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Sort by descriptor distance
    good_matches.sort(key=lambda x: x.distance)

    # Draw best matches
    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    output_name = (
        f"{path1.stem}_{path2.stem}_matches.jpg"
    )

    output_path = OUTPUT_DIR / output_name

    cv2.imwrite(str(output_path), result)

    print(
        f"{path1.name} <-> {path2.name}: "
        f"{len(good_matches)} good matches"
    )

    print("Saved:", output_name)

print("\nFeature matching completed!")
