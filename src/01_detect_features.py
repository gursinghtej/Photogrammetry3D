
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "images"
OUTPUT_DIR = ROOT / "outputs" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize SIFT
sift = cv2.SIFT_create(nfeatures=2000)

# Find images
extensions = {".jpg", ".jpeg", ".png"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
]

if not images:
    print("No images found!")
    print("Place photographs inside:", IMAGE_DIR)
    raise SystemExit(1)

print(f"Found {len(images)} images")

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Cannot read:", image_path.name)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(
        gray, None
    )

    result = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    output_path = OUTPUT_DIR / f"{image_path.stem}_features.jpg"

    cv2.imwrite(str(output_path), result)

    print(f"{image_path.name}: {len(keypoints)} keypoints")

print("\nFeature detection completed!")
print("Output:", OUTPUT_DIR)
