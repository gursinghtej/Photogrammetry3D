
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "images"

for name in ["image13.jpg", "image14.jpg"]:

    image = cv2.imread(str(IMAGE_DIR / name))

    if image is None:
        print("Cannot load:", name)
        continue

    # Resize for convenient inspection
    preview = cv2.resize(
        image,
        (576, 1020)
    )

    cv2.imshow(name, preview)

    print("Inspecting:", name)

    cv2.waitKey(0)

cv2.destroyAllWindows()
