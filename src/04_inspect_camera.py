
import cv2
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "images"

images = sorted(IMAGE_DIR.glob("*"))

if not images:
    raise SystemExit("No images found")

image_path = images[0]

# Image dimensions using OpenCV
image = cv2.imread(str(image_path))
height, width = image.shape[:2]

print("IMAGE:", image_path.name)
print("Width:", width)
print("Height:", height)

# Read EXIF metadata
with Image.open(image_path) as img:
    exif = img.getexif()

    print("\nCAMERA METADATA")

    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)

        if tag in [
            "Make",
            "Model",
            "FocalLength",
            "FocalLengthIn35mmFilm",
            "Orientation",
            "LensModel"
        ]:
            print(f"{tag}: {value}")

print("\nCamera inspection completed!")
