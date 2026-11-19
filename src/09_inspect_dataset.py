
from pathlib import Path
import cv2
import json
import re

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "gso_cereal_box"

IMAGE_DIR = DATASET_ROOT / "cereal_box" / "rgb"


def natural_sort_key(path):
    """Sort image_2 before image_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def inspect_images():
    print("\n" + "=" * 55)
    print("          GSO CEREAL BOX DATASET INSPECTION")
    print("=" * 55)

    if not IMAGE_DIR.exists():
        print(f"ERROR: Image directory not found: {IMAGE_DIR}")
        return False

    images = sorted(
        IMAGE_DIR.glob("*"),
        key=natural_sort_key
    )

    images = [
        p for p in images
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    print(f"\nDataset location: {DATASET_ROOT}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Total images found: {len(images)}")

    if not images:
        print("No images found.")
        return False

    dimensions = {}
    unreadable = []

    print("\nIMAGE DETAILS")
    print("-" * 55)

    for index, image_path in enumerate(images):

        image = cv2.imread(str(image_path))

        if image is None:
            unreadable.append(image_path.name)
            print(f"{index + 1:02d}. {image_path.name} - UNREADABLE")
            continue

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        dimensions[(width, height)] = (
            dimensions.get((width, height), 0) + 1
        )

        print(
            f"{index + 1:02d}. {image_path.name:<30}"
            f" {width} x {height} | Channels: {channels}"
        )

    print("\nSUMMARY")
    print("-" * 55)

    print(f"Total images: {len(images)}")
    print(f"Readable images: {len(images) - len(unreadable)}")
    print(f"Unreadable images: {len(unreadable)}")

    print("\nImage dimensions:")
    for size, count in dimensions.items():
        print(f"{size[0]} x {size[1]} : {count} images")

    if unreadable:
        print("\nUnreadable files:", unreadable)

    return True


def inspect_metadata():
    print("\n" + "=" * 55)
    print("             METADATA INSPECTION")
    print("=" * 55)

    # Search for metadata files without assuming
    # their exact nested directory structure.
    candidates = [
        p for p in DATASET_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".json"
    ]

    if not candidates:
        print("\nNo JSON metadata files found.")
        return

    for path in candidates:

        print(f"\nFILE: {path.relative_to(DATASET_ROOT)}")
        print("-" * 55)

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            print("Type:", type(data).__name__)

            if isinstance(data, dict):
                print("Top-level keys:", list(data.keys()))

            # Print a readable preview, not an entire
            # potentially large metadata file.
            preview = json.dumps(data, indent=2)

            print("\nJSON preview:")
            print(preview[:5000])

            if len(preview) > 5000:
                print("\n... Preview truncated ...")

        except Exception as error:
            print("Could not read JSON:", error)


def main():
    success = inspect_images()

    inspect_metadata()

    print("\n" + "=" * 55)

    if success:
        print("DATASET INSPECTION COMPLETED")
    else:
        print("DATASET INSPECTION FAILED")

    print("=" * 55)


if __name__ == "__main__":
    main()
