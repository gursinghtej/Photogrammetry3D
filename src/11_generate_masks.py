
import cv2
import numpy as np
import os

IMAGE_DIR = "datasets/gso_cereal_box/cereal_box/rgb"
MASK_DIR = "outputs/masks"
PREVIEW_DIR = "outputs/mask_previews"

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Four representative views: narrow side, back, front, angled
test_views = [0, 10, 40, 55]

for view in test_views:

    filename = f"cereal_box_{view}_rgb.png"
    image_path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load {filename}")
        continue

    height, width = image.shape[:2]

    # Initial rectangle around the central object
    rect = (
        int(width * 0.30),
        int(height * 0.08),
        int(width * 0.40),
        int(height * 0.82)
    )

    # Initialize GrabCut
    mask = np.zeros((height, width), np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Keep definite and probable foreground
    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Save binary mask
    mask_path = os.path.join(
        MASK_DIR,
        f"mask_{view}.png"
    )

    cv2.imwrite(mask_path, binary_mask)

    # Create a green overlay for visual inspection
    overlay = image.copy()

    green = np.zeros_like(image)
    green[:, :] = (0, 255, 0)

    foreground = binary_mask > 0

    overlay[foreground] = cv2.addWeighted(
        image[foreground], 0.5,
        green[foreground], 0.5,
        0
    )

    preview = np.hstack((image, overlay))

    preview_path = os.path.join(
        PREVIEW_DIR,
        f"preview_{view}.jpg"
    )

    cv2.imwrite(preview_path, preview)

    print(f"Processed view {view}")

print("\nSegmentation test completed.")
