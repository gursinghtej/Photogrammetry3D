import cv2
import os
import glob
import math

IMAGE_DIR = "datasets/gso_cereal_box/cereal_box/rgb"
OUTPUT_DIR = "outputs/dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

cols = 8
rows = math.ceil(len(files) / cols)

thumb_w = 240
thumb_h = 160
label_h = 25

sheet = 255 * (
    __import__("numpy").ones(
        (rows * (thumb_h + label_h), cols * thumb_w, 3),
        dtype="uint8"
    )
)

for i, path in enumerate(files):
    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (thumb_w, thumb_h))

    r = i // cols
    c = i % cols

    y = r * (thumb_h + label_h)
    x = c * thumb_w

    sheet[y:y + thumb_h, x:x + thumb_w] = img

    cv2.putText(
        sheet,
        f"View {i}",
        (x + 5, y + thumb_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1
    )

output_path = os.path.join(OUTPUT_DIR, "contact_sheet.jpg")
cv2.imwrite(output_path, sheet)

print(f"Contact sheet saved: {output_path}")