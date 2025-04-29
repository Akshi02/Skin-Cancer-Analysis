import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# === Von Luschan reference RGB values ===
von_luschan_rgb = [
    (1,  (244, 242, 245)), (2,  (236, 235, 233)), (3,  (250, 249, 247)), (4,  (253, 251, 230)),
    (5,  (253, 246, 230)), (6,  (254, 247, 229)), (7,  (250, 240, 239)), (8,  (243, 234, 229)),
    (9,  (244, 241, 234)), (10, (251, 252, 244)), (11, (252, 248, 237)), (12, (254, 246, 225)),
    (13, (255, 249, 225)), (14, (255, 249, 225)), (15, (241, 231, 195)), (16, (239, 226, 173)),
    (17, (224, 210, 147)), (18, (242, 226, 151)), (19, (235, 214, 159)), (20, (235, 217, 133)),
    (21, (227, 196, 103)), (22, (225, 193, 106)), (23, (223, 193, 123)), (24, (222, 184, 119)),
    (25, (199, 164, 100)), (26, (188, 151,  98)), (27, (156, 107,  67)), (28, (142,  88,  62)),
    (29, (121,  77,  48)), (30, (100,  49,  22)), (31, (101,  48,  32)), (32, ( 96,  49,  33)),
    (33, ( 87,  50,  41)), (34, ( 64,  32,  21)), (35, ( 49,  37,  44)), (36, ( 27,  28,  44)),
]

def rgb_to_von_luschan(rgb_val):
    distances = [np.linalg.norm(np.array(rgb_val) - np.array(rgb)) for _, rgb in von_luschan_rgb]
    return von_luschan_rgb[np.argmin(distances)][0]

def von_luschan_to_fitzpatrick(index):
    if index <= 6:
        return "Type I (FST1)"
    elif index <= 13:
        return "Type II (FST2)"
    elif index <= 20:
        return "Type III (FST3)"
    elif index <= 27:
        return "Type IV (FST4)"
    elif index <= 34:
        return "Type V (FST5)"
    else:
        return "Type VI (FST6)"

def extract_avg_skin_color(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(img_rgb, (15, 15), 0)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    masked_rgb = cv2.bitwise_and(blurred, blurred, mask=mask)
    non_zero_pixels = masked_rgb[mask != 0]
    if len(non_zero_pixels) == 0:
        return None
    avg_rgb = np.mean(non_zero_pixels, axis=0)
    return avg_rgb

# === Main Pipeline ===

# Input dataset directory (change this if needed)
dataset_path = "../Dataset"  # relative to your script

# Output directory
output_dir = "Clustered_Dataset"
os.makedirs(output_dir, exist_ok=True)

# Supported extensions
exts = (".jpg", ".jpeg", ".png")

# Process all images
for root, _, files in os.walk(dataset_path):
    for fname in tqdm(files):
        if not fname.lower().endswith(exts):
            continue
        img_path = os.path.join(root, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        avg_rgb = extract_avg_skin_color(img)
        if avg_rgb is None:
            continue

        luschan_index = rgb_to_von_luschan(avg_rgb)
        fitz_label = von_luschan_to_fitzpatrick(luschan_index)

        save_dir = os.path.join(output_dir, fitz_label)
        os.makedirs(save_dir, exist_ok=True)

        shutil.copy2(img_path, os.path.join(save_dir, fname))

print(f"\n✅ Images grouped by Fitzpatrick skin type into '{output_dir}'")
