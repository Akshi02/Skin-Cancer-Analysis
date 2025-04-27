# === Full Pipeline: Skin RGB Analysis + Fitzpatrick Mapping ===

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# --- Step 1: Load images ---
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    filenames.append(img_path)
    return images, filenames

# --- Step 2: Extract average skin color (ignore dark moles) ---
def extract_avg_skin_color(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(img_rgb, (15, 15), 0)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    masked_rgb = cv2.bitwise_and(blurred, blurred, mask=mask)
    non_zero_pixels = masked_rgb[mask != 0]

    if len(non_zero_pixels) == 0:
        return np.array([0, 0, 0])
    avg_rgb = np.mean(non_zero_pixels, axis=0)
    return avg_rgb

# --- Step 3: Map RGB centers to Fitzpatrick Skin Types ---
def map_rgb_to_fitzpatrick(center_rgb):
    brightness = np.mean(center_rgb)
    if brightness >= 220:
        return "Type I (FST1): Very fair skin"
    elif brightness >= 190:
        return "Type II (FST2): Fair skin"
    elif brightness >= 160:
        return "Type III (FST3): Medium skin tone"
    elif brightness >= 130:
        return "Type IV (FST4): Olive skin tone"
    elif brightness >= 100:
        return "Type V (FST5): Brown skin"
    else:
        return "Type VI (FST6): Dark brown or black skin"

# --- Main Pipeline ---

# Set your dataset path
dataset_path = "../Dataset"  # Adjust this if needed

# Load images
images, filenames = load_images_from_folder(dataset_path)
print(f"Loaded {len(images)} images.")

# Extract average skin colors
avg_colors = []
for img in images:
    avg_rgb = extract_avg_skin_color(img)
    avg_colors.append(avg_rgb)

avg_colors = np.array(avg_colors)

# Remove empty entries (bad images)
non_empty_indices = [i for i, color in enumerate(avg_colors) if not np.all(color == 0)]
avg_colors = avg_colors[non_empty_indices]
filenames = [filenames[i] for i in non_empty_indices]

print(f"Extracted average skin colors for {len(avg_colors)} images.")

# Cluster the skin tones
num_clusters = 6  # For FST1-FST6
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(avg_colors)
cluster_centers = kmeans.cluster_centers_

# Map cluster centers to Fitzpatrick types
fitzpatrick_labels = [map_rgb_to_fitzpatrick(center) for center in cluster_centers]

print("\nCluster mapping to Fitzpatrick Types:")
for idx, label in enumerate(fitzpatrick_labels):
    print(f"Cluster {idx}: {label}")

# Save clustering results
df = pd.DataFrame({
    'filename': filenames,
    'cluster_label': labels
})
df.to_csv('skin_tone_clusters.csv', index=False)
print("\nSaved clustering results to 'skin_tone_clusters.csv'.")

# --- Plot random sample images per cluster ---
num_samples_per_cluster = 5

for cluster_idx in range(len(cluster_centers)):
    cluster_name = fitzpatrick_labels[cluster_idx]
    cluster_image_indices = np.where(labels == cluster_idx)[0]
    
    if len(cluster_image_indices) == 0:
        continue

    selected_indices = random.sample(list(cluster_image_indices), min(num_samples_per_cluster, len(cluster_image_indices)))

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"{cluster_name} (Cluster {cluster_idx})", fontsize=16)

    for i, idx in enumerate(selected_indices):
        img = cv2.imread(filenames[idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, num_samples_per_cluster, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(os.path.basename(filenames[idx]))

    plt.show()

# --- Group images into folders ---
output_base = "Grouped_Images"
os.makedirs(output_base, exist_ok=True)

for idx, file_path in enumerate(filenames):
    label = labels[idx]
    fitz_name = fitzpatrick_labels[label]

    if "benign" in file_path.lower():
        category = "benign"
    elif "malignant" in file_path.lower():
        category = "malignant"
    else:
        category = "unknown"

    output_folder = os.path.join(output_base, fitz_name, category)
    os.makedirs(output_folder, exist_ok=True)

    filename_only = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename_only)
    shutil.copy2(file_path, output_path)

print(f"\nImages successfully grouped into '{output_base}/' based on Fitzpatrick types and categories.")

# --- Plot SCIN true vs predicted Fitzpatrick distributions ---

# Load SCIN true Fitzpatrick data
real_data = pd.read_csv('/mnt/data/scin_cases.csv')
real_skin_types = real_data['fitzpatrick_skin_type'].dropna()
true_counts = real_skin_types.value_counts()

# Predicted distribution
predicted_fitz_names = [fitzpatrick_labels[label] for label in labels]
pred_counts = pd.Series(predicted_fitz_names).value_counts()

# Align categories
fitz_order = [
    "Type I (FST1): Very fair skin",
    "Type II (FST2): Fair skin",
    "Type III (FST3): Medium skin tone",
    "Type IV (FST4): Olive skin tone",
    "Type V (FST5): Brown skin",
    "Type VI (FST6): Dark brown or black skin"
]

true_counts = true_counts.reindex(fitz_order, fill_value=0)
pred_counts = pred_counts.reindex(fitz_order, fill_value=0)

# Plot side-by-side bar chart
x = np.arange(len(fitz_order))
width = 0.35

fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width/2, true_counts.values, width, label='True (SCIN Dataset)')
rects2 = ax.bar(x + width/2, pred_counts.values, width, label='Predicted (Clustered Images)')

ax.set_ylabel('Number of Samples')
ax.set_title('Fitzpatrick Skin Type Distribution Comparison')
ax.set_xticks(x)
ax.set_xticklabels(fitz_order, rotation=45, ha='right')
ax.legend()

def annotate_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

annotate_bars(rects1)
annotate_bars(rects2)

plt.tight_layout()
plt.show()
