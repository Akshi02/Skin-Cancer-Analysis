import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Step 1: Load Images from Dataset Folder ---
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

# --- Step 2: Extract Average Skin Color (Ignoring Mole) ---
def extract_avg_skin_color(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Blur to reduce mole influence
    blurred = cv2.GaussianBlur(img_rgb, (15, 15), 0)

    # Create mask to exclude dark regions
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Apply mask
    masked_rgb = cv2.bitwise_and(blurred, blurred, mask=mask)

    # Extract non-zero pixels
    non_zero_pixels = masked_rgb[mask != 0]

    # If no skin detected, return a placeholder
    if len(non_zero_pixels) == 0:
        return np.array([0, 0, 0])

    avg_rgb = np.mean(non_zero_pixels, axis=0)
    return avg_rgb

# --- Step 3: Main Processing ---
dataset_path = "../Dataset"

images, filenames = load_images_from_folder(dataset_path)

avg_colors = []
for img in images:
    avg_rgb = extract_avg_skin_color(img)
    avg_colors.append(avg_rgb)

avg_colors = np.array(avg_colors)

print(f"Extracted average skin colors for {len(avg_colors)} images.")

# --- Step 4: Clustering ---
num_clusters = 3  # Change based on how many groups you want
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(avg_colors)

# --- Step 5: Visualization ---
# Show color centers
cluster_centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 4))
for idx, center in enumerate(cluster_centers):
    patch = np.ones((50, 100, 3), dtype=np.uint8) * np.uint8(center)
    plt.subplot(1, num_clusters, idx + 1)
    plt.imshow(patch)
    plt.axis('off')
    plt.title(f"Cluster {idx+1}")
plt.suptitle("Skin Tone Clusters", fontsize=16)
plt.show()

'''
# --- Step 6: (Optional) Save Clustering Info ---
import pandas as pd

df = pd.DataFrame({
    'filename': filenames,
    'cluster_label': labels
})

# Save as CSV
df.to_csv('skin_tone_clusters.csv', index=False)
print("Saved skin tone clustering results to 'skin_tone_clusters.csv'.")
'''
