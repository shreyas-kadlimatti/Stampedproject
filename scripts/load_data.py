# ===================================================
# SHANGHAITECH HEAD ANNOTATION VISUALIZATION
# ===================================================

import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os

# PATHS
image_path = "/content/Stampede/dataset/ShanghaiTech/part_A/test_data/images/IMG_1.jpg"

gt_path = "/content/Stampede/dataset/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_1.mat"

# LOAD IMAGE
img = cv2.imread(image_path)

if img is None:
    print("IMAGE NOT FOUND")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Image Shape:", img.shape)


# LOAD GROUND TRUTH

mat = loadmat(gt_path)


# EXTRACT COORDINATES
points = mat["image_info"][0,0][0,0][0]

# FORCE FLOAT ARRAY
points = np.array(points, dtype=np.float32)
head_count = len(points)

print("Total Head Count:", head_count)

print("\nPOINTS SHAPE:")
print(points.shape)

print("\nFIRST 5 POINTS:")
print(points[:5])

# DISPLAY IMAGE + HEAD POINTS
plt.figure(figsize=(6,6))

plt.imshow(img)

# DRAW RED DOTS
for i in range(len(points)):

    x = points[i][0]
    y = points[i][1]

    plt.scatter(
        x,
        y,
        color='red',
        s=25
    )

plt.title("ShanghaiTech Head Annotations")

plt.axis("off")

plt.show()