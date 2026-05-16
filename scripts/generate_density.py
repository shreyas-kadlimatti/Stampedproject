import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# ===================================================
# PATHS
# ===================================================
image_path = "/content/Stampede/dataset/ShanghaiTech/part_A/test_data/images/IMG_50.jpg"
gt_path = r"dataset/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_50.mat"

# ===================================================
# LOAD IMAGE
# ===================================================
img = cv2.imread(image_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, _ = img.shape

# ===================================================
# LOAD HEAD POINTS
# ===================================================
mat = loadmat(gt_path)

points = mat["image_info"][0,0][0,0][0]

points = np.array(points, dtype=np.float32)

print("Head Count:", len(points))

# ===================================================
# CREATE EMPTY DENSITY MAP
# ===================================================
density = np.zeros((height, width), dtype=np.float32)

# ===================================================
# PLACE HEAD POINTS
# ===================================================
for p in points:

    x = min(width-1, max(0, int(p[0])))
    y = min(height-1, max(0, int(p[1])))

    density[y, x] = 1

# ===================================================
# APPLY GAUSSIAN FILTER
# ===================================================
density = gaussian_filter(density, sigma=15)

# ===================================================
# DISPLAY RESULTS
# ===================================================
plt.figure(figsize=(6,6))

plt.imshow(density, cmap='jet')

plt.title("Density Map")

plt.axis("off")

plt.colorbar()

plt.show()

# ===================================================
# ESTIMATED COUNT
# ===================================================
print("Density Sum:", density.sum())