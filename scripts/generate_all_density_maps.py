import os
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os
import sys

# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# ===================================================
# ADD ROOT TO PYTHON PATH
# ===================================================
sys.path.append(BASE_DIR)
# ===================================================
# DATASET PATHS
# ===================================================
image_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "train_data",
    "images"
)

gt_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "train_data",
    "ground-truth"
)

output_dir = os.path.join(
    BASE_DIR,
    "density_maps"
)

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# ===================================================
# GET IMAGE LIST
# ===================================================
image_files = [f for f in os.listdir(image_dir)
               if f.endswith(".jpg")]

print("Total Images:", len(image_files))

# ===================================================
# PROCESS ALL IMAGES
# ===================================================
for img_name in tqdm(image_files):

    # -----------------------------------------------
    # IMAGE PATH
    # -----------------------------------------------
    img_path = os.path.join(image_dir, img_name)

    # -----------------------------------------------
    # GT PATH
    # -----------------------------------------------
    gt_name = "GT_" + img_name.replace(".jpg", ".mat")

    gt_path = os.path.join(gt_dir, gt_name)

    # -----------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------
    img = cv2.imread(img_path)

    if img is None:
        continue

    height, width, _ = img.shape

    # -----------------------------------------------
    # LOAD ANNOTATIONS
    # -----------------------------------------------
    mat = loadmat(gt_path)

    points = mat["image_info"][0,0][0,0][0]

    points = np.array(points, dtype=np.float32)

    # -----------------------------------------------
    # CREATE DENSITY MAP
    # -----------------------------------------------
    density = np.zeros((height, width),
                       dtype=np.float32)

    for p in points:

        x = min(width - 1, max(0, int(p[0])))
        y = min(height - 1, max(0, int(p[1])))

        density[y, x] = 1

    # -----------------------------------------------
    # APPLY GAUSSIAN FILTER
    # -----------------------------------------------
    density = gaussian_filter(density,
                              sigma=4)

    # -----------------------------------------------
    # SAVE DENSITY MAP
    # -----------------------------------------------
    save_name = img_name.replace(".jpg", ".npy")

    save_path = os.path.join(output_dir,
                             save_name)

    np.save(save_path, density)

print("\nALL DENSITY MAPS GENERATED")