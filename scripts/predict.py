import os
import sys

# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from models.csrnet import CSRNet




# ===================================================
# SETTINGS
# ===================================================

# Set to None for complete test-set evaluation.
# Set to an integer such as 20 only for quick testing.
MAX_IMAGES = None

# Saving 182 plots takes time and storage.
SAVE_VISUALIZATIONS = True

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)

# ===================================================
# DEVICE
# ===================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
elif device.type == "mps":
    print("GPU: Apple Silicon (Metal)")



# ===================================================
# LOAD MODEL
# ===================================================
model = CSRNet().to(device)

model_path = os.path.join(
    BASE_DIR,
    "checkpoints",
    "final_csrnet.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!")

# ===================================================
# DATASET PATHS
# ===================================================
model_path = os.path.join(
    BASE_DIR,
    "checkpoints",
    "final_csrnet.pth"
)

test_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech Dataset",
    "ShanghaiTech",
    "part_A",
    "test_data",
    "images"
)

gt_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech Dataset",
    "ShanghaiTech",
    "part_A",
    "test_data",
    "ground-truth"
)

# ===================================================
# OUTPUT DIRECTORY
# ===================================================
output_dir = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(output_dir, exist_ok=True)

# ===================================================
# LOAD TEST IMAGES
# ===================================================
image_files = sorted(
    os.listdir(test_dir)
)[:20]

# ===================================================
# PROCESS TEST IMAGES
# ===================================================
for img_name in image_files:

    # ===================================================
    # IMAGE PATH
    # ===================================================
    image_path = os.path.join(
        test_dir,
        img_name
    )

    print("\nProcessing:", img_name)

    # ===================================================
    # LOAD IMAGE
    # ===================================================
    img = cv2.imread(image_path)

    if img is None:

        print("Image not found!")
        continue

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # ===================================================
    # PREPROCESS IMAGE
    # ===================================================
    img_tensor = img_rgb.astype(
        np.float32
    ) / 255.0

    img_tensor = torch.tensor(
        img_tensor
    ).permute(2,0,1).unsqueeze(0)

    img_tensor = img_tensor.to(device)

    # ===================================================
    # MODEL PREDICTION
    # ===================================================
    with torch.no_grad():

        output = model(img_tensor)

    density_map = output.squeeze().cpu().numpy()

    # ===================================================
    # CROWD COUNT
    # ===================================================
    predicted_count = density_map.sum()

    # ===================================================
    # LOAD GROUND TRUTH
    # ===================================================
    gt_name = "GT_" + img_name.replace(
        ".jpg",
        ".mat"
    )

    gt_path = os.path.join(
        gt_dir,
        gt_name
    )

    mat = loadmat(gt_path)

    points = mat["image_info"][0,0][0,0][0]

    actual_count = len(points)

    # ===================================================
    # DEBUG INFO
    # ===================================================
    print(
        "Output Shape:",
        output.shape
    )

    print(
        "Density Min:",
        density_map.min()
    )

    print(
        "Density Max:",
        density_map.max()
    )

    print(
        "Actual Count:",
        actual_count
    )

    print(
        "Predicted Count:",
        int(predicted_count)
    )

    print(
        "Error:",
        abs(
            int(predicted_count) - actual_count
        )
    )

    # ===================================================
    # DISPLAY RESULTS
    # ===================================================
    plt.figure(figsize=(14,6))

    # ORIGINAL IMAGE
    plt.subplot(1,2,1)

    plt.imshow(img_rgb)

    plt.title(
        f"{img_name}\nActual Count: {actual_count}"
    )

    plt.axis("off")

    # ===================================================
    # DENSITY MAP
    # ===================================================
    plt.subplot(1,2,2)

    plt.imshow(
        density_map,
        cmap='jet'
    )

    plt.title(
        f"Predicted Count: {int(predicted_count)}"
    )

    plt.axis("off")

    plt.colorbar()

    plt.tight_layout()

    # ===================================================
    # SAVE OUTPUT IMAGE
    # ===================================================
    save_path = os.path.join(
        output_dir,
        f"{img_name}.png"
    )

    plt.savefig(save_path)

    plt.close()

print("\nPrediction Completed!")