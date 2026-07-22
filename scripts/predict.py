import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)

from models.csrnet import CSRNet

# ===================================================
# DEVICE
# ===================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ===================================================
# PATHS
# ===================================================
model_path = os.path.join(
    BASE_DIR,
    "checkpoints",
    "final_csrnet.pth"
)

test_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "test_data",
    "images"
)

gt_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "test_data",
    "ground-truth"
)

output_dir = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(output_dir, exist_ok=True)

# ===================================================
# CHECK PATHS
# ===================================================
print("Model Exists:", os.path.exists(model_path))
print("Test Images Exists:", os.path.exists(test_dir))
print("GT Exists:", os.path.exists(gt_dir))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test image folder not found: {test_dir}")

if not os.path.exists(gt_dir):
    raise FileNotFoundError(f"GT folder not found: {gt_dir}")

# ===================================================
# LOAD MODEL
# ===================================================
model = CSRNet().to(device)

state_dict = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("Model Loaded Successfully!")

# ===================================================
# LOAD TEST IMAGES
# ===================================================
image_files = [
    f for f in os.listdir(test_dir)
    if f.endswith(".jpg")
]

image_files.sort()

image_files = image_files[:20]

# ===================================================
# EVALUATION VARIABLES
# ===================================================
total_abs_error = 0.0
total_sq_error = 0.0
valid_images = 0

print("\nStarting Evaluation...")

# ===================================================
# PROCESS TEST IMAGES
# ===================================================
for img_name in image_files:

    image_path = os.path.join(test_dir, img_name)

    print("\nProcessing:", img_name)

    # -----------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found, skipping.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------
    # PREPROCESS IMAGE
    # -----------------------------------------------
    img_tensor = img_rgb.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)

    img_tensor = img_tensor.float().to(device)

    # -----------------------------------------------
    # PREDICT
    # -----------------------------------------------
    with torch.no_grad():
        output = model(img_tensor)

    density_map = output.squeeze().cpu().numpy()

    pred_count = float(density_map.sum())

    # -----------------------------------------------
    # LOAD GROUND TRUTH
    # -----------------------------------------------
    gt_name = "GT_" + img_name.replace(".jpg", ".mat")

    gt_path = os.path.join(gt_dir, gt_name)

    if not os.path.exists(gt_path):
        print("GT file not found, skipping:", gt_path)
        continue

    mat = loadmat(gt_path)

    points = mat["image_info"][0, 0][0, 0][0]

    gt_count = len(points)

    # -----------------------------------------------
    # ERRORS
    # -----------------------------------------------
    abs_error = abs(pred_count - gt_count)
    sq_error = (pred_count - gt_count) ** 2

    total_abs_error += abs_error
    total_sq_error += sq_error
    valid_images += 1

    # -----------------------------------------------
    # PRINT DETAILS
    # -----------------------------------------------
    print("Output Shape:", output.shape)
    print("Density Min:", density_map.min())
    print("Density Max:", density_map.max())
    print(f"Predicted Count: {pred_count:.2f}")
    print("GT Count:", gt_count)
    print(f"Error: {abs_error:.2f}")

    # -----------------------------------------------
    # DISPLAY / SAVE RESULT
    # -----------------------------------------------
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"{img_name}\nGT Count: {gt_count}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap="jet")
    plt.title(f"Predicted Count: {pred_count:.2f}")
    plt.axis("off")
    plt.colorbar()

    plt.tight_layout()

    save_name = img_name.replace(".jpg", "_prediction.png")

    save_path = os.path.join(output_dir, save_name)

    plt.savefig(save_path)

    plt.close()

# ===================================================
# FINAL METRICS
# ===================================================
if valid_images > 0:

    mae = total_abs_error / valid_images
    mse = (total_sq_error / valid_images) ** 0.5

    print("\n========== FINAL EVALUATION ==========")
    print("Images Evaluated:", valid_images)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {mse:.2f}")

else:

    print("No valid images evaluated.")