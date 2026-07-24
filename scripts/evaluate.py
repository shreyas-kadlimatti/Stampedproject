import os
import sys
import cv2
import torch
import numpy as np
from scipy.io import loadmat

# ===================================================
# PROJECT ROOT
# ===================================================
try:
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
except NameError:
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)

from models.csrnet import CSRNet

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
# ===================================================
# datasets PATHS
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

checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

clean_state = {}

for key, value in state_dict.items():
    if key.startswith("module."):
        clean_state[key[7:]] = value
    else:
        clean_state[key] = value

model.load_state_dict(clean_state, strict=True)

model.eval()

print("Model Loaded Successfully!")

# ===================================================
# TEST IMAGE FILES
# ===================================================
image_files = sorted([
    f for f in os.listdir(test_dir)
    if f.endswith(".jpg")
])

# For only first 20 images, uncomment this:
# image_files = image_files[:20]

# ===================================================
# METRIC LISTS
# ===================================================
absolute_errors = []
squared_errors = []
accuracies = []

print("\nStarting Evaluation...\n")

# ===================================================
# EVALUATION LOOP
# ===================================================
for img_name in image_files:

    image_path = os.path.join(test_dir, img_name)

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found:", img_name)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = img_rgb.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)

    img_tensor = img_tensor.float().to(device)

    # ===================================================
    # PREDICTION
    # ===================================================
    with torch.no_grad():
        output = model(img_tensor)

    density_map = output.squeeze().cpu().numpy()

    pred_count = float(density_map.sum())

    # ===================================================
    # LOAD GROUND TRUTH
    # ===================================================
    gt_name = "GT_" + img_name.replace(".jpg", ".mat")

    gt_path = os.path.join(gt_dir, gt_name)

    if not os.path.exists(gt_path):
        print("GT not found:", gt_name)
        continue

    mat = loadmat(gt_path)

    points = mat["image_info"][0, 0][0, 0][0]

    gt_count = len(points)

    # ===================================================
    # ERROR METRICS
    # ===================================================
    error = abs(pred_count - gt_count)

    squared_error = (pred_count - gt_count) ** 2

    accuracy = max(
        0,
        100 - (error / max(gt_count, 1)) * 100
    )

    absolute_errors.append(error)
    squared_errors.append(squared_error)
    accuracies.append(accuracy)

    print(
        f"{img_name} | "
        f"GT: {gt_count:.0f} | "
        f"Pred: {pred_count:.2f} | "
        f"Error: {error:.2f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

# ===================================================
# FINAL RESULTS
# ===================================================
mae = np.mean(absolute_errors)
rmse = np.sqrt(np.mean(squared_errors))
avg_accuracy = np.mean(accuracies)

print("\n==============================")
print("Evaluation Results")
print("==============================")
print("Total Test Images:", len(absolute_errors))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Average Accuracy: {avg_accuracy:.2f}%")
print("==============================")