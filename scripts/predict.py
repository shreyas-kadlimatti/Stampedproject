import os
import re
import sys
import csv

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat


# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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

visualization_dir = os.path.join(
    output_dir,
    "test_predictions"
)

results_csv_path = os.path.join(
    output_dir,
    "test_results.csv"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

if SAVE_VISUALIZATIONS:
    os.makedirs(
        visualization_dir,
        exist_ok=True
    )


# ===================================================
# CHECK PATHS
# ===================================================
print("Model Path:", model_path)
print("Model Exists:", os.path.isfile(model_path))
print("Test Images Exist:", os.path.isdir(test_dir))
print("GT Folder Exists:", os.path.isdir(gt_dir))

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )

if not os.path.isdir(test_dir):
    raise FileNotFoundError(
        f"Test image folder not found: {test_dir}"
    )

if not os.path.isdir(gt_dir):
    raise FileNotFoundError(
        f"Ground-truth folder not found: {gt_dir}"
    )


# ===================================================
# NATURAL SORT
# ===================================================
def natural_sort_key(file_name):
    """
    Sorts image names numerically.

    Example:
    IMG_1.jpg, IMG_2.jpg, IMG_10.jpg
    instead of:
    IMG_1.jpg, IMG_10.jpg, IMG_2.jpg
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", file_name)
    ]


# ===================================================
# LOAD CHECKPOINT
# ===================================================
def load_checkpoint(model, checkpoint_path, map_location):
    print("\nLoading checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False
    )

    # Full training checkpoint
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

        if "epoch" in checkpoint:
            print(
                "Saved Epoch:",
                checkpoint["epoch"]
            )

        if "best_val_mae" in checkpoint:
            print(
                "Saved Best Validation MAE:",
                checkpoint["best_val_mae"]
            )

        elif "val_mae" in checkpoint:
            print(
                "Saved Validation MAE:",
                checkpoint["val_mae"]
            )

    # Alternative checkpoint format
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    # Plain model.state_dict()
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The checkpoint does not contain a valid state dictionary."
        )

    # Remove DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            clean_key = key[len("module."):]
        else:
            clean_key = key

        cleaned_state_dict[clean_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Checkpoint Loaded Successfully!")


# ===================================================
# IMAGE PREPROCESSING
# ===================================================
def preprocess_image(image_path):
    """
    Loads and normalizes an image exactly as required by
    the pretrained VGG16 frontend.
    """
    image_bgr = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_array = image_rgb.astype(
        np.float32
    ) / 255.0

    image_array = (
        image_array - IMAGENET_MEAN
    ) / IMAGENET_STD

    image_array = np.ascontiguousarray(
        image_array.transpose(2, 0, 1)
    )

    image_tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0)

    return image_rgb, image_tensor


# ===================================================
# LOAD GROUND-TRUTH COUNT
# ===================================================
def load_ground_truth_count(gt_path):
    """
    Loads ShanghaiTech Part A head annotations.

    Ground-truth crowd count is the number of annotated
    head coordinates.
    """
    mat_data = loadmat(gt_path)

    if "image_info" not in mat_data:
        raise KeyError(
            f"'image_info' not found in: {gt_path}"
        )

    points = mat_data[
        "image_info"
    ][0, 0][0, 0][0]

    return int(len(points))


# ===================================================
# LOAD MODEL
# ===================================================
# pretrained=False is important during prediction.
# The trained checkpoint already contains frontend weights.
model = CSRNet(
    pretrained=False
).to(device)

load_checkpoint(
    model=model,
    checkpoint_path=model_path,
    map_location=device
)

model.eval()

print("Model Ready for Evaluation!")


# ===================================================
# LOAD TEST IMAGE NAMES
# ===================================================
image_files = [
    file_name
    for file_name in os.listdir(test_dir)
    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

image_files.sort(
    key=natural_sort_key
)

if MAX_IMAGES is not None:
    image_files = image_files[:MAX_IMAGES]

print(
    "\nNumber of Images Selected:",
    len(image_files)
)


# ===================================================
# EVALUATION VARIABLES
# ===================================================
errors = []
results = []

print("\nStarting Test Evaluation...")


# ===================================================
# PROCESS TEST IMAGES
# ===================================================
for image_index, img_name in enumerate(
    image_files,
    start=1
):
    image_path = os.path.join(
        test_dir,
        img_name
    )

    image_stem = os.path.splitext(
        img_name
    )[0]

    gt_name = f"GT_{image_stem}.mat"

    gt_path = os.path.join(
        gt_dir,
        gt_name
    )

    print(
        f"\n[{image_index}/{len(image_files)}] "
        f"Processing: {img_name}"
    )

    if not os.path.isfile(gt_path):
        print(
            "Ground-truth file not found. Skipping:",
            gt_path
        )
        continue

    try:
        # -------------------------------------------
        # IMAGE PREPROCESSING
        # -------------------------------------------
        image_rgb, image_tensor = preprocess_image(
            image_path
        )

        image_tensor = image_tensor.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True
        )

        # -------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------
        with torch.inference_mode():
            output = model(image_tensor)

        # Convert output to float32 before CPU/NumPy.
        density_map = (
            output[0, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        # Calculate count directly from the original
        # model output. Do not resize before summing.
        pred_count = float(
            density_map.sum()
        )

        # -------------------------------------------
        # GROUND TRUTH
        # -------------------------------------------
        gt_count = load_ground_truth_count(
            gt_path
        )

        # -------------------------------------------
        # ERROR
        # -------------------------------------------
        signed_error = (
            pred_count - gt_count
        )

        abs_error = abs(
            signed_error
        )

        errors.append(
            signed_error
        )

        results.append(
            {
                "image": img_name,
                "ground_truth": gt_count,
                "prediction": pred_count,
                "absolute_error": abs_error
            }
        )

        # -------------------------------------------
        # PRINT RESULT
        # -------------------------------------------
        print("Input Shape:", tuple(image_tensor.shape))
        print("Output Shape:", tuple(output.shape))
        print(
            f"Density Range: "
            f"{density_map.min():.8f} to "
            f"{density_map.max():.8f}"
        )
        print(f"GT Count: {gt_count}")
        print(f"Predicted Count: {pred_count:.2f}")
        print(f"Absolute Error: {abs_error:.2f}")

        # -------------------------------------------
        # SAVE VISUALIZATION
        # -------------------------------------------
        if SAVE_VISUALIZATIONS:
            figure = plt.figure(
                figsize=(14, 6)
            )

            axis_image = figure.add_subplot(
                1,
                2,
                1
            )

            axis_image.imshow(
                image_rgb
            )

            axis_image.set_title(
                f"{img_name}\n"
                f"Ground Truth: {gt_count}"
            )

            axis_image.axis("off")

            axis_density = figure.add_subplot(
                1,
                2,
                2
            )

            heatmap = axis_density.imshow(
                density_map,
                cmap="jet"
            )

            axis_density.set_title(
                f"Predicted Count: {pred_count:.2f}\n"
                f"Absolute Error: {abs_error:.2f}"
            )

            axis_density.axis("off")

            figure.colorbar(
                heatmap,
                ax=axis_density,
                fraction=0.046,
                pad=0.04
            )

            figure.tight_layout()

            save_path = os.path.join(
                visualization_dir,
                f"{image_stem}_prediction.png"
            )

            figure.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(figure)

    except Exception as error:
        print(
            f"Failed to process {img_name}: {error}"
        )


# ===================================================
# FINAL METRICS
# ===================================================
if errors:
    errors_array = np.asarray(
        errors,
        dtype=np.float64
    )

    mae = float(
        np.mean(
            np.abs(errors_array)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(errors_array)
            )
        )
    )

    mean_signed_error = float(
        np.mean(errors_array)
    )

    print(
        "\n========== FINAL TEST EVALUATION =========="
    )

    print(
        "Images Evaluated:",
        len(errors_array)
    )

    print(
        f"Test MAE: {mae:.2f}"
    )

    print(
        f"Test RMSE: {rmse:.2f}"
    )

    print(
        f"Mean Signed Error: {mean_signed_error:.2f}"
    )

    if mean_signed_error < 0:
        print(
            "Overall tendency: Model is undercounting."
        )
    elif mean_signed_error > 0:
        print(
            "Overall tendency: Model is overcounting."
        )
    else:
        print(
            "Overall tendency: No counting bias."
        )

    # -----------------------------------------------
    # SAVE CSV RESULTS
    # -----------------------------------------------
    with open(
        results_csv_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "ground_truth",
                "prediction",
                "absolute_error"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        "Results CSV:",
        results_csv_path
    )

    if SAVE_VISUALIZATIONS:
        print(
            "Prediction Images:",
            visualization_dir
        )

else:
    print(
        "\nNo valid images were evaluated."
    )