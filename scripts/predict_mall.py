import os
import re
import sys
import csv
import random

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


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
MAX_IMAGES = 150

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
# PATHS
# ===================================================
model_path = os.path.join(
    BASE_DIR,
    "checkpoints",
    "final_csrnet.pth"
)

mall_images_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "mall_dataset",
    "frames"
)

output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "mall_predictions"
)

csv_path = os.path.join(
    BASE_DIR,
    "outputs",
    "mall_predictions.csv"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

os.makedirs(
    os.path.dirname(csv_path),
    exist_ok=True
)


# ===================================================
# CHECK PATHS
# ===================================================
print("Model Path:", model_path)
print("Model Exists:", os.path.isfile(model_path))
print(
    "Mall Images Folder:",
    mall_images_dir
)
print(
    "Mall Images Exist:",
    os.path.isdir(mall_images_dir)
)

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )

if not os.path.isdir(mall_images_dir):
    raise FileNotFoundError(
        f"Mall dataset image folder not found: "
        f"{mall_images_dir}"
    )


# ===================================================
# NATURAL SORT
# ===================================================
def natural_sort_key(file_name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            file_name
        )
    ]


# ===================================================
# LOAD CHECKPOINT
# ===================================================
def load_checkpoint(
    model,
    checkpoint_path,
    map_location
):
    print(
        "\nLoading Checkpoint:",
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

        if "epoch" in checkpoint:
            print(
                "Checkpoint Epoch:",
                checkpoint["epoch"]
            )

        if "best_val_mae" in checkpoint:
            print(
                "Best Validation MAE:",
                checkpoint["best_val_mae"]
            )

        elif "val_mae" in checkpoint:
            print(
                "Validation MAE:",
                checkpoint["val_mae"]
            )

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid "
            "model state dictionary."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Checkpoint Loaded Successfully!")


# ===================================================
# IMAGE PREPROCESSING
# ===================================================
def preprocess_image(image_path):
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
# LOAD MODEL
# ===================================================
model = CSRNet().to(device)

load_checkpoint(
    model=model,
    checkpoint_path=model_path,
    map_location=device
)

model.eval()

print("Model Ready!")


# ===================================================
# LOAD MALL IMAGES
# ===================================================
image_files = [
    file_name
    for file_name in os.listdir(
        mall_images_dir
    )
    if file_name.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
]



image_files.sort(key=natural_sort_key)

if MAX_IMAGES is not None:
    random.seed(42)  # Same random selection every run
    image_files = random.sample(
        image_files,
        min(MAX_IMAGES, len(image_files))
    )
    image_files.sort(key=natural_sort_key)

print(
    "\nMall Images Selected:",
    len(image_files)
)


# ===================================================
# PREDICTION RESULTS
# ===================================================
results = []

print("\nStarting Mall Dataset Prediction...")


# ===================================================
# PROCESS IMAGES
# ===================================================
for index, image_name in enumerate(
    image_files,
    start=1
):
    image_path = os.path.join(
        mall_images_dir,
        image_name
    )

    image_stem = os.path.splitext(
        image_name
    )[0]

    print(
        f"\n[{index}/{len(image_files)}] "
        f"Processing: {image_name}"
    )

    try:
        # -------------------------------------------
        # PREPROCESS IMAGE
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
        # PREDICTION
        # -------------------------------------------
        with torch.inference_mode():
            output = model(
                image_tensor
            )

        density_map = (
            output[0, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        predicted_count = float(
            density_map.sum()
        )

        print(
            "Input Shape:",
            tuple(image_tensor.shape)
        )

        print(
            "Output Shape:",
            tuple(output.shape)
        )

        print(
            f"Predicted Count: "
            f"{predicted_count:.2f}"
        )

        print(
            f"Density Range: "
            f"{density_map.min():.8f} to "
            f"{density_map.max():.8f}"
        )

        results.append(
            {
                "frame": image_name,
                "predicted_count": predicted_count
            }
        )

        # -------------------------------------------
        # SAVE VISUALIZATION
        # -------------------------------------------
        if SAVE_VISUALIZATIONS:
            figure = plt.figure(
                figsize=(16, 6)
            )

            original_axis = figure.add_subplot(
                1,
                3,
                1
            )

            original_axis.imshow(
                image_rgb
            )

            original_axis.set_title(
                f"Original Frame\n{image_name}"
            )

            original_axis.axis("off")

            density_axis = figure.add_subplot(
                1,
                3,
                2
            )

            density_plot = density_axis.imshow(
                density_map,
                cmap="jet"
            )

            density_axis.set_title(
                "Predicted Density Map"
            )

            density_axis.axis("off")

            figure.colorbar(
                density_plot,
                ax=density_axis,
                fraction=0.046,
                pad=0.04
            )

            # ---------------------------------------
            # OVERLAY DENSITY MAP
            # ---------------------------------------
            original_height, original_width = (
                image_rgb.shape[:2]
            )

            resized_density = cv2.resize(
                density_map,
                (
                    original_width,
                    original_height
                ),
                interpolation=cv2.INTER_CUBIC
            )

            density_min = resized_density.min()
            density_max = resized_density.max()

            normalized_density = (
                resized_density - density_min
            ) / (
                density_max
                - density_min
                + 1e-8
            )

            heatmap_uint8 = np.uint8(
                normalized_density * 255
            )

            heatmap_bgr = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET
            )

            heatmap_rgb = cv2.cvtColor(
                heatmap_bgr,
                cv2.COLOR_BGR2RGB
            )

            overlay = cv2.addWeighted(
                image_rgb,
                0.65,
                heatmap_rgb,
                0.35,
                0
            )

            overlay_axis = figure.add_subplot(
                1,
                3,
                3
            )

            overlay_axis.imshow(
                overlay
            )

            overlay_axis.set_title(
                f"Density Overlay\n"
                f"Predicted Count: "
                f"{predicted_count:.2f}"
            )

            overlay_axis.axis("off")

            figure.tight_layout()

            save_path = os.path.join(
                output_dir,
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
            f"Failed to process "
            f"{image_name}: {error}"
        )


# ===================================================
# SAVE CSV
# ===================================================
if results:
    with open(
        csv_path,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "predicted_count"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    predicted_counts = np.array(
        [
            item["predicted_count"]
            for item in results
        ],
        dtype=np.float64
    )

    print(
        "\n========== MALL PREDICTION SUMMARY =========="
    )

    print(
        "Frames Processed:",
        len(results)
    )

    print(
        f"Average Predicted Count: "
        f"{predicted_counts.mean():.2f}"
    )

    print(
        f"Minimum Predicted Count: "
        f"{predicted_counts.min():.2f}"
    )

    print(
        f"Maximum Predicted Count: "
        f"{predicted_counts.max():.2f}"
    )

    print(
        "CSV Results:",
        csv_path
    )

    print(
        "Prediction Images:",
        output_dir
    )

else:
    print(
        "\nNo Mall dataset images were processed."
    )