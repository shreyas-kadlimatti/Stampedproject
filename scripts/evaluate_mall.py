import os
import re
import sys
import csv

import cv2
import torch
import numpy as np
from scipy.io import loadmat


# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.csrnet import CSRNet


# ===================================================
# SETTINGS
# ===================================================
# Use 100 for your current cross-dataset test.
# Use None to evaluate every available Mall frame.
MAX_IMAGES = 100

# True = select 100 evenly distributed frames.
# False = select the first 100 frames.
USE_EVEN_SAMPLING = True


# ===================================================
# PATHS - mac
# ===================================================
MODEL_PATH = r"/Users/shreyaskadlimatti/Documents/stampede/project_stamped/checkpoints/final_csrnet.pth"

DATASET_ROOT = r"/Users/shreyaskadlimatti/Documents/stampede/project_stamped/dataset"

OUTPUT_DIR = r"/Users/shreyaskadlimatti/Documents/stampede/project_stamped/outputs"

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "mall_cross_dataset_results.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ===================================================
# AUTOMATIC MALL DATASET PATH DETECTION
# ===================================================
def find_mall_image_directory(root_directory):
    """
    Finds the directory containing seq_*.jpg Mall frames.
    """

    for current_root, directories, files in os.walk(
        root_directory
    ):
        mall_images = [
            file_name
            for file_name in files
            if (
                file_name.lower().startswith("seq_")
                and file_name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            )
        ]

        if mall_images:
            return current_root

    raise FileNotFoundError(
        f"Could not find Mall images under: "
        f"{root_directory}"
    )


def find_mall_ground_truth(root_directory):
    """
    Finds mall_gt.mat anywhere inside the dataset folder.
    """

    for current_root, directories, files in os.walk(
        root_directory
    ):
        for file_name in files:
            if file_name.lower() == "mall_gt.mat":
                return os.path.join(
                    current_root,
                    file_name
                )

    raise FileNotFoundError(
        f"Could not find mall_gt.mat under: "
        f"{root_directory}"
    )


MALL_IMAGE_DIR = find_mall_image_directory(
    DATASET_ROOT
)

MALL_GT_PATH = find_mall_ground_truth(
    DATASET_ROOT
)

print(
    "Detected Mall Image Directory:",
    MALL_IMAGE_DIR
)

print(
    "Detected Mall GT File:",
    MALL_GT_PATH
)


# ===================================================
# IMAGENET NORMALIZATION
# ===================================================
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
# VERIFY PATHS
# ===================================================
print("Model Exists:", os.path.isfile(MODEL_PATH))
print(
    "Mall Images Exist:",
    os.path.isdir(MALL_IMAGE_DIR)
)
print(
    "Mall GT Exists:",
    os.path.isfile(MALL_GT_PATH)
)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.isdir(MALL_IMAGE_DIR):
    raise FileNotFoundError(
        f"Mall image folder not found: "
        f"{MALL_IMAGE_DIR}"
    )

if not os.path.isfile(MALL_GT_PATH):
    raise FileNotFoundError(
        f"Mall ground truth not found: "
        f"{MALL_GT_PATH}"
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
# LOAD MODEL CHECKPOINT
# ===================================================
def load_checkpoint(
    model,
    checkpoint_path,
    map_location
):
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
                "Saved Best Validation MAE:",
                checkpoint["best_val_mae"]
            )

        elif "val_mae" in checkpoint:
            print(
                "Saved Validation MAE:",
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
            "Invalid checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        clean_key = (
            key[len("module."):]
            if key.startswith("module.")
            else key
        )

        cleaned_state_dict[
            clean_key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Model checkpoint loaded successfully.")


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
            f"Could not read image: {image_path}"
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

    return image_tensor


# ===================================================
# MALL GT HELPERS
# ===================================================
def unwrap_mat_value(value):
    """
    Repeatedly unwrap one-element MATLAB arrays.
    """
    while (
        isinstance(value, np.ndarray)
        and value.size == 1
    ):
        value = value.flat[0]

    return value


def extract_points_from_frame(frame_item):
    """
    Supports common Mall Dataset structures such as:

        frame[i][0][0][0][0]
        frame[i].loc
        structured arrays containing a 'loc' field
    """
    frame_item = unwrap_mat_value(
        frame_item
    )

    # scipy mat_struct style
    if hasattr(frame_item, "loc"):
        points = frame_item.loc
        points = unwrap_mat_value(points)

        return np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 2)

    # NumPy structured-array style
    if (
        isinstance(frame_item, np.void)
        and frame_item.dtype.names
        and "loc" in frame_item.dtype.names
    ):
        points = frame_item["loc"]
        points = unwrap_mat_value(points)

        return np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 2)

    if (
        isinstance(frame_item, np.ndarray)
        and frame_item.dtype.names
        and "loc" in frame_item.dtype.names
    ):
        points = frame_item["loc"]
        points = unwrap_mat_value(points)

        return np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 2)

    # Common nested MATLAB-cell fallback
    value = frame_item

    for _ in range(8):
        value = unwrap_mat_value(value)

        array = np.asarray(value)

        if (
            array.ndim == 2
            and array.shape[1] == 2
            and np.issubdtype(
                array.dtype,
                np.number
            )
        ):
            return array.astype(
                np.float32
            )

        if (
            isinstance(value, np.ndarray)
            and value.size > 0
        ):
            value = value.flat[0]
        else:
            break

    raise ValueError(
        "Could not extract Mall head coordinates "
        "from this frame annotation."
    )


def load_mall_ground_truth(gt_path):
    """
    Returns a list of head-position arrays,
    one array for each Mall frame.
    """
    mat = loadmat(
        gt_path,
        squeeze_me=True,
        struct_as_record=False
    )

    public_keys = [
        key
        for key in mat.keys()
        if not key.startswith("__")
    ]

    print(
        "Ground-truth MAT keys:",
        public_keys
    )

    if "frame" not in mat:
        raise KeyError(
            "Expected key 'frame' was not found in "
            f"{gt_path}. Available keys: {public_keys}"
        )

    frames = np.atleast_1d(
        mat["frame"]
    ).reshape(-1)

    annotations = []

    for frame_index, frame_item in enumerate(
        frames,
        start=1
    ):
        try:
            points = extract_points_from_frame(
                frame_item
            )

        except Exception as error:
            raise ValueError(
                f"Could not read annotation for Mall "
                f"frame {frame_index}: {error}"
            ) from error

        annotations.append(
            points
        )

    return annotations


# ===================================================
# LOAD MODEL
# ===================================================
model = CSRNet().to(device)

load_checkpoint(
    model=model,
    checkpoint_path=MODEL_PATH,
    map_location=device
)

model.eval()


# ===================================================
# LOAD MALL ANNOTATIONS
# ===================================================
mall_annotations = load_mall_ground_truth(
    MALL_GT_PATH
)

print(
    "Mall Annotation Frames:",
    len(mall_annotations)
)


# ===================================================
# LOAD IMAGE FILES
# ===================================================
image_files = [
    file_name
    for file_name in os.listdir(
        MALL_IMAGE_DIR
    )
    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

image_files.sort(
    key=natural_sort_key
)

print(
    "Mall Image Frames:",
    len(image_files)
)

usable_count = min(
    len(image_files),
    len(mall_annotations)
)

image_files = image_files[
    :usable_count
]

mall_annotations = mall_annotations[
    :usable_count
]

if len(image_files) != len(mall_annotations):
    raise RuntimeError(
        "Image and annotation counts could not "
        "be aligned."
    )


# ===================================================
# SELECT 100 FRAMES
# ===================================================
all_indices = np.arange(
    usable_count
)

if (
    MAX_IMAGES is not None
    and usable_count > MAX_IMAGES
):
    if USE_EVEN_SAMPLING:
        selected_indices = np.linspace(
            0,
            usable_count - 1,
            num=MAX_IMAGES,
            dtype=int
        )
    else:
        selected_indices = all_indices[
            :MAX_IMAGES
        ]
else:
    selected_indices = all_indices

print(
    "Frames Selected for Evaluation:",
    len(selected_indices)
)


# ===================================================
# EVALUATION
# ===================================================
signed_errors = []
percentage_errors = []
results = []

print(
    "\nStarting Mall Cross-Dataset Evaluation...\n"
)

for evaluation_number, dataset_index in enumerate(
    selected_indices,
    start=1
):
    image_name = image_files[
        dataset_index
    ]

    image_path = os.path.join(
        MALL_IMAGE_DIR,
        image_name
    )

    points = mall_annotations[
        dataset_index
    ]

    gt_count = int(
        len(points)
    )

    try:
        image_tensor = preprocess_image(
            image_path
        )

        image_tensor = image_tensor.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True
        )

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

        pred_count = float(
            density_map.sum()
        )

        signed_error = (
            pred_count - gt_count
        )

        absolute_error = abs(
            signed_error
        )

        percentage_error = (
            absolute_error
            / max(gt_count, 1)
        ) * 100.0

        signed_errors.append(
            signed_error
        )

        percentage_errors.append(
            percentage_error
        )

        results.append(
            {
                "dataset_index": int(
                    dataset_index + 1
                ),
                "image": image_name,
                "ground_truth": gt_count,
                "prediction": pred_count,
                "signed_error": signed_error,
                "absolute_error": absolute_error,
                "percentage_error": percentage_error
            }
        )

        print(
            f"[{evaluation_number}/"
            f"{len(selected_indices)}] "
            f"{image_name} | "
            f"GT: {gt_count} | "
            f"Pred: {pred_count:.2f} | "
            f"Error: {absolute_error:.2f}"
        )

    except Exception as error:
        print(
            f"Failed to process "
            f"{image_name}: {error}"
        )


# ===================================================
# FINAL METRICS
# ===================================================
if not signed_errors:
    raise RuntimeError(
        "No Mall images were evaluated."
    )

errors_array = np.asarray(
    signed_errors,
    dtype=np.float64
)

percentage_array = np.asarray(
    percentage_errors,
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

mape = float(
    np.mean(percentage_array)
)

# Informal percentage only.
# MAE and RMSE remain the standard metrics.
average_percentage_accuracy = max(
    0.0,
    100.0 - mape
)


# ===================================================
# SAVE CSV
# ===================================================
with open(
    RESULTS_CSV,
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "dataset_index",
            "image",
            "ground_truth",
            "prediction",
            "signed_error",
            "absolute_error",
            "percentage_error"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ===================================================
# PRINT RESULTS
# ===================================================
print(
    "\n=========================================="
)
print(
    "MALL CROSS-DATASET EVALUATION"
)
print(
    "=========================================="
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Images Evaluated:",
    len(errors_array)
)

print(
    f"MAE: {mae:.2f}"
)

print(
    f"RMSE: {rmse:.2f}"
)

print(
    f"Mean Signed Error: "
    f"{mean_signed_error:.2f}"
)

print(
    f"MAPE: {mape:.2f}%"
)

print(
    f"Approximate Percentage Accuracy: "
    f"{average_percentage_accuracy:.2f}%"
)

if mean_signed_error > 0:
    print(
        "Model tendency: Overcounting"
    )
elif mean_signed_error < 0:
    print(
        "Model tendency: Undercounting"
    )
else:
    print(
        "Model tendency: No overall bias"
    )

print(
    "Results CSV:",
    RESULTS_CSV
)

print(
    "=========================================="
)