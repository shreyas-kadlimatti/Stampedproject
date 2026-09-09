import os
import re
import sys
import csv

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
    sys.path.insert(
        0,
        BASE_DIR
    )


from models.csrnet import CSRNet


# ===================================================
# SETTINGS
# ===================================================

# None means process ALL Mall dataset frames
MAX_IMAGES = None

# Do not generate thousands of prediction images
SAVE_VISUALIZATIONS = False


# ===================================================
# IMAGENET NORMALIZATION
# ===================================================

IMAGENET_MEAN = np.array(
    [
        0.485,
        0.456,
        0.406
    ],
    dtype=np.float32
).reshape(
    1,
    1,
    3
)


IMAGENET_STD = np.array(
    [
        0.229,
        0.224,
        0.225
    ],
    dtype=np.float32
).reshape(
    1,
    1,
    3
)


# ===================================================
# DEVICE
# ===================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Using Device:",
    device
)


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
    "best_csrnet.pth"
)


# IMPORTANT:
# SAME FOLDER USED BY OPTICAL FLOW
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
    "crowd_features.csv"
)


os.makedirs(
    output_dir,
    exist_ok=True
)


os.makedirs(
    os.path.dirname(
        csv_path
    ),
    exist_ok=True
)


# ===================================================
# CHECK PATHS
# ===================================================

print(
    "Model Path:",
    model_path
)


print(
    "Model Exists:",
    os.path.isfile(
        model_path
    )
)


print(
    "Mall Images Folder:",
    mall_images_dir
)


print(
    "Mall Images Exist:",
    os.path.isdir(
        mall_images_dir
    )
)


if not os.path.isfile(
    model_path
):

    raise FileNotFoundError(
        f"Model not found: "
        f"{model_path}"
    )


if not os.path.isdir(
    mall_images_dir
):

    raise FileNotFoundError(
        f"Mall dataset image folder "
        f"not found: {mall_images_dir}"
    )


# ===================================================
# NATURAL SORT
# ===================================================

def natural_sort_key(
    file_name
):

    return [

        int(part)
        if part.isdigit()
        else part.lower()

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

        isinstance(
            checkpoint,
            dict
        )

        and

        "model_state_dict"
        in checkpoint

    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]


        if "epoch" in checkpoint:

            print(
                "Checkpoint Epoch:",
                checkpoint[
                    "epoch"
                ]
            )


        if (
            "best_val_mae"
            in checkpoint
        ):

            print(
                "Best Validation MAE:",
                checkpoint[
                    "best_val_mae"
                ]
            )


        elif (
            "val_mae"
            in checkpoint
        ):

            print(
                "Validation MAE:",
                checkpoint[
                    "val_mae"
                ]
            )


    elif (

        isinstance(
            checkpoint,
            dict
        )

        and

        "state_dict"
        in checkpoint

    ):

        state_dict = checkpoint[
            "state_dict"
        ]


    else:

        state_dict = checkpoint


    if not isinstance(
        state_dict,
        dict
    ):

        raise TypeError(
            "Checkpoint does not contain "
            "a valid model state dictionary."
        )


    cleaned_state_dict = {}


    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module."):
            ]


        cleaned_state_dict[
            key
        ] = value


    model.load_state_dict(

        cleaned_state_dict,

        strict=True

    )


    print(
        "Checkpoint Loaded Successfully!"
    )


# ===================================================
# IMAGE PREPROCESSING
# ===================================================

def preprocess_image(
    image_path
):

    image_bgr = cv2.imread(

        image_path,

        cv2.IMREAD_COLOR

    )


    if image_bgr is None:

        raise ValueError(
            f"Unable to read image: "
            f"{image_path}"
        )


    image_rgb = cv2.cvtColor(

        image_bgr,

        cv2.COLOR_BGR2RGB

    )


    image_array = image_rgb.astype(
        np.float32
    ) / 255.0


    image_array = (

        image_array

        -

        IMAGENET_MEAN

    ) / IMAGENET_STD


    image_array = np.ascontiguousarray(

        image_array.transpose(

            2,
            0,
            1

        )

    )


    image_tensor = torch.from_numpy(

        image_array

    ).unsqueeze(
        0
    )


    return (

        image_rgb,

        image_tensor

    )


# ===================================================
# LOAD MODEL
# ===================================================

model = CSRNet(

    pretrained=False

).to(
    device
)


load_checkpoint(

    model=model,

    checkpoint_path=model_path,

    map_location=device

)


model.eval()


print(
    "Model Ready!"
)


# ===================================================
# LOAD MALL FRAMES
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


image_files.sort(

    key=natural_sort_key

)


# ===================================================
# LIMIT IMAGES
# ===================================================

if MAX_IMAGES is not None:

    image_files = image_files[
        :MAX_IMAGES
    ]


print(
    "\nMall Images Selected:",
    len(
        image_files
    )
)


# ===================================================
# RESULTS
# ===================================================

results = []


print(
    "\nStarting Sequential "
    "Mall Dataset Prediction..."
)


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
        f"\n[{index}/"
        f"{len(image_files)}] "
        f"Processing: "
        f"{image_name}"
    )


    try:


        # ===========================================
        # PREPROCESS IMAGE
        # ===========================================

        image_rgb, image_tensor = (
            preprocess_image(
                image_path
            )
        )


        image_tensor = image_tensor.to(

            device=device,

            dtype=torch.float32,

            non_blocking=True

        )


        # ===========================================
        # CSRNET PREDICTION
        # ===========================================

        with torch.inference_mode():

            output = model(
                image_tensor
            )


        density_map = (

            output[
                0,
                0
            ]

            .detach()

            .float()

            .cpu()

            .numpy()

        )


        # ===========================================
        # CROWD COUNT
        # ===========================================

        predicted_count = float(

            density_map.sum()

        )


        # ===========================================
        # DENSITY FEATURES
        # ===========================================

        mean_density = float(

            np.mean(
                density_map
            )

        )


        max_density = float(

            np.max(
                density_map
            )

        )


        std_density = float(

            np.std(
                density_map
            )

        )


        # ===========================================
        # PRINT RESULTS
        # ===========================================

        print(
            f"Predicted Count: "
            f"{predicted_count:.2f}"
        )


        print(
            f"Mean Density: "
            f"{mean_density:.8f}"
        )


        print(
            f"Maximum Density: "
            f"{max_density:.8f}"
        )


        print(
            f"Density Std: "
            f"{std_density:.8f}"
        )


        # ===========================================
        # STORE RESULTS
        # ===========================================

        results.append(

            {

                "frame":
                image_name,


                "predicted_count":
                predicted_count,


                "mean_density":
                mean_density,


                "max_density":
                max_density,


                "std_density":
                std_density

            }

        )


        # ===========================================
        # OPTIONAL VISUALIZATION
        # ===========================================

        if SAVE_VISUALIZATIONS:


            figure = plt.figure(

                figsize=(
                    16,
                    6
                )

            )


            # ---------------------------------------
            # ORIGINAL IMAGE
            # ---------------------------------------

            original_axis = (
                figure.add_subplot(

                    1,

                    3,

                    1

                )
            )


            original_axis.imshow(

                image_rgb

            )


            original_axis.set_title(

                f"Original Frame\n"
                f"{image_name}"

            )


            original_axis.axis(
                "off"
            )


            # ---------------------------------------
            # DENSITY MAP
            # ---------------------------------------

            density_axis = (
                figure.add_subplot(

                    1,

                    3,

                    2

                )
            )


            density_plot = (
                density_axis.imshow(

                    density_map,

                    cmap="jet"

                )
            )


            density_axis.set_title(

                "Predicted Density Map"

            )


            density_axis.axis(
                "off"
            )


            figure.colorbar(

                density_plot,

                ax=density_axis,

                fraction=0.046,

                pad=0.04

            )


            # ---------------------------------------
            # DENSITY OVERLAY
            # ---------------------------------------

            original_height = (
                image_rgb.shape[
                    0
                ]
            )


            original_width = (
                image_rgb.shape[
                    1
                ]
            )


            resized_density = cv2.resize(

                density_map,

                (

                    original_width,

                    original_height

                ),

                interpolation=(
                    cv2.INTER_CUBIC
                )

            )


            density_min = (
                resized_density.min()
            )


            density_max = (
                resized_density.max()
            )


            normalized_density = (

                resized_density

                -

                density_min

            ) / (

                density_max

                -

                density_min

                +

                1e-8

            )


            heatmap_uint8 = np.uint8(

                normalized_density

                *

                255

            )


            heatmap_bgr = (
                cv2.applyColorMap(

                    heatmap_uint8,

                    cv2.COLORMAP_JET

                )
            )


            heatmap_rgb = (
                cv2.cvtColor(

                    heatmap_bgr,

                    cv2.COLOR_BGR2RGB

                )
            )


            overlay = (
                cv2.addWeighted(

                    image_rgb,

                    0.65,

                    heatmap_rgb,

                    0.35,

                    0

                )
            )


            overlay_axis = (
                figure.add_subplot(

                    1,

                    3,

                    3

                )
            )


            overlay_axis.imshow(
                overlay
            )


            overlay_axis.set_title(

                f"Density Overlay\n"

                f"Predicted Count: "

                f"{predicted_count:.2f}"

            )


            overlay_axis.axis(
                "off"
            )


            figure.tight_layout()


            save_path = os.path.join(

                output_dir,

                f"{image_stem}"
                f"_prediction.png"

            )


            figure.savefig(

                save_path,

                dpi=150,

                bbox_inches="tight"

            )


            plt.close(
                figure
            )


    except Exception as error:


        print(

            f"Failed to process "

            f"{image_name}: "

            f"{error}"

        )


# ===================================================
# SAVE CROWD FEATURES CSV
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

                "predicted_count",

                "mean_density",

                "max_density",

                "std_density"

            ]

        )


        writer.writeheader()


        writer.writerows(
            results
        )


    predicted_counts = np.array(

        [

            item[
                "predicted_count"
            ]

            for item in results

        ],

        dtype=np.float64

    )


    print(

        "\n"
        + "=" * 50

    )


    print(
        "CROWD FEATURE SUMMARY"
    )


    print(
        "=" * 50
    )


    print(

        "Frames Processed:",

        len(
            results
        )

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

        "Crowd Features CSV:",

        csv_path

    )


    print(

        "Prediction Images:",

        output_dir

    )


else:


    print(

        "\nNo Mall dataset images "

        "were processed."

    )