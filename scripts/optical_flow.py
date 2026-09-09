import os
import re
import sys
import csv

import cv2
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


# ===================================================
# SETTINGS
# ===================================================

# None = process all consecutive frame pairs
MAX_PAIRS = None

# Save visualization images
SAVE_VISUALIZATIONS = True

# Minimum magnitude considered meaningful motion
MOTION_THRESHOLD = 1.0

# Distance between arrows
ARROW_STEP = 25

# Multiply optical-flow vectors for clearer arrows
ARROW_SCALE = 3.0

# Number of direction bins
DIRECTION_BINS = 36


# ===================================================
# PATHS
# ===================================================

mall_frames_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "mall_dataset",
    "frames"
)


output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "optical_flow"
)


csv_path = os.path.join(
    BASE_DIR,
    "outputs",
    "motion_features.csv"
)


os.makedirs(
    output_dir,
    exist_ok=True
)


# ===================================================
# CHECK DATASET PATH
# ===================================================

print(
    "Mall Frames Folder:",
    mall_frames_dir
)

print(
    "Mall Frames Exist:",
    os.path.isdir(
        mall_frames_dir
    )
)


if not os.path.isdir(
    mall_frames_dir
):

    raise FileNotFoundError(
        f"Mall frames directory not found: "
        f"{mall_frames_dir}"
    )


# ===================================================
# NATURAL SORT
# ===================================================

def natural_sort_key(file_name):

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
# LOAD FRAME LIST
# ===================================================

frame_files = [

    file_name

    for file_name in os.listdir(
        mall_frames_dir
    )

    if file_name.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )

]


frame_files.sort(
    key=natural_sort_key
)


print(
    "Total Frames Found:",
    len(frame_files)
)


if len(frame_files) < 2:

    raise RuntimeError(
        "At least two frames are required "
        "for optical flow."
    )


# ===================================================
# LIMIT FRAME PAIRS
# ===================================================

if MAX_PAIRS is not None:

    frame_files = frame_files[
        :MAX_PAIRS + 1
    ]


# ===================================================
# RESULTS
# ===================================================

results = []


# ===================================================
# PROCESS CONSECUTIVE FRAME PAIRS
# ===================================================

for index in range(
    len(frame_files) - 1
):

    frame_name_1 = frame_files[
        index
    ]

    frame_name_2 = frame_files[
        index + 1
    ]


    frame_path_1 = os.path.join(
        mall_frames_dir,
        frame_name_1
    )


    frame_path_2 = os.path.join(
        mall_frames_dir,
        frame_name_2
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        f"[{index + 1}/"
        f"{len(frame_files) - 1}]"
    )

    print(
        f"Frame t: "
        f"{frame_name_1}"
    )

    print(
        f"Frame t+1: "
        f"{frame_name_2}"
    )


    # ===============================================
    # LOAD FRAMES
    # ===============================================

    frame1 = cv2.imread(
        frame_path_1
    )


    frame2 = cv2.imread(
        frame_path_2
    )


    if (
        frame1 is None
        or
        frame2 is None
    ):

        print(
            "Unable to read frame pair."
        )

        continue


    # ===============================================
    # CHECK FRAME SIZE
    # ===============================================

    if frame1.shape != frame2.shape:

        print(
            "Frame sizes differ."
        )

        print(
            "Resizing Frame t+1."
        )

        frame2 = cv2.resize(
            frame2,
            (
                frame1.shape[1],
                frame1.shape[0]
            )
        )


    # ===============================================
    # CONVERT TO GRAYSCALE
    # ===============================================

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )


    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )


    # ===============================================
    # FARNEBACK DENSE OPTICAL FLOW
    # ===============================================

    flow = cv2.calcOpticalFlowFarneback(

        gray1,
        gray2,
        None,

        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0

    )


    # ===============================================
    # MAGNITUDE AND DIRECTION
    # ===============================================

    magnitude, angle = cv2.cartToPolar(

        flow[:, :, 0],
        flow[:, :, 1],

        angleInDegrees=True

    )


    # ===============================================
    # BASIC MOTION FEATURES
    # ===============================================

    mean_magnitude = float(
        np.mean(
            magnitude
        )
    )


    max_magnitude = float(
        np.max(
            magnitude
        )
    )


    std_magnitude = float(
        np.std(
            magnitude
        )
    )


    # ===============================================
    # MOVING PIXEL RATIO
    # ===============================================

    moving_mask = (
        magnitude
        > MOTION_THRESHOLD
    )


    moving_pixel_ratio = float(
        np.mean(
            moving_mask
        )
    )


    # ===============================================
    # HORIZONTAL MOTION
    # ===============================================

    horizontal_flow = flow[
        :,
        :,
        0
    ]


    mean_horizontal_flow = float(
        np.mean(
            horizontal_flow
        )
    )


    # ===============================================
    # VERTICAL MOTION
    # ===============================================

    vertical_flow = flow[
        :,
        :,
        1
    ]


    mean_vertical_flow = float(
        np.mean(
            vertical_flow
        )
    )


    # ===============================================
    # DIRECTION HISTOGRAM
    # ===============================================

    significant_angles = angle[
        moving_mask
    ]


    if len(
        significant_angles
    ) > 0:


        histogram, bin_edges = np.histogram(

            significant_angles,

            bins=DIRECTION_BINS,

            range=(
                0,
                360
            )

        )


        # ===========================================
        # DOMINANT MOTION DIRECTION
        # ===========================================

        dominant_bin = int(
            np.argmax(
                histogram
            )
        )


        dominant_direction = float(

            (
                bin_edges[
                    dominant_bin
                ]

                +

                bin_edges[
                    dominant_bin + 1
                ]

            )

            / 2

        )


        # ===========================================
        # DIRECTIONAL ENTROPY
        # ===========================================

        total_motion_pixels = float(
            np.sum(
                histogram
            )
        )


        probabilities = (

            histogram.astype(
                np.float64
            )

            /

            total_motion_pixels

        )


        # Remove zero probabilities before log()
        non_zero_probabilities = probabilities[
            probabilities > 0
        ]


        directional_entropy = float(

            -np.sum(

                non_zero_probabilities

                *

                np.log(
                    non_zero_probabilities
                )

            )

        )


        # ===========================================
        # NORMALIZED DIRECTIONAL ENTROPY
        # ===========================================

        max_entropy = np.log(
            DIRECTION_BINS
        )


        normalized_directional_entropy = float(

            directional_entropy

            /

            (
                max_entropy
                +
                1e-12
            )

        )


    else:

        dominant_direction = 0.0

        directional_entropy = 0.0

        normalized_directional_entropy = 0.0


    # ===============================================
    # PRINT MOTION FEATURES
    # ===============================================

    print(
        f"Mean Motion Magnitude: "
        f"{mean_magnitude:.4f}"
    )


    print(
        f"Maximum Motion Magnitude: "
        f"{max_magnitude:.4f}"
    )


    print(
        f"Motion Standard Deviation: "
        f"{std_magnitude:.4f}"
    )


    print(
        f"Moving Pixel Ratio: "
        f"{moving_pixel_ratio:.4f}"
    )


    print(
        f"Mean Horizontal Flow: "
        f"{mean_horizontal_flow:.4f}"
    )


    print(
        f"Mean Vertical Flow: "
        f"{mean_vertical_flow:.4f}"
    )


    print(
        f"Dominant Direction: "
        f"{dominant_direction:.2f} degrees"
    )


    print(
        f"Directional Entropy: "
        f"{directional_entropy:.4f}"
    )


    print(
        f"Normalized Directional Entropy: "
        f"{normalized_directional_entropy:.4f}"
    )


    # ===============================================
    # SAVE NUMERICAL FEATURES
    # ===============================================

    results.append(

        {

            "frame_t":
            frame_name_1,


            "frame_t_plus_1":
            frame_name_2,


            "mean_magnitude":
            mean_magnitude,


            "max_magnitude":
            max_magnitude,


            "std_magnitude":
            std_magnitude,


            "moving_pixel_ratio":
            moving_pixel_ratio,


            "mean_horizontal_flow":
            mean_horizontal_flow,


            "mean_vertical_flow":
            mean_vertical_flow,


            "dominant_direction":
            dominant_direction,


            "directional_entropy":
            directional_entropy,


            "normalized_directional_entropy":
            normalized_directional_entropy

        }

    )


    # ===============================================
    # VISUALIZATION
    # ===============================================

    if SAVE_VISUALIZATIONS:


        # ===========================================
        # COLOR OPTICAL FLOW MAP
        # ===========================================

        hsv = np.zeros_like(
            frame1
        )


        hsv[..., 1] = 255


        hsv[..., 0] = np.uint8(
            angle / 2
        )


        hsv[..., 2] = cv2.normalize(

            magnitude,

            None,

            0,

            255,

            cv2.NORM_MINMAX

        ).astype(
            np.uint8
        )


        flow_visualization = cv2.cvtColor(

            hsv,

            cv2.COLOR_HSV2BGR

        )


        # ===========================================
        # ARROW VISUALIZATION
        # ===========================================

        arrow_visualization = (
            frame1.copy()
        )


        height = frame1.shape[0]

        width = frame1.shape[1]


        for y in range(

            ARROW_STEP // 2,

            height,

            ARROW_STEP

        ):


            for x in range(

                ARROW_STEP // 2,

                width,

                ARROW_STEP

            ):


                fx = float(
                    flow[
                        y,
                        x,
                        0
                    ]
                )


                fy = float(
                    flow[
                        y,
                        x,
                        1
                    ]
                )


                local_magnitude = float(

                    np.sqrt(

                        fx ** 2

                        +

                        fy ** 2

                    )

                )


                if (

                    local_magnitude
                    < MOTION_THRESHOLD

                ):

                    continue


                start_point = (

                    int(x),

                    int(y)

                )


                end_x = int(

                    x

                    +

                    fx
                    *
                    ARROW_SCALE

                )


                end_y = int(

                    y

                    +

                    fy
                    *
                    ARROW_SCALE

                )


                end_point = (

                    end_x,

                    end_y

                )


                cv2.arrowedLine(

                    arrow_visualization,

                    start_point,

                    end_point,

                    (
                        0,
                        255,
                        0
                    ),

                    1,

                    tipLength=0.3

                )


        # ===========================================
        # CREATE FIGURE
        # ===========================================

        figure = plt.figure(
            figsize=(
                22,
                6
            )
        )


        # FRAME T
        axis1 = figure.add_subplot(
            1,
            4,
            1
        )

        axis1.imshow(
            cv2.cvtColor(
                frame1,
                cv2.COLOR_BGR2RGB
            )
        )

        axis1.set_title(
            f"Frame t\n"
            f"{frame_name_1}"
        )

        axis1.axis(
            "off"
        )


        # FRAME T+1
        axis2 = figure.add_subplot(
            1,
            4,
            2
        )

        axis2.imshow(
            cv2.cvtColor(
                frame2,
                cv2.COLOR_BGR2RGB
            )
        )

        axis2.set_title(
            f"Frame t+1\n"
            f"{frame_name_2}"
        )

        axis2.axis(
            "off"
        )


        # ARROW MOTION MAP
        axis3 = figure.add_subplot(
            1,
            4,
            3
        )

        axis3.imshow(
            cv2.cvtColor(
                arrow_visualization,
                cv2.COLOR_BGR2RGB
            )
        )

        axis3.set_title(
            "Motion Direction Arrows\n"
            f"Dominant: "
            f"{dominant_direction:.1f}°\n"
            f"Entropy: "
            f"{normalized_directional_entropy:.3f}"
        )

        axis3.axis(
            "off"
        )


        # COLOR OPTICAL FLOW MAP
        axis4 = figure.add_subplot(
            1,
            4,
            4
        )

        axis4.imshow(
            cv2.cvtColor(
                flow_visualization,
                cv2.COLOR_BGR2RGB
            )
        )

        axis4.set_title(
            "Optical Flow Color Map\n"
            f"Mean Motion: "
            f"{mean_magnitude:.4f}\n"
            f"Direction Disorder: "
            f"{normalized_directional_entropy:.3f}"
        )

        axis4.axis(
            "off"
        )


        figure.tight_layout()


        output_name = (

            "flow_"

            +

            os.path.splitext(
                frame_name_1
            )[0]

            +

            ".png"

        )


        save_path = os.path.join(
            output_dir,
            output_name
        )


        figure.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close(
            figure
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

                "frame_t",

                "frame_t_plus_1",

                "mean_magnitude",

                "max_magnitude",

                "std_magnitude",

                "moving_pixel_ratio",

                "mean_horizontal_flow",

                "mean_vertical_flow",

                "dominant_direction",

                "directional_entropy",

                "normalized_directional_entropy"

            ]

        )


        writer.writeheader()


        writer.writerows(
            results
        )


    print(
        "\n"
        + "=" * 60
    )


    print(
        "OPTICAL FLOW SUMMARY"
    )


    print(
        "=" * 60
    )


    print(
        "Frame Pairs Processed:",
        len(results)
    )


    print(
        "Motion Features CSV:",
        csv_path
    )


    print(
        "Optical Flow Images:",
        output_dir
    )


else:

    print(
        "\nNo frame pairs were processed."
    )