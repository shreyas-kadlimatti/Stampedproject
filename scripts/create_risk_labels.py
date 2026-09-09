import os
import sys

import numpy as np
import pandas as pd


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
# PATHS
# ===================================================

input_csv_path = os.path.join(
    BASE_DIR,
    "outputs",
    "fused_features.csv"
)


output_csv_path = os.path.join(
    BASE_DIR,
    "outputs",
    "labeled_features.csv"
)


# ===================================================
# CHECK INPUT FILE
# ===================================================

if not os.path.isfile(
    input_csv_path
):

    raise FileNotFoundError(
        f"Fused features file not found:\n"
        f"{input_csv_path}"
    )


# ===================================================
# LOAD DATA
# ===================================================

print(
    "Loading fused features..."
)

df = pd.read_csv(
    input_csv_path
)


print(
    "Total samples:",
    len(df)
)


# ===================================================
# CONVERT DIRECTION TO SIN / COS
# ===================================================

print(
    "Converting dominant direction..."
)


direction_radians = np.deg2rad(
    df["dominant_direction"]
)


df["direction_sin"] = np.sin(
    direction_radians
)


df["direction_cos"] = np.cos(
    direction_radians
)


# ===================================================
# MIN-MAX NORMALIZATION FUNCTION
# ===================================================

def min_max_normalize(
    series
):

    minimum = series.min()

    maximum = series.max()


    if maximum == minimum:

        return pd.Series(
            np.zeros(
                len(series)
            ),
            index=series.index
        )


    return (

        series - minimum

    ) / (

        maximum - minimum

    )


# ===================================================
# NORMALIZE RISK INDICATORS
# ===================================================

print(
    "Normalizing risk indicators..."
)


df["norm_count"] = min_max_normalize(

    df["predicted_count"]

)


df["norm_motion"] = min_max_normalize(

    df["mean_magnitude"]

)


df["norm_motion_std"] = min_max_normalize(

    df["std_magnitude"]

)


df["norm_moving_ratio"] = min_max_normalize(

    df["moving_pixel_ratio"]

)


# ===================================================
# CALCULATE PROXY RISK SCORE
# ===================================================

# Crowd density importance
COUNT_WEIGHT = 0.35

# Overall crowd movement
MOTION_WEIGHT = 0.30

# Motion instability
MOTION_STD_WEIGHT = 0.20

# Spatial extent of movement
MOVING_RATIO_WEIGHT = 0.15


df["risk_score"] = (

    COUNT_WEIGHT
    *
    df["norm_count"]

    +

    MOTION_WEIGHT
    *
    df["norm_motion"]

    +

    MOTION_STD_WEIGHT
    *
    df["norm_motion_std"]

    +

    MOVING_RATIO_WEIGHT
    *
    df["norm_moving_ratio"]

)


# ===================================================
# CREATE RISK LABELS
# ===================================================

# Divide the dataset distribution into
# three equally sized risk groups.

lower_threshold = df[
    "risk_score"
].quantile(
    0.33
)


upper_threshold = df[
    "risk_score"
].quantile(
    0.67
)


def assign_risk_label(
    risk_score
):

    if risk_score <= lower_threshold:

        return 0

    elif risk_score <= upper_threshold:

        return 1

    else:

        return 2


df["risk_label"] = df[
    "risk_score"
].apply(
    assign_risk_label
)


# ===================================================
# HUMAN-READABLE LABEL
# ===================================================

risk_label_names = {

    0: "Safe",

    1: "Moderate",

    2: "High"

}


df["risk_category"] = df[
    "risk_label"
].map(
    risk_label_names
)


# ===================================================
# REMOVE TEMPORARY NORMALIZED COLUMNS
# ===================================================

df = df.drop(

    columns=[

        "norm_count",

        "norm_motion",

        "norm_motion_std",

        "norm_moving_ratio"

    ]

)


# ===================================================
# SAVE RESULTS
# ===================================================

df.to_csv(

    output_csv_path,

    index=False

)


# ===================================================
# SUMMARY
# ===================================================

print(
    "\n"
    + "=" * 55
)

print(
    "RISK LABEL GENERATION COMPLETE"
)

print(
    "=" * 55
)


print(
    "\nRisk Score Thresholds:"
)

print(
    f"Safe Threshold: <= "
    f"{lower_threshold:.4f}"
)

print(
    f"Moderate Threshold: <= "
    f"{upper_threshold:.4f}"
)

print(
    f"High Risk: > "
    f"{upper_threshold:.4f}"
)


print(
    "\nRisk Distribution:"
)


risk_counts = df[
    "risk_category"
].value_counts()


for category, count in risk_counts.items():

    percentage = (

        count

        /

        len(df)

    ) * 100


    print(

        f"{category}: "

        f"{count} samples "

        f"({percentage:.2f}%)"

    )


print(
    "\nOutput File:"
)

print(
    output_csv_path
)


print(
    "\nNew Columns Added:"
)

print(
    "- direction_sin"
)

print(
    "- direction_cos"
)

print(
    "- risk_score"
)

print(
    "- risk_label"
)

print(
    "- risk_category"
)


print(
    "\nFirst 10 Rows:"
)

print(
    df.head(10)
)


print(
    "\n"
    + "=" * 55
)