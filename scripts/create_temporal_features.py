import os
import sys
import re

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 10


# ============================================================
# PATHS
# ============================================================

input_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "fused_features.csv"
)


output_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "temporal_features.csv"
)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(
    input_csv
):
    raise FileNotFoundError(
        f"Fused features file not found:\n"
        f"{input_csv}"
    )


print("=" * 70)
print("TEMPORAL FEATURE GENERATION")
print("=" * 70)

print(
    "\nLoading:",
    input_csv
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    input_csv
)


print(
    "\nOriginal Shape:",
    df.shape
)


print(
    "\nColumns:"
)

for column in df.columns:
    print(
        "-",
        column
    )


# ============================================================
# NATURAL SORT BY FRAME NUMBER
# ============================================================

def extract_frame_number(
    frame_name
):

    numbers = re.findall(
        r"\d+",
        str(frame_name)
    )

    if len(numbers) == 0:
        return -1

    return int(
        numbers[-1]
    )


df[
    "frame_number"
] = df[
    "frame_t"
].apply(
    extract_frame_number
)


df = df.sort_values(
    by="frame_number"
).reset_index(
    drop=True
)


# ============================================================
# CHECK FOR MISSING VALUES
# ============================================================

print(
    "\nMissing Values:"
)

missing_values = df.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# ============================================================
# FILL MISSING VALUES IF ANY
# ============================================================

df = df.ffill()
df = df.bfill()


# ============================================================
# DIRECTION FEATURE TRANSFORMATION
# ============================================================

print(
    "\nConverting dominant direction "
    "to sine and cosine..."
)


direction_radians = np.deg2rad(
    df[
        "dominant_direction"
    ]
)


df[
    "direction_sin"
] = np.sin(
    direction_radians
)


df[
    "direction_cos"
] = np.cos(
    direction_radians
)


# ============================================================
# TEMPORAL CHANGE FEATURES
# ============================================================

print(
    "Calculating temporal changes..."
)


# ------------------------------------------------------------
# CROWD COUNT CHANGE
# ------------------------------------------------------------

df[
    "count_change"
] = df[
    "predicted_count"
].diff()


df[
    "count_change_abs"
] = df[
    "count_change"
].abs()


# ------------------------------------------------------------
# DENSITY CHANGE
# ------------------------------------------------------------

df[
    "density_change"
] = df[
    "mean_density"
].diff()


df[
    "density_change_abs"
] = df[
    "density_change"
].abs()


# ------------------------------------------------------------
# MOTION CHANGE
# ------------------------------------------------------------

df[
    "motion_change"
] = df[
    "mean_magnitude"
].diff()


df[
    "motion_change_abs"
] = df[
    "motion_change"
].abs()


# ------------------------------------------------------------
# MOTION ACCELERATION
# ------------------------------------------------------------

df[
    "motion_acceleration"
] = df[
    "motion_change"
].diff()


df[
    "motion_acceleration_abs"
] = df[
    "motion_acceleration"
].abs()


# ------------------------------------------------------------
# MOVING PIXEL CHANGE
# ------------------------------------------------------------

df[
    "moving_ratio_change"
] = df[
    "moving_pixel_ratio"
].diff()


df[
    "moving_ratio_change_abs"
] = df[
    "moving_ratio_change"
].abs()


# ============================================================
# ROLLING TEMPORAL FEATURES
# ============================================================

print(
    "Calculating rolling temporal statistics..."
)


ROLLING_WINDOW = 5


# ------------------------------------------------------------
# CROWD COUNT
# ------------------------------------------------------------

df[
    "count_rolling_mean"
] = df[
    "predicted_count"
].rolling(
    window=ROLLING_WINDOW,
    min_periods=1
).mean()


df[
    "count_rolling_std"
] = df[
    "predicted_count"
].rolling(
    window=ROLLING_WINDOW,
    min_periods=1
).std().fillna(
    0
)


# ------------------------------------------------------------
# MOTION
# ------------------------------------------------------------

df[
    "motion_rolling_mean"
] = df[
    "mean_magnitude"
].rolling(
    window=ROLLING_WINDOW,
    min_periods=1
).mean()


df[
    "motion_rolling_std"
] = df[
    "mean_magnitude"
].rolling(
    window=ROLLING_WINDOW,
    min_periods=1
).std().fillna(
    0
)


# ------------------------------------------------------------
# DIRECTIONAL DISORDER
# ------------------------------------------------------------

df[
    "entropy_rolling_mean"
] = df[
    "normalized_directional_entropy"
].rolling(
    window=ROLLING_WINDOW,
    min_periods=1
).mean()


df[
    "entropy_rolling_std"
] = df[
    "normalized_directional_entropy"
].rolling(
    window=ROLLING_WINDOW,
    min_periods=1
).std().fillna(
    0
)


# ============================================================
# FILL NaN VALUES CREATED BY DIFFERENCING
# ============================================================

temporal_numeric_columns = [

    "count_change",
    "count_change_abs",

    "density_change",
    "density_change_abs",

    "motion_change",
    "motion_change_abs",

    "motion_acceleration",
    "motion_acceleration_abs",

    "moving_ratio_change",
    "moving_ratio_change_abs"

]


for column in temporal_numeric_columns:

    df[
        column
    ] = df[
        column
    ].fillna(
        0
    )


# ============================================================
# CREATE SEQUENCE ID
# ============================================================

print(
    "Creating temporal sequence IDs..."
)


sequence_ids = []


for index in range(
    len(df)
):

    sequence_id = (
        index //
        SEQUENCE_LENGTH
    )

    sequence_ids.append(
        sequence_id
    )


df[
    "sequence_id"
] = sequence_ids


# ============================================================
# POSITION INSIDE SEQUENCE
# ============================================================

sequence_positions = []


for index in range(
    len(df)
):

    position = (
        index %
        SEQUENCE_LENGTH
    )

    sequence_positions.append(
        position
    )


df[
    "sequence_position"
] = sequence_positions


# ============================================================
# SAVE OUTPUT
# ============================================================

os.makedirs(
    os.path.dirname(
        output_csv
    ),
    exist_ok=True
)


df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TEMPORAL FEATURE GENERATION COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nInput Rows:",
    len(
        pd.read_csv(
            input_csv
        )
    )
)


print(
    "Output Rows:",
    len(df)
)


print(
    "Total Columns:",
    len(
        df.columns
    )
)


print(
    "Sequence Length:",
    SEQUENCE_LENGTH
)


print(
    "Total Sequences:",
    df[
        "sequence_id"
    ].nunique()
)


print(
    "\nOutput File:"
)

print(
    output_csv
)


print(
    "\nFirst 5 Rows:"
)

print(
    df.head()
)


print(
    "\nLast 5 Rows:"
)

print(
    df.tail()
)


print(
    "\nFeature Generation Finished Successfully."
)