import os
import sys

import pandas as pd
import numpy as np


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
# INPUT AND OUTPUT PATHS
# ============================================================

input_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "temporal_features.csv"
)


output_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "risk_scores.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STAMPede RISK SCORE CALCULATION")
print("=" * 70)

print(
    "\nInput CSV:",
    input_csv
)


# ------------------------------------------------------------
# CHECK INPUT FILE
# ------------------------------------------------------------

if not os.path.isfile(
    input_csv
):

    raise FileNotFoundError(
        f"\nInput CSV not found:\n{input_csv}"
    )


# ------------------------------------------------------------
# LOAD CSV
# ------------------------------------------------------------

df = pd.read_csv(
    input_csv
)


print(
    "\nTotal Rows:",
    len(df)
)


print(
    "\nAvailable Columns:"
)


for column in df.columns:

    print(
        "-",
        column
    )


# ============================================================
# REQUIRED FEATURES
# ============================================================

required_columns = [

    # Crowd information
    "predicted_count",

    # Motion information
    "mean_magnitude",

    "moving_pixel_ratio",

    # Directional disorder
    "normalized_directional_entropy",

    # Temporal crowd behaviour
    "count_change_abs",

    "motion_change_abs",

    "motion_acceleration_abs",

    "moving_ratio_change_abs"

]


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    print(
        "\nMissing Columns:"
    )


    for column in missing_columns:

        print(
            "-",
            column
        )


    raise ValueError(
        "\nThe input CSV does not contain "
        "all required features."
    )


print(
    "\nAll required risk features found."
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print(
    "\nChecking missing values..."
)


for column in required_columns:

    missing_count = df[
        column
    ].isna().sum()


    print(
        f"{column}: "
        f"{missing_count} missing values"
    )


# Replace missing values with zero
df[
    required_columns
] = df[
    required_columns
].fillna(
    0
)


# ============================================================
# MIN-MAX NORMALIZATION FUNCTION
# ============================================================

def min_max_normalize(
    series
):

    minimum = series.min()

    maximum = series.max()


    # Prevent division by zero
    if maximum == minimum:

        return pd.Series(

            np.zeros(
                len(series)
            ),

            index=series.index

        )


    normalized = (

        series - minimum

    ) / (

        maximum - minimum

    )


    return normalized


# ============================================================
# NORMALIZE CROWD FEATURES
# ============================================================

print(
    "\nNormalizing features..."
)


# ------------------------------------------------------------
# 1. CROWD COUNT RISK
# ------------------------------------------------------------

df[
    "count_risk"
] = min_max_normalize(

    df[
        "predicted_count"
    ]

)


# ------------------------------------------------------------
# 2. SUDDEN CROWD COUNT CHANGE
# ------------------------------------------------------------

df[
    "count_change_risk"
] = min_max_normalize(

    df[
        "count_change_abs"
    ]

)


# ------------------------------------------------------------
# 3. MOTION INTENSITY
# ------------------------------------------------------------

df[
    "motion_risk"
] = min_max_normalize(

    df[
        "mean_magnitude"
    ]

)


# ------------------------------------------------------------
# 4. SUDDEN MOTION CHANGE
# ------------------------------------------------------------

df[
    "motion_change_risk"
] = min_max_normalize(

    df[
        "motion_change_abs"
    ]

)


# ------------------------------------------------------------
# 5. MOTION ACCELERATION
# ------------------------------------------------------------

df[
    "motion_acceleration_risk"
] = min_max_normalize(

    df[
        "motion_acceleration_abs"
    ]

)


# ------------------------------------------------------------
# 6. MOVING PIXEL RATIO
# ------------------------------------------------------------

df[
    "moving_ratio_risk"
] = min_max_normalize(

    df[
        "moving_pixel_ratio"
    ]

)


# ------------------------------------------------------------
# 7. SUDDEN CHANGE IN MOVEMENT
# ------------------------------------------------------------

df[
    "moving_ratio_change_risk"
] = min_max_normalize(

    df[
        "moving_ratio_change_abs"
    ]

)


# ------------------------------------------------------------
# 8. DIRECTIONAL DISORDER
# ------------------------------------------------------------

df[
    "entropy_risk"
] = min_max_normalize(

    df[
        "normalized_directional_entropy"
    ]

)


# ============================================================
# RISK SCORE WEIGHTS
# ============================================================

# Total weight = 1.00


COUNT_WEIGHT = 0.20


COUNT_CHANGE_WEIGHT = 0.15


MOTION_WEIGHT = 0.15


MOTION_CHANGE_WEIGHT = 0.15


MOTION_ACCELERATION_WEIGHT = 0.10


MOVING_RATIO_WEIGHT = 0.10


MOVING_RATIO_CHANGE_WEIGHT = 0.10


ENTROPY_WEIGHT = 0.05


# ============================================================
# CALCULATE WEIGHTED RISK SCORE
# ============================================================

print(
    "\nCalculating weighted risk score..."
)


df[
    "risk_score"
] = (

    # Crowd density
    COUNT_WEIGHT
    *
    df[
        "count_risk"
    ]


    +

    # Sudden change in crowd
    COUNT_CHANGE_WEIGHT
    *
    df[
        "count_change_risk"
    ]


    +

    # Overall motion intensity
    MOTION_WEIGHT
    *
    df[
        "motion_risk"
    ]


    +

    # Sudden change in motion
    MOTION_CHANGE_WEIGHT
    *
    df[
        "motion_change_risk"
    ]


    +

    # Acceleration in motion
    MOTION_ACCELERATION_WEIGHT
    *
    df[
        "motion_acceleration_risk"
    ]


    +

    # Amount of moving area
    MOVING_RATIO_WEIGHT
    *
    df[
        "moving_ratio_risk"
    ]


    +

    # Sudden change in moving area
    MOVING_RATIO_CHANGE_WEIGHT
    *
    df[
        "moving_ratio_change_risk"
    ]


    +

    # Disorder in movement direction
    ENTROPY_WEIGHT
    *
    df[
        "entropy_risk"
    ]

)


# ============================================================
# CALCULATE RISK PERCENTAGE
# ============================================================

df[
    "risk_percentage"
] = (

    df[
        "risk_score"
    ]

    *
    100

)


# ============================================================
# RISK LEVEL CLASSIFICATION
# ============================================================

def classify_risk(
    score
):

    if score < 0.25:

        return "Low"


    elif score < 0.50:

        return "Moderate"


    elif score < 0.75:

        return "High"


    else:

        return "Critical"


# Apply classification

df[
    "risk_level"
] = df[
    "risk_score"
].apply(
    classify_risk
)


# ============================================================
# ROUND FINAL VALUES
# ============================================================

df[
    "risk_score"
] = df[
    "risk_score"
].round(
    4
)


df[
    "risk_percentage"
] = df[
    "risk_percentage"
].round(
    2
)


# ============================================================
# SAVE OUTPUT
# ============================================================

df.to_csv(

    output_csv,

    index=False

)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "RISK ANALYSIS COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nOutput File:"
)

print(
    output_csv
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print(
    "\nRisk Level Distribution:"
)


print(

    df[
        "risk_level"
    ].value_counts()

)


# ============================================================
# RISK SCORE STATISTICS
# ============================================================

print(
    "\nRisk Score Statistics:"
)


print(

    df[
        "risk_score"
    ].describe()

)


# ============================================================
# SHOW SAMPLE RESULTS
# ============================================================

print(
    "\nFirst 20 Risk Results:"
)


display_columns = [

    "frame_t",

    "predicted_count",

    "mean_magnitude",

    "moving_pixel_ratio",

    "normalized_directional_entropy",

    "count_change_abs",

    "motion_change_abs",

    "motion_acceleration_abs",

    "moving_ratio_change_abs",

    "risk_score",

    "risk_percentage",

    "risk_level"

]


# Keep only columns that exist

available_display_columns = [

    column

    for column in display_columns

    if column in df.columns

]


print(

    df[
        available_display_columns
    ].head(
        20
    )

)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\nRisk score calculation finished successfully."
)