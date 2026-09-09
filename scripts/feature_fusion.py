import os
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


# ===================================================
# PATHS
# ===================================================

motion_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "motion_features.csv"
)


crowd_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "crowd_features.csv"
)


output_csv = os.path.join(
    BASE_DIR,
    "outputs",
    "fused_features.csv"
)


# ===================================================
# CHECK FILES
# ===================================================

if not os.path.exists(motion_csv):

    raise FileNotFoundError(
        f"Motion features file not found:\n"
        f"{motion_csv}"
    )


if not os.path.exists(crowd_csv):

    raise FileNotFoundError(
        f"Crowd features file not found:\n"
        f"{crowd_csv}"
    )


# ===================================================
# LOAD CSV FILES
# ===================================================

print("\nLoading motion features...")

motion_df = pd.read_csv(
    motion_csv
)


print(
    "Motion feature rows:",
    len(motion_df)
)


print(
    "Motion feature columns:"
)

print(
    motion_df.columns.tolist()
)


print("\nLoading crowd features...")

crowd_df = pd.read_csv(
    crowd_csv
)


print(
    "Crowd feature rows:",
    len(crowd_df)
)


print(
    "Crowd feature columns:"
)

print(
    crowd_df.columns.tolist()
)


# ===================================================
# VALIDATE REQUIRED COLUMNS
# ===================================================

required_motion_columns = [

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


required_crowd_columns = [

    "frame",

    "predicted_count",

    "mean_density",

    "max_density",

    "std_density"

]


missing_motion = [

    column

    for column in required_motion_columns

    if column not in motion_df.columns

]


missing_crowd = [

    column

    for column in required_crowd_columns

    if column not in crowd_df.columns

]


if missing_motion:

    raise ValueError(
        f"Missing motion columns:\n"
        f"{missing_motion}"
    )


if missing_crowd:

    raise ValueError(
        f"Missing crowd columns:\n"
        f"{missing_crowd}"
    )


# ===================================================
# MERGE FEATURES
# ===================================================

print(
    "\nMerging features..."
)


fused_df = motion_df.merge(

    crowd_df,

    how="left",

    left_on="frame_t_plus_1",

    right_on="frame"

)


# ===================================================
# REMOVE DUPLICATE FRAME COLUMN
# ===================================================

fused_df = fused_df.drop(
    columns=[
        "frame"
    ]
)


# ===================================================
# CHECK FOR MISSING VALUES
# ===================================================

print(
    "\nChecking missing values..."
)


missing_values = fused_df.isnull().sum()


print(
    missing_values[
        missing_values > 0
    ]
)


# ===================================================
# REMOVE ROWS WITH MISSING CROWD FEATURES
# ===================================================

before_rows = len(
    fused_df
)


fused_df = fused_df.dropna()


after_rows = len(
    fused_df
)


print(
    "\nRows before removing missing values:",
    before_rows
)


print(
    "Rows after removing missing values:",
    after_rows
)


# ===================================================
# SAVE FUSED FEATURES
# ===================================================

os.makedirs(
    os.path.dirname(
        output_csv
    ),
    exist_ok=True
)


fused_df.to_csv(

    output_csv,

    index=False

)


# ===================================================
# SUMMARY
# ===================================================

print(
    "\n=================================================="
)

print(
    "FEATURE FUSION COMPLETE"
)

print(
    "=================================================="
)


print(
    "\nOutput File:"
)

print(
    output_csv
)


print(
    "\nTotal Fused Rows:"
)

print(
    len(fused_df)
)


print(
    "\nFinal Feature Columns:"
)


for column in fused_df.columns:

    print(
        "-",
        column
    )


print(
    "\nFirst 5 Rows:"
)


print(
    fused_df.head()
)


print(
    "\n=================================================="
)