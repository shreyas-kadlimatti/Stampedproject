import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model


# ==========================================================
# PROJECT ROOT
# ==========================================================

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


# ==========================================================
# PATHS
# ==========================================================

OUTPUTS_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm_risk_model.keras"
)


OUTPUT_CSV = os.path.join(
    OUTPUTS_DIR,
    "lstm_risk_predictions.csv"
)


# ==========================================================
# SETTINGS
# ==========================================================

SEQUENCE_LENGTH = 10


# ==========================================================
# IMPORTANT:
# THESE MUST MATCH THE FEATURES USED DURING LSTM TRAINING
# ==========================================================

FEATURE_COLUMNS = [

    "predicted_count",

    "mean_density",

    "mean_magnitude",

    "std_magnitude",

    "moving_pixel_ratio",

    "normalized_directional_entropy",

    "count_change_abs",

    "density_change_abs",

    "motion_change_abs",

    "motion_acceleration_abs",

    "moving_ratio_change_abs",

    "count_rolling_std",

    "motion_rolling_std"

]


# ==========================================================
# CHECK OUTPUT DIRECTORY
# ==========================================================

print(
    "\n=================================================="
)

print(
    "LSTM RISK PREDICTION"
)

print(
    "=================================================="
)


print(
    "\nProject root:"
)

print(
    BASE_DIR
)


print(
    "\nOutputs directory:"
)

print(
    OUTPUTS_DIR
)


if not os.path.isdir(
    OUTPUTS_DIR
):

    raise FileNotFoundError(
        f"\nOutputs directory not found:\n"
        f"{OUTPUTS_DIR}"
    )


# ==========================================================
# CHECK MODEL FILE
# ==========================================================

if not os.path.isfile(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"\nLSTM model not found:\n"
        f"{MODEL_PATH}"
    )


print(
    "\nLSTM model found:"
)

print(
    MODEL_PATH
)


# ==========================================================
# FIND INPUT CSV AUTOMATICALLY
# ==========================================================

print(
    "\nSearching for feature CSV..."
)


csv_files = [

    file_name

    for file_name in os.listdir(
        OUTPUTS_DIR
    )

    if file_name.lower().endswith(
        ".csv"
    )

    and file_name != os.path.basename(
        OUTPUT_CSV
    )

]


if not csv_files:

    raise FileNotFoundError(
        f"\nNo input CSV files found in:\n"
        f"{OUTPUTS_DIR}"
    )


INPUT_CSV = None

best_row_count = 0


for file_name in csv_files:

    csv_path = os.path.join(
        OUTPUTS_DIR,
        file_name
    )


    try:

        test_df = pd.read_csv(
            csv_path
        )


        has_required_features = all(

            column in test_df.columns

            for column in FEATURE_COLUMNS

        )


        if has_required_features:

            row_count = len(
                test_df
            )


            # Prefer the largest valid feature file.
            # This helps select the full 1999-row dataset.
            if row_count > best_row_count:

                INPUT_CSV = csv_path

                best_row_count = row_count


    except Exception as error:

        print(
            f"\nSkipping CSV:"
        )

        print(
            file_name
        )

        print(
            "Reason:",
            error
        )


# ==========================================================
# CHECK INPUT CSV FOUND
# ==========================================================

if INPUT_CSV is None:

    print(
        "\n=================================================="
    )

    print(
        "NO SUITABLE INPUT CSV FOUND"
    )

    print(
        "=================================================="
    )


    print(
        "\nCSV files checked:"
    )


    for file_name in csv_files:

        print(
            "-",
            file_name
        )


    print(
        "\nRequired features:"
    )


    for column in FEATURE_COLUMNS:

        print(
            "-",
            column
        )


    raise ValueError(
        "\nCould not find a CSV containing "
        "all required LSTM features."
    )


# ==========================================================
# LOAD INPUT DATA
# ==========================================================

print(
    "\n=================================================="
)

print(
    "INPUT CSV SELECTED"
)

print(
    "=================================================="
)


print(
    INPUT_CSV
)


print(
    "\nLoading sequence features..."
)


df = pd.read_csv(
    INPUT_CSV
)


print(
    "Total rows:",
    len(df)
)


print(
    "\nAvailable columns:"
)


for column in df.columns:

    print(
        "-",
        column
    )


# ==========================================================
# VERIFY REQUIRED FEATURES
# ==========================================================

missing_columns = [

    column

    for column in FEATURE_COLUMNS

    if column not in df.columns

]


if missing_columns:

    print(
        "\nMissing columns:"
    )


    for column in missing_columns:

        print(
            "-",
            column
        )


    raise ValueError(
        "\nRequired LSTM features are missing."
    )


# ==========================================================
# DISPLAY FEATURES
# ==========================================================

print(
    "\n=================================================="
)

print(
    "LSTM INPUT FEATURES"
)

print(
    "=================================================="
)


for index, column in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"{index}. {column}"
    )


print(
    "\nTotal features:",
    len(FEATURE_COLUMNS)
)


# ==========================================================
# EXTRACT FEATURES
# ==========================================================

df_features = df[
    FEATURE_COLUMNS
].copy()


# ==========================================================
# CONVERT TO NUMERIC
# ==========================================================

for column in FEATURE_COLUMNS:

    df_features[column] = pd.to_numeric(

        df_features[column],

        errors="coerce"

    )


# ==========================================================
# HANDLE INFINITE VALUES
# ==========================================================

df_features = df_features.replace(

    [
        np.inf,
        -np.inf
    ],

    np.nan

)


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

print(
    "\nMissing values before filling:",
    int(
        df_features.isna().sum().sum()
    )
)


df_features = df_features.fillna(
    0
)


print(
    "Missing values after filling:",
    int(
        df_features.isna().sum().sum()
    )
)


# ==========================================================
# NORMALIZE FEATURES
# ==========================================================

print(
    "\nNormalizing features..."
)


# IMPORTANT:
# Ideally, this scaler should be the SAME scaler used
# during training. For now, this reproduces the current
# workflow using StandardScaler on the full feature dataset.

scaler = StandardScaler()


scaled_features = scaler.fit_transform(
    df_features
)


scaled_features = scaled_features.astype(
    np.float32
)


print(
    "Scaled feature shape:",
    scaled_features.shape
)


# ==========================================================
# CREATE LSTM SEQUENCES
# ==========================================================

print(
    "\nCreating LSTM sequences..."
)


X_sequences = []

sequence_indices = []


for index in range(
    SEQUENCE_LENGTH,
    len(scaled_features)
):

    sequence = scaled_features[

        index - SEQUENCE_LENGTH:
        index

    ]


    X_sequences.append(
        sequence
    )


    sequence_indices.append(
        index
    )


X_sequences = np.array(

    X_sequences,

    dtype=np.float32

)


if len(X_sequences) == 0:

    raise ValueError(
        "\nNo LSTM sequences could be created."
        "\nCheck that the CSV has more rows than "
        "SEQUENCE_LENGTH."
    )


print(
    "\nLSTM Input Shape:"
)

print(
    X_sequences.shape
)


# ==========================================================
# LOAD TRAINED LSTM MODEL
# ==========================================================

print(
    "\nLoading trained LSTM model..."
)


model = load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)


# ==========================================================
# DISPLAY MODEL INPUT SHAPE
# ==========================================================

print(
    "\n=================================================="
)

print(
    "MODEL INPUT VALIDATION"
)

print(
    "=================================================="
)


print(
    "\nModel input shape:"
)

print(
    model.input_shape
)


expected_sequence_length = model.input_shape[1]

expected_feature_count = model.input_shape[2]


actual_sequence_length = X_sequences.shape[1]

actual_feature_count = X_sequences.shape[2]


print(
    "\nModel expected sequence length:",
    expected_sequence_length
)


print(
    "Actual sequence length:",
    actual_sequence_length
)


print(
    "\nModel expected feature count:",
    expected_feature_count
)


print(
    "Actual feature count:",
    actual_feature_count
)


# ==========================================================
# VALIDATE SEQUENCE LENGTH
# ==========================================================

if actual_sequence_length != expected_sequence_length:

    raise ValueError(

        f"\nSEQUENCE LENGTH MISMATCH\n"

        f"Model expects: "
        f"{expected_sequence_length}\n"

        f"Actual input: "
        f"{actual_sequence_length}"

    )


# ==========================================================
# VALIDATE FEATURE COUNT
# ==========================================================

if actual_feature_count != expected_feature_count:

    raise ValueError(

        f"\nFEATURE COUNT MISMATCH\n"

        f"Model expects: "
        f"{expected_feature_count} features\n"

        f"Actual input: "
        f"{actual_feature_count} features\n"

        f"\nCurrent feature list:\n"
        f"{FEATURE_COLUMNS}"

    )


print(
    "\nInput shape matches the trained LSTM model."
)


# ==========================================================
# GENERATE PREDICTIONS
# ==========================================================

print(
    "\n=================================================="
)

print(
    "GENERATING LSTM PREDICTIONS"
)

print(
    "=================================================="
)


predictions = model.predict(

    X_sequences,

    verbose=1

)


# ==========================================================
# CONVERT PREDICTIONS
# ==========================================================

predictions = np.asarray(
    predictions
)


print(
    "\nRaw prediction shape:"
)

print(
    predictions.shape
)


# ==========================================================
# HANDLE SINGLE OUTPUT MODEL
# ==========================================================

if predictions.ndim == 2:

    if predictions.shape[1] == 1:

        predictions = predictions.flatten()


    else:

        raise ValueError(

            "\nThe LSTM model returned multiple outputs "
            "per sequence."

            f"\nPrediction shape: "
            f"{predictions.shape}"

            "\nThis prediction script currently expects "
            "one risk value per sequence."

        )


elif predictions.ndim == 1:

    predictions = predictions.flatten()


else:

    raise ValueError(

        "\nUnexpected prediction shape: "
        f"{predictions.shape}"

    )


# ==========================================================
# ADD PREDICTIONS TO DATAFRAME
# ==========================================================

df["lstm_prediction"] = np.nan


for prediction_index, dataframe_index in enumerate(
    sequence_indices
):

    df.loc[

        dataframe_index,

        "lstm_prediction"

    ] = float(

        predictions[
            prediction_index
        ]

    )


# ==========================================================
# ADD SEQUENCE INFORMATION
# ==========================================================

df["lstm_sequence_length"] = np.nan

df["lstm_sequence_start_index"] = np.nan

df["lstm_sequence_end_index"] = np.nan


for dataframe_index in sequence_indices:

    df.loc[

        dataframe_index,

        "lstm_sequence_length"

    ] = SEQUENCE_LENGTH


    df.loc[

        dataframe_index,

        "lstm_sequence_start_index"

    ] = dataframe_index - SEQUENCE_LENGTH


    df.loc[

        dataframe_index,

        "lstm_sequence_end_index"

    ] = dataframe_index - 1


# ==========================================================
# OPTIONAL RISK LEVEL FROM LSTM PREDICTION
# ==========================================================

# The exact thresholds depend on the scale of the target
# used during LSTM training.

def get_lstm_risk_level(
    value
):

    if pd.isna(
        value
    ):

        return np.nan


    # If your model predicts normalized risk between 0 and 1

    if value < 0.33:

        return "Low"


    elif value < 0.66:

        return "Moderate"


    else:

        return "High"


df[
    "lstm_prediction_level"
] = df[
    "lstm_prediction"
].apply(
    get_lstm_risk_level
)


# ==========================================================
# SAVE OUTPUT
# ==========================================================

os.makedirs(

    OUTPUTS_DIR,

    exist_ok=True

)


print(
    "\nSaving output CSV..."
)


df.to_csv(

    OUTPUT_CSV,

    index=False

)


# ==========================================================
# SUMMARY
# ==========================================================

print(
    "\n=================================================="
)

print(
    "LSTM PREDICTION COMPLETE"
)

print(
    "=================================================="
)


print(
    "\nInput CSV:"
)

print(
    INPUT_CSV
)


print(
    "\nOutput CSV:"
)

print(
    OUTPUT_CSV
)


print(
    "\nTotal rows:"
)

print(
    len(df)
)


print(
    "\nSequence length:"
)

print(
    SEQUENCE_LENGTH
)


print(
    "\nTotal LSTM predictions:"
)

print(
    len(predictions)
)


print(
    "\nRows without LSTM prediction:"
)

print(
    int(
        df[
            "lstm_prediction"
        ].isna().sum()
    )
)


print(
    "\nRows with LSTM prediction:"
)

print(
    int(
        df[
            "lstm_prediction"
        ].notna().sum()
    )
)


print(
    "\nPrediction statistics:"
)


valid_predictions = df[
    "lstm_prediction"
].dropna()


print(
    "Minimum:",
    valid_predictions.min()
)


print(
    "Maximum:",
    valid_predictions.max()
)


print(
    "Mean:",
    valid_predictions.mean()
)


print(
    "Standard Deviation:",
    valid_predictions.std()
)


print(
    "\nFirst 10 predictions:"
)


columns_to_display = [

    "frame_t",

    "predicted_count",

    "risk_score",

    "actual_risk_score",

    "lstm_prediction",

    "lstm_prediction_level"

]


# Only display columns that actually exist

columns_to_display = [

    column

    for column in columns_to_display

    if column in df.columns

]


print(

    df[
        columns_to_display
    ].dropna(
        subset=[
            "lstm_prediction"
        ]
    ).head(10)

)


print(
    "\n=================================================="
)

print(
    "PROCESS FINISHED SUCCESSFULLY"
)

print(
    "=================================================="
)