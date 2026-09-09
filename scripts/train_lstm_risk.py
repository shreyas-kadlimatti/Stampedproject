import os
import sys
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


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
# RANDOM SEED
# ==========================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ==========================================================
# SETTINGS
# ==========================================================

SEQUENCE_LENGTH = 10

TEST_RATIO = 0.20

EPOCHS = 50

BATCH_SIZE = 32


# ==========================================================
# PATHS
# ==========================================================

INPUT_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "risk_scores.csv"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_risk_model.keras"
)


# Test predictions only
TEST_OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "lstm_test_predictions.csv"
)


# Predictions for entire dataset
ALL_OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "all_lstm_predictions.csv"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================================
# CHECK INPUT FILE
# ==========================================================

print("=" * 70)

print("INPUT CSV:")

print(INPUT_CSV)

print("=" * 70)


if not os.path.isfile(INPUT_CSV):

    raise FileNotFoundError(
        f"\nRisk score CSV not found:\n"
        f"{INPUT_CSV}"
    )


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    INPUT_CSV
)


print(
    "\nDataset Shape:",
    df.shape
)


print(
    "\nAvailable Columns:"
)


for column in df.columns:

    print(
        "-",
        column
    )


# ==========================================================
# FEATURE COLUMNS
# ==========================================================

FEATURE_COLUMNS = [

    # Crowd features

    "predicted_count",

    "mean_density",


    # Motion features

    "mean_magnitude",

    "moving_pixel_ratio",


    # Directional feature

    "normalized_directional_entropy",


    # Temporal change features

    "count_change_abs",

    "density_change_abs",

    "motion_change_abs",

    "motion_acceleration_abs",

    "moving_ratio_change_abs",


    # Temporal variability features

    "count_rolling_std",

    "motion_rolling_std",

    "entropy_rolling_std"

]


# ==========================================================
# TARGET COLUMN
# ==========================================================

TARGET_COLUMN = "risk_score"


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = (
    FEATURE_COLUMNS
    +
    [TARGET_COLUMN]
)


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
        "\nRequired features are missing "
        "from the input CSV."
    )


# ==========================================================
# SORT DATA TEMPORALLY
# ==========================================================

if "frame_number" in df.columns:

    df = df.sort_values(
        "frame_number"
    ).reset_index(
        drop=True
    )


print(
    "\nTotal Frame Pair Records:",
    len(df)
)


# ==========================================================
# HANDLE INVALID VALUES
# ==========================================================

df[
    FEATURE_COLUMNS
] = df[
    FEATURE_COLUMNS
].replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


df[
    FEATURE_COLUMNS
] = df[
    FEATURE_COLUMNS
].fillna(
    0
)


df[
    TARGET_COLUMN
] = df[
    TARGET_COLUMN
].replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


df[
    TARGET_COLUMN
] = df[
    TARGET_COLUMN
].fillna(
    0
)


# ==========================================================
# EXTRACT FEATURES AND TARGET
# ==========================================================

X_raw = df[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)


y_raw = df[
    TARGET_COLUMN
].values.astype(
    np.float32
)


print(
    "\nNumber of Features:",
    len(FEATURE_COLUMNS)
)


print(
    "\nFeature Names:"
)


for feature in FEATURE_COLUMNS:

    print(
        "-",
        feature
    )


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================
# Temporal split:
#
# Earlier 80% -> Training
# Later 20%   -> Testing
# ==========================================================

split_index = int(
    len(df)
    *
    (
        1
        -
        TEST_RATIO
    )
)


print(
    "\nTraining Split Index:",
    split_index
)


print(
    "Testing Records:",
    len(df)
    -
    split_index
)


# ==========================================================
# NORMALIZE FEATURES
# ==========================================================
# IMPORTANT:
# Fit scaler ONLY on training data.
# This prevents future test data leakage.
# ==========================================================

scaler = StandardScaler()


scaler.fit(
    X_raw[
        :split_index
    ]
)


X_scaled = scaler.transform(
    X_raw
).astype(
    np.float32
)


# ==========================================================
# CREATE LSTM SEQUENCES
# ==========================================================

def create_sequences(
    features,
    targets,
    sequence_length
):

    X_sequences = []

    y_sequences = []

    target_indices = []


    for end_index in range(
        sequence_length - 1,
        len(features)
    ):

        start_index = (
            end_index
            -
            sequence_length
            +
            1
        )


        sequence = features[
            start_index:
            end_index + 1
        ]


        target = targets[
            end_index
        ]


        X_sequences.append(
            sequence
        )


        y_sequences.append(
            target
        )


        target_indices.append(
            end_index
        )


    return (

        np.array(
            X_sequences,
            dtype=np.float32
        ),

        np.array(
            y_sequences,
            dtype=np.float32
        ),

        np.array(
            target_indices
        )

    )


# ==========================================================
# CREATE SEQUENCES FOR COMPLETE DATASET
# ==========================================================

X_sequences, y_sequences, target_indices = create_sequences(

    X_scaled,

    y_raw,

    SEQUENCE_LENGTH

)


print(
    "\nComplete Sequence Shape:",
    X_sequences.shape
)


print(
    "Complete Target Shape:",
    y_sequences.shape
)


print(
    "\nTotal Possible LSTM Predictions:",
    len(X_sequences)
)


# ==========================================================
# SPLIT SEQUENCES
# ==========================================================
# Split based on target frame position.
# ==========================================================

train_mask = (
    target_indices
    <
    split_index
)


test_mask = (
    target_indices
    >=
    split_index
)


X_train = X_sequences[
    train_mask
]


y_train = y_sequences[
    train_mask
]


X_test = X_sequences[
    test_mask
]


y_test = y_sequences[
    test_mask
]


test_indices = target_indices[
    test_mask
]


print(
    "\nTraining Sequences:",
    len(X_train)
)


print(
    "Testing Sequences:",
    len(X_test)
)


# ==========================================================
# BUILD LSTM MODEL
# ==========================================================

model = Sequential(

    [

        LSTM(

            64,

            return_sequences=True,

            input_shape=(

                SEQUENCE_LENGTH,

                len(
                    FEATURE_COLUMNS
                )

            )

        ),


        Dropout(
            0.20
        ),


        LSTM(
            32
        ),


        Dropout(
            0.20
        ),


        Dense(

            16,

            activation="relu"

        ),


        Dense(
            1
        )

    ]

)


# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer="adam",

    loss="mse",

    metrics=[

        "mae"

    ]

)


print(
    "\n"
    +
    "="
    *
    70
)


print(
    "LSTM MODEL SUMMARY"
)


print(
    "="
    *
    70
)


model.summary()


# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)


# ==========================================================
# TRAIN MODEL
# ==========================================================

print(
    "\nStarting LSTM Training..."
)


history = model.fit(

    X_train,

    y_train,


    validation_split=0.20,


    epochs=EPOCHS,


    batch_size=BATCH_SIZE,


    callbacks=[

        early_stopping

    ],


    verbose=1

)


# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    MODEL_PATH
)


print(
    "\nModel Saved:"
)


print(
    MODEL_PATH
)


# ==========================================================
# TEST SET PREDICTIONS
# ==========================================================

print(
    "\nGenerating Test Predictions..."
)


predicted_risk_test = model.predict(

    X_test,

    verbose=0

).flatten()


# ==========================================================
# TEST EVALUATION
# ==========================================================

mae = np.mean(

    np.abs(

        predicted_risk_test
        -
        y_test

    )

)


rmse = np.sqrt(

    np.mean(

        (

            predicted_risk_test
            -
            y_test

        )
        **
        2

    )

)


print(
    "\n"
    +
    "="
    *
    70
)


print(
    "LSTM TEST EVALUATION"
)


print(
    "="
    *
    70
)


print(

    "MAE:",

    f"{mae:.6f}"

)


print(

    "RMSE:",

    f"{rmse:.6f}"

)


# ==========================================================
# RISK LEVEL FUNCTION
# ==========================================================

def risk_level_from_score(
    score
):

    if score < 0.33:

        return "Low"


    elif score < 0.66:

        return "Moderate"


    else:

        return "High"


# ==========================================================
# CREATE TEST PREDICTION OUTPUT
# ==========================================================

test_prediction_rows = df.iloc[
    test_indices
].copy()


test_prediction_rows[
    "actual_risk_score"
] = y_test


test_prediction_rows[
    "lstm_predicted_risk_score"
] = predicted_risk_test


test_prediction_rows[
    "actual_risk_level"
] = [

    risk_level_from_score(
        score
    )

    for score in y_test

]


test_prediction_rows[
    "lstm_predicted_risk_level"
] = [

    risk_level_from_score(
        score
    )

    for score in predicted_risk_test

]


# ==========================================================
# SAVE TEST PREDICTIONS
# ==========================================================

test_prediction_rows.to_csv(

    TEST_OUTPUT_CSV,

    index=False

)


print(
    "\nTest Prediction CSV Saved:"
)


print(
    TEST_OUTPUT_CSV
)


# ==========================================================
# PREDICT FOR ALL POSSIBLE SEQUENCES
# ==========================================================
# For 1999 frame pairs and sequence length 10:
#
# Predictions = 1999 - 9 = 1990
#
# The first 9 rows do not have enough history.
# ==========================================================

print(
    "\n"
    +
    "="
    *
    70
)


print(
    "GENERATING PREDICTIONS FOR ALL SEQUENCES"
)


print(
    "="
    *
    70
)


predicted_risk_all = model.predict(

    X_sequences,

    verbose=1

).flatten()


# ==========================================================
# CREATE COMPLETE PREDICTION DATAFRAME
# ==========================================================

all_prediction_rows = df.iloc[
    target_indices
].copy()


all_prediction_rows[
    "actual_risk_score"
] = y_sequences


all_prediction_rows[
    "lstm_predicted_risk_score"
] = predicted_risk_all


all_prediction_rows[
    "actual_risk_level"
] = [

    risk_level_from_score(
        score
    )

    for score in y_sequences

]


all_prediction_rows[
    "lstm_predicted_risk_level"
] = [

    risk_level_from_score(
        score
    )

    for score in predicted_risk_all

]


# ==========================================================
# ADD SEQUENCE INFORMATION
# ==========================================================

all_prediction_rows[
    "sequence_length"
] = SEQUENCE_LENGTH


all_prediction_rows[
    "sequence_start_frame_index"
] = (

    target_indices
    -
    SEQUENCE_LENGTH
    +
    1

)


all_prediction_rows[
    "sequence_end_frame_index"
] = target_indices


# ==========================================================
# SAVE ALL PREDICTIONS
# ==========================================================

all_prediction_rows.to_csv(

    ALL_OUTPUT_CSV,

    index=False

)


print(
    "\n"
    +
    "="
    *
    70
)


print(
    "ALL LSTM PREDICTIONS SAVED"
)


print(
    "="
    *
    70
)


print(
    "\nOutput File:"
)


print(
    ALL_OUTPUT_CSV
)


print(
    "\nTotal Original Frame Pair Records:",
    len(df)
)


print(
    "Sequence Length:",
    SEQUENCE_LENGTH
)


print(
    "Total LSTM Predictions:",
    len(all_prediction_rows)
)


print(
    "Rows Without Prediction:",
    SEQUENCE_LENGTH - 1
)


# ==========================================================
# SAMPLE ALL DATASET RESULTS
# ==========================================================

print(
    "\n"
    +
    "="
    *
    70
)


print(
    "SAMPLE ALL-DATASET PREDICTIONS"
)


print(
    "="
    *
    70
)


display_columns = [

    "frame_t",

    "frame_t_plus_1",

    "actual_risk_score",

    "lstm_predicted_risk_score",

    "actual_risk_level",

    "lstm_predicted_risk_level"

]


available_display_columns = [

    column

    for column in display_columns

    if column in all_prediction_rows.columns

]


print(

    all_prediction_rows[
        available_display_columns
    ].head(
        15
    )

)


print(
    "\n"
    +
    "="
    *
    70
)


print(
    "LSTM TRAINING AND ALL-DATASET PREDICTION COMPLETED"
)


print(
    "="
    *
    70
)