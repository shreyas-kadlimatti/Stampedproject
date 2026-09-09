import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)


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


# ==========================================================
# FIND PREDICTION CSV
# ==========================================================

print(
    "\nSearching for LSTM prediction CSV..."
)


if not os.path.isdir(
    OUTPUTS_DIR
):

    raise FileNotFoundError(
        f"Outputs directory not found:\n"
        f"{OUTPUTS_DIR}"
    )


csv_files = [

    file_name

    for file_name in os.listdir(
        OUTPUTS_DIR
    )

    if file_name.lower().endswith(
        ".csv"
    )

]


if not csv_files:

    raise FileNotFoundError(
        f"No CSV files found in:\n"
        f"{OUTPUTS_DIR}"
    )


INPUT_CSV = None


REQUIRED_COLUMNS = [

    "actual_risk_score",

    "lstm_predicted_risk_score"

]


for file_name in csv_files:

    csv_path = os.path.join(
        OUTPUTS_DIR,
        file_name
    )

    try:

        test_df = pd.read_csv(
            csv_path,
            nrows=5
        )

        if all(

            column in test_df.columns

            for column in REQUIRED_COLUMNS

        ):

            INPUT_CSV = csv_path

            print(
                "\nCorrect prediction CSV found:"
            )

            print(
                INPUT_CSV
            )

            break


    except Exception as error:

        print(
            f"Skipping {file_name}: "
            f"{error}"
        )


if INPUT_CSV is None:

    print(
        "\nCSV files found:"
    )

    for file_name in csv_files:

        print(
            "-",
            file_name
        )


    raise ValueError(
        "\nCould not find a CSV containing "
        "actual_risk_score and "
        "lstm_predicted_risk_score."
    )


# ==========================================================
# LOAD DATA
# ==========================================================

print(
    "\nLoading prediction data..."
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
# REQUIRED COLUMNS
# ==========================================================

REQUIRED_COLUMNS = [

    "actual_risk_score",

    "lstm_predicted_risk_score"

]


missing_columns = [

    column

    for column in REQUIRED_COLUMNS

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
        "\nRequired evaluation columns are missing."
    )


# ==========================================================
# SELECT VALID PREDICTIONS
# ==========================================================

evaluation_df = df[

    REQUIRED_COLUMNS

].copy()


# Convert to numeric

evaluation_df[
    "actual_risk_score"
] = pd.to_numeric(

    evaluation_df[
        "actual_risk_score"
    ],

    errors="coerce"

)


evaluation_df[
    "lstm_predicted_risk_score"
] = pd.to_numeric(

    evaluation_df[
        "lstm_predicted_risk_score"
    ],

    errors="coerce"

)


# Remove missing values

evaluation_df = evaluation_df.dropna()


print(
    "\nValid evaluation samples:",
    len(evaluation_df)
)


if len(evaluation_df) == 0:

    raise ValueError(
        "\nNo valid rows available for evaluation."
    )


# ==========================================================
# ACTUAL AND PREDICTED VALUES
# ==========================================================

y_true = evaluation_df[
    "actual_risk_score"
].values


y_pred = evaluation_df[
    "lstm_predicted_risk_score"
].values


# ==========================================================
# REGRESSION METRICS
# ==========================================================

mae = mean_absolute_error(
    y_true,
    y_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)


r2 = r2_score(
    y_true,
    y_pred
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print(
    "\n=================================================="
)

print(
    "LSTM RISK SCORE EVALUATION"
)

print(
    "=================================================="
)


print(
    f"\nSamples Evaluated: "
    f"{len(y_true)}"
)


print(
    f"\nMAE: "
    f"{mae:.6f}"
)


print(
    f"RMSE: "
    f"{rmse:.6f}"
)


print(
    f"R² Score: "
    f"{r2:.6f}"
)


# ==========================================================
# OPTIONAL RISK LEVEL EVALUATION
# ==========================================================

RISK_LEVEL_COLUMNS = [

    "actual_risk_level",

    "lstm_predicted_risk_level"

]


if all(

    column in df.columns

    for column in RISK_LEVEL_COLUMNS

):

    level_df = df[

        RISK_LEVEL_COLUMNS

    ].copy()


    level_df = level_df.dropna()


    if len(level_df) > 0:

        y_true_level = level_df[
            "actual_risk_level"
        ]


        y_pred_level = level_df[
            "lstm_predicted_risk_level"
        ]


        accuracy = accuracy_score(

            y_true_level,

            y_pred_level

        )


        print(
            "\n=================================================="
        )

        print(
            "RISK LEVEL CLASSIFICATION EVALUATION"
        )

        print(
            "=================================================="
        )


        print(
            f"\nClassification Accuracy: "
            f"{accuracy:.4f}"
        )


        print(
            "\nClassification Report:\n"
        )


        print(

            classification_report(

                y_true_level,

                y_pred_level,

                zero_division=0

            )

        )


        print(
            "\nConfusion Matrix:"
        )


        print(

            confusion_matrix(

                y_true_level,

                y_pred_level

            )

        )


else:

    print(
        "\nRisk-level columns not found."
    )

    print(
        "Skipping classification evaluation."
    )


# ==========================================================
# SAVE EVALUATION RESULTS
# ==========================================================

RESULTS_CSV = os.path.join(

    OUTPUTS_DIR,

    "lstm_evaluation_results.csv"

)


results_df = pd.DataFrame({

    "metric": [

        "MAE",

        "RMSE",

        "R2_SCORE",

        "TOTAL_EVALUATED_SAMPLES"

    ],

    "value": [

        mae,

        rmse,

        r2,

        len(y_true)

    ]

})


results_df.to_csv(

    RESULTS_CSV,

    index=False

)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print(
    "\n=================================================="
)

print(
    "EVALUATION COMPLETE"
)

print(
    "=================================================="
)


print(
    "\nEvaluation results saved to:"
)


print(
    RESULTS_CSV
)


print(
    "\n=================================================="
)