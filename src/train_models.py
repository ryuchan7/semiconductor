from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

HIGH_MISSING_THRESHOLD = 0.50

MIN_FAIL_RECALL = 0.40


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================================
# HELPER: THRESHOLD SEARCH
# ==========================================================

def search_threshold(
    y_true,
    scores,
    thresholds,
    min_recall=0.40,
):
    """
    Select a threshold that:

    1. Achieves at least min_recall
    2. Among those thresholds, produces the fewest false alarms
    3. Uses F2 as a secondary ranking criterion
    """

    y_binary = (
        np.asarray(y_true) == 1
    ).astype(int)

    rows = []

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_binary,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_binary,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_binary,
            predictions,
            zero_division=0,
        )

        f2 = fbeta_score(
            y_binary,
            predictions,
            beta=2,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_binary,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "F2": float(f2),
                "False Alarms": int(fp),
                "Missed FAIL": int(fn),
                "Detected FAIL": int(tp),
            }
        )

    results = pd.DataFrame(rows)

    candidates = results[
        results["Recall"] >= min_recall
    ].copy()

    if len(candidates) > 0:

        selected = (
            candidates
            .sort_values(
                by=[
                    "False Alarms",
                    "F2",
                    "Threshold",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

    else:
        # Fallback if no threshold reaches required recall
        selected = (
            results
            .sort_values(
                by="F2",
                ascending=False,
            )
            .iloc[0]
        )

    return (
        float(selected["Threshold"]),
        results,
    )


# ==========================================================
# HELPER: MODEL METRICS
# ==========================================================

def calculate_metrics(
    y_true,
    predictions,
    probability_scores,
):
    """
    Calculate metrics with FAIL = positive class.
    """

    y_binary = (
        np.asarray(y_true) == 1
    ).astype(int)

    pred_binary = (
        np.asarray(predictions) == 1
    ).astype(int)

    cm = confusion_matrix(
        y_binary,
        pred_binary,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(
            accuracy_score(
                y_binary,
                pred_binary,
            )
        ),

        "fail_precision": float(
            precision_score(
                y_binary,
                pred_binary,
                zero_division=0,
            )
        ),

        "fail_recall": float(
            recall_score(
                y_binary,
                pred_binary,
                zero_division=0,
            )
        ),

        "fail_f1": float(
            f1_score(
                y_binary,
                pred_binary,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_binary,
                probability_scores,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_binary,
                probability_scores,
            )
        ),

        "correct_pass": int(tn),
        "false_alarms": int(fp),
        "missed_fail": int(fn),
        "detected_fail": int(tp),
        "total_fail": int(fn + tp),
    }


# ==========================================================
# STEP 1: LOAD SECOM
# ==========================================================

print("=" * 65)
print("SEMICONDUCTOR AI TRAINING PIPELINE")
print("=" * 65)

print("\n[1/9] Loading UCI SECOM dataset...")

secom = fetch_ucirepo(
    id=179
)

df = secom.data.original.copy()

print(
    f"Raw dataset shape: {df.shape}"
)


# ==========================================================
# STEP 2: PREPARE RAW FEATURES / TARGET
# ==========================================================

print("\n[2/9] Preparing features and target...")

y = (
    df["class"]
    .astype(int)
)

X = df.drop(
    columns=[
        "class",
        "timestamp",
    ],
    errors="ignore",
)

# Make sure every process feature is numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce",
)

print(
    f"Raw process features: {X.shape[1]}"
)

print(
    f"PASS samples: {(y == -1).sum()}"
)

print(
    f"FAIL samples: {(y == 1).sum()}"
)


# ==========================================================
# STEP 3: TRAIN / TEST SPLIT
# ==========================================================

print("\n[3/9] Creating train/test split...")

(
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
) = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(
    f"Training samples: {len(X_train_raw)}"
)

print(
    f"Testing samples:  {len(X_test_raw)}"
)


# ==========================================================
# STEP 4: FEATURE CLEANING
# IMPORTANT: TRAINING DATA ONLY
# ==========================================================

print(
    "\n[4/9] Learning feature cleaning rules "
    "from training data only..."
)

# ----------------------------------------------------------
# Remove constant features
# ----------------------------------------------------------

constant_columns = (
    X_train_raw
    .columns[
        X_train_raw.nunique(
            dropna=True
        ) <= 1
    ]
    .tolist()
)

X_train_temp = X_train_raw.drop(
    columns=constant_columns
)

# ----------------------------------------------------------
# Remove >50% missing features
# ----------------------------------------------------------

training_missing_rate = (
    X_train_temp
    .isnull()
    .mean()
)

high_missing_columns = (
    training_missing_rate[
        training_missing_rate
        > HIGH_MISSING_THRESHOLD
    ]
    .index
    .tolist()
)

feature_names = [
    column
    for column in X_train_temp.columns
    if column not in high_missing_columns
]

X_train = X_train_raw[
    feature_names
].copy()

X_test = X_test_raw[
    feature_names
].copy()


print(
    f"Constant features removed: "
    f"{len(constant_columns)}"
)

print(
    f"High-missing features removed: "
    f"{len(high_missing_columns)}"
)

print(
    f"Final usable features: "
    f"{len(feature_names)}"
)


# ==========================================================
# STEP 5: SUBTRAIN / VALIDATION SPLIT
# ==========================================================

print(
    "\n[5/9] Creating validation set "
    "for threshold tuning..."
)

(
    X_subtrain,
    X_val,
    y_subtrain,
    y_val,
) = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

print(
    f"Sub-training samples: "
    f"{len(X_subtrain)}"
)

print(
    f"Validation samples:   "
    f"{len(X_val)}"
)


# ==========================================================
# STEP 6: RANDOM FOREST THRESHOLD TUNING
# ==========================================================

print(
    "\n[6/9] Training Random Forest "
    "for threshold tuning..."
)

rf_tune_imputer = SimpleImputer(
    strategy="median"
)

X_subtrain_rf = (
    rf_tune_imputer.fit_transform(
        X_subtrain
    )
)

X_val_rf = (
    rf_tune_imputer.transform(
        X_val
    )
)

rf_tune_model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_tune_model.fit(
    X_subtrain_rf,
    y_subtrain,
)

fail_class_index = list(
    rf_tune_model.classes_
).index(1)

rf_val_probability = (
    rf_tune_model.predict_proba(
        X_val_rf
    )[:, fail_class_index]
)

rf_threshold_candidates = np.arange(
    0.01,
    0.51,
    0.01,
)

(
    rf_threshold,
    rf_threshold_results,
) = search_threshold(
    y_true=y_val,
    scores=rf_val_probability,
    thresholds=rf_threshold_candidates,
    min_recall=MIN_FAIL_RECALL,
)

print(
    f"Selected Random Forest threshold: "
    f"{rf_threshold:.4f}"
)

rf_threshold_results.to_csv(
    RESULTS_DIR
    / "random_forest_threshold_search.csv",
    index=False,
)


# ==========================================================
# STEP 7: FINAL RANDOM FOREST
# ==========================================================

print(
    "\n[7/9] Training final Random Forest..."
)

rf_imputer = SimpleImputer(
    strategy="median"
)

X_train_rf = (
    rf_imputer.fit_transform(
        X_train
    )
)

X_test_rf = (
    rf_imputer.transform(
        X_test
    )
)

final_rf_model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_rf_model.fit(
    X_train_rf,
    y_train,
)

fail_class_index = list(
    final_rf_model.classes_
).index(1)

rf_test_probability = (
    final_rf_model.predict_proba(
        X_test_rf
    )[:, fail_class_index]
)

rf_test_prediction = np.where(
    rf_test_probability
    >= rf_threshold,
    1,
    -1,
)

rf_metrics = calculate_metrics(
    y_true=y_test,
    predictions=rf_test_prediction,
    probability_scores=rf_test_probability,
)

print("\nRandom Forest Test Results")

print(
    f"Accuracy:       "
    f"{rf_metrics['accuracy']:.3f}"
)

print(
    f"FAIL Precision: "
    f"{rf_metrics['fail_precision']:.3f}"
)

print(
    f"FAIL Recall:    "
    f"{rf_metrics['fail_recall']:.3f}"
)

print(
    f"FAIL F1:        "
    f"{rf_metrics['fail_f1']:.3f}"
)

print(
    f"ROC-AUC:        "
    f"{rf_metrics['roc_auc']:.3f}"
)

print(
    f"PR-AUC:         "
    f"{rf_metrics['pr_auc']:.3f}"
)

print(
    f"Detected FAIL:  "
    f"{rf_metrics['detected_fail']}"
    f"/{rf_metrics['total_fail']}"
)


# ==========================================================
# STEP 8: ISOLATION FOREST
# ==========================================================

print(
    "\n[8/9] Training Isolation Forest..."
)

# Train anomaly detector only on normal PASS samples
normal_mask = (
    y_subtrain == -1
)

X_normal_subtrain = (
    X_subtrain.loc[
        normal_mask
    ]
)

anomaly_imputer = SimpleImputer(
    strategy="median"
)

X_normal_iso = (
    anomaly_imputer.fit_transform(
        X_normal_subtrain
    )
)

X_val_iso = (
    anomaly_imputer.transform(
        X_val
    )
)

iso_model = IsolationForest(
    n_estimators=500,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

iso_model.fit(
    X_normal_iso
)

# Higher score = more abnormal
val_anomaly_scores = (
    -iso_model.score_samples(
        X_val_iso
    )
)

anomaly_threshold_candidates = (
    np.linspace(
        val_anomaly_scores.min(),
        val_anomaly_scores.max(),
        200,
    )
)

(
    anomaly_threshold,
    anomaly_threshold_results,
) = search_threshold(
    y_true=y_val,
    scores=val_anomaly_scores,
    thresholds=anomaly_threshold_candidates,
    min_recall=MIN_FAIL_RECALL,
)

print(
    f"Selected Isolation Forest threshold: "
    f"{anomaly_threshold:.6f}"
)

anomaly_threshold_results.to_csv(
    RESULTS_DIR
    / "isolation_forest_threshold_search.csv",
    index=False,
)


# ----------------------------------------------------------
# Evaluate anomaly detector on untouched test set
# ----------------------------------------------------------

X_test_iso = (
    anomaly_imputer.transform(
        X_test
    )
)

test_anomaly_scores = (
    -iso_model.score_samples(
        X_test_iso
    )
)

iso_test_prediction_binary = (
    test_anomaly_scores
    >= anomaly_threshold
).astype(int)

# convert back to -1 / 1 for metrics helper
iso_test_prediction = np.where(
    iso_test_prediction_binary == 1,
    1,
    -1,
)

iso_metrics = calculate_metrics(
    y_true=y_test,
    predictions=iso_test_prediction,
    probability_scores=test_anomaly_scores,
)


print("\nIsolation Forest Test Results")

print(
    f"Accuracy:       "
    f"{iso_metrics['accuracy']:.3f}"
)

print(
    f"FAIL Precision: "
    f"{iso_metrics['fail_precision']:.3f}"
)

print(
    f"FAIL Recall:    "
    f"{iso_metrics['fail_recall']:.3f}"
)

print(
    f"FAIL F1:        "
    f"{iso_metrics['fail_f1']:.3f}"
)

print(
    f"ROC-AUC:        "
    f"{iso_metrics['roc_auc']:.3f}"
)

print(
    f"PR-AUC:         "
    f"{iso_metrics['pr_auc']:.3f}"
)

print(
    f"Detected FAIL:  "
    f"{iso_metrics['detected_fail']}"
    f"/{iso_metrics['total_fail']}"
)


# ==========================================================
# STEP 9: SAVE MODELS AND PROJECT ARTIFACTS
# ==========================================================

print(
    "\n[9/9] Saving models and artifacts..."
)


# ----------------------------------------------------------
# Random Forest
# ----------------------------------------------------------

joblib.dump(
    final_rf_model,
    MODELS_DIR
    / "random_forest.pkl",
)

joblib.dump(
    rf_imputer,
    MODELS_DIR
    / "rf_imputer.pkl",
)


# ----------------------------------------------------------
# Isolation Forest
# ----------------------------------------------------------

joblib.dump(
    iso_model,
    MODELS_DIR
    / "isolation_forest.pkl",
)

joblib.dump(
    anomaly_imputer,
    MODELS_DIR
    / "anomaly_imputer.pkl",
)


# ----------------------------------------------------------
# Feature names
# ----------------------------------------------------------

joblib.dump(
    feature_names,
    MODELS_DIR
    / "feature_names.pkl",
)


# ----------------------------------------------------------
# Thresholds
# ----------------------------------------------------------

threshold_data = {
    "rf_threshold": float(
        rf_threshold
    ),
    "anomaly_threshold": float(
        anomaly_threshold
    ),
}

joblib.dump(
    threshold_data,
    MODELS_DIR
    / "thresholds.pkl",
)


# ----------------------------------------------------------
# Demo test samples
# ----------------------------------------------------------

demo_samples = (
    X_test.reset_index(
        drop=True
    )
    .copy()
)

demo_samples[
    "Actual_Class"
] = (
    y_test
    .reset_index(
        drop=True
    )
)

demo_samples.to_csv(
    DATA_DIR
    / "demo_samples.csv",
    index=False,
)


# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

metrics_data = {
    "dataset": {
        "samples": int(
            len(X)
        ),

        "original_features": int(
            X.shape[1]
        ),

        "final_features": int(
            len(feature_names)
        ),

        "constant_features_removed": int(
            len(constant_columns)
        ),

        "high_missing_features_removed": int(
            len(high_missing_columns)
        ),
    },

    "random_forest": {
        "threshold": float(
            rf_threshold
        ),
        **rf_metrics,
    },

    "isolation_forest": {
        "threshold": float(
            anomaly_threshold
        ),
        **iso_metrics,
    },
}

with open(
    RESULTS_DIR
    / "metrics.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics_data,
        file,
        indent=4,
    )


# ==========================================================
# COMPLETE
# ==========================================================

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print(
    f"\nUsable process features: "
    f"{len(feature_names)}"
)

print(
    f"Random Forest threshold: "
    f"{rf_threshold:.4f}"
)

print(
    f"Isolation threshold: "
    f"{anomaly_threshold:.6f}"
)

print("\nSaved files:")

print(
    "models/random_forest.pkl"
)

print(
    "models/rf_imputer.pkl"
)

print(
    "models/isolation_forest.pkl"
)

print(
    "models/anomaly_imputer.pkl"
)

print(
    "models/feature_names.pkl"
)

print(
    "models/thresholds.pkl"
)

print(
    "data/demo_samples.csv"
)

print(
    "results/metrics.json"
)

print(
    "results/random_forest_threshold_search.csv"
)

print(
    "results/isolation_forest_threshold_search.csv"
)

print("\nPipeline completed successfully.")