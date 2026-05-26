from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")
IN_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_hit_trials_features.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_threshold_tuning_loso_results.csv"

df = pd.read_csv(IN_FILE)

feature_cols = [
    "rt_filled",
    "rt_is_missing",
    "responded",
    "correct",
    "miss",
    "error",
    "block",
    "previous_rt",
    "previous_correct",
    "previous_responded",
    "rolling_mean_rt",
    "rolling_accuracy",
    "rolling_response_rate",
]

X = df[feature_cols]
y = df["target"]
subjects = sorted(df["subject"].unique())

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    ),
}


def find_best_threshold(y_true, y_proba):
    """
    Choose the threshold that maximizes balanced accuracy.
    This is done only on inner-validation data, not on the outer test subject.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_score = -1

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


all_results = []

print("Loaded data:", df.shape)
print("\nSubjects:")
print(df["subject"].value_counts().sort_index())
print("\nTarget distribution:")
print(df["target_label"].value_counts())

for test_subject in subjects:
    print("\n" + "#" * 80)
    print(f"OUTER TEST SUBJECT: {test_subject}")
    print("#" * 80)

    outer_train_mask = df["subject"] != test_subject
    outer_test_mask = df["subject"] == test_subject

    X_outer_train = X[outer_train_mask]
    y_outer_train = y[outer_train_mask]

    X_outer_test = X[outer_test_mask]
    y_outer_test = y[outer_test_mask]

    train_subjects = sorted(df.loc[outer_train_mask, "subject"].unique())

    print("Outer train subjects:", train_subjects)
    print("Outer test subject:", test_subject)
    print("Outer test target counts:")
    print(y_outer_test.value_counts().sort_index())

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        # -----------------------------
        # Inner LOSO on training subjects only
        # -----------------------------
        inner_y_true = []
        inner_y_proba = []

        for inner_val_subject in train_subjects:
            inner_train_mask = outer_train_mask & (df["subject"] != inner_val_subject)
            inner_val_mask = outer_train_mask & (df["subject"] == inner_val_subject)

            X_inner_train = X[inner_train_mask]
            y_inner_train = y[inner_train_mask]

            X_inner_val = X[inner_val_mask]
            y_inner_val = y[inner_val_mask]

            model.fit(X_inner_train, y_inner_train)
            proba = model.predict_proba(X_inner_val)[:, 1]

            inner_y_true.extend(y_inner_val.tolist())
            inner_y_proba.extend(proba.tolist())

        inner_y_true = np.array(inner_y_true)
        inner_y_proba = np.array(inner_y_proba)

        best_threshold, inner_bal_acc = find_best_threshold(inner_y_true, inner_y_proba)

        print("Chosen threshold from inner validation:", round(best_threshold, 2))
        print("Inner validation balanced accuracy:", round(inner_bal_acc, 3))

        # -----------------------------
        # Fit on full outer train
        # -----------------------------
        model.fit(X_outer_train, y_outer_train)
        test_proba = model.predict_proba(X_outer_test)[:, 1]

        # Default threshold = 0.5
        y_pred_default = (test_proba >= 0.5).astype(int)

        # Tuned threshold
        y_pred_tuned = (test_proba >= best_threshold).astype(int)

        for threshold_type, y_pred, threshold in [
            ("default_0.5", y_pred_default, 0.5),
            ("tuned", y_pred_tuned, best_threshold),
        ]:
            bal_acc = balanced_accuracy_score(y_outer_test, y_pred)
            f1 = f1_score(y_outer_test, y_pred)
            auc = roc_auc_score(y_outer_test, test_proba)
            cm = confusion_matrix(y_outer_test, y_pred)

            print(f"\nThreshold type: {threshold_type}")
            print("Threshold:", round(threshold, 2))
            print("Confusion matrix:")
            print(cm)
            print("Balanced accuracy:", round(bal_acc, 3))
            print("F1:", round(f1, 3))
            print("ROC-AUC:", round(auc, 3))

            all_results.append({
                "test_subject": test_subject,
                "model": model_name,
                "threshold_type": threshold_type,
                "threshold": threshold,
                "inner_bal_acc": inner_bal_acc,
                "n_test": len(y_outer_test),
                "n_low_test": int((y_outer_test == 0).sum()),
                "n_high_test": int((y_outer_test == 1).sum()),
                "balanced_accuracy": bal_acc,
                "f1": f1,
                "roc_auc": auc,
                "tn": cm[0, 0],
                "fp": cm[0, 1],
                "fn": cm[1, 0],
                "tp": cm[1, 1],
            })


results = pd.DataFrame(all_results)

print("\n" + "#" * 80)
print("SUMMARY BY MODEL AND THRESHOLD TYPE")
print("#" * 80)

summary = (
    results
    .groupby(["model", "threshold_type"])[["balanced_accuracy", "f1", "roc_auc"]]
    .agg(["mean", "std"])
)

print(summary)

print("\nDetailed results:")
print(results)

results.to_csv(OUT_FILE, index=False)
print(f"\nSaved threshold tuning results to: {OUT_FILE}")