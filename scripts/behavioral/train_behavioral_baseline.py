from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
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
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_baseline_loso_results.csv"

df = pd.read_csv(IN_FILE)

print("Loaded data:", df.shape)
print("\nSubjects:")
print(df["subject"].value_counts().sort_index())

print("\nTarget distribution:")
print(df["target_label"].value_counts())


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
y = df["target"]  # 0 = low, 1 = high

subjects = sorted(df["subject"].unique())

models = {
    "Dummy most_frequent": DummyClassifier(strategy="most_frequent"),

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

all_results = []

for test_subject in subjects:
    print("\n" + "#" * 80)
    print(f"TEST SUBJECT: {test_subject}")
    print("#" * 80)

    train_mask = df["subject"] != test_subject
    test_mask = df["subject"] == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("Train target distribution:")
    print(y_train.value_counts().sort_index())

    print("Test target distribution:")
    print(y_test.value_counts().sort_index())

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)

        cm = confusion_matrix(y_test, y_pred)

        print("Confusion matrix:")
        print(cm)

        print("Balanced accuracy:", round(bal_acc, 3))
        print("F1 score:", round(f1, 3))
        print("ROC-AUC:", round(auc, 3) if auc is not None else None)

        all_results.append({
            "test_subject": test_subject,
            "model": model_name,
            "n_test": len(y_test),
            "n_low_test": int((y_test == 0).sum()),
            "n_high_test": int((y_test == 1).sum()),
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
print("SUMMARY BY MODEL")
print("#" * 80)

summary = (
    results
    .groupby("model")[["balanced_accuracy", "f1", "roc_auc"]]
    .agg(["mean", "std"])
)

print(summary)

print("\nDetailed results:")
print(results)

results.to_csv(OUT_FILE, index=False)
print(f"\nSaved LOSO results to: {OUT_FILE}")


# Optional: Logistic Regression coefficients trained on all subjects
print("\n" + "#" * 80)
print("LOGISTIC REGRESSION COEFFICIENTS TRAINED ON ALL SUBJECTS")
print("#" * 80)

log_reg = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

log_reg.fit(X, y)

coef = log_reg.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coef,
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

print(coef_df[["feature", "coefficient"]])