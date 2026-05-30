from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

IN_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_trials_binary_clean.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_hit_trials_features_subject_norm.csv"

df = pd.read_csv(IN_FILE)

print("Input shape:", df.shape)

# Keep hit trials only for the first behavioral baseline
df = df[df["hittrials"] == 1].copy()

# Mark whether participant actually responded
df["responded"] = df["rt"].notna().astype(int)
df["rt_is_missing"] = df["rt"].isna().astype(int)

# Fill current RT with global median first, so every row has a numeric RT
global_rt_median = df["rt"].median()
df["rt_filled"] = df["rt"].fillna(global_rt_median)

# Sort for previous/rolling features
df = df.sort_values(
    ["subject", "session", "nback_level", "block", "trial_index_clean"]
).reset_index(drop=True)

group_cols = ["subject", "session", "nback_level"]

# Previous trial features
df["previous_rt"] = df.groupby(group_cols)["rt"].shift(1)
df["previous_correct"] = df.groupby(group_cols)["correct"].shift(1)
df["previous_responded"] = df.groupby(group_cols)["responded"].shift(1)

# Rolling features based only on previous trials
df["rolling_mean_rt"] = (
    df.groupby(group_cols)["rt"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

df["rolling_accuracy"] = (
    df.groupby(group_cols)["correct"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

df["rolling_response_rate"] = (
    df.groupby(group_cols)["responded"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

# Fill missing previous/rolling values
for col in ["previous_rt", "rolling_mean_rt"]:
    df[col] = df[col].fillna(df[col].median())

for col in ["previous_correct", "previous_responded", "rolling_accuracy", "rolling_response_rate"]:
    df[col] = df[col].fillna(0)

# -----------------------------
# Subject-wise RT normalization
# -----------------------------
# Compute mean/std using each subject's hit-trial RT distribution.
# This is exploratory and helps reduce between-subject RT scale differences.
subject_rt_stats = (
    df.groupby("subject")["rt_filled"]
    .agg(["mean", "std"])
    .rename(columns={"mean": "subject_rt_mean", "std": "subject_rt_std"})
)

df = df.merge(subject_rt_stats, on="subject", how="left")

# Avoid division by zero just in case
df["subject_rt_std"] = df["subject_rt_std"].replace(0, np.nan)

df["rt_z_subject"] = (
    (df["rt_filled"] - df["subject_rt_mean"]) / df["subject_rt_std"]
)

df["previous_rt_z_subject"] = (
    (df["previous_rt"] - df["subject_rt_mean"]) / df["subject_rt_std"]
)

df["rolling_mean_rt_z_subject"] = (
    (df["rolling_mean_rt"] - df["subject_rt_mean"]) / df["subject_rt_std"]
)

# Fill any remaining NaNs defensively
for col in ["rt_z_subject", "previous_rt_z_subject", "rolling_mean_rt_z_subject"]:
    df[col] = df[col].fillna(0)

feature_cols = [
    # original behavioral features
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

    # subject-normalized RT features
    "rt_z_subject",
    "previous_rt_z_subject",
    "rolling_mean_rt_z_subject",
]

id_cols = [
    "subject",
    "session",
    "nback_level",
    "target",
    "target_label",
    "trial_index_clean",
    "source_file",
]

keep_cols = id_cols + feature_cols

features = df[keep_cols].copy()

print("Feature table shape:", features.shape)

print("\nTarget counts:")
print(features["target_label"].value_counts())

print("\nRT z-score summary by subject and target:")
print(
    features.groupby(["subject", "target_label"])["rt_z_subject"]
    .agg(["mean", "median", "std"])
)

print("\nMissing values per column:")
print(features.isna().sum())

features.to_csv(OUT_FILE, index=False)
print(f"\nSaved to: {OUT_FILE}")