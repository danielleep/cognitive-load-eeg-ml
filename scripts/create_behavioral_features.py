from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

IN_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_trials_binary_clean.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_hit_trials_features.csv"

df = pd.read_csv(IN_FILE)

print("Input shape:", df.shape)

# Keep hit trials only for the first behavioral baseline
df = df[df["hittrials"] == 1].copy()

# Mark whether the participant actually responded
df["responded"] = df["rt"].notna().astype(int)

# Sort so previous/rolling features are calculated in the correct order
df = df.sort_values(
    ["subject", "session", "nback_level", "block", "trial_index_clean"]
).reset_index(drop=True)

group_cols = ["subject", "session", "nback_level"]

# Previous trial features within the same subject/session/condition
df["previous_rt"] = df.groupby(group_cols)["rt"].shift(1)
df["previous_correct"] = df.groupby(group_cols)["correct"].shift(1)
df["previous_responded"] = df.groupby(group_cols)["responded"].shift(1)

# Rolling features based only on previous trials, not the current trial
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

# Fill missing previous/rolling features in a simple, explicit way
# This is needed for baseline models like Logistic Regression.
for col in ["previous_rt", "rolling_mean_rt"]:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

for col in ["previous_correct", "previous_responded", "rolling_accuracy", "rolling_response_rate"]:
    df[col] = df[col].fillna(0)

# For current RT, keep both:
# 1. rt_is_missing indicates miss/no response
# 2. rt_filled allows models to use a numeric value
df["rt_is_missing"] = df["rt"].isna().astype(int)
df["rt_filled"] = df["rt"].fillna(df["rt"].median())

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

print("\nResponded counts by target:")
print(pd.crosstab(features["target_label"], features["responded"]))

print("\nRT summary by target:")
print(features.groupby("target_label")["rt_filled"].describe())

print("\nMissing values per column:")
print(features.isna().sum())

features.to_csv(OUT_FILE, index=False)
print(f"\nSaved to: {OUT_FILE}")