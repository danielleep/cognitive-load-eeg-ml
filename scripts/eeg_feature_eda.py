from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

IN_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05_log.csv"

df = pd.read_csv(IN_FILE)

print("Loaded EEG features:", df.shape)

print("\nTarget counts:")
print(df["target_label"].value_counts())

print("\nCounts by subject:")
print(df["subject"].value_counts().sort_index())

print("\nCounts by session:")
print(df["session"].value_counts().sort_index())

print("\nMissing values:")
print(df.isna().sum())

feature_cols = [
    col for col in df.columns
    if col.startswith(("theta_", "alpha_", "beta_"))
]

print("\nEEG feature columns:")
print(feature_cols)

print("\nRaw EEG feature summary:")
print(df[feature_cols].describe().T)

print("\nRaw EEG feature means by target:")
means_by_target = df.groupby("target_label")[feature_cols].mean().T
print(means_by_target)

# Difference high - low
diff = means_by_target["high"] - means_by_target["low"]
diff_df = diff.to_frame(name="high_minus_low")
diff_df["abs_diff"] = diff_df["high_minus_low"].abs()
diff_df = diff_df.sort_values("abs_diff", ascending=False)

print("\nLargest raw mean differences: high - low")
print(diff_df)

# Check if all feature values are positive
print("\nMinimum values per feature:")
print(df[feature_cols].min())

# Create log-transformed features
# EEG power values are positive and very small, so log10 makes them easier to model.
log_df = df.copy()

epsilon = 1e-20

for col in feature_cols:
    log_df[f"log_{col}"] = np.log10(log_df[col] + epsilon)

log_feature_cols = [f"log_{col}" for col in feature_cols]

print("\nLog EEG feature summary:")
print(log_df[log_feature_cols].describe().T)

print("\nLog EEG feature means by target:")
log_means_by_target = log_df.groupby("target_label")[log_feature_cols].mean().T
print(log_means_by_target)

log_diff = log_means_by_target["high"] - log_means_by_target["low"]
log_diff_df = log_diff.to_frame(name="high_minus_low")
log_diff_df["abs_diff"] = log_diff_df["high_minus_low"].abs()
log_diff_df = log_diff_df.sort_values("abs_diff", ascending=False)

print("\nLargest log mean differences: high - low")
print(log_diff_df)

log_df.to_csv(OUT_FILE, index=False)
print(f"\nSaved log-transformed EEG features to: {OUT_FILE}")