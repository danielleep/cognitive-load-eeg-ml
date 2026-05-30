from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

IN_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05_engineered.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05_session_norm.csv"

df = pd.read_csv(IN_FILE)

print("Loaded EEG engineered features:", df.shape)

feature_cols = [col for col in df.columns if col.startswith("log_")]

print("\nNumber of original EEG features:", len(feature_cols))
print(feature_cols)

group_cols = ["subject", "session"]

norm_df = df.copy()

for col in feature_cols:
    mean = norm_df.groupby(group_cols)[col].transform("mean")
    std = norm_df.groupby(group_cols)[col].transform("std")

    std = std.replace(0, np.nan)

    norm_df[f"z_{col}"] = (norm_df[col] - mean) / std
    norm_df[f"z_{col}"] = norm_df[f"z_{col}"].fillna(0)

z_feature_cols = [f"z_{col}" for col in feature_cols]

print("\nNumber of normalized EEG features:", len(z_feature_cols))

print("\nTarget counts:")
print(norm_df["target_label"].value_counts())

print("\nMissing values in normalized features:")
print(norm_df[z_feature_cols].isna().sum().sum())

print("\nZ-feature summary:")
print(norm_df[z_feature_cols].describe().T.head(30))

print("\nMean z-features by target:")
means = norm_df.groupby("target_label")[z_feature_cols].mean().T
print(means.head(30))

diff = means["high"] - means["low"]
diff_df = diff.to_frame(name="high_minus_low")
diff_df["abs_diff"] = diff_df["high_minus_low"].abs()
diff_df = diff_df.sort_values("abs_diff", ascending=False)

print("\nLargest mean differences after session normalization:")
print(diff_df.head(20))

norm_df.to_csv(OUT_FILE, index=False)
print(f"\nSaved session-normalized EEG features to: {OUT_FILE}")