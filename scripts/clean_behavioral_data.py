from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")
IN_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_trials_sub01_sub05.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_trials_binary_clean.csv"

df = pd.read_csv(IN_FILE)

print("Original shape:", df.shape)

# Keep only low and high load for initial binary classification
df = df[df["nback_level"].isin([0, 2])].copy()

# Remove outliers
df = df[df["outlier"] == 0].copy()

# Add binary target
df["target"] = df["nback_level"].map({0: 0, 2: 1})
df["target_label"] = df["nback_level"].map({0: "low", 2: "high"})

# Add response indicator
df["has_rt"] = df["rt"].notna().astype(int)

# Create trial index after filtering
df["trial_index_clean"] = (
    df.groupby(["subject", "session", "nback_level"]).cumcount() + 1
)

print("After keeping 0-back and 2-back, removing outliers:", df.shape)

print("\nTarget counts:")
print(df["target_label"].value_counts())

print("\nRT missing rate by target:")
print(df.groupby("target_label")["rt"].apply(lambda x: x.isna().mean()))

print("\nCorrect counts by target:")
print(pd.crosstab(df["target_label"], df["correct"]))

print("\nHit trials by target:")
print(pd.crosstab(df["target_label"], df["hittrials"]))

# Hit-trial-only version for first behavioral baseline
hit_df = df[df["hittrials"] == 1].copy()

print("\nHit-trials only shape:", hit_df.shape)
print("\nHit-trials RT missing rate:")
print(hit_df["rt"].isna().mean())

print("\nHit-trials RT summary by target:")
print(hit_df.groupby("target_label")["rt"].describe())

print("\nHit-trials correctness by target:")
print(pd.crosstab(hit_df["target_label"], hit_df["correct"]))

df.to_csv(OUT_FILE, index=False)
print(f"\nSaved clean binary behavioral table to: {OUT_FILE}")