from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")
CSV_DIR = PROJECT_ROOT / "data" / "processed" / "behavioral_csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_trials_sub01_sub05.csv"

csv_files = sorted(CSV_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    # Safety check
    required_cols = {
        "rt", "correct", "miss", "error", "condition",
        "block", "outlier", "subject", "session", "nback_level", "load_label"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file.name} is missing columns: {missing}")

    # Add source filename for traceability
    df["source_file"] = file.name

    dfs.append(df)

behavior = pd.concat(dfs, ignore_index=True)

# Add trial index within each subject/session/condition
behavior["trial_index"] = (
    behavior
    .groupby(["subject", "session", "nback_level"])
    .cumcount() + 1
)

print("\nCombined behavioral table:")
print(behavior.shape)

print("\nSubjects:")
print(sorted(behavior["subject"].unique()))

print("\nSessions:")
print(behavior["session"].value_counts().sort_index())

print("\nN-back levels:")
print(behavior["nback_level"].value_counts().sort_index())

print("\nLoad labels:")
print(behavior["load_label"].value_counts())

print("\nRT missing rate by load label:")
print(behavior.groupby("load_label")["rt"].apply(lambda x: x.isna().mean()))

print("\nCorrect counts by load label:")
print(pd.crosstab(behavior["load_label"], behavior["correct"]))

print("\nOutlier counts:")
print(behavior["outlier"].value_counts(dropna=False))

behavior.to_csv(OUT_FILE, index=False)
print(f"\nSaved to: {OUT_FILE}")