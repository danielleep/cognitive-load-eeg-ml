from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")
IN_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_hit_trials_features.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "behavioral_subject_summary.csv"

df = pd.read_csv(IN_FILE)

print("Loaded data:", df.shape)

# Basic counts by subject and target
counts = (
    df.groupby(["subject", "target_label"])
    .size()
    .unstack(fill_value=0)
)

print("\nCounts by subject and target:")
print(counts)

# RT summary by subject and target
rt_summary = (
    df.groupby(["subject", "target_label"])["rt_filled"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print("\nRT summary by subject and target:")
print(rt_summary)

# Accuracy / correctness summary
accuracy_summary = (
    df.groupby(["subject", "target_label"])
    .agg(
        n_trials=("target", "size"),
        mean_correct=("correct", "mean"),
        response_rate=("responded", "mean"),
        miss_rate=("miss", "mean"),
        error_rate=("error", "mean"),
        rt_missing_rate=("rt_is_missing", "mean"),
        mean_rt=("rt_filled", "mean"),
        median_rt=("rt_filled", "median"),
        mean_rolling_rt=("rolling_mean_rt", "mean"),
        mean_rolling_accuracy=("rolling_accuracy", "mean"),
    )
    .reset_index()
)

print("\nBehavioral summary by subject and target:")
print(accuracy_summary)

# Pivoted view: easier to compare low vs high per subject
pivot = accuracy_summary.pivot(
    index="subject",
    columns="target_label",
    values=["n_trials", "mean_correct", "response_rate", "rt_missing_rate", "mean_rt", "median_rt"]
)

print("\nPivot summary:")
print(pivot)

# Difference high - low per subject
wide = accuracy_summary.pivot(index="subject", columns="target_label")

diff = pd.DataFrame(index=wide.index)
diff["mean_rt_high_minus_low"] = wide[("mean_rt", "high")] - wide[("mean_rt", "low")]
diff["median_rt_high_minus_low"] = wide[("median_rt", "high")] - wide[("median_rt", "low")]
diff["correct_high_minus_low"] = wide[("mean_correct", "high")] - wide[("mean_correct", "low")]
diff["response_rate_high_minus_low"] = wide[("response_rate", "high")] - wide[("response_rate", "low")]
diff["rt_missing_high_minus_low"] = wide[("rt_missing_rate", "high")] - wide[("rt_missing_rate", "low")]

print("\nHigh - Low differences by subject:")
print(diff)

# Save one clean summary file
accuracy_summary.to_csv(OUT_FILE, index=False)
print(f"\nSaved subject summary to: {OUT_FILE}")