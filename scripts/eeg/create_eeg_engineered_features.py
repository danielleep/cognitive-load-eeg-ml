from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

IN_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05_log.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05_engineered.csv"

df = pd.read_csv(IN_FILE)

print("Loaded EEG log features:", df.shape)

regions = ["frontal", "central", "parietal", "occipital"]
bands = ["theta", "alpha", "beta"]

# --------------------------------------------------
# 1. Global log band power across regions
# --------------------------------------------------

for band in bands:
    cols = [f"log_{band}_{region}" for region in regions]
    df[f"log_{band}_global"] = df[cols].mean(axis=1)

# --------------------------------------------------
# 2. Band ratio features using log-power differences
# --------------------------------------------------
# Since these are log10(power), subtraction is equivalent to log10(power ratio).
# Example:
# log_theta_alpha_ratio = log10(theta_power / alpha_power)

for region in regions:
    df[f"log_theta_alpha_ratio_{region}"] = (
        df[f"log_theta_{region}"] - df[f"log_alpha_{region}"]
    )

    df[f"log_theta_beta_ratio_{region}"] = (
        df[f"log_theta_{region}"] - df[f"log_beta_{region}"]
    )

    df[f"log_alpha_beta_ratio_{region}"] = (
        df[f"log_alpha_{region}"] - df[f"log_beta_{region}"]
    )

# Global ratios
df["log_theta_alpha_ratio_global"] = (
    df["log_theta_global"] - df["log_alpha_global"]
)

df["log_theta_beta_ratio_global"] = (
    df["log_theta_global"] - df["log_beta_global"]
)

df["log_alpha_beta_ratio_global"] = (
    df["log_alpha_global"] - df["log_beta_global"]
)

# --------------------------------------------------
# 3. Basic checks
# --------------------------------------------------

feature_cols = [
    col for col in df.columns
    if col.startswith("log_theta_")
    or col.startswith("log_alpha_")
    or col.startswith("log_beta_")
]

print("\nNumber of EEG feature columns:", len(feature_cols))

print("\nTarget counts:")
print(df["target_label"].value_counts())

print("\nMissing values in engineered features:")
print(df[feature_cols].isna().sum().sum())

print("\nEngineered feature summary:")
print(df[feature_cols].describe().T.head(30))

print("\nMean engineered features by target:")
means = df.groupby("target_label")[feature_cols].mean().T
print(means.head(30))

diff = means["high"] - means["low"]
diff_df = diff.to_frame(name="high_minus_low")
diff_df["abs_diff"] = diff_df["high_minus_low"].abs()
diff_df = diff_df.sort_values("abs_diff", ascending=False)

print("\nLargest mean differences after feature engineering:")
print(diff_df.head(20))

df.to_csv(OUT_FILE, index=False)
print(f"\nSaved engineered EEG features to: {OUT_FILE}")