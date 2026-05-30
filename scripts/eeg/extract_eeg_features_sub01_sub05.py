from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sub05.csv"

SUBJECTS_TO_USE = {f"sub-{i:02d}" for i in range(1, 6)}
CONDITIONS_TO_USE = {"zeroBACK", "twoBACK"}

CONDITION_INFO = {
    "zeroBACK": {"target": 0, "target_label": "low"},
    "twoBACK": {"target": 1, "target_label": "high"},
}

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

REGIONS = {
    "frontal": [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6"
    ],
    "central": [
        "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"
    ],
    "parietal": [
        "P7", "P3", "Pz", "P4", "P8", "POz"
    ],
    "occipital": [
        "O1", "Oz", "O2"
    ],
}


def parse_subject_session_condition(path: Path):
    path_str = str(path)

    subject_matches = re.findall(r"sub-\d{2}", path_str)
    session_matches = re.findall(r"ses-S\d", path_str)

    if not subject_matches or not session_matches:
        raise ValueError(f"Could not parse subject/session from path: {path}")

    subject = subject_matches[-1]
    session = session_matches[-1]
    condition = path.stem

    return subject, session, condition


def bandpower_welch(data, sfreq, band):
    low, high = band

    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=min(int(sfreq * 2), data.shape[1]),
        axis=1,
    )

    band_mask = (freqs >= low) & (freqs < high)

    if not np.any(band_mask):
        return np.full(data.shape[0], np.nan)

    return psd[:, band_mask].mean(axis=1)


def extract_features_for_file(path: Path, window_sec=5):
    subject, session, condition = parse_subject_session_condition(path)

    print("\n" + "=" * 80)
    print(f"Processing {subject} | {session} | {condition}")
    print("=" * 80)

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")

    if "ECG1" in raw.ch_names:
        raw.drop_channels(["ECG1"])

    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")

    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    data = raw.get_data()

    window_samples = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_samples

    print("Channels:", len(ch_names))
    print("Sampling rate:", sfreq)
    print("Duration sec:", round(data.shape[1] / sfreq, 2))
    print("Full windows:", n_windows)

    rows = []

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        window_data = data[:, start:end]

        row = {
            "subject": subject,
            "session": session,
            "condition": condition,
            "target": CONDITION_INFO[condition]["target"],
            "target_label": CONDITION_INFO[condition]["target_label"],
            "window_index": w + 1,
            "window_start_sec": start / sfreq,
            "window_end_sec": end / sfreq,
        }

        for band_name, band_range in BANDS.items():
            channel_power = bandpower_welch(window_data, sfreq, band_range)
            channel_power_map = dict(zip(ch_names, channel_power))

            for region_name, region_channels in REGIONS.items():
                available_channels = [
                    ch for ch in region_channels
                    if ch in channel_power_map
                ]

                if not available_channels:
                    row[f"{band_name}_{region_name}"] = np.nan
                else:
                    values = [channel_power_map[ch] for ch in available_channels]
                    row[f"{band_name}_{region_name}"] = float(np.mean(values))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    eeg_files = []

    for condition in CONDITIONS_TO_USE:
        eeg_files.extend(RAW_DIR.rglob(f"{condition}.set"))

    parsed_files = []

    for path in eeg_files:
        subject, session, condition = parse_subject_session_condition(path)

        if subject in SUBJECTS_TO_USE and condition in CONDITIONS_TO_USE:
            parsed_files.append(path)

    parsed_files = sorted(parsed_files)

    print(f"Found {len(parsed_files)} EEG files to process.")

    for path in parsed_files:
        print(path)

    all_features = []

    for path in parsed_files:
        features = extract_features_for_file(path, window_sec=5)
        all_features.append(features)

    if not all_features:
        raise RuntimeError("No EEG features were extracted.")

    eeg_features = pd.concat(all_features, ignore_index=True)

    print("\n" + "#" * 80)
    print("FINAL EEG FEATURE TABLE")
    print("#" * 80)

    print("Shape:", eeg_features.shape)

    print("\nTarget counts:")
    print(eeg_features["target_label"].value_counts())

    print("\nCounts by subject:")
    print(eeg_features["subject"].value_counts().sort_index())

    print("\nCounts by session:")
    print(eeg_features["session"].value_counts().sort_index())

    print("\nCounts by condition:")
    print(eeg_features["condition"].value_counts())

    print("\nMissing values:")
    print(eeg_features.isna().sum())

    eeg_features.to_csv(OUT_FILE, index=False)
    print(f"\nSaved EEG features to: {OUT_FILE}")


if __name__ == "__main__":
    main()