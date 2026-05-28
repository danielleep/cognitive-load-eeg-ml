from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

OUT_FILE = PROJECT_ROOT / "data" / "processed" / "eeg_features_sub01_sesS1.csv"

FILES = [
    {
        "subject": "sub-01",
        "session": "ses-S1",
        "condition": "zeroBACK",
        "target": 0,
        "target_label": "low",
        "path": PROJECT_ROOT / "data" / "raw" / "sub-01" / "sub-01" / "ses-S1" / "eeg" / "zeroBACK.set",
    },
    {
        "subject": "sub-01",
        "session": "ses-S1",
        "condition": "twoBACK",
        "target": 1,
        "target_label": "high",
        "path": PROJECT_ROOT / "data" / "raw" / "sub-01" / "sub-01" / "ses-S1" / "eeg" / "twoBACK.set",
    },
]

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


def bandpower_welch(data, sfreq, band):
    """
    Compute average power in a frequency band using Welch PSD.

    data shape: channels x samples
    """
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


def extract_features_for_file(file_info, window_sec=5):
    print("\n" + "=" * 80)
    print(f"Processing {file_info['subject']} {file_info['session']} {file_info['condition']}")
    print("=" * 80)

    raw = mne.io.read_raw_eeglab(file_info["path"], preload=True, verbose="ERROR")

    # Drop ECG channel if present
    if "ECG1" in raw.ch_names:
        raw.drop_channels(["ECG1"])

    # Basic filtering for EEG band-power features
    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")

    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    data = raw.get_data()  # channels x samples

    window_samples = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_samples

    print("Channels:", len(ch_names))
    print("Sampling rate:", sfreq)
    print("Total duration sec:", round(data.shape[1] / sfreq, 2))
    print("Window length sec:", window_sec)
    print("Number of full windows:", n_windows)

    rows = []

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples

        window_data = data[:, start:end]

        row = {
            "subject": file_info["subject"],
            "session": file_info["session"],
            "condition": file_info["condition"],
            "target": file_info["target"],
            "target_label": file_info["target_label"],
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

                if len(available_channels) == 0:
                    row[f"{band_name}_{region_name}"] = np.nan
                else:
                    values = [channel_power_map[ch] for ch in available_channels]
                    row[f"{band_name}_{region_name}"] = float(np.mean(values))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    all_features = []

    for file_info in FILES:
        features = extract_features_for_file(file_info, window_sec=5)
        all_features.append(features)

    eeg_features = pd.concat(all_features, ignore_index=True)

    print("\nFinal EEG feature table:")
    print(eeg_features.shape)
    print(eeg_features.head())

    print("\nTarget counts:")
    print(eeg_features["target_label"].value_counts())

    print("\nMissing values:")
    print(eeg_features.isna().sum())

    eeg_features.to_csv(OUT_FILE, index=False)
    print(f"\nSaved EEG features to: {OUT_FILE}")


if __name__ == "__main__":
    main()