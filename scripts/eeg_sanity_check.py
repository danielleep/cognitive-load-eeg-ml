from pathlib import Path
import mne

PROJECT_ROOT = Path("/Users/danielleepstein/Documents/cognitive-load-eeg-ml")

files = {
    "low_0back": PROJECT_ROOT / "data/raw/sub-01/sub-01/ses-S1/eeg/zeroBACK.set",
    "high_2back": PROJECT_ROOT / "data/raw/sub-01/sub-01/ses-S1/eeg/twoBACK.set",
}

for label, path in files.items():
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    raw = mne.io.read_raw_eeglab(path, preload=False)

    print("\nBasic info:")
    print(raw.info)

    print("\nNumber of channels:", len(raw.ch_names))
    print("First 20 channel names:")
    print(raw.ch_names[:20])

    print("\nSampling frequency:", raw.info["sfreq"])
    print("Duration in seconds:", raw.times[-1])

    print("\nChannel types:")
    print(raw.get_channel_types())

    print("\nAnnotations:")
    print(raw.annotations[:20])
    print("Number of annotations:", len(raw.annotations))