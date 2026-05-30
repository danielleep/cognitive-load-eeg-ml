# Predicting Cognitive Load from Behavioral and EEG Features

## Project Goal

This project aims to predict cognitive load from behavioral and EEG-derived features using an N-Back EEG dataset.

Main question:

> Can cognitive load be predicted from behavioral and EEG features, and does EEG add useful information beyond behavioral performance?

## Dataset

Dataset: **COG-BCI**  
Task: **N-Back**

Binary target definition:

- **Low cognitive load:** 0-back
- **High cognitive load:** 2-back
- **1-back:** reserved for optional later analysis

A previous candidate dataset, OpenNeuro `ds007169`, was evaluated but not used because reaction time was not clearly available.

## Data Used

Current experiments use:

- Subjects: `sub-01` to `sub-10`
- Sessions: `ses-S1`, `ses-S2`, `ses-S3`
- Conditions: 0-back and 2-back

EEG files are in EEGLAB format (`.set` + `.fdt`) and are read using MNE.

## Pipeline

### Behavioral Features

Behavioral `.mat` files are converted to CSV and processed into trial-level features.

Main behavioral features include:

- reaction time
- correctness
- response/miss indicators
- previous-trial features
- rolling reaction time
- rolling accuracy
- subject-normalized reaction time features

### EEG Features

EEG recordings are split into fixed-length time windows.

For each window, band-power features are extracted:

- theta: 4–8 Hz
- alpha: 8–13 Hz
- beta: 13–30 Hz

Features are averaged across broad scalp regions:

- frontal
- central
- parietal
- occipital

Additional EEG features include global band power, band-power ratios, and session-normalized EEG features.

## Evaluation

Models are evaluated using **Leave-One-Subject-Out cross-validation**.

This ensures that the test subject is not included in training and helps reduce subject-level leakage.

## Current Results

### Behavioral-only baseline

| Feature set | Best model | Balanced Accuracy |
|---|---:|---:|
| Subject-normalized behavioral features | Random Forest | ~0.91 |
| Subject-normalized behavioral features | Logistic Regression | ~0.90 |

### EEG-only baseline

| Feature set | Best model | Balanced Accuracy |
|---|---:|---:|
| Log band-power features | Logistic Regression | ~0.55 |
| Engineered EEG features | Logistic Regression | ~0.55 |
| Session-normalized EEG features | Random Forest | ~0.59 |

## Current Interpretation

Behavioral features provide a strong signal for distinguishing low vs high cognitive load.

EEG-only features are weaker, but session-level normalization improves performance slightly, suggesting that EEG features are sensitive to subject/session variability.

The normalized models should be interpreted as **calibration-aware** models, since they assume access to some subject/session data for normalization.

## Repository Structure

```text
scripts/
├── behavioral/     # Behavioral preprocessing, feature creation, and baselines
├── eeg/            # EEG feature extraction, feature engineering, and baselines
└── exploration/    # Notes about exploratory checks

notes/              # Experiment notes and project documentation
data/               # Raw and processed data, not tracked by Git
````

## Next Step

Build a first combined behavioral + EEG model and test whether EEG features add information beyond behavioral features.

```
