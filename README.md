# Predicting Cognitive Load from Behavioral and EEG Features

## Project goal
This project aims to predict cognitive load from behavioral and EEG-derived features using an N-Back EEG dataset.

The main research question is:

> Can cognitive load be predicted from behavioral and EEG-derived features, and does combining both modalities improve prediction compared to using each modality alone?

## Dataset
Selected dataset: **COG-BCI**

Task used in this project: **N-Back**

Initial target definition:
- **Low cognitive load:** 0-back
- **High cognitive load:** 2-back
- **1-back:** reserved for optional later analysis

A previous candidate dataset, OpenNeuro `ds007169`, was evaluated but not selected because reaction time was not clearly available.

## Dataset feasibility check
Initial checks confirmed that COG-BCI is suitable for this project:

- Behavioral `.mat` files exist for 0-back, 1-back, and 2-back.
- Behavioral tables include reaction time (`rt`) and correctness (`correct`).
- EEG files exist in EEGLAB format: `.set` + `.fdt`.
- MNE successfully reads the EEG files.
- Example EEG file checked: `zeroBACK.set`
- Sampling rate: 500 Hz
- Channels: 62 EEG + 1 ECG

## Current progress

### Behavioral data processing
Behavioral files were converted from MATLAB `.mat` format to CSV for:

- Subjects: `sub-01` to `sub-05`
- Sessions: `ses-S1`, `ses-S2`, `ses-S3`
- Conditions: 0-back, 1-back, 2-back

A combined behavioral table was created with:

- 5 subjects
- 3 sessions
- 3 N-Back conditions
- 6480 total trials

For the initial binary classification task, the data was filtered to:

- 0-back and 2-back only
- outlier trials removed
- hit trials used for the first behavioral baseline

### Behavioral features
The first behavioral feature table includes:

- reaction time
- correctness
- response/miss indicators
- previous trial features
- rolling reaction time
- rolling accuracy
- subject-normalized RT features

Subject-wise RT normalization was added after exploratory analysis showed strong between-subject variability, especially for `sub-05`.

## Behavioral baseline results

Evaluation method: **Leave-One-Subject-Out cross-validation**

This avoids trial-level leakage by ensuring that the test subject is not included in the training set.

### Raw RT features
Initial behavioral baseline using raw RT features:

| Model | Balanced Accuracy |
|---|---:|
| Logistic Regression | ~0.79 |
| Random Forest | ~0.78 |

### Subject-normalized RT features
After adding subject-normalized RT features:

| Model | Balanced Accuracy |
|---|---:|
| Logistic Regression | ~0.92 |
| Random Forest | ~0.91 |

This suggests that normalizing reaction time relative to each subject helps address between-subject differences in response speed.

## Next step
Begin EEG feature extraction for the N-Back task, starting with:

- 0-back EEG files
- 2-back EEG files

Initial EEG features will likely include band-power features such as theta, alpha, and beta power.## Project goal
This project aims to predict cognitive load from behavioral and EEG-derived features using an N-Back EEG dataset.

## Current dataset decision
Selected dataset: COG-BCI  
Task: N-Back  
Initial target definition:
- Low cognitive load: 0-back
- High cognitive load: 2-back
- 1-back will be reserved for optional later analysis

## Dataset feasibility check
Completed initial checks on sub-01, session S1:
- Behavioral files exist for 0-back, 1-back, 2-back
- Behavioral tables include RT and correctness
- EEG files exist in EEGLAB format: .set + .fdt
- MNE successfully reads zeroBACK.set
- Sampling rate: 500 Hz
- Channels: 62 EEG + 1 ECG

## Notes
The previous dataset ds007169 was not selected because reaction time was not clearly available.

## Current progress

### Dataset selection
Selected dataset: COG-BCI.

The previous candidate dataset, OpenNeuro ds007169, was evaluated but not selected because reaction time was not clearly available.

### Current task
N-Back task from COG-BCI.

Initial binary classification setup:
- Low cognitive load: 0-back
- High cognitive load: 2-back
- 1-back is reserved for optional later analysis

### Behavioral data status
Behavioral `.mat` files were converted to CSV for subjects sub-01 to sub-05, across 3 sessions and 3 N-Back conditions.

A combined behavioral table was created:
- 5 subjects
- 3 sessions
- 3 conditions
- 6480 total trials

A clean binary behavioral table was created using:
- 0-back and 2-back only
- outlier trials removed

A behavioral hit-trial feature table was created with:
- reaction time
- correctness
- response/miss indicators
- previous trial features
- rolling RT and rolling accuracy features

### Behavioral baseline results

Leave-One-Subject-Out evaluation was used to avoid trial-level leakage across subjects.

Initial behavioral baseline with raw RT features:
- Logistic Regression balanced accuracy: approximately 0.79
- Random Forest balanced accuracy: approximately 0.78

EDA showed strong between-subject variability, especially for sub-05, whose low-load RTs were slower than other subjects.

After adding subject-normalized RT features:
- Logistic Regression balanced accuracy: approximately 0.92
- Random Forest balanced accuracy: approximately 0.91

This suggests that subject-wise RT normalization helps address between-subject differences in behavioral response speed.

### Next step
Begin EEG feature extraction for the N-Back task, starting with 0-back and 2-back EEG files.
