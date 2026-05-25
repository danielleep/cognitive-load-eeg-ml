# Predicting Cognitive Load from Behavioral and EEG Features

## Project goal
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
