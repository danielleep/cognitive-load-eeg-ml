# Experiment Log

## Dataset decision
Selected COG-BCI because it includes N-Back EEG, reaction time, and correctness.
OpenNeuro ds007169 was not selected because RT was not clearly available.

## Behavioral baseline
- Raw behavioral features: Logistic Regression balanced accuracy ~0.79
- Subject-normalized behavioral features: Logistic Regression balanced accuracy ~0.92
- Interpretation: subject-wise RT normalization helped due to between-subject RT variability.

## EEG baseline
- Raw log band-power features: near chance
- Engineered EEG features: near chance
- Session-normalized EEG features: Logistic Regression balanced accuracy ~0.61
- Interpretation: EEG signal exists but is weaker than behavioral signal; EEG features are sensitive to subject/session variability.

## Current next steps
- Decide whether to expand to more subjects or start combined behavioral + EEG model.
- Keep README concise.
- Use final report later for full motivation, methods, results, and limitations.
