% Convert COG-BCI N-Back behavioral .mat files to CSV
% Uses absolute paths for Danielle's project folder.

clear; clc;

raw_dir = "/Users/danielleepstein/Documents/cognitive-load-eeg-ml/data/raw";
out_dir = "/Users/danielleepstein/Documents/cognitive-load-eeg-ml/data/processed/behavioral_csv";

if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

conditions = ["0-Back.mat", "1-Back.mat", "2-Back.mat"];

all_files = [];

for i = 1:length(conditions)
    files_i = dir(fullfile(raw_dir, "**", conditions(i)));
    all_files = [all_files; files_i];
end

fprintf("Found %d N-Back behavioral files.\n", length(all_files));

for i = 1:length(all_files)
    in_file = fullfile(all_files(i).folder, all_files(i).name);

    sub_match = regexp(in_file, "sub-\d+", "match");
    ses_match = regexp(in_file, "ses-S\d+", "match");
    cond_match = regexp(all_files(i).name, "\d-Back", "match");

    if isempty(sub_match) || isempty(ses_match) || isempty(cond_match)
        fprintf("Could not parse metadata from: %s\n", in_file);
        continue;
    end

    % Use last subject match because sometimes paths may contain sub-01 twice
    subject_id = sub_match{end};
    session_id = ses_match{end};
    condition_name = cond_match{1};

    fprintf("Processing: %s | %s | %s\n", subject_id, session_id, condition_name);

    S = load(in_file);

    if ~isfield(S, "nback")
        fprintf("Skipping, no nback table: %s\n", in_file);
        continue;
    end

    T = S.nback;

    % Add metadata columns
    T.subject = repmat(string(subject_id), height(T), 1);
    T.session = repmat(string(session_id), height(T), 1);

    if condition_name == "0-Back"
        T.nback_level = zeros(height(T), 1);
        T.load_label = repmat("low", height(T), 1);
    elseif condition_name == "1-Back"
        T.nback_level = ones(height(T), 1);
        T.load_label = repmat("medium", height(T), 1);
    elseif condition_name == "2-Back"
        T.nback_level = repmat(2, height(T), 1);
        T.load_label = repmat("high", height(T), 1);
    end

    out_file = fullfile(out_dir, sprintf("%s_%s_%s.csv", subject_id, session_id, condition_name));
    writetable(T, out_file);

    fprintf("Saved: %s\n", out_file);
end

fprintf("Done.\n");