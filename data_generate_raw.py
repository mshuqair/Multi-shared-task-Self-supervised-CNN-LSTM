import numpy as np
from scipy.io import loadmat
from data_processing import segment


# === Configuration ===
fs = 64  # Sampling frequency (Hz)
window_width = 5  # Window size (seconds)
window_length = int(fs * window_width)  # Window size (samples)
overlap = 0  # Overlap between windows
num_channels = 6  # Number of channels
ind_for_y = 1  # Index of the target label (e.g., UPDRS = 1, Tremor = 4, etc.)
mat_file_path = 'data/Data_W_v7.mat'
save_path = 'data/data_gyro_raw.npz'  # Update name if needed

# === Load Data ===
print('Loading data started...')
data_mat = loadmat(mat_file_path)

# Extract from loaded MAT file
data_info = data_mat['Data_Gyro_W'][0]  # shape: (n_rounds,)
rounds_info = data_mat['SampleInfo_W']  # shape: (n_rounds, info_cols)
samples_info = data_mat['SampleInfo_W2'][0]  # list of sample-level info
patients_id = data_mat['patientsNosD1'][0]

print('Loading data finished!')

# === Segment and Process Data ===
print('Starting segmentation...')
print('Total number of rounds:', data_info.shape[0])

# Initialize containers
data_raw = []
target_scores = []
activities = []
patient_nos = []
round_nos = []

for round_i in range(len(data_info)):
    features = data_info[round_i]
    n_samples, n_cols = features.shape

    # Create full-length label array for segmentation
    round_score = rounds_info[round_i, ind_for_y]
    labels = np.full(n_samples, round_score)

    # Segment data
    x_segmented, y_segmented = segment(features=features,
                                       targets=labels,
                                       window_size=window_length,
                                       num_cols=n_cols,
                                       overlap_rate=overlap)

    data_raw.append(x_segmented)
    target_scores.append(y_segmented)

    # Activity labeling for each window
    sample_activity = samples_info[round_i][3]  # Activity labels at the sample level
    for i in range(x_segmented.shape[0]):
        start_idx = i * window_length
        end_idx = start_idx + window_length
        window_activities = sample_activity[start_idx:end_idx]
        most_common_activity = np.bincount(window_activities).argmax()
        activities.append(most_common_activity)

    # Patient ID and round number for each window
    patient_id = rounds_info[round_i, 0]
    n_windows = x_segmented.shape[0]
    patient_nos.extend([patient_id] * n_windows)
    round_nos.extend([round_i + 1] * n_windows)

print('Segmentation finished!')

# === Finalize Data ===
data_raw = np.concatenate(data_raw, axis=0)
target_scores = np.concatenate(target_scores)
activities = np.array(activities)
patient_nos = np.array(patient_nos)
round_nos = np.array(round_nos)

print('Resulting data shape:', data_raw.shape)

# === Save Output ===
print('Saving the data...')
np.savez_compressed(file=save_path,
                    x_raw=data_raw,
                    y_raw=target_scores,
                    pNo=patient_nos,
                    roundNo=round_nos,
                    actsNo=activities)
print('Data saved to:', save_path)
