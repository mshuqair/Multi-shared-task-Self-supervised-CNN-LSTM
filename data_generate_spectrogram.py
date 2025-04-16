import numpy as np
from scipy.io import loadmat
from scipy.signal import spectrogram
from scipy.signal.windows import kaiser
from data_processing import segment, next_pow_2


# === Configuration ===
fs = 64  # Sampling frequency (Hz)
window_width = 5  # Main window size in seconds
window_length = fs * window_width  # Main window size in samples
overlap = 0  # Overlap between windows
num_channels = 6
ind_for_y = 1  # Index of the target label (1=UPDRS, 4=Tremor, etc.)
mat_file_path = 'data/Data_W_v7.mat'
save_path = 'data/data_gyro_spectro.npz'

# === Spectrogram Parameters ===
spec_win_sec = 1  # Spectrogram window length (in seconds)
spec_win_samples = spec_win_sec * fs
spec_overlap = round(spec_win_samples * 0.9)
nfft = next_pow_2(spec_win_samples)
f_max = 15  # Max frequency to retain

# Frequency & Time axes for output shape
freq_bins = np.arange(0, fs / 2 + fs / nfft, fs / nfft)
freq_mask = freq_bins < f_max
time_steps = np.arange((spec_win_samples - spec_overlap) * 5,
                       window_length - (spec_win_samples - spec_overlap) * 5,
                       (spec_win_samples - spec_overlap)) / fs

# === Load Data ===
print('Loading data...')
data_mat = loadmat(mat_file_path)
data_info = data_mat['Data_Gyro_W'][0]
rounds_info = data_mat['SampleInfo_W']
samples_info = data_mat['SampleInfo_W2'][0]
patients_id = data_mat['patientsNosD1'][0]
print('Data loaded. Total rounds:', len(data_info))

# === Initialize Output Containers ===
data_spectrogram = []
target_scores = []
patient_ids = []
round_ids = []

print('Generating spectrograms...')

for round_i in range(len(data_info)):
    features = data_info[round_i]
    n_samples = features.shape[0]

    # Target label for the entire round
    label = rounds_info[round_i, ind_for_y]
    labels = np.full(n_samples, label)

    # Segment signal into windows
    x_segmented, y_segmented = segment(features=features,
                                       targets=labels,
                                       window_size=window_length,
                                       num_cols=features.shape[1],
                                       overlap_rate=overlap)

    # Initialize array for this round’s spectrograms
    x_spec = np.zeros((x_segmented.shape[0], num_channels,
                       freq_mask.sum(), time_steps.shape[0]))

    # Compute spectrograms
    for i in range(x_segmented.shape[0]):
        for ch in range(num_channels):
            f, t, Sxx = spectrogram(x_segmented[i, :, ch],
                                    fs=fs,
                                    window=kaiser(spec_win_samples, beta=5),
                                    noverlap=spec_overlap,
                                    nfft=nfft,
                                    axis=-1,
                                    mode='complex')
            x_spec[i, ch, :, :] = np.abs(Sxx[f < f_max])

    data_spectrogram.append(x_spec)
    target_scores.append(y_segmented)
    patient_ids.extend([rounds_info[round_i, 0]] * x_spec.shape[0])
    round_ids.extend([round_i + 1] * x_spec.shape[0])

# === Finalize Data ===
x_spect_all = np.concatenate(data_spectrogram, axis=0)
y_spect_all = np.concatenate(target_scores)
p_no_all = np.array(patient_ids)
round_no_all = np.array(round_ids)

print('Spectrogram generation complete.')
print('Final data shape:', x_spect_all.shape)

# === Save Output ===
print('Saving data...')
np.savez_compressed(save_path,
                    x_spect=x_spect_all,
                    y_spect=y_spect_all,
                    pNo=p_no_all,
                    roundNo=round_no_all)
print('Data saved to:', save_path)
