from scipy.io import loadmat
import numpy as np
from scipy.signal import spectrogram
from scipy.signal.windows import kaiser
from data_processing import *


def load_data(file_path, subject_filter):
    """
    Load gyroscope data and apply subject filtering.

    Args:
        file_path (str): Path to the .mat file.
        subject_filter (list): List of subject numbers to select.

    Returns:
        np.ndarray: Filtered gyroscope data.
        np.ndarray: Corresponding sample information.
    """
    print(f'Loading data from {file_path}...')
    Data_W = loadmat(file_path)
    data_gyro = Data_W['Data_Gyro_W'][0]
    rounds_info = Data_W['SampleInfo_W']
    samples_info = Data_W['SampleInfo_W2'][0, :]

    # Select subjects based on subject_filter
    ind_set2 = rounds_info[:, 0] < 25
    data_gyro = data_gyro[ind_set2]
    samples_info = samples_info[ind_set2]

    # Concatenate data from all rounds
    num_samples = sum([d.shape[0] for d in data_gyro])
    data_gyro_arr = np.zeros((num_samples, 6))  # For gyroscope data
    sub_num_arr = np.zeros((num_samples,))  # For subject numbers
    target_arr = np.zeros((num_samples,))  # For activity labels

    # Fill arrays with the data
    idx = 0
    for round_i in range(len(data_gyro)):
        data_gyro_arr[idx:idx + data_gyro[round_i].shape[0], :] = data_gyro[round_i]
        sub_num_arr[idx:idx + data_gyro[round_i].shape[0]] = samples_info[round_i][0, :]
        target_arr[idx:idx + data_gyro[round_i].shape[0]] = samples_info[round_i][1, :]
        idx += data_gyro[round_i].shape[0]

    return data_gyro_arr, sub_num_arr, target_arr


def apply_augmentation(data, augmentation_list, labels_da_pseudo, list_da):
    """
    Apply different data augmentations based on the provided list.

    Args:
        data (np.ndarray): Raw gyroscope data.
        augmentation_list (list): List of augmentations to apply (e.g., ['Original', 'Jitter']).
        labels_da_pseudo (np.ndarray): The labels for augmentation tasks.
        list_da (list): List of augmentation task names.

    Returns:
        np.ndarray: Augmented data.
        np.ndarray: Augmented labels.
    """
    x = data
    y = np.array([])

    # Define augmentation methods
    augmentation_methods = {
        'Original': da_original,
        'Jitter': da_jitter,
        'Scaling': da_scaling,
        'Rotation': da_rotation,
        'Permutation': da_permutation,
        'Time-Warping': da_time_warp,
        'Magnitude-Warping': da_mag_warp
    }

    for aug in augmentation_list:
        if aug in augmentation_methods:
            print(f'Task: {aug}')
            augmented_data = np.concatenate(
                [augmentation_methods[aug](data[:, 0:3]), augmentation_methods[aug](data[:, 3:6])], axis=1
            )
            labels_da = np.full((augmented_data.shape[0], 1), fill_value=labels_da_pseudo[list_da.index(aug)])
            x = np.concatenate((x, augmented_data), axis=0)
            y = np.concatenate((y, labels_da), axis=0) if y.size else labels_da

    return x, y


def segment_data(data, labels, window_length, overlap):
    """
    Segment the data into smaller windows with overlap.

    Args:
        data (np.ndarray): Data to be segmented.
        labels (np.ndarray): Corresponding labels.
        window_length (int): Window size in samples.
        overlap (int): Number of overlapping samples between windows.

    Returns:
        np.ndarray: Segmented data.
        np.ndarray: Corresponding segmented labels.
    """
    return segment(data, labels, window_length, data.shape[1], overlap)


def generate_spectrogram(data, fs, windowL_spec, windowO_spec, nfft, f_max):
    """
    Generate spectrograms for each segment of the data.

    Args:
        data (np.ndarray): Segmented data.
        fs (int): Sampling frequency.
        windowL_spec (int): Window length for spectrogram.
        windowO_spec (int): Overlap for spectrogram.
        nfft (int): Number of FFT points.
        f_max (float): Maximum frequency for the spectrogram.

    Returns:
        np.ndarray: Spectrograms of the data.
    """
    num_samples = data.shape[0]
    num_channels = data.shape[2]
    freq_steps = np.arange(start=0, step=fs / nfft, stop=(fs / 2 + fs / nfft))
    freq_steps_s = freq_steps[freq_steps < f_max]
    time_steps = np.arange(
        start=(windowL_spec - windowO_spec) * 5,
        step=(windowL_spec - windowO_spec),
        stop=(windowL_spec - windowO_spec) * 5 - (windowL_spec - windowO_spec)
    ) / fs

    spectrograms = np.zeros((num_samples, num_channels, freq_steps_s.shape[0], time_steps.shape[0]))

    for index_i in range(num_samples):
        for axis_i in range(num_channels):
            f, t, Sxx = spectrogram(data[index_i, :, axis_i], fs, window=kaiser(windowL_spec, beta=5),
                                    noverlap=windowO_spec, nfft=nfft, axis=-1, mode='complex')
            Sxx_new = abs(Sxx[f < f_max])
            spectrograms[index_i, axis_i, :, :] = np.flip(Sxx_new, axis=0)

    return spectrograms


def prepare_data_gyro(window_length, overlap, sub_array, list_da, file_path='data/Data_W_v7.mat'):
    print('Preparing data for Pretext task using Gyroscope data...')

    # Load data
    data_gyro_arr, sub_num_arr, target_arr = load_data(file_path, sub_array)

    # Data augmentation
    labels_da_pseudo = np.arange(start=1, stop=len(list_da) + 1, dtype=int)
    data_aug, labels_aug = apply_augmentation(data_gyro_arr, list_da, labels_da_pseudo, list_da)

    # Data segmentation
    x_seg, y_seg = segment_data(data_aug, labels_aug, window_length, overlap)

    # Generate spectrogram
    x_spectrogram = generate_spectrogram(x_seg, fs=64, windowL_spec=64, windowO_spec=57, nfft=128, f_max=15)

    # Calculate scaling parameters
    mean = np.mean(x_spectrogram, axis=0)
    std = np.std(x_spectrogram, axis=0)

    print('Preparation finished!')
    return x_spectrogram, y_seg, mean, std
