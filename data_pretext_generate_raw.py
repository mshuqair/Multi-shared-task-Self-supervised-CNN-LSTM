from data_processing import *
from scipy.io import loadmat
import numpy as np


# A script to load, augment and segment dataset
# two function for each case: Acc, Gyro


def prepare_data_acc(window_length, overlap, sub_array, list_da):

    print('Preparing data for Pretext task using Accelerometer data...')

    # signal parameters
    # window_length: window size in samples
    # overlap: overlap between windows, => 4s ==> rate = 4/5 = 0.8
    # sub_array: subject numbers to be selected

    # list of data augmentation functions
    labels_da_pseudo = np.arange(start=1, stop=len(list_da)+1, dtype=int)

    # Import the dataset
    print('Loading data started...')
    Data_W = loadmat('data/Data_W_v7.mat')  # Reading the data that was not segmented.
    # List of n rounds of activities, each item is (number of samples x number of axes) accelerometer and gyroscope
    # data using wrist and ankle sensors
    data_acc = Data_W['Data_Acc_W'][0]
    rounds_info = Data_W['SampleInfo_W']  # The general information of n rounds of activities
    samples_info = Data_W['SampleInfo_W2'][0, :]  # A list of n rounds and each item has the information each sample
    del Data_W

    # selecting the dataset
    ind_set2 = rounds_info[:, 0] < 25
    data_acc = data_acc[ind_set2]
    samples_info = samples_info[ind_set2]

    # concatenating the signals from all the subjects into one array
    num_samples = 0
    for round_i in range(data_acc.__len__()):
        num_samples += data_acc[round_i].shape[0]

    data_acc_arr = np.zeros((num_samples, 6))  # gyroscope data for all subjects
    sub_num_arr = np.zeros((num_samples, ))     # subject number for each sample
    target_arr = np.zeros((num_samples, ))     # label (activity) for each sample
    num_samples = 0
    for round_i in range(data_acc.__len__()):
        data_acc_arr[num_samples:num_samples + data_acc[round_i].shape[0], :] = data_acc[round_i]
        sub_num_arr[num_samples:num_samples + data_acc[round_i].shape[0], ] = samples_info[round_i][0, :]
        target_arr[num_samples:num_samples + data_acc[round_i].shape[0], ] = samples_info[round_i][1, :]  # target in mat file
        num_samples += data_acc[round_i].shape[0]
    del data_acc, samples_info
    print('Loading data finished!')

    # Data Augmentation
    print('Data augmentation started...')
    # selecting subject from PD for training
    data_acc_arr = data_acc_arr[np.isin(sub_num_arr, sub_array)]

    # data augmentation methods
    # Original
    print('Task: Original')
    data_da_o = np.concatenate((da_original(data_acc_arr[:, 0:3]), da_original(data_acc_arr[:, 3:6])), axis=1)
    labels_da_o = np.full((data_da_o.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Original')])
    x = data_da_o
    y = labels_da_o

    # Jitter
    if 'Jitter' in list_da:
        print('Task: Jitter')
        data_da_j = np.concatenate((da_jitter(data_acc_arr[:, 0:3]), da_jitter(data_acc_arr[:, 3:6])), axis=1)
        labels_da_j = np.full((data_da_j.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Jitter')])
        x = np.concatenate((x, data_da_j), axis=0)
        y = np.concatenate((y, labels_da_j), axis=0)

    # Scaling
    if 'Scaling' in list_da:
        print('Task: Scaling')
        data_da_s = np.concatenate((da_scaling(data_acc_arr[:, 0:3]), da_scaling(data_acc_arr[:, 3:6])), axis=1)
        labels_da_s = np.full((data_da_s.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Scaling')])
        x = np.concatenate((x, data_da_s), axis=0)
        y = np.concatenate((y, labels_da_s), axis=0)

    # Rotation
    if 'Rotation' in list_da:
        print('Task: Rotation')
        data_da_r = np.concatenate((da_rotation(data_acc_arr[:, 0:3]), da_rotation(data_acc_arr[:, 3:6])), axis=1)
        labels_da_r = np.full((data_da_r.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Rotation')])
        x = np.concatenate((x, data_da_r), axis=0)
        y = np.concatenate((y, labels_da_r), axis=0)

    # Permutation
    if 'Permutation' in list_da:
        print('Task: Permutation')
        data_da_p = np.concatenate((da_permutation(data_acc_arr[:, 0:3]), da_permutation(data_acc_arr[:, 3:6])), axis=1)
        labels_da_p = np.full((data_da_p.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Permutation')])
        x = np.concatenate((x, data_da_p), axis=0)
        y = np.concatenate((y, labels_da_p), axis=0)

    # Time-Warping
    if 'Time-Warping' in list_da:
        print('Task: Time-Warping')
        data_da_t = np.concatenate((da_time_warp(data_acc_arr[:, 0:3]), da_time_warp(data_acc_arr[:, 3:6])), axis=1)
        labels_da_t = np.full((data_da_t.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Time-Warping')])
        x = np.concatenate((x, data_da_t), axis=0)
        y = np.concatenate((y, labels_da_t), axis=0)

    # Magnitude-Warping
    if 'Magnitude-Warping' in list_da:
        print('Task: Magnitude-Warping')
        data_da_m = np.concatenate((da_mag_warp(data_acc_arr[:, 0:3]), da_mag_warp(data_acc_arr[:, 3:6])), axis=1)
        labels_da_m = np.full((data_da_m.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Magnitude-Warping')])
        x = np.concatenate((x, data_da_m), axis=0)
        y = np.concatenate((y, labels_da_m), axis=0)

    # calculate data scaling parameters
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    print('Data augmentation finished!')
    print('Data generated labels: ', labels_da_pseudo)

    # Data segmentation
    print('Data segmentation started...')
    # segmenting the signals, and removing the segments that share multiple activities
    x_seg, y_seg = segment(x, y, window_length, x.shape[1], overlap)
    del x, y
    print('Data segmentation finished!')

    return x_seg, y_seg, mean, std


def prepare_data_gyro(window_length, overlap, sub_array, list_da):

    print('Preparing data for Pretext task using Gyroscope data...')

    # signal parameters
    # window_length: window size in samples
    # overlap: overlap between windows, => 4s ==> rate = 4/5 = 0.8
    # sub_array: subject numbers to be selected

    # list of data augmentation functions
    labels_da_pseudo = np.arange(start=1, stop=len(list_da)+1, dtype=int)

    # Import the dataset
    print('Loading data started...')
    Data_W = loadmat('data/Data_W_v7.mat')  # Reading the data that was not segmented.
    # List of n rounds of activities, each item is (number of samples x number of axes) accelerometer and gyroscope
    # data using wrist and ankle sensors
    data_gyro = Data_W['Data_Gyro_W'][0]
    rounds_info = Data_W['SampleInfo_W']  # The general information of n rounds of activities
    samples_info = Data_W['SampleInfo_W2'][0, :]  # A list of n rounds and each item has the information each sample
    del Data_W

    # selecting the dataset
    ind_set2 = rounds_info[:, 0] < 25
    data_gyro = data_gyro[ind_set2]
    samples_info = samples_info[ind_set2]

    # concatenating the signals from all the subjects into one array
    num_samples = 0
    for round_i in range(data_gyro.__len__()):
        num_samples += data_gyro[round_i].shape[0]

    data_gyro_arr = np.zeros((num_samples, 6))  # gyroscope data for all subjects
    sub_num_arr = np.zeros((num_samples, ))     # subject number for each sample
    target_arr = np.zeros((num_samples, ))     # label (activity) for each sample
    num_samples = 0
    for round_i in range(data_gyro.__len__()):
        data_gyro_arr[num_samples:num_samples + data_gyro[round_i].shape[0], :] = data_gyro[round_i]
        sub_num_arr[num_samples:num_samples + data_gyro[round_i].shape[0], ] = samples_info[round_i][0, :]
        target_arr[num_samples:num_samples + data_gyro[round_i].shape[0], ] = samples_info[round_i][1, :]  # target in mat file
        num_samples += data_gyro[round_i].shape[0]
    del data_gyro, samples_info
    print('Loading data finished!')

    # Data Augmentation
    print('Data augmentation started...')
    # selecting subject from PD for training
    data_gyro_arr = data_gyro_arr[np.isin(sub_num_arr, sub_array)]

    # data augmentation methods
    # Original
    print('Task: Original')
    data_da_o = np.concatenate((da_original(data_gyro_arr[:, 0:3]), da_original(data_gyro_arr[:, 3:6])), axis=1)
    labels_da_o = np.full((data_da_o.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Original')])
    x = data_da_o
    y = labels_da_o

    # Jitter
    if 'Jitter' in list_da:
        print('Task: Jitter')
        data_da_j = np.concatenate((da_jitter(data_gyro_arr[:, 0:3]), da_jitter(data_gyro_arr[:, 3:6])), axis=1)
        labels_da_j = np.full((data_da_j.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Jitter')])
        x = np.concatenate((x, data_da_j), axis=0)
        y = np.concatenate((y, labels_da_j), axis=0)

    # Scaling
    if 'Scaling' in list_da:
        print('Task: Scaling')
        data_da_s = np.concatenate((da_scaling(data_gyro_arr[:, 0:3]), da_scaling(data_gyro_arr[:, 3:6])), axis=1)
        labels_da_s = np.full((data_da_s.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Scaling')])
        x = np.concatenate((x, data_da_s), axis=0)
        y = np.concatenate((y, labels_da_s), axis=0)

    # Rotation
    if 'Rotation' in list_da:
        print('Task: Rotation')
        data_da_r = np.concatenate((da_rotation(data_gyro_arr[:, 0:3]), da_rotation(data_gyro_arr[:, 3:6])), axis=1)
        labels_da_r = np.full((data_da_r.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Rotation')])
        x = np.concatenate((x, data_da_r), axis=0)
        y = np.concatenate((y, labels_da_r), axis=0)

    # Permutation
    if 'Permutation' in list_da:
        print('Task: Permutation')
        data_da_p = np.concatenate((da_permutation(data_gyro_arr[:, 0:3]), da_permutation(data_gyro_arr[:, 3:6])), axis=1)
        labels_da_p = np.full((data_da_p.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Permutation')])
        x = np.concatenate((x, data_da_p), axis=0)
        y = np.concatenate((y, labels_da_p), axis=0)

    # Time-Warping
    if 'Time-Warping' in list_da:
        print('Task: Time-Warping')
        data_da_t = np.concatenate((da_time_warp(data_gyro_arr[:, 0:3]), da_time_warp(data_gyro_arr[:, 3:6])), axis=1)
        labels_da_t = np.full((data_da_t.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Time-Warping')])
        x = np.concatenate((x, data_da_t), axis=0)
        y = np.concatenate((y, labels_da_t), axis=0)

    # Magnitude-Warping
    if 'Magnitude-Warping' in list_da:
        print('Task: Magnitude-Warping')
        data_da_m = np.concatenate((da_mag_warp(data_gyro_arr[:, 0:3]), da_mag_warp(data_gyro_arr[:, 3:6])), axis=1)
        labels_da_m = np.full((data_da_m.shape[0], 1), fill_value=labels_da_pseudo[list_da.index('Magnitude-Warping')])
        x = np.concatenate((x, data_da_m), axis=0)
        y = np.concatenate((y, labels_da_m), axis=0)

    # calculate data scaling parameters
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    print('Data augmentation finished!')
    print('Data generated labels: ', labels_da_pseudo)

    # Data segmentation
    print('Data segmentation started...')
    # segmenting the signals, and removing the segments that share multiple activities
    x_seg, y_seg = segment(x, y, window_length, x.shape[1], overlap)
    del x, y
    print('Data segmentation finished!')

    return x_seg, y_seg, mean, std
