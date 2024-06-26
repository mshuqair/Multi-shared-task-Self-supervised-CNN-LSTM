from scipy.io import loadmat
import numpy as np
from data_processing import segment


# Import amd segment the PD data from the .mat file
# The data is segmented based on subjects and rounds
# The output file contains the windowed data, labels, patient number, and round number
# This is to replace loading the data from the mat file each time
# The code was updated to include activities per segmented window


# signal parameters
fs = 64  # sampling frequency of the original signal
window_width = 5  # window size in seconds
window_length = int(fs * window_width)  # window size in samples
overlap = 0  # overlap between windows, => 4s ==> rate = 4/5 = 0.8
numChannels = 6


# import the dataset
print('Loading data started...')
Data_W = loadmat('data/Data_W_v7.mat')  # Reading the data that was not segmented.
# List of n rounds of activities, each item is (number of samples x number of axes) accelerometer and gyroscope
# data using wrist and ankle sensors
data_info = Data_W['Data_Gyro_W'][0]    # Replace by Gyro or Accel data
rounds_info = Data_W['SampleInfo_W']  # The general information of n rounds of activities
samples_info = Data_W['SampleInfo_W2'][0, :]  # A list of n rounds and each item has the information each sample
patients_id1 = Data_W['patientsNosD1'][0,]
ind_for_y = 1  # 1 for UPDRS, 4 Tremor, 5 Bradykinesia, 8 Rigidity, 3 for activities
print('Loading data finished!')


print('Starting segmentation...')
print('Total number of rounds:', data_info.shape[0])
actsNo = np.array([], dtype=int)
roundNo = np.array([], dtype=int)
pNo = np.array([], dtype=int)
targetScore = np.array([], dtype=int)
dataRaw = np.zeros(shape=(1, window_length, numChannels))

for round_i in range(data_info.shape[0]):
    x, y = segment(features=data_info[round_i],
                   targets=np.full(shape=(data_info[round_i].shape[0]), fill_value=rounds_info[round_i, ind_for_y]),
                   window_size=window_length,
                   num_cols=data_info[round_i].shape[1],
                   overlap_rate=overlap)
    dataRaw = np.append(dataRaw, x, axis=0)
    targetScore = np.append(targetScore, y, axis=0)
    index_1, index_2 = 0, x.shape[1]
    for actI in range(x.shape[0]):
        temp = samples_info[round_i][3, index_1:index_2]
        unique, unique_counts = np.unique(temp, return_counts=True)
        act_temp = int(unique[np.argmax(unique_counts)])
        actsNo = np.append(actsNo, act_temp)
        index_1, index_2 = index_1 + x.shape[1], index_2 + x.shape[1]
    pNo = np.append(pNo, np.full(shape=(x.shape[0]), fill_value=rounds_info[round_i, 0]), axis=0)
    roundNo = np.append(roundNo, np.full(shape=(x.shape[0]), fill_value=round_i+1), axis=0)
dataRaw = dataRaw[1:, :, :]   # to deselect the first row containing zeros
print('Segmentation finished!')
print('Resulting data shape:', dataRaw.shape)


# saving final data data
print('Saving the data...')
np.savez_compressed(file='data/data_gyro_raw.npz', x_raw=dataRaw, y_raw=targetScore,
                    pNo=pNo, roundNo=roundNo, actsNo=actsNo)   # change name if needed
print('Data saved!')
