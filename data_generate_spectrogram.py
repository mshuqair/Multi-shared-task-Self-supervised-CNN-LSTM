from scipy.io import loadmat
import numpy as np
from data_processing import segment, next_pow_2
from scipy.signal import spectrogram
from scipy.signal.windows import kaiser


# Import amd segment the PD data from the .mat file
# Generate spectrogram data
# The data is segmented based on subjects and rounds
# The output file contains the windowed data, labels, patient number, and round number
# This is to replace loading the data from the mat file each time
# Spectrogram of 5-s windows


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
ind_for_y = 1  # 1 for UPDRS, 4 Tremor, 5 Bradykinesia, 8 Rigidity


print('Total number of rounds:', data_info.shape[0])

print('Generating spectrogram...')
# Spectrogram parameters
fs = 64
windowL_brady = 5      # 60 seconds window length
windowL_spec = 1 * fs
windowO_spec = round(windowL_spec * 0.9)  # sec
nfft = next_pow_2(windowL_spec)  # for windowL_spec-sec rounds
f_max = 15      # max frequency
freq_steps = np.arange(start=0, step=fs/nfft, stop=(fs/2 + fs/nfft))
freq_steps_s = freq_steps[freq_steps < f_max]
time_steps = np.arange(start=(windowL_spec-windowO_spec)*5, step=(windowL_spec-windowO_spec),
                       stop=windowL_brady*fs-(windowL_spec-windowO_spec)*5-(windowL_spec-windowO_spec))/fs  # 111

# initialize empty arrays
roundNo = np.array([], dtype=int)
pNo = np.array([], dtype=int)
targetScore = np.array([], dtype=int)
dataSpectrogram = np.zeros(shape=(1, numChannels, freq_steps_s.shape[0], time_steps.shape[0]))

for round_i in range(data_info.shape[0]):
    x, y = segment(features=data_info[round_i],
                   targets=np.full(shape=(data_info[round_i].shape[0]), fill_value=rounds_info[round_i, ind_for_y]),
                   window_size=window_length,
                   num_cols=data_info[round_i].shape[1],
                   overlap_rate=overlap)
    x_spectrogram = np.zeros(shape=(x.shape[0], numChannels, freq_steps_s.shape[0], time_steps.shape[0]))
    for index_i in range(x.shape[0]):
        for axis_i in range(numChannels):
            f, t, Sxx = spectrogram(x=x[index_i, :, axis_i], fs=fs, window=kaiser(windowL_spec, beta=5),
                                    noverlap=windowO_spec, nfft=nfft, axis=-1, mode='complex')
            x_spectrogram[index_i, axis_i, :, :] = abs(Sxx[f < f_max])
    dataSpectrogram = np.append(dataSpectrogram, x_spectrogram, axis=0)
    targetScore = np.append(targetScore, y, axis=0)
    pNo = np.append(pNo, np.full(shape=(x.shape[0]), fill_value=rounds_info[round_i, 0]), axis=0)
    roundNo = np.append(roundNo, np.full(shape=(x.shape[0]), fill_value=round_i+1), axis=0)
dataSpectrogram = dataSpectrogram[1:, :, :, :]   # to deselect the first row containing zeros
print('Resulting data shape:', dataSpectrogram.shape)


# saving final data data
print('Saving the data...')
np.savez_compressed(file='data/data_gyro_spectro.npz', x_spect=dataSpectrogram, y_spect=targetScore,
                    pNo=pNo, roundNo=roundNo)   # change name if needed
