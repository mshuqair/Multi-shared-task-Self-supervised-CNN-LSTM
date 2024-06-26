import os
from keras.backend import clear_session
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Concatenate, GlobalMaxPooling2D, Conv2D, \
    MaxPooling2D, LSTM, TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers.initializers_v2 import GlorotUniform
import numpy as np
import pickle
import time


# Task Baseline leave one subject out testing
# Using Gyroscope Raw and Spectrogram data
# The model is 1D+2D CNN-LSTM


def func_model():
    # model parameters
    optimizer = Adam(learning_rate=0.0001)
    dropout_prob_1 = 0.1
    dropout_prob_2 = 0.2
    dropout_prob_3 = 0.3
    dropout_prob_lstm_1 = 0.1

    # classification layers
    dense_units = 256

    # LSTM units
    lstm_units = 128

    # This is for the Gyroscope Raw data part
    # input / output setup
    data_shape = (window_length, numChannels)
    num_classes = 1

    # conv. blocks hyperparameters
    padding_method = 'same'
    conv1_filters = 64
    conv2_filters = 128
    conv1_kernel = 32
    conv2_kernel = 8
    pool_size = 16
    pool_strides = 4

    # input layer
    layer_input_1 = Input(shape=data_shape, name='input_raw')

    # convolutional block layers
    layer_cnn_1_a = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_input_1)
    layer_cnn_1_b = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_1_a)
    layer_dropout_1 = Dropout(dropout_prob_1)(layer_cnn_1_b)
    layer_pooling_1 = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(layer_dropout_1)

    layer_cnn_2_a = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_pooling_1)
    layer_cnn_2_b = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_2_a)
    layer_dropout_2 = Dropout(dropout_prob_2)(layer_cnn_2_b)

    layer_lstm = (LSTM(units=lstm_units, kernel_initializer=kernel_initializer, dropout=dropout_prob_lstm_1)
                  (layer_dropout_2))
    layer_final_1 = layer_lstm

    # This is for Gyroscope Spectrogram
    # model setup
    dropout_prob_1 = 0.1
    dropout_prob_2 = 0.1
    dropout_prob_lstm_2 = 0.2

    # input / output setup
    data_shape = (None, num_freq_steps, num_time_steps, numChannels)

    # conv. blocks hyperparameters
    conv1_filters = 64
    conv2_filters = 128
    conv1_kernel = (5, 5)
    conv2_kernel = (3, 3)
    pool_size = (2, 2)
    pool_strides = None

    # input layer
    layer_input_2 = Input(shape=data_shape, name='input_spect')

    # convolutional block layers
    layer_cnn_1_a = (TimeDistributed(Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))
                     (layer_input_2))
    layer_cnn_1_b = (TimeDistributed(Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))
                     (layer_cnn_1_a))
    layer_dropout_1 = TimeDistributed(Dropout(dropout_prob_1))(layer_cnn_1_b)
    layer_pooling_1 = TimeDistributed(MaxPooling2D(pool_size=pool_size, strides=pool_strides))(layer_dropout_1)

    layer_cnn_2_a = (TimeDistributed(Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))
                     (layer_pooling_1))
    layer_cnn_2_b = (TimeDistributed(Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))
                     (layer_cnn_2_a))
    layer_dropout_2 = TimeDistributed(Dropout(dropout_prob_2))(layer_cnn_2_b)
    layer_global_pooling = TimeDistributed(GlobalMaxPooling2D())(layer_dropout_2)

    layer_lstm = (LSTM(units=lstm_units, kernel_initializer=kernel_initializer, dropout=dropout_prob_lstm_2)
                  (layer_global_pooling))
    layer_final_2 = layer_lstm

    # features fusion layer
    layer_concatenate = Concatenate()([layer_final_1, layer_final_2])

    # classification layers
    layer_dense_1 = ((Dense(units=dense_units, activation='relu', kernel_initializer=kernel_initializer))
                     (layer_concatenate))
    layer_dropout_3 = Dropout(dropout_prob_3)(layer_dense_1)
    layer_output = (Dense(units=num_classes, activation=None, name='output', kernel_initializer=kernel_initializer)
                    (layer_dropout_3))

    # inputs and outputs
    outputs = [layer_output]
    inputs = [layer_input_1, layer_input_2]

    # construct the model
    model = Model(inputs=inputs, outputs=outputs)

    # compile and model summary
    model.compile(optimizer=optimizer, loss='huber_loss')
    # model.summary(show_trainable=True)

    return model


def scheduler(epoch, lr):
    if epoch <= (0.1*epochs):
        lr_new = lr
    else:
        lr_new = lr - lr_decay
    return lr_new


def select_fold(array, f, leave_out):
    array = np.roll(array, -leave_out * (f - 1))
    train = array[leave_out:]
    test = array[0:leave_out]
    return train, test


# Main code
start_time = time.time()

# results parameters
save_history = False  # whether to save the training history in pickle file or not
save_predictions = True  # whether to save the model predictions
SaveDirParent = './results/' + os.path.basename(__file__)[0:-3] + '/'
if not os.path.isdir(SaveDirParent):
    os.mkdir(SaveDirParent)

# training settings
epochs = 35
batch_size = 32
learning_rate = 0.0001
lr_decay = learning_rate / epochs
kernel_initializer = GlorotUniform(seed=1)

# subjects for cross-testing
sub_array = np.arange(start=1, stop=25, dtype=int)  # the patient number at stop is not included
num_sub_leave_out = 1  # number of subjects to leave out for cross-testing
num_folds = int(sub_array.shape[0] / num_sub_leave_out)  # number of folds, always integer
num_fold = np.arange(1, num_folds + 1)

# metrics calculation
y_test_all = np.array([])
y_predicted_all = np.array([])

# import gyroscope spectrogram data
data = np.load('data/data_gyro_spectro.npz')
x_spect = data['x_spect']
y_spect = data['y_spect']
numChannels = x_spect.shape[1]
num_freq_steps = x_spect.shape[2]
num_time_steps = x_spect.shape[3]

# import the gyroscope raw data
data = np.load('data/data_gyro_raw.npz')
x_raw = data['x_raw']
y_raw = data['y_raw']
pNo = data['pNo']
roundNo = data['roundNo']
window_length = x_raw.shape[1]


print('Number of folds: ' + str(num_folds))
for fold in num_fold:
    print('Fold number ' + str(fold))
    sub_train, sub_test = select_fold(sub_array, fold, num_sub_leave_out)
    print('Training subjects: ', sub_train)
    print('Testing subjects: ', sub_test)

    # tuning data
    x_train_raw = x_raw[np.isin(pNo, sub_train)]
    mean_raw = np.mean(x_train_raw, axis=0)
    std_raw = np.std(x_train_raw, axis=0)
    x_train_raw = (x_train_raw - mean_raw) / std_raw

    x_train_spect = x_spect[np.isin(pNo, sub_train)]
    mean_spect = np.mean(x_train_spect, axis=0)
    std_spect = np.std(x_train_spect, axis=0)
    x_train_spect = (x_train_spect - mean_spect) / std_spect
    x_train_spect = np.moveaxis(x_train_spect, source=1, destination=-1)

    x_train = {'input_raw': x_train_raw, 'input_spect': np.expand_dims(x_train_spect, axis=1)}
    y_train = y_raw[np.isin(pNo, sub_train)]

    # testing data
    x_test_raw = x_raw[np.isin(pNo, sub_test)]
    x_test_raw = (x_test_raw - mean_raw) / std_raw

    x_test_spect = x_spect[np.isin(pNo, sub_test)]
    x_test_spect = (x_test_spect - mean_spect) / std_spect
    x_test_spect = np.moveaxis(x_test_spect, source=1, destination=-1)

    x_test = {'input_raw': x_test_raw, 'input_spect': np.expand_dims(x_test_spect, axis=1)}
    y_test = y_raw[np.isin(pNo, sub_test)]

    model_baseline = func_model()

    # creating required model callbacks
    callback_scheduler = LearningRateScheduler(scheduler)
    callback_stopping = EarlyStopping(monitor='loss', patience=5)

    history = model_baseline.fit(x=x_train, y=y_train,
                                 batch_size=batch_size, epochs=epochs,
                                 callbacks=[callback_scheduler, callback_stopping],
                                 verbose=1)

    # save the training history
    if save_history:
        model_history = history.history
        with open('results/model_train_history_f_' + str(fold) + '.pkl', 'wb') as file:
            pickle.dump(model_history, file)

    # calculate model testing metrics
    print('Testing metrics...')
    y_predicted = model_baseline.predict(x_test)
    y_test_all = np.concatenate((y_test_all, y_test), axis=0)
    y_predicted_all = np.concatenate((y_predicted_all, y_predicted[:, 0]), axis=0)

    clear_session()  # close the session for training based on this fold


if save_predictions:
    model_predictions = [y_test_all, y_predicted_all, pNo, roundNo]
    with open(SaveDirParent+'model_predictions.pkl', 'wb') as file:
        pickle.dump(model_predictions, file)


# calculate and print elapsed time
elapsed_time = format((time.time() - start_time) / 60, '.2f')
print("Elapsed time: " + str(elapsed_time) + " minutes")
