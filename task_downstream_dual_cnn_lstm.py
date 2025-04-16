import os
from keras.backend import clear_session
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Concatenate, GlobalMaxPooling2D, Conv2D, \
    MaxPooling2D, LSTM, TimeDistributed
from keras.optimizers.optimizer_v2.adam import Adam
from keras.models import load_model, Model
from keras.initializers.initializers_v2 import GlorotUniform
import numpy as np
import pickle
import time


# Task Downstream leave one subject out testing
# Using Gyroscope Raw and Spectrogram data
# The model is 1D+2D CNN-LSTM


def func_model():
    """
    Constructs and returns the CNN-LSTM model for classifying raw gyroscope and spectrogram data.
    """
    # Model parameters
    optimizer = Adam(learning_rate=0.0001)
    dropout_prob_1 = 0.1
    dropout_prob_2 = 0.2
    dropout_prob_3 = 0.3
    dropout_prob_lstm_1 = 0.1
    dense_units = 256
    lstm_units = 128
    num_classes = 1
    padding_method = 'same'
    kernel_initializer = GlorotUniform(seed=57)

    # Gyroscope Raw Data Part
    data_shape = (window_length, numChannels)
    conv1_filters = 64
    conv2_filters = 128
    conv1_kernel = 32
    conv2_kernel = 8
    pool_size = 16
    pool_strides = 4

    layer_input_1 = Input(shape=data_shape, name='input_raw')

    # Conv Block 1
    layer_cnn_1_a = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_input_1)
    layer_cnn_1_b = Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_1_a)
    layer_dropout_1 = Dropout(dropout_prob_1)(layer_cnn_1_b)
    layer_pooling_1 = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(layer_dropout_1)

    # Conv Block 2
    layer_cnn_2_a = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_pooling_1)
    layer_cnn_2_b = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_2_a)
    layer_dropout_2 = Dropout(dropout_prob_2)(layer_cnn_2_b)

    layer_lstm = LSTM(units=lstm_units, kernel_initializer=kernel_initializer, dropout=dropout_prob_lstm_1)(layer_dropout_2)
    layer_final_1 = layer_lstm

    # Gyroscope Spectrogram Part
    dropout_prob_lstm_2 = 0.2
    data_shape = (None, num_freq_steps, num_time_steps, numChannels)
    conv1_filters = 64
    conv2_filters = 128
    conv1_kernel = (5, 5)
    conv2_kernel = (3, 3)
    pool_size = (2, 2)

    layer_input_2 = Input(shape=data_shape, name='input_spect')

    # Conv Block 1 (Spectrogram)
    layer_cnn_1_a = TimeDistributed(Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))(layer_input_2)
    layer_cnn_1_b = TimeDistributed(Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))(layer_cnn_1_a)
    layer_dropout_1 = TimeDistributed(Dropout(dropout_prob_1))(layer_cnn_1_b)
    layer_pooling_1 = TimeDistributed(MaxPooling2D(pool_size=pool_size))(layer_dropout_1)

    # Conv Block 2 (Spectrogram)
    layer_cnn_2_a = TimeDistributed(Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))(layer_pooling_1)
    layer_cnn_2_b = TimeDistributed(Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu',
                                            padding=padding_method, kernel_initializer=kernel_initializer))(layer_cnn_2_a)
    layer_dropout_2 = TimeDistributed(Dropout(dropout_prob_2))(layer_cnn_2_b)
    layer_global_pooling = TimeDistributed(GlobalMaxPooling2D())(layer_dropout_2)

    layer_lstm = LSTM(units=lstm_units, kernel_initializer=kernel_initializer, dropout=dropout_prob_lstm_2)(layer_global_pooling)
    layer_final_2 = layer_lstm

    # Features Fusion
    layer_concatenate = Concatenate()([layer_final_1, layer_final_2])

    # Fully connected layers
    layer_dense_1 = Dense(units=dense_units, activation='relu', kernel_initializer=kernel_initializer)(layer_concatenate)
    layer_dropout_3 = Dropout(dropout_prob_3)(layer_dense_1)
    layer_output = Dense(units=num_classes, activation=None, name='output', kernel_initializer=kernel_initializer)(layer_dropout_3)

    # Construct the model
    model = Model(inputs=[layer_input_1, layer_input_2], outputs=[layer_output])

    # Compile the model
    model.compile(optimizer=optimizer, loss='huber_loss')

    return model


def func_downstream_model(fold_no):
    # model parameters and setup
    optimizer = Adam(learning_rate=learning_rate)

    # finetune settings
    # pretext layers 2, 4, 10, 12 are for Gyro Raw branch
    # pretext layers 3, 5, 11, 13 are for Gyro Spectro branch
    # downstream layers indices are alternated
    conv_range_pretext = [2, 3, 4, 5, 10, 11, 12, 13]  # specify which layers weights to transfer by index
    conv_range_downstream = [3, 2, 5, 4, 11, 10, 13, 12]  # specify which layers weights to transfer by index
    conv_range_freeze = [3, 2, 5, 4]  # specify which layers to freeze weights by index

    # loading both models
    # model_pretext = build_model_pretext(num_tasks=4)     # in case we want to evaluate untrained Conv. blocks
    model_pretext = load_model('models/model_pretext_multitask_trained_gyro_raw_spectro_' + str(fold_no) + '.h5')
    model_downstream = func_model()

    # transfer conv. weights from pretext to downstream
    # i+1 because of the Time Distributed layer
    for i, j in zip(conv_range_downstream, conv_range_pretext):
        model_downstream.layers[i].set_weights(model_pretext.layers[j].get_weights())
        # print(model_downstream.layers[i])
        # print(model_pretext.layers[i])

    # freeze conv. weights from pretext to downstream
    for i in conv_range_freeze:
        model_downstream.layers[i].trainable = False
        # print(model_downstream.layers[i])

    # compile the new model, because we froze the conv. layers
    model_downstream.compile(optimizer=optimizer, loss='huber_loss')

    return model_downstream


def scheduler(epoch, lr):
    """
    Schedules the learning rate during training.
    """
    if epoch <= (0.1 * epochs):
        return lr
    else:
        return lr - lr_decay


def select_fold(array, f, leave_out):
    """
    Selects training and testing folds for cross-validation.
    """
    array = np.roll(array, -leave_out * (f - 1))
    train = array[leave_out:]
    test = array[:leave_out]
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
pNo = data['pNo']
roundNo = data['roundNo']
numChannels = x_spect.shape[1]
num_freq_steps = x_spect.shape[2]
num_time_steps = x_spect.shape[3]

# use scaling info of pretext task
scaling_info = np.load('data/data_pretext_scaling_info_gyro_spectro.npz')
mean = scaling_info['mean']
std = scaling_info['std']
x_spect = (x_spect - mean) / std

x_spect = np.moveaxis(x_spect, source=1, destination=-1)

# import the gyroscope raw data
data = np.load('data/data_gyro_raw.npz')
x_raw = data['x_raw']
y_raw = data['y_raw']
window_length = x_raw.shape[1]

# use scaling info of pretext task
scaling_info = np.load('data/data_pretext_scaling_info_gyro_raw.npz')
mean = scaling_info['mean']
std = scaling_info['std']
x_raw = (x_raw - mean) / std


print('Number of folds: ' + str(num_folds))
for fold in num_fold:
    print('Fold number ' + str(fold))
    sub_tune, sub_valid = select_fold(sub_array, fold, num_sub_leave_out)
    print('Tuning / Training subjects: ', sub_tune)
    print('Testing subjects: ', sub_valid)

    # tuning data
    x_tune_raw = x_raw[np.isin(pNo, sub_tune)]
    x_tune_spect = x_spect[np.isin(pNo, sub_tune)]
    x_tune = {'input_raw': x_tune_raw, 'input_spect': np.expand_dims(x_tune_spect, axis=1)}
    y_tune = y_raw[np.isin(pNo, sub_tune)]

    # validation data
    x_test_raw = x_raw[np.isin(pNo, sub_valid)]
    x_test_spect = x_spect[np.isin(pNo, sub_valid)]
    x_test = {'input_raw': x_test_raw, 'input_spect': np.expand_dims(x_test_spect, axis=1)}
    y_test = y_raw[np.isin(pNo, sub_valid)]

    model = func_downstream_model(fold)

    # creating required model callbacks
    callback_scheduler = LearningRateScheduler(scheduler)
    callback_stopping = EarlyStopping(monitor='loss', patience=5)       # to use for testing

    history = model.fit(x=x_tune, y=y_tune,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[callback_stopping, callback_scheduler],
                        verbose=1)

    # save the training history
    if save_history:
        model_history = history.history
        with open('results/model_train_history_f_' + str(fold) + '.pkl', 'wb') as file:
            pickle.dump(model_history, file)

    # calculate model testing metrics
    print('Testing metrics...')
    model = load_model('models/model_downstream_best.h5')
    y_predicted = model.predict(x_test)
    y_test_all = np.concatenate((y_test_all, y_test), axis=0)
    y_predicted_all = np.concatenate((y_predicted_all, y_predicted[:, 0]), axis=0)

    clear_session()  # close the session for training based on this fold


if save_predictions:
    model_predictions = [y_test_all, y_predicted_all, pNo, roundNo]
    with open(SaveDirParent + 'model_predictions.pkl', 'wb') as file:
        pickle.dump(model_predictions, file)

# calculate and print elapsed time
elapsed_time = format((time.time() - start_time) / 60, '.2f')
print("Elapsed time: " + str(elapsed_time) + " minutes")
