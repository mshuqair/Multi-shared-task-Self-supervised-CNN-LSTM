import sys
from keras import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.initializers.initializers_v2 import GlorotUniform
from keras.layers import (Concatenate, Dense, Dropout, GlobalMaxPooling2D, Conv2D, MaxPooling2D, GlobalAveragePooling1D,
                          Conv1D, MaxPooling1D, Input)
from keras.optimizers import Adam
import data_pretext_generate_raw
import data_pretext_generate_spectrogram
import numpy as np
import time
import pickle


# Pretext task
# Multi-shared-task learning
# Up to 7 tasks in total
# Gyroscope and Gyroscope Spectrogram data is used
# The model is a 1D + 2D CNN


# learning rate scheduler decay function
def scheduler(epoch, lr):
    if epoch <= (0.1 * epochs):
        lr_new = lr
    else:
        lr_new = lr - lr_decay
    return lr_new


def func_model(num_tasks):
    # model parameters
    optimizer = Adam(learning_rate=0.0001)
    dropout_prob_1 = 0.1
    dropout_prob_2 = 0.2

    # classification layers
    dense_units = 128

    # This is for the Gyroscope data part
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
    layer_input_1 = Input(shape=data_shape, name='input_1')

    # convolutional block layers
    layer_cnn_1_a = (Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                            kernel_initializer=kernel_initializer)(layer_input_1))
    layer_cnn_1_b = (Conv1D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method,
                            kernel_initializer=kernel_initializer)(layer_cnn_1_a))
    layer_dropout_1 = Dropout(dropout_prob_1)(layer_cnn_1_b)
    layer_pooling_1 = MaxPooling1D(pool_size=pool_size, strides=pool_strides)(layer_dropout_1)

    layer_cnn_2_a = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_pooling_1)
    layer_cnn_2_b = Conv1D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method,
                           kernel_initializer=kernel_initializer)(layer_cnn_2_a)
    layer_dropout_2 = Dropout(dropout_prob_2)(layer_cnn_2_b)
    layer_global_pooling = GlobalAveragePooling1D()(layer_dropout_2)
    layer_final_1 = layer_global_pooling

    # This is for spectrogram
    # model setup
    # input / output setup
    data_shape = (num_freq_steps, num_time_steps, numChannels)

    # conv. blocks hyperparameters
    conv1_filters = 64
    conv2_filters = 128
    conv1_kernel = (5, 5)
    conv2_kernel = (3, 3)
    pool_size = (2, 2)
    pool_strides = None

    # input layer
    layer_input_2 = Input(shape=data_shape, name='input_2')

    # convolutional block layers
    layer_cnn_1_a = (Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method)
                     (layer_input_2))
    layer_cnn_1_b = (Conv2D(filters=conv1_filters, kernel_size=conv1_kernel, activation='relu', padding=padding_method)
                     (layer_cnn_1_a))
    layer_dropout_1 = Dropout(dropout_prob_1)(layer_cnn_1_b)
    layer_pooling_1 = MaxPooling2D(pool_size=pool_size, strides=pool_strides)(layer_dropout_1)

    layer_cnn_2_a = (Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method)
                     (layer_pooling_1))
    layer_cnn_2_b = (Conv2D(filters=conv2_filters, kernel_size=conv2_kernel, activation='relu', padding=padding_method)
                     (layer_cnn_2_a))
    layer_dropout_2 = Dropout(dropout_prob_2)(layer_cnn_2_b)
    layer_global_pooling = GlobalMaxPooling2D()(layer_dropout_2)
    layer_final_2 = layer_global_pooling

    # classification layers
    layer_task_1 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_1 = Dense(units=num_classes, activation='sigmoid', name='output_task_1')(layer_task_1)
    layer_task_2 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_2 = Dense(units=num_classes, activation='sigmoid', name='output_task_2')(layer_task_2)
    layer_task_3 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_3 = Dense(units=num_classes, activation='sigmoid', name='output_task_3')(layer_task_3)
    layer_task_4 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_4 = Dense(units=num_classes, activation='sigmoid', name='output_task_4')(layer_task_4)
    layer_task_5 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_5 = Dense(units=num_classes, activation='sigmoid', name='output_task_5')(layer_task_5)
    layer_task_6 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_6 = Dense(units=num_classes, activation='sigmoid', name='output_task_6')(layer_task_6)
    layer_task_7 = Dense(units=dense_units, activation='relu')(Concatenate()([layer_final_1, layer_final_2]))
    layer_output_7 = Dense(units=num_classes, activation='sigmoid', name='output_task_7')(layer_task_7)

    if num_tasks == 7:
        outputs = [layer_output_1, layer_output_2, layer_output_3, layer_output_4, layer_output_5, layer_output_6,
                   layer_output_7]
    elif num_tasks == 6:
        outputs = [layer_output_1, layer_output_2, layer_output_3, layer_output_4, layer_output_5, layer_output_6]
    elif num_tasks == 5:
        outputs = [layer_output_1, layer_output_2, layer_output_3, layer_output_4, layer_output_5]
    elif num_tasks == 4:
        outputs = [layer_output_1, layer_output_2, layer_output_3, layer_output_4]
    elif num_tasks == 3:
        outputs = [layer_output_1, layer_output_2, layer_output_3]
    elif num_tasks == 2:
        outputs = [layer_output_1, layer_output_2]
    else:
        print('Invalid number of tasks selected')
        sys.exit()

    inputs = [layer_input_1, layer_input_2]

    # assign weights to the loss of each task based on its difficulty
    # larger weights are assigned for harder tasks
    # equal weights for all tasks is the default
    # loss_weights = None
    loss_weights = [0.3, 0.2, 0.3, 0.2]

    # construct the model
    model = Model(inputs=inputs, outputs=outputs)

    # compile and model summary
    model.compile(optimizer=optimizer, metrics='accuracy', loss='binary_crossentropy', loss_weights=loss_weights)
    model.summary(show_trainable=True)

    return model


def select_fold(array, f, leave_out):
    array = np.roll(array, -leave_out * (f - 1))
    tune = array[leave_out:]
    valid = array[0:leave_out]
    return tune, valid


# Main code
start_time = time.time()

# saving parameters
save_history = True  # whether to save the training history in pickle file or not
save_model = True  # whether to save the trained model in .h5 format

# signal parameters
fs = 64  # sampling frequency of the original signal
window_width = 5  # window size in seconds, 5 for gyro and 60 for spectrogram
window_length = int(fs * window_width)  # window size in samples
overlap = 0  # overlap between windows, => 4s ==> rate = 4/5 = 0.8

# training settings
epochs = 30
batch_size = 32
learning_rate = 0.0001
lr_decay = learning_rate / epochs
kernel_initializer = GlorotUniform(seed=1)

# subjects for cross-testing
sub_array = np.arange(start=1, stop=25, dtype=int)  # the patient number at stop is not included
num_sub_leave_out = 1  # number of subjects to leave out for cross-testing
num_folds = int(sub_array.shape[0] / num_sub_leave_out)  # number of folds, always integer
num_fold = np.arange(1, num_folds + 1)

# list of tasks for multitask learning (at least two tasks to be selected):
# 'Original', 'Jitter', 'Scaling', 'Rotation', 'Permutation', 'Time-Warping', 'Magnitude-Warping'
list_da = ['Original', 'Rotation', 'Permutation', 'Time-Warping']

print('Number of folds: ' + str(num_folds))
for fold in num_fold:
    print('Fold number ' + str(fold))
    sub_train, sub_test = select_fold(sub_array, fold, num_sub_leave_out)
    print('Training subjects: ', sub_train)
    print('Testing subjects: ', sub_test)

    # loading and scaling gyro raw data
    x_gyro, y_gyro, mean_raw, std_raw = data_pretext_generate_raw.prepare_data_gyro(window_length=window_length,
                                                                                    overlap=overlap,
                                                                                    sub_array=sub_train,
                                                                                    list_da=list_da)
    x_gyro = (x_gyro - mean_raw) / std_raw
    print('Shape of Pretext Gyro Raw data:', x_gyro.shape)

    # loading and scaling gyro spectrogram data
    x_spect, y_train_spect, mean_spect, std_spect = data_pretext_generate_spectrogram.prepare_data_gyro(
        window_length=window_length, overlap=overlap, sub_array=sub_train, list_da=list_da)
    x_spect = (x_spect - mean_spect) / std_spect
    numChannels = x_spect.shape[1]
    num_freq_steps = x_spect.shape[2]
    num_time_steps = x_spect.shape[3]
    print('Shape of Pretext Gyro Spectro data:', x_spect.shape)

    # move axis from (None, channels, f, t) to (None, f, t, channels) to make it compatible with model input
    x_spect = np.moveaxis(x_spect, source=1, destination=-1)

    # saving scaling info
    print('Saving the scaling info...')
    np.savez(file='data/data_pretext_scaling_info_gyro_raw.npz', mean=mean_raw, std=std_raw)
    np.savez(file='data/data_pretext_scaling_info_gyro_spectro.npz', mean=mean_spect, std=std_spect)

    # arrange and remap labels to [0, 1] for multitask learning
    y = {}
    for task in range(1, len(list_da) + 1):
        y['output_task_' + str(task)] = np.where(y_gyro == task, 1, 0)

    # arrange inputs
    x = {'input_1': x_gyro, 'input_2': x_spect}

    # loading pretext model
    model_pretext = func_model(num_tasks=len(list_da))
    model_pretext.summary()

    # creating a learning rate scheduler
    callback_scheduler = LearningRateScheduler(scheduler)

    # training the model
    history = model_pretext.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                                callbacks=[callback_scheduler], verbose=1)

    # save the training history
    if save_history:
        model_history = history.history
        with open('models/model_pretext_multitask_train_history_gyro_raw_spectro_'+str(fold)+'.pkl', 'wb') as file:
            pickle.dump(model_history, file)

    # save the trained model
    if save_model:
        model_pretext.save('models/model_pretext_multitask_trained_gyro_raw_spectro_ '+str(fold)+'.h5')

# calculate and print elapsed time
elapsed_time = format((time.time() - start_time) / 60, '.2f')
print("Elapsed time: " + str(elapsed_time) + " minutes")
