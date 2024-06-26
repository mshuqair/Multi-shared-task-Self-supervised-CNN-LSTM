import matplotlib.pyplot as plt
import numpy as np
import pickle


# Code to plot the training history from the pretext task
# Built for 4 tasks


def plot_loss_curve(loss_train, epoch):
    plt.figure(figsize=(6.5, 3))
    plt.plot(epoch, loss_train)
    plt.grid(visible=True, which='major', axis='both', linestyle='--')
    plt.ylim([-0.01, 0.5])
    plt.xlabel(xlabel='Epoch#', fontsize='large')
    plt.ylabel(ylabel='Cross-entropy', fontsize='large')
    plt.title(label='Total network loss', fontsize='x-large', weight='bold')
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.tight_layout()
    if save_figure:
        plt.savefig('./figures/pretext_network_loss.png')
    plt.show()


def plot_loss_curves(loss_train, epoch):
    plt.figure(figsize=(6.5, 3))
    for i in range(loss_train.shape[0]):
        plt.plot(epoch, loss_train[i, :])
    plt.legend(['Original', 'Rotation', 'Permutation', 'Time-Warping'])
    plt.grid(visible=True, which='major', axis='both', linestyle='--')
    plt.ylim([-0.01, 0.5])
    plt.xlabel('Epoch#', fontsize='large')
    plt.ylabel('Cross-entropy', fontsize='large')
    plt.title('Tasks loss', fontsize='x-large', weight='bold')
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.tight_layout()
    if save_figure:
        plt.savefig('./figures/pretext_tasks_loss.png')
    plt.show()


def plot_acc_curves(acc_train, epoch):
    plt.figure(figsize=(6.5, 3))
    for i in range(acc_train.shape[0]):
        plt.plot(epoch, acc_train[i, :])
    plt.legend(['Original', 'Rotation', 'Permutation', 'Time-Warping'])
    plt.grid(visible=True, which='major', axis='both', linestyle='--')
    plt.ylim([0.79, 1.01])
    plt.xlabel('Epoch#', fontsize='large')
    plt.ylabel('Accuracy', fontsize='large')
    plt.title('Tasks accuracy', fontsize='x-large', weight='bold')
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.tight_layout()
    if save_figure:
        plt.savefig('./figures/pretext_tasks_accuracy.png')
    plt.show()


# Main code
save_figure = True
with open('models/model_pretext_multitask_train_history_gyro_raw_spectro.pkl', 'rb') as file:
    model_history = pickle.load(file)

loss_model = np.array(model_history['loss'])
loss_tasks = np.array([model_history['output_task_1_loss'], model_history['output_task_2_loss'],
                       model_history['output_task_3_loss'], model_history['output_task_4_loss'],])
accuracy_tasks = np.array([model_history['output_task_1_accuracy'], model_history['output_task_2_accuracy'],
                           model_history['output_task_3_accuracy'], model_history['output_task_4_accuracy']])

# plotting curves
epochs = np.arange(1, loss_model.shape[0]+1)
plot_loss_curve(loss_model, epochs)
plot_loss_curves(loss_tasks, epochs)
plot_acc_curves(accuracy_tasks, epochs)
