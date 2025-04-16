import matplotlib.pyplot as plt
import numpy as np
import pickle

# Configuration
save_figure = True
history_file = 'models/model_pretext_multitask_train_history_gyro_raw_spectro.pkl'


# Plotting Functions
def plot_loss_curve(loss_train, epoch):
    """Plot total network loss over epochs."""
    plt.figure(figsize=(6.5, 3))
    plt.plot(epoch, loss_train, label='Total Loss', color='tab:blue')
    plt.grid(True, which='major', axis='both', linestyle='--')
    plt.ylim([-0.01, 0.5])
    plt.xlabel('Epoch#', fontsize='large')
    plt.ylabel('Cross-entropy', fontsize='large')
    plt.title('Total Network Loss', fontsize='x-large', weight='bold')
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.tight_layout()
    if save_figure:
        plt.savefig('./figures/pretext_network_loss.png')
    plt.show()


def plot_loss_curves(loss_train, epoch):
    """Plot loss curves for each self-supervised task."""
    plt.figure(figsize=(6.5, 3))
    for i, label in enumerate(['Original', 'Rotation', 'Permutation', 'Time-Warping']):
        plt.plot(epoch, loss_train[i], label=label)
    plt.legend()
    plt.grid(True, which='major', axis='both', linestyle='--')
    plt.ylim([-0.01, 0.5])
    plt.xlabel('Epoch#', fontsize='large')
    plt.ylabel('Cross-entropy', fontsize='large')
    plt.title('Task Losses', fontsize='x-large', weight='bold')
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.tight_layout()
    if save_figure:
        plt.savefig('./figures/pretext_tasks_loss.png')
    plt.show()


def plot_acc_curves(acc_train, epoch):
    """Plot accuracy curves for each self-supervised task."""
    plt.figure(figsize=(6.5, 3))
    for i, label in enumerate(['Original', 'Rotation', 'Permutation', 'Time-Warping']):
        plt.plot(epoch, acc_train[i], label=label)
    plt.legend()
    plt.grid(True, which='major', axis='both', linestyle='--')
    plt.ylim([0.79, 1.01])
    plt.xlabel('Epoch#', fontsize='large')
    plt.ylabel('Accuracy', fontsize='large')
    plt.title('Task Accuracies', fontsize='x-large', weight='bold')
    plt.xticks(fontsize='medium')
    plt.yticks(fontsize='medium')
    plt.tight_layout()
    if save_figure:
        plt.savefig('./figures/pretext_tasks_accuracy.png')
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Load training history
    with open(history_file, 'rb') as file:
        model_history = pickle.load(file)

    # Extract losses and accuracies
    loss_model = np.array(model_history['loss'])
    loss_tasks = np.array([
        model_history['output_task_1_loss'],
        model_history['output_task_2_loss'],
        model_history['output_task_3_loss'],
        model_history['output_task_4_loss']
    ])
    accuracy_tasks = np.array([
        model_history['output_task_1_accuracy'],
        model_history['output_task_2_accuracy'],
        model_history['output_task_3_accuracy'],
        model_history['output_task_4_accuracy']
    ])

    # Define epochs
    epochs = np.arange(1, loss_model.shape[0] + 1)

    # Plot
    plot_loss_curve(loss_model, epochs)
    plot_loss_curves(loss_tasks, epochs)
    plot_acc_curves(accuracy_tasks, epochs)
