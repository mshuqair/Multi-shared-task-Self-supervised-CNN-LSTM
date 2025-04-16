import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def load_predictions(file_path):
    """Load model predictions from a pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def aggregate_predictions(y, y_predicted, pNo, roundNo):
    """Aggregate predictions and labels per round."""
    unique_rounds = np.unique(roundNo)
    y_rounds, y_pred_rounds, pNo_rounds = [], [], []

    for r in unique_rounds:
        mask = roundNo == r
        y_rounds.append(np.mean(y[mask]))
        pred_mean = np.rint(np.clip(np.mean(y_predicted[mask]), a_min=0, a_max=None))
        y_pred_rounds.append(pred_mean)
        pNo_rounds.append(np.unique(pNo[mask])[0])

    return np.array(y_rounds), np.array(y_pred_rounds), np.array(pNo_rounds)


def compute_mae_by_score(y_true, y_pred, scores):
    """Compute MAE for each UPDRS-III score."""
    mae = np.zeros(len(scores))
    for i, score in enumerate(scores):
        mask = y_true == score
        mae[i] = mean_absolute_error(y_true[mask], y_pred[mask])
    return mae


def plot_bar_mae(scores, mae_dict, save_path=None):
    """Plot bar chart of MAEs."""
    width = 0.75
    colors = ['tab:blue', 'tab:olive']
    fig, ax = plt.subplots(figsize=(6.5, 3))
    multiplier = 0
    z_order = 3
    for i, (label, mae_values) in enumerate(mae_dict.items()):
        offset = width * multiplier
        ax.bar(scores + offset, mae_values, width=width, label=label,
               color=colors[i], zorder=z_order)
        multiplier += 0.75
        z_order -= 1

    ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_ylim(0, 21)
    ax.set_xlim(-1, 64)
    ax.set_yticks(np.arange(0, 25, 5))
    ax.set_xticks(np.arange(0, 70, 10))
    ax.set_ylabel('Mean absolute error', fontsize='large')
    ax.set_xlabel('UPDRS-III scores', fontsize='large')
    ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_box_mae(mae_ssl, mae_cnn, save_path=None):
    """Plot box plot of MAEs."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    box = ax.boxplot([mae_ssl, mae_cnn], tick_labels=['M-SSL', 'Supervised'],
                     widths=0.3, patch_artist=True, showfliers=False, zorder=3)

    for patch, color in zip(box['boxes'], ['tab:blue', 'tab:olive']):
        patch.set_facecolor(color)

    ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_ylim(-1, 21)
    ax.set_yticks(np.arange(0, 25, 2.5))
    ax.set_ylabel('Mean absolute error', fontsize='large')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    save_figure = True

    # Supervised CNN-LSTM
    y, y_pred, pNo, roundNo = load_predictions('./results/task_baseline_dual_cnn_lstm/model_predictions.pkl')
    y_rounds_cnn, y_pred_rounds_cnn, _ = aggregate_predictions(y, y_pred, pNo, roundNo)
    mae_cnn = compute_mae_by_score(y_rounds_cnn, y_pred_rounds_cnn, np.unique(y_rounds_cnn))

    # M-SSL CNN-LSTM
    y, y_pred, pNo, roundNo = load_predictions('./results/task_downstream_dual_cnn_lstm/model_predictions.pkl')
    y_rounds_ssl, y_pred_rounds_ssl, _ = aggregate_predictions(y, y_pred, pNo, roundNo)
    mae_ssl = compute_mae_by_score(y_rounds_ssl, y_pred_rounds_ssl, np.unique(y_rounds_ssl))

    # Plotting
    mae_dict = {
        'M-SSL': mae_ssl,
        'Supervised': mae_cnn
    }
    plot_bar_mae(np.unique(y_rounds_cnn), mae_dict,
                 save_path='./figures/mae_bars.png' if save_figure else None)

    plot_box_mae(mae_ssl, mae_cnn,
                 save_path='./figures/mae_box.png' if save_figure else None)


if __name__ == "__main__":
    main()
