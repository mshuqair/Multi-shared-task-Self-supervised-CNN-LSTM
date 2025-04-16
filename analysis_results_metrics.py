import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


def load_predictions(file_path):
    """Load predictions from a pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def compute_aggregated_metrics(y, y_predicted, pNo, roundNo):
    """Aggregate predictions and compute metrics per round."""
    unique_rounds = np.unique(roundNo)
    y_rounds, y_predicted_rounds, pNo_rounds = [], [], []

    for round_id in unique_rounds:
        round_mask = roundNo == round_id
        y_rounds.append(np.mean(y[round_mask]))
        mean_pred = np.mean(y_predicted[round_mask])
        y_predicted_rounds.append(np.rint(np.clip(mean_pred, a_min=0, a_max=None)))
        pNo_rounds.append(np.unique(pNo[round_mask])[0])  # assumes one patient per round

    return (
        np.array(y_rounds),
        np.array(y_predicted_rounds),
        np.array(pNo_rounds)
    )


def print_metrics(y_true, y_pred, model_name):
    """Print regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, p_val = pearsonr(y_true, y_pred)

    print(f"{model_name} Metrics:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.2f}")
    print(f"  Correlation: {corr:.2f} (p = {p_val:.4f})\n")


def main():
    print("Regression metrics for the M-SSL CNN-LSTM:")
    file_mssl = './results/task_downstream_dual_cnn_lstm/model_predictions.pkl'
    y, y_predicted, pNo, roundNo = load_predictions(file_mssl)
    y_rounds, y_pred_rounds, _ = compute_aggregated_metrics(y, y_predicted, pNo, roundNo)
    print_metrics(y_rounds, y_pred_rounds, model_name="M-SSL CNN-LSTM")

    print("Regression metrics for the Supervised CNN-LSTM:")
    file_supervised = './results/task_baseline_dual_cnn_lstm/model_predictions.pkl'
    _, y_predicted_sup, _, roundNo_sup = load_predictions(file_supervised)

    y_pred_rounds_sup = []
    for round_id in np.unique(roundNo_sup):
        preds = y_predicted_sup[roundNo_sup == round_id]
        y_pred_rounds_sup.append(np.rint(np.clip(np.mean(preds), a_min=0, a_max=None)))

    y_pred_rounds_sup = np.array(y_pred_rounds_sup)
    print_metrics(y_rounds, y_pred_rounds_sup, model_name="Supervised CNN-LSTM")


if __name__ == "__main__":
    main()
